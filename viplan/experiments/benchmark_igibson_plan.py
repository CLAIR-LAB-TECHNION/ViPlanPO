import os
import copy
import fire
import json
import torch
import random
import transformers

from collections import deque

from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from viplan.code_helpers import get_logger, parse_output, get_unique_id
from viplan.planning.planning_utils import get_plan, update_problem
from viplan.policies.policy_interface import (
    PolicyObservation,
    resolve_policy_class,
)
from viplan.policies.policy_plan import DefaultPlanningPolicy, predicate_questions, get_questions, cast_to_yes_no, \
    update_vlm_state, check_action
from viplan.log_utils import get_img_output_dir
from viplan.planning.igibson_client_env import iGibsonClient


# State enumeration

def get_enumeration_results(env, model, questions, base_prompt, logger, batch_size=64):
    responses = []
    base_prompt = open(base_prompt, 'r').read()
    q_list = [ base_prompt + "\n" + question[0] for question in questions.values() ]
    
    for i in range(0, len(questions), batch_size):
        logger.info(f"Processing questions {i} to {min(i + batch_size, len(q_list))} of {len(q_list)}")
        batch_prompts = q_list[i:i + min(batch_size, len(q_list) - i)]
        img = env.render()
        batch_images = [img] * len(batch_prompts)
        batch_responses = model.generate(prompts=batch_prompts, images=batch_images, return_probs=True)
        responses.extend(batch_responses)
        
    assert len(responses) == len(q_list), "Some answers were not generated."
    
    # Convert to a format that works with update_vlm_state
    results = {}
    for i, (question, response) in enumerate(zip(questions, responses)):
        logger.debug(f"Question: {question}, Response: {response}")
        question_str = questions[question][0]
        answer = 'yes' if questions[question][1] else 'no'
        response, tags_found = parse_output(response[0], answer_tags=["answer", "explanation"])
        parsed_answer = response['answer'] if tags_found and 'answer' in response else response
        parsed_answer = parsed_answer.strip().lower().rstrip('.,!?') if isinstance(parsed_answer, str) else parsed_answer
        parsed_answer = cast_to_yes_no(parsed_answer, logger)
        logger.debug(f"Parsed answer: {parsed_answer}")
        
        results[question] = (parsed_answer, answer.lower())
    
    return results

def compute_enumeration_metrics(results):
    enum_results = {
        'accuracy': 0,
        'yes_accuracy': 0,
        'yes_correct': 0,
        'yes_total': 0,
        'no_accuracy': 0,
        'no_correct': 0,
        'no_total': 0,
        'predicates': {}
    }
    # For enumeration, there is no expected value from the PDDL model (as we're testing everything), so the accuracy is already based on the environment
    for question, (answer, expected_answer) in results.items():
        if answer == expected_answer:
            enum_results['accuracy'] += 1
            if expected_answer == 'yes':
                enum_results['yes_correct'] += 1
            else:
                enum_results['no_correct'] += 1
                
        if expected_answer == 'yes':
            enum_results['yes_total'] += 1
        else:
            enum_results['no_total'] += 1
        
        predicate = question.split(" ")[0]
        if predicate not in enum_results['predicates']:
            enum_results['predicates'][predicate] = {
                'accuracy': 0,
                'yes_accuracy': 0,
                'yes_correct': 0,
                'yes_total': 0,
                'no_accuracy': 0,
                'no_correct': 0,
                'no_total': 0
            }
        if answer == expected_answer:
            enum_results['predicates'][predicate]['accuracy'] += 1
            if expected_answer == 'yes':
                enum_results['predicates'][predicate]['yes_correct'] += 1
            else:
                enum_results['predicates'][predicate]['no_correct'] += 1
                
        if expected_answer == 'yes':
            enum_results['predicates'][predicate]['yes_total'] += 1
        else:
            enum_results['predicates'][predicate]['no_total'] += 1
    
    enum_results['accuracy'] /= len(results)
    enum_results['yes_accuracy'] = (enum_results['yes_correct'] / enum_results['yes_total']) if enum_results['yes_total'] > 0 else None
    enum_results['no_accuracy'] = (enum_results['no_correct'] / enum_results['no_total']) if enum_results['no_total'] > 0 else None
    
    for predicate in enum_results['predicates']:
        enum_results['predicates'][predicate]['accuracy'] /= (enum_results['predicates'][predicate]['yes_total'] + enum_results['predicates'][predicate]['no_total'])
        enum_results['predicates'][predicate]['yes_accuracy'] = (enum_results['predicates'][predicate]['yes_correct'] / enum_results['predicates'][predicate]['yes_total']) if enum_results['predicates'][predicate]['yes_total'] > 0 else None
        enum_results['predicates'][predicate]['no_accuracy'] = (enum_results['predicates'][predicate]['no_correct'] / enum_results['predicates'][predicate]['no_total']) if enum_results['predicates'][predicate]['no_total'] > 0 else None
        
    return enum_results


def check_plan(env,
               plan,
               problem,
               vlm_state, # Initial state as perceived by the VLM
               model,
               base_prompt,
               logger,
               img_log_info,
               replan=False,
               text_only=False,
               max_actions=20,
               enumerate_replan=False,
               enum_batch_size=64,
               policy_class=DefaultPlanningPolicy,
               policy_kwargs=None):

    all_correct = True
    results = []
    replans = []

    def instantiate_policy(current_plan):
        queue = deque(current_plan.actions)
        kwargs = dict(policy_kwargs or {})
        kwargs.setdefault('predicate_language', predicate_questions)
        kwargs.setdefault('logger', logger)
        kwargs.setdefault('problem', problem)
        kwargs.setdefault('model', model)
        kwargs.setdefault('base_prompt', base_prompt)
        kwargs['plan'] = current_plan
        kwargs['action_queue'] = queue
        policy = policy_class(**kwargs)
        return policy, queue

    policy, action_queue = instantiate_policy(plan)
    most_recent_action = None

    while len(results) < max_actions:
        if not action_queue:
            logger.info('All actions completed')
            break

        observation = PolicyObservation(
            image=env.render(),
            problem=problem,
            predicate_language=policy.predicate_language,
            predicate_groundings=env.visible_predicates,
            previous_actions=[
                {'action': r['action'], 'success': r.get('action_state_correct')}
                for r in results
            ],
            context={'vlm_state': copy.deepcopy(vlm_state)},
        )

        policy_action = policy.next_action(observation)
        if policy_action is None:
            logger.info('Policy returned no action; stopping episode')
            break

        action = policy_action.metadata.get('plan_action')
        if action is None:
            logger.warning('Policy did not provide plan_action metadata; using queue order')
            action = action_queue.popleft()
        else:
            try:
                action_queue.remove(action)
            except ValueError:
                pass

        logger.info(f"Applying action {action}")

        try:
            action_correct, preconditions_results, non_visible_precond_results, effects_results, action_state_correct, action_info = check_action(env, action, vlm_state, model, base_prompt, logger, img_log_info, text_only=text_only)
        except Exception as e:
            logger.warning(f"Error while checking action {action}: {e}")
            import traceback
            traceback.print_exc()

            action_correct = False
            preconditions_results = {}
            non_visible_precond_results = {}
            effects_results = None
            action_state_correct = False
            action_info = None
            break

        policy.observe_outcome(
            policy_action,
            success=action_correct,
            info={'preconditions': preconditions_results, 'effects': effects_results},
        )

        results.append({
            'action': str(action),
            'action_correct': action_correct,
            'action_state_correct': action_state_correct,
            'preconditions_results': preconditions_results,
            'non_visible_precond_results': non_visible_precond_results,
            'effects_results': effects_results,
            'action_info': action_info,
        })
        if not action_correct:
            if 'all_correct' in preconditions_results and not preconditions_results['all_correct']:
                reason = 'Preconditions not satisfied'
            elif effects_results is None:
                reason = 'Action was not legal'
            elif not effects_results['all_correct']:
                reason = 'Not all effects were observed as expected'
            else:
                reason = 'Unknown'
            logger.warning(f"Action {action} failed: {reason}")
            if reason == 'Action was not legal' and str(action) == str(most_recent_action):
                logger.warning("Action was not legal, but it was the same as the most recent action. Stopping as we're likely in a loop.")
                break
            try:
                all_correct = False

                if replan:
                    replans.append({})
                    logger.info('Replanning from newly observed state')

                    if enumerate_replan:
                        questions = get_questions(env.visible_predicates)
                        enum_results = get_enumeration_results(env, model, questions, base_prompt, logger, batch_size=enum_batch_size)
                        enum_metrics = compute_enumeration_metrics(enum_results)
                        replans[-1]['enum_results'] = enum_results
                        replans[-1]['enum_metrics'] = enum_metrics

                        vlm_state, changed = update_vlm_state(copy.deepcopy(env.state), enum_results)
                else:
                    break
            except Exception as e:
                logger.warning(f"Error while updating VLM state: {e}")
                import traceback
                traceback.print_exc()
                all_correct = False
                break

            new_problem = update_problem(vlm_state, env.problem)
            plan_result = get_plan(new_problem, logger)
            if plan_result is None:
                logger.warning('Breaking out of episode due to error in the planner')
                break
            elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
                logger.warning('No plan found after replanning')
                break
            else:
                logger.info('Replan found')
                new_plan = plan_result.plan
                policy, action_queue = instantiate_policy(new_plan)
                replans[-1].update({
                    'step': len(results),
                    'actions': [str(a) for a in new_plan.actions]
                })

        if len(results) >= max_actions:
            logger.warning('Max actions reached')
            break

        most_recent_action = copy.deepcopy(action)

    goal_reached = env.goal_reached
    logger.info(f"Goal reached: {goal_reached}")

    if all_correct and not goal_reached:
        logger.warning(f"All actions executed correctly, but goal not reached")

    return all_correct, results, replans, action_queue, goal_reached


def compute_metrics(results, logger):
    # task accuracy = task was fully completed, 
    # action_accuracy = fraction of individual actions that were correctly predicted (in full)
    # predicate_accuracy = fraction of predicates that were correctly predicted
    # macro_predicate_accuracy = fraction of predicates that were correctly predicted, equally weighted independently of the number of predicates
    # fail_ratio = fraction of problems that never had a plan (wrong initial state)

    task_accuracy = sum([results[problem]['goal_reached'] for problem in results if 'goal_reached' in results[problem]]) / len(results)
    problem_stats = {}
    predicate_stats = {}
    
    # Problem stats for action accuracy
    for problem in results:
        problem_stats[problem] = {}
        try:
            # Action accuracy is computed on the actual state correctness (e.g. did the VLM correctly answer as to what it was seeing)
            problem_stats[problem]['action_correct'] = sum([action['action_state_correct'] for action in results[problem]['action_results'] if 'action_state_correct' in action])
            problem_stats[problem]['action_total'] = len([action for action in results[problem]['action_results'] if 'action_state_correct' in action])
            # print(f"Problem: {problem}, actions: {n_actions}, action_accuracy: {action_accuracy}")
            problem_stats[problem]['action_total'] += len(results[problem]['remaining_actions'])
            problem_stats[problem]['remaining_actions'] = results[problem]['remaining_actions']
            problem_stats[problem]['action_accuracy'] = problem_stats[problem]['action_correct'] / problem_stats[problem]['action_total'] if problem_stats[problem]['action_total'] > 0 else 0
            problem_stats[problem]['failed'] = False
            # print(f"Problem: {problem}, actions: {n_actions}, action_accuracy: {action_accuracy} (after remaining actions {len(results[problem]['remaining_actions'])})")
        except Exception as e:
            problem_stats[problem]['action_correct'] = 0
            problem_stats[problem]['action_total'] = 1 # count this as a first action that failed to normalize the metric
            problem_stats[problem]['failed'] = True
            print(f"Problem {problem} had no actions, likely never started due to wrong initial state")
            continue
    logger.debug(f"Problem stats: {problem_stats}")
    
    # Predicate stats for predicate accuracy
    for problem in results:
        # First add the enumeration results
        if 'initial_state_enum' in results[problem]:
            logger.debug(f"Adding initial state enumeration results for problem {problem}")
            for predicate in results[problem]['initial_state_enum']['statistics']['predicates']:
                if predicate not in predicate_stats:
                    predicate_stats[predicate] = {
                        'accuracy': 0,
                        'yes_accuracy': 0,
                        'yes_correct': 0,
                        'yes_total': 0,
                        'no_accuracy': 0,
                        'no_correct': 0,
                        'no_total': 0
                    }
                predicate_stats[predicate]['yes_correct'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['yes_correct']
                predicate_stats[predicate]['yes_total'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['yes_total']
                predicate_stats[predicate]['no_correct'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['no_correct']
                predicate_stats[predicate]['no_total'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['no_total']
        else:
            logger.info(f"No initial state enumeration results for problem {problem}")
        
        # Check if there is enumeration in the replans
        if 'replans' in results[problem] and results[problem]['replans'] is not None:
            logger.debug(f"Adding replan enumeration results for problem {problem}")
            for replan in results[problem]['replans']:
                if 'enum_metrics' in replan:
                    for predicate in replan['enum_metrics']['predicates']:
                        if predicate not in predicate_stats:
                            predicate_stats[predicate] = {
                                'accuracy': 0,
                                'yes_accuracy': 0,
                                'yes_correct': 0,
                                'yes_total': 0,
                                'no_accuracy': 0,
                                'no_correct': 0,
                                'no_total': 0
                            }
                        predicate_stats[predicate]['yes_correct'] += replan['enum_metrics']['predicates'][predicate]['yes_correct']
                        predicate_stats[predicate]['yes_total'] += replan['enum_metrics']['predicates'][predicate]['yes_total']
                        predicate_stats[predicate]['no_correct'] += replan['enum_metrics']['predicates'][predicate]['no_correct']
                        predicate_stats[predicate]['no_total'] += replan['enum_metrics']['predicates'][predicate]['no_total']
        else:
            logger.info(f"No replan enumeration results for problem {problem}")

            def update_predicate_stats(predicate_stats, results):
                for key, res in results.items():
                    if key in ('all_correct', 'all_state_correct'):
                        continue
                    predicate = key.split(" ")[0]
                    model_answer = res[0]
                    state_correct = res[6]
                    if predicate not in predicate_stats:
                        predicate_stats[predicate] = {
                            'accuracy': 0,
                            'yes_accuracy': 0,
                            'yes_correct': 0,
                            'yes_total': 0,
                            'no_accuracy': 0,
                            'no_correct': 0,
                            'no_total': 0
                        }
                    if model_answer == 'yes':
                        if state_correct:
                            predicate_stats[predicate]['yes_correct'] += 1
                        predicate_stats[predicate]['yes_total'] += 1
                    elif model_answer == 'no':
                        if not state_correct:
                            predicate_stats[predicate]['no_correct'] += 1
                        predicate_stats[predicate]['no_total'] += 1
                    else:
                        logger.warning(f"Unexpected answer {model_answer} for predicate {predicate}")
                        continue

                for action in results[problem]['action_results']:
                    update_predicate_stats(predicate_stats, action.get('preconditions_results', {}))
                    update_predicate_stats(predicate_stats, action.get('effects_results', {}))
        
    action_accuracy = sum([problem_stats[problem]['action_correct'] for problem in problem_stats]) / sum([problem_stats[problem]['action_total'] for problem in problem_stats])
    fail_ratio = sum([problem_stats[problem]['failed'] for problem in problem_stats]) / len(problem_stats)
            
    for predicate in predicate_stats:
        predicate_stats[predicate]['correct'] = predicate_stats[predicate]['yes_correct'] + predicate_stats[predicate]['no_correct']
        predicate_stats[predicate]['total'] = predicate_stats[predicate]['yes_total'] + predicate_stats[predicate]['no_total']
        predicate_stats[predicate]['accuracy'] = predicate_stats[predicate]['correct'] / predicate_stats[predicate]['total'] if predicate_stats[predicate]['total'] > 0 else 0
        
        predicate_stats[predicate]['yes_accuracy'] = predicate_stats[predicate]['yes_correct'] / predicate_stats[predicate]['yes_total'] if predicate_stats[predicate]['yes_total'] > 0 else 0
        predicate_stats[predicate]['no_accuracy'] = predicate_stats[predicate]['no_correct'] / predicate_stats[predicate]['no_total'] if predicate_stats[predicate]['no_total'] > 0 else 0
    
    logger.debug(f"Predicate stats: {predicate_stats}")
    predicate_accuracy = sum([predicate_stats[predicate]['correct'] for predicate in predicate_stats]) / sum([predicate_stats[predicate]['total'] for predicate in predicate_stats])
    macro_predicate_accuracy = sum([predicate_stats[predicate]['accuracy'] for predicate in predicate_stats]) / len(predicate_stats) if len(predicate_stats) > 0 else 0

    return predicate_accuracy, macro_predicate_accuracy, action_accuracy, task_accuracy, problem_stats, predicate_stats, fail_ratio    

def main(
    problems_dir: os.PathLike,
    domain_file: os.PathLike,
    model_name: str,
    prompt_path: os.PathLike,
    base_url: str,
    seed: int = 1,
    output_dir: os.PathLike = None,
    hf_cache_dir: os.PathLike = None,
    log_level ='info',
    replan: bool = True, # Try to replan if an action fails
    text_only: bool = False, # Instead of asking based on the image, ask based on a textual description
    fail_probability: float = 0.0, # Probability of action failure
    enumerate_initial_state: bool = False, # Enumerate initial state predicates (instead of using oracle)
    enumerate_replan: bool = True, # Enumerate predicates before replanning if there is a failure
    enum_batch_size: int = 64, # Batch size for enumeration
    max_steps: int = 20, # Max number of steps to take in the environment
    policy_cls: str = None,
    **kwargs):
    
    # Ensure deterministic behavior (in theory)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    
    logger = get_logger(log_level=log_level)
    unique_id = get_unique_id(logger)
        
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", None)
        logger.debug(f"Using HF cache dir: {hf_cache_dir}")
        
    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts

    # Load model
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}." if torch.cuda.is_available() else "Using CPU.")
    if torch.cuda.is_available() and ('A100' in torch.cuda.get_device_name() or 'H100' in torch.cuda.get_device_name() or 'H200' in torch.cuda.get_device_name()):
        use_flash_attn = True
    else:
        use_flash_attn = False
    logger.info(f"Use flash attention: {use_flash_attn}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if text_only:
        logger.info("Using LLM for text-only evaluation")
        from viplan.models import HuggingFaceLLM
        model = HuggingFaceLLM(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=0, device=device, dtype=dtype, use_flash_attention=use_flash_attn, **kwargs)
    else:
        from viplan.code_helpers import load_vlm
        model = load_vlm(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=0, device=device, dtype=dtype, use_flash_attention=use_flash_attn, **kwargs)
    logger.info(f"Loaded model {model_name} on device {device} with dtype {dtype}")
    
    results = {}
    metadata = os.path.join(problems_dir, "metadata.json")
    assert os.path.exists(metadata), f"Metadata file {metadata} not found"
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    problem_files = [problem for problem in metadata.keys()]
    problem_files = [f"{problems_dir}/{problem}" for problem in problem_files]
    
    PolicyClass = resolve_policy_class(policy_cls, DefaultPlanningPolicy)
    policy_kwargs = {'predicate_language': predicate_questions}

    for problem_file in problem_files:
        logger.info(f"Loading problem {problem_file}")
        
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        task = metadata[os.path.basename(problem_file)]['activity_name']
        scene_instance_pairs = metadata[os.path.basename(problem_file)]['scene_instance_pairs']
        
        for scene_id, instance_id in scene_instance_pairs:

            results[f"{problem_file}_{scene_id}_{instance_id}"] = {}
            logger.info(f"Loading problem {problem_file}")
            try:
                env = iGibsonClient(task=task, scene_id=scene_id, instance_id=instance_id, problem=problem, base_url=base_url, logger=logger)
            except Exception as e:
                logger.error(f"Could not load problem {problem_file}: {e}")
                results[f"{problem_file}_{scene_id}_{instance_id}"] = {
                    'all_correct': False,
                    'action_results': None,
                    'replans': None,
                    'remaining_actions': None
                }
                continue
            
            if enumerate_initial_state:
                raise NotImplementedError("Initial state enumeration not implemented for partial observable envs")
            else:
                vlm_state = copy.deepcopy(env.state) # Oracle state
            
            plan_result = get_plan(problem, logger)
            if plan_result is None:
                logger.warning("Breaking out of episode due to error in the planner")
                results[f"{problem_file}_{scene_id}_{instance_id}"].update({
                    'all_correct': False,
                    'action_results': None,
                    'replans': None,
                    'remaining_actions': None
                })
                continue
            elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
                logger.warning("No plan found.")
                results[f"{problem_file}_{scene_id}_{instance_id}"].update({
                    'all_correct': False,
                    'action_results': None,
                    'replans': None,
                    'remaining_actions': None
                })
                continue
            plan = plan_result.plan

            img_output_dir = get_img_output_dir('plan', instance_id, scene_id, task)
            img_log_info = {
                'img_output_dir': img_output_dir,
                'problem_file': problem_file,
                'scene_id': scene_id,
                'instance_id': instance_id,
                'image_counter': 0,
            }

            all_correct, action_results, replans, action_queue, goal_reached = check_plan(
                env,
                plan,
                problem,
                vlm_state,
                model,
                prompt_path,
                logger,
                img_log_info,
                replan=replan,
                text_only=text_only,
                enumerate_replan=enumerate_replan,
                enum_batch_size=enum_batch_size,
                max_actions=max_steps,
                policy_class=PolicyClass,
                policy_kwargs=policy_kwargs,
            )
            results[f"{problem_file}_{scene_id}_{instance_id}"].update({
                'all_correct': all_correct,
                'goal_reached': goal_reached,
                'action_results': action_results,
                'replans': replans,
                'remaining_actions': [str(a) for a in action_queue]
            })

        #     break
        # break
    
    predicate_accuracy, macro_predicate_accuracy, action_accuracy, task_accuracy, problem_stats, predicate_stats, fail_ratio = compute_metrics(results, logger)
    logger.info(f"Predicate accuracy: {predicate_accuracy:.2f}, Macro predicate accuracy: {macro_predicate_accuracy:.2f}, Action accuracy: {action_accuracy:.2f}, Task accuracy: {task_accuracy:.2f}")
    logger.info(f"Fail ratio: {fail_ratio:.2f}")
    results['problem_stats'] = problem_stats
    results['predicate_stats'] = predicate_stats
    results['predicate_accuracy'] = predicate_accuracy
    results['macro_predicate_accuracy'] = macro_predicate_accuracy
    results['action_accuracy'] = action_accuracy
    results['task_accuracy'] = task_accuracy
    results['fail_ratio'] = fail_ratio
    results['metadata'] = {
        'model_name': model_name,
        'prompt_path': prompt_path,
        'problems_dir': problems_dir,
        'seed': seed,
        'replan': replan,
        'fail_probability': fail_probability,
        'enumerate_initial_state': enumerate_initial_state,
        'job_id': unique_id,
    }
    
    if enumerate_initial_state:
        problem_keys = [k for k in results.keys() if isinstance(results[k], dict) and 'initial_state_enum' in results[k]]
        enumeration_accuracy = sum([results[problem]['initial_state_enum']['statistics']['accuracy'] for problem in problem_keys]) / len(problem_keys) if len(problem_keys) > 0 else None
        results['enumeration_accuracy'] = enumeration_accuracy
        if enumeration_accuracy is not None:
            logger.info(f"Enumeration average accuracy: {enumeration_accuracy:.2f}")
        else:
            logger.info("No problems were enumerated")
        
        predicate_enumeration_accuracy = {}
        for problem in problem_keys:
            predicate_enum_stats = results[problem]['initial_state_enum']['statistics']['predicates']
            for predicate in predicate_enum_stats:
                if predicate not in predicate_enumeration_accuracy:
                    predicate_enumeration_accuracy[predicate] = []
                predicate_enumeration_accuracy[predicate].append(predicate_enum_stats[predicate]['accuracy'])
                        
        for predicate in predicate_enumeration_accuracy:
            avg_accuracy = sum(predicate_enumeration_accuracy[predicate]) / len(predicate_enumeration_accuracy[predicate]) if len(predicate_enumeration_accuracy[predicate]) > 0 else None
            if avg_accuracy is not None:
                logger.info(f"Predicate {predicate} average enumeration accuracy: {avg_accuracy:.2f}")
            else:
                logger.info(f"Predicate {predicate} had no enumeration accuracy")
            
    if output_dir is None:
        output_dir = os.curdir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{unique_id}.json")
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    fire.Fire(main)