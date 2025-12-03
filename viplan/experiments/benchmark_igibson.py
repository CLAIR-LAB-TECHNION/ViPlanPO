import os
import traceback

import fire
import json
import copy
import time

import torch
import random
import transformers

from collections import deque
from typing import Optional

from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from viplan.log_utils import get_img_output_dir, get_task_logger
from viplan.planning.igibson_client_env import iGibsonClient
from viplan.code_helpers import get_logger, load_vlm, get_unique_id
from viplan.policies.policy_interface import (
    Policy,
    PolicyAction,
    PolicyObservation,
    resolve_policy_class,
)

from viplan.policies.policy_plan import DefaultPlanningPolicy
from viplan.policies.policy_vila import DefaultVILAPolicy
from viplan.policies.policy_cpp import PolicyCPP

goal_templates = {
    'reachable': {
        True:  "the {0} needs to be reachable by the agent",
        False: "the {0} needs to be unreachable by the agent"
    },
    'holding': {
        True:  "the agent needs to be holding the {0}",
        False: "the agent must not be holding the {0}"
    },
    'open': {
        True:  "the {0} needs to be open",
        False: "the {0} needs to be closed"
    },
    'ontop': {
        True:  "the {0} needs to be on top of the {1}",
        False: "the {0} must not be on top of the {1}"
    },
    'inside': {
        True:  "the {0} needs to be inside the {1}",
        False: "the {0} must not be inside the {1}"
    },
    'nextto': {
        True:  "the {0} needs to be next to the {1}",
        False: "the {0} must not be next to the {1}"
    }
}

"""
def get_goal_str(env):
    goal_fluents = env.goal_fluents
    goal_string = ""

    for fluent in goal_fluents:
        name = fluent.fluent().name
        args = fluent.args

        descr = goal_templates[str(name)].format(*[arg.object().name for arg in args])
        goal_string += descr + "\n"
    
    goal_string = goal_string.strip()
    return goal_string
"""

def get_goal_str(env):
    goal_fluents = env.goal_fluents
    goal_string = ""

    for fluent in goal_fluents:
        value = True
        if fluent.is_not():
            value = False
            fluent = fluent.args[0]
        name = fluent.fluent().name
        args = fluent.args

        descr = goal_templates[str(name)][value].format(*[arg.object().name for arg in args])
        goal_string += descr + "\n"
    
    goal_string = goal_string.strip()
    return goal_string


def planning_loop(
    env,
    policy,
    problem,
    logger,
    tasks_logger,
    log_extra,
    img_output_dir,
    max_steps=50,
    use_predicate_groundings=True,
):
    previous_actions = []
    problem_results = {
        'plans': [],
        'actions': [],
        'previous_actions': previous_actions,
        'completed': False,
    }

    initial_max_steps = copy.deepcopy(max_steps)

    while not env.goal_reached and max_steps > 0:
        log_action_extra = log_extra.copy()
        step = initial_max_steps - max_steps + 1
        log_action_extra['step'] = step
        logger.info(f"Step {step}")
        logger.info(f"Environment state before action:\n{env.state}")
        start_time = time.time()
        img = env.render()
        tasks_logger.info(
            "Rendered env image",
            extra=log_action_extra | {'compute_time': time.time() - start_time},
        )
        observation = PolicyObservation(
            image=img,
            problem=problem,
            predicate_groundings=env.priviledged_predicates if use_predicate_groundings else None,
            previous_actions=previous_actions,
            context={'step': step},
        )

        policy_action = None
        policy_error = 'unknown'
        try:
            start_time = time.time()
            policy_action = policy.next_action(observation, log_action_extra)
        except json.JSONDecodeError as e:
            traceback.print_exc()
            logger.error(f"Could not parse VLM output: {e}")
            policy_error = 'Could not parse VLM output'
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"Policy failed to return an action: {exc}", stack_info=True)
            policy_error = 'Policy failed to return an action'

        if policy_action is None:
            logger.warning("Policy returned no action; stopping episode")
            log_extra['policy_error'] = policy_error # Goes back up to for logging by caller
            break

        problem_results['plans'].append(policy_action.raw_response)
        action_repr = policy_action.as_env_command()
        problem_results['actions'].append(action_repr)
        logger.info(
            f"First action: {policy_action.name}({', '.join(policy_action.parameters)})"
        )
        log_action_extra['action'] = policy_action.name
        log_action_extra['parameters'] = policy_action.parameters
        tasks_logger.info(
            "Next action decided",
            extra=log_action_extra | {
                'compute_time': time.time() - start_time,
            },
        )

        try:
            start_time = time.time()
            success, info = env.apply_action(
                action=policy_action.name, params=list(policy_action.parameters)
            )
        except Exception as e:
            logger.error(f"Unexpected error applying action: {e}")
            problem_results['actions'][-1]['success'] = False
            problem_results['actions'][-1]['info'] = None
            wrong_parameters = False  # can't assume it's input-related
        else:
            problem_results['actions'][-1]['success'] = success
            problem_results['actions'][-1]['info'] = info
            wrong_parameters = not success
            policy.observe_outcome(policy_action, success=success, info=info)

        log_action_extra['wrong_parameters'] = wrong_parameters
        log_action_extra['action_success'] = problem_results['actions'][-1]['success']
        log_action_extra['action_info'] = problem_results['actions'][-1]['info']

        if policy_action:
            params_str = "-".join(policy_action.parameters)
            action_details_str = f'{policy_action.name}_{params_str}'
        else:
            action_details_str = 'failed'
        img.save(
            os.path.join(
                img_output_dir, f"env_render_{step}_{action_details_str}.png"
            )
        )
        tasks_logger.info(
            "Action completed",
            extra=log_action_extra | {
                'compute_time': time.time() - start_time,
            },
        )

        # Failsafe for actions that don't exist
        try:
            available_actions = [a.name for a in problem.actions]
            logger.debug(
                f"Available actions: {available_actions}, first action: {policy_action.name}"
            )
            if policy_action.name not in available_actions:
                logger.warn(
                    f"Action {policy_action.name} does not exist in the environment."
                )
                previous_actions.append(
                    {'action': policy_action.name, 'outcome': 'action does not exist'}
                )
            else:
                if wrong_parameters is not None and wrong_parameters:
                    previous_actions.append(
                        {
                            'action': policy_action.name,
                            'parameters': list(policy_action.parameters),
                            'outcome': 'parameters incorrectly specified'
                        }
                    )
                else:
                    previous_actions.append(
                        {
                            'action': policy_action.name,
                            'parameters': list(policy_action.parameters),
                            'outcome': 'executed' if success else 'failed'
                        }
                    )
        except Exception as e:
            logger.error(f"Something went wrong (e.g. plan is empty): {e}")
            tasks_logger.info(
                "Something went wrong (e.g. plan is empty)",
                extra=log_action_extra | {
                    'compute_time': time.time() - start_time,
                },
            )
            break

        logger.info(f"Action outcome: {'executed' if success else 'failed'}")
        if info is not None:
            logger.info(f"Info about action outcome: {info}")
        logger.info(f"Previous actions: {previous_actions}")
        logger.info(f"Environment state after action:\n{env.state}")
        tasks_logger.info(
            "Saved image",
            extra=log_action_extra | {
                'img_path': os.path.join(
                    img_output_dir, f"env_render_{step}_{action_details_str}.png"
                ),
            },
        )
        problem_results['previous_actions'] = previous_actions

        max_steps -= 1

    if env.goal_reached:
        logger.info("Goal reached!")
        problem_results['completed'] = True

    return problem_results


def main(
    problems_dir: os.PathLike,
    domain_file: os.PathLike,
    base_url: str,
    model_name: str,
    seed: int = 1,
    output_dir: os.PathLike = None,
    hf_cache_dir: os.PathLike = None,
    log_level = 'info',
    max_steps: int = 10,
    policy_cls: str = 'PolicyCPP',
    use_predicate_groundings: bool = True,
    **kwargs):
    
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    
    logger = get_logger(log_level=log_level)
    unique_id = get_unique_id(logger)
    tasks_logger = get_task_logger(out_dir=output_dir, unique_id=unique_id)
        
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", None)
        logger.debug(f"Using HF cache dir: {hf_cache_dir}")

    model = load_vlm(model_name, hf_cache_dir=hf_cache_dir, logger=logger, **kwargs)

    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts

    PolicyCls = resolve_policy_class(policy_cls, DefaultVILAPolicy)

    results = {}
    metadata = os.path.join(problems_dir, "metadata.json")
    assert os.path.exists(metadata), f"Metadata file {metadata} not found"
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    problem_files = [problem for problem in metadata.keys()]
    problem_files = [f"{problems_dir}/{problem}" for problem in problem_files]
    
    for problem_file in problem_files:
        logger.info(f"Loading problem {problem_file}")
        
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        task = metadata[os.path.basename(problem_file)]['activity_name']
        scene_instance_pairs = metadata[os.path.basename(problem_file)]['scene_instance_pairs']
        for scene_id, instance_id in scene_instance_pairs:
            env = iGibsonClient(task=task, scene_id=scene_id, instance_id=instance_id, problem=problem, base_url=base_url, logger=logger)
            env.reset() # Reset = send a request to the server to re-initialize the task (also needed when switching tasks)

            goal_string = get_goal_str(env)
            logger.info(f"Goal: {goal_string}")

            img_output_dir = get_img_output_dir('vila', instance_id, scene_id, task)

            log_extra = {
                'problem_file': problem_file,
                'task': task,
                'scene_id': scene_id,
                'instance_id': instance_id,
                'policy_cls': PolicyCls.__name__,
                'use_predicate_groundings': use_predicate_groundings,
                'model': model_name,
                'img_output_dir': img_output_dir,
            }

            if issubclass(PolicyCls, DefaultPlanningPolicy):
                action_queue = deque()
                policy = PolicyCls(
                    action_queue=action_queue,
                    logger=logger,
                )
            else:
                policy = PolicyCls(
                    domain_file=domain_file,
                    problem_file=problem_file,
                    model=model,
                    model_name=model_name,
                    goal_string=goal_string,
                    tasks_logger=tasks_logger,
                    log_extra=log_extra,
                    logger=logger,
                    problem=problem,
                )

            # Run planning loop
            logger.info("Starting planning loop...")
            start_time = time.time()
            problem_results = planning_loop(
                env,
                policy,
                problem,
                logger,
                tasks_logger,
                log_extra,
                img_output_dir,
                max_steps=max_steps,
                use_predicate_groundings=use_predicate_groundings,
            )
            results[f"{problem_file}_{scene_id}_{instance_id}"] = problem_results
            log_extra['elapsed_time'] = time.time() - start_time
            log_extra['completed'] = problem_results['completed']
            tasks_logger.info('Finished planning loop', extra=log_extra)
    
    # Compute some statistics
    total_actions = 0
    total_success = 0
    total_failed = 0
    total_tasks_completed = 0
    
    for problem_file, problem_results in results.items():
        if problem_results is None:
            continue
        total_actions += len(problem_results['actions'])
        total_success += sum([1 for a in problem_results['actions'] if a.get('success', False)])
        total_failed += sum([1 for a in problem_results['actions'] if not a.get('success', False)])
        total_tasks_completed += 1 if problem_results['completed'] else 0
        
    action_success_rate = total_success / total_actions if total_actions > 0 else 0
    action_failure_rate = total_failed / total_actions if total_actions > 0 else 0
    task_completion_rate = total_tasks_completed / len(results) if len(results) > 0 else 0
    
    results['statistics'] = {
        'total_actions': total_actions,
        'total_success': total_success,
        'total_failed': total_failed,
        'total_tasks_completed': total_tasks_completed,
        'action_success_rate': action_success_rate,
        'action_failure_rate': action_failure_rate,
        'task_completion_rate': task_completion_rate,
    }
    
    results['metadata'] = {
        'model': model_name,
        'seed': seed,
        'max_steps': max_steps,
        'job_id': unique_id,
        'use_predicate_groundings': use_predicate_groundings,
    }

    logger.info(f"Action success rate: {action_success_rate}")
    logger.info(f"Action failure rate: {action_failure_rate}")
    logger.info(f"Task completion rate: {task_completion_rate}")
    
    if output_dir is None:
        output_dir = os.curdir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{unique_id}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)