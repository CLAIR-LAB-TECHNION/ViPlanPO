import copy
from pathlib import Path
from typing import Optional, Dict, Any

from unified_planning.shortcuts import up

from viplan.code_helpers import get_logger, parse_output
from viplan.log_utils import save_vlm_question_images
from viplan.planning.planning_utils import get_plan
from viplan.policies.natural_language_utils import PREDICATE_QUESTIONS, load_prompt
from viplan.policies.policy_interface import Policy, PolicyAction, PolicyObservation


class DefaultPlanningPolicy(Policy):
    """Default policy that sequentially executes planner actions."""

    def __init__(
            self,
            logger,
            problem,
            **kwargs
    ):
        super().__init__()
        plan_result = get_plan(problem, logger)
        if plan_result is None:
            logger.warning('Breaking out of episode due to error in the planner')
        elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
            logger.warning('No plan found after replanning')
        else:
            logger.info('Replan found')
        self.plan = plan_result.plan
        self.action_queue = self.plan.action_queue
        self.logger = logger

    def next_action(self, observation: PolicyObservation, log_extra: Dict[str, Any]) -> Optional[PolicyAction]:
        logger = self.logger

        if self.action_queue.empty():
            return None

        env = observation.env

        # === Consider last action ===
        if observation.previous_actions:
            legal = observation.previous_actions[-1]['success']
        else:
            legal = True

        if not legal:
            logger.warning("Action was not legal")
            return None

        effects_results, vlm_state = check_effects(env, vlm_state, effects, grounded_params, model, base_prompt,
                                                   previous_state, logger, img_log_info, text_only=text_only)
        vlm_state, changed = update_vlm_state(vlm_state, effects_results)
        if len(changed) > 0:
            logger.debug("VLM state changed after effects:", changed)

        precond_all_correct = preconditions_results['all_correct'] if 'all_correct' in preconditions_results else True
        effects_all_correct = effects_results['all_correct'] if 'all_correct' in effects_results else True
        all_correct = precond_all_correct and effects_all_correct

        # Effects state correct can be different from all_correct if there was a failure in the environment and the VLM detected it
        precond_state_correct = preconditions_results[
            'all_state_correct'] if 'all_state_correct' in preconditions_results else True
        effects_state_correct = effects_results['all_state_correct'] if 'all_state_correct' in effects_results else True
        all_state_correct = precond_state_correct and effects_state_correct

        # === Prepare for next action ===
        preconditions = action.action.preconditions
        effects = action.action.effects
        grounded_params = {param.name: str(value) for param, value in
                           zip(action.action.parameters, action.actual_parameters)}
        previous_state = copy.deepcopy(env.state)
        logger.info("Environment state before action\n" + str(env))

        preconditions_results, non_visible_precond_results = check_preconditions(env, preconditions, grounded_params,
                                                                                 model, base_prompt, logger,
                                                                                 text_only=text_only,
                                                                                 img_log_info=img_log_info)
        if not non_visible_precond_results['all_correct']:
            logger.warning("Non visible preconditions not satisfied")
            return False, non_visible_precond_results, None, False, None  # TODO return both precond results later on

        vlm_state, changed = update_vlm_state(vlm_state, preconditions_results)
        if len(changed) > 0:
            logger.debug("VLM state changed after preconditions:", changed)

        # If the preconditions are not satisfied according to the PDDL model, the action can not be taken
        if 'all_correct' in preconditions_results and not preconditions_results['all_correct']:
            logger.warning("Preconditions not satisfied")
            return False, preconditions_results, non_visible_precond_results, None, False, None

        self.logger.debug(f"Policy selecting action {action}")
        return PolicyAction(
            name=action.action.name,
            parameters=[str(p) for p in action.actual_parameters],
            metadata={'plan_action': action},
        )


def get_questions(predicates):
    questions = {}
    if isinstance(predicates, dict):
        for predicate in predicates:
            for args in predicates[predicate]:
                key = predicate + " " + args
                value = predicates[predicate][args]
                if predicate in PREDICATE_QUESTIONS:
                    questions[key] = (PREDICATE_QUESTIONS[predicate].format(*args.split(",")), value)
                else:
                    raise ValueError(f"Unknown predicate {predicate}")
    else:
        for pred in predicates:
            key = list(pred.keys())[0]
            predicate = key.split(" ")[0]
            value = pred[key]
            args = key.split(" ")[1].split(",")
            if predicate in PREDICATE_QUESTIONS:
                questions[predicate + " " + ",".join(args)] = (PREDICATE_QUESTIONS[predicate].format(*args), value)
            else:
                raise ValueError(f"Unknown predicate {predicate}")

    return questions


def get_predicates_for_question(env, node, grounded_args, top_level=True, default_value=True):
    result = []

    if not node.is_fluent_exp():

        if node.is_or():
            # We skip all "or" nodes (which only happens once in our domain) as our method asks questions independently to the VLM, and thus a disjunction is not possible
            # Since the only disjunction (in navigate-to) is by definition never visible by the VLM (as it asks for objects inside containers), we can ignore it
            return {}

        if node.is_not():
            child = node.args[0]
            exps = get_predicates_for_question(env, child, grounded_args, False, default_value)
            for exp in exps:
                result.append({k: not v for k, v in exp.items()})
        elif node.is_forall():
            assert len(node.variables()) == 1, "Only single forall supported"
            var = node.variables()[0]

            for value in env.all_objects[str(var.type)]:
                if str(value) in grounded_args.values():
                    continue
                grounded_args[var.name] = str(value)
                exps = get_predicates_for_question(env, node.args[0], grounded_args, False, default_value)
                result.extend(exps)

        else:
            for child in node.args:
                exps = get_predicates_for_question(env, child, grounded_args, False, default_value)
                result.extend(exps)

    elif node.is_fluent_exp():
        fluent_name = node.fluent().name
        arg_names = [str(arg) for arg in node.args]
        actual_args = [str(grounded_args[arg]) for arg in arg_names]
        args_key = ",".join(actual_args)
        key = fluent_name + " " + args_key
        value = default_value # by default assume precondition is asking for predicate to be default_value (for effects it can also be False)
        bool_value = value if isinstance(value, bool) else value.is_true()
        result = {key: bool_value}

    else:
        raise ValueError("Unknown node type", node)

    return [result] if type(result) is dict else result


def get_effect_predicates(env, effect, grounded_args, previous_state):
    all_preds = []

    if effect.is_forall():
        vars_ = effect.forall
        # Support single or double forall
        if len(vars_) == 1:
            var = vars_[0]
            for value in env.all_objects[str(var.type)]:
                grounded_args[var.name] = str(value)
                # If conditional, check against previous state
                if effect.is_conditional():
                    if not env._check_value(effect.condition, grounded_args, previous_state):
                        continue
                preds = get_predicates_for_question(
                    env, effect.fluent, grounded_args,
                    top_level=False, default_value=effect.value
                )
                if isinstance(preds, dict):
                    all_preds.append(preds)
                else:
                    all_preds.extend(preds)
        elif len(vars_) == 2:
            var1, var2 = vars_
            for v1 in env.all_objects[str(var1.type)]:
                grounded_args[var1.name] = str(v1)
                for v2 in env.all_objects[str(var2.type)]:
                    grounded_args[var2.name] = str(v2)
                    if effect.is_conditional():
                        if not env._check_value(effect.condition, grounded_args, previous_state):
                            continue
                    preds = get_predicates_for_question(
                        env, effect.fluent, grounded_args,
                        top_level=False, default_value=effect.value
                    )
                    if isinstance(preds, dict):
                        all_preds.append(preds)
                    else:
                        all_preds.extend(preds)
        else:
            raise NotImplementedError("Only up to 2 nested foralls are supported")

    elif effect.is_conditional():
        condition = effect.condition
        # For conditional effects, check condition against previous_state.
        if env._check_value(condition, grounded_args, previous_state):
            preds = get_predicates_for_question(env, effect.fluent, grounded_args, top_level=False, default_value=effect.value)
            if isinstance(preds, dict):
                all_preds.append(preds)
            else:
                all_preds.extend(preds)

    else:
        preds = get_predicates_for_question(env, effect.fluent, grounded_args, top_level=False, default_value=effect.value)
        if isinstance(preds, dict):
            all_preds.append(preds)
        else:
            all_preds.extend(preds)

    return all_preds


def get_preconditions_predicates(env, preconditions, grounded_args):
    precond_list = []
    for precondition in preconditions:
        if precondition.is_and():
            precond_list.extend(precondition.args)

    preconditions_predicates = []
    for precondition in precond_list:
        preds = get_predicates_for_question(env, precondition, grounded_args)
        preconditions_predicates.extend(preds)
    return preconditions_predicates


def get_effects_predicates(env, effects, grounded_args, previous_state):
    all_preds = []
    for effect in effects:
        preds = get_effect_predicates(env, effect, grounded_args, previous_state)
        all_preds.extend(preds)
    return all_preds


def cast_to_yes_no(parsed_answer, logger):
    if parsed_answer is None: # empty answer or invalid format
        return "invalid answer"

    if not parsed_answer.startswith("yes") and not parsed_answer.startswith("no"):
        for word in parsed_answer.split(" "):
            if word.startswith("yes"):
                parsed_answer = "yes"
                logger.debug(f"Found 'yes' in answer: {parsed_answer}")
                break
            elif word.startswith("no"):
                parsed_answer = "no"
                logger.debug(f"Found 'no' in answer: {parsed_answer}")
                break
    elif parsed_answer.startswith("yes"):
        parsed_answer = "yes"
    elif parsed_answer.startswith("no"):
        parsed_answer = "no"

    return parsed_answer


DEFAULT_PLAN_PROMPT_PATH = Path("benchmark/igibson/prompt_po_all-BB.md")
DEFAULT_PLAN_PROMPT = load_prompt(DEFAULT_PLAN_PROMPT_PATH)


def ask_vlm(questions, image, model, base_prompt, logger, img_log_info, check_type=None, **kwargs):
    base_prompt = load_prompt(base_prompt)
    prompts = [base_prompt + q[0] for q in questions.values()]
    images = [image for _ in questions]

    if len(prompts) > 0:
        save_vlm_question_images(questions, image, img_log_info, check_type, logger)

    outputs = model.generate(prompts=prompts, images=images, return_probs=True, **kwargs)

    results = {}
    for j, (key) in enumerate(questions.keys()):
        answer, yes_prob, no_prob = outputs[j]
        original_output = copy.deepcopy(answer)
        answer, tags_found = parse_output(answer, answer_tags=["answer", "explanation"])
        parsed_answer = answer['answer'] if tags_found and 'answer' in answer else answer
        parsed_answer = parsed_answer.strip().lower().rstrip('.,!?') if isinstance(parsed_answer, str) else parsed_answer
        parsed_answer = cast_to_yes_no(parsed_answer, logger)

        parsed_explanation = answer['explanation'] if tags_found and 'explanation' in answer else None

        logger.info(f"Q: {questions[key][0]}, A: {parsed_answer}, Yes: {yes_prob:.2f}, No: {no_prob:.2f}")
        if parsed_explanation is not None:
            logger.info(f"Explanation (CoT): {parsed_explanation}")

        # Answer match = the answer is what the PDDL model would expect -> if false, then replan
        answer_match = parsed_answer == 'yes' if questions[key][1] else parsed_answer == 'no'

        results[key] = (parsed_answer, yes_prob, no_prob, parsed_explanation, answer_match, original_output, True)

    # All correct = all predicates are correct according to the PDDL model -> no replan needed
    # State match can only be used for metrics -> in the real world, the ground truth is not known
    results['all_correct'] = all([results[result][4] for result in results])
    results['all_state_correct'] = all([results[k][6] for k in results if k not in ['all_correct', 'all_state_correct']])

    return results


def update_vlm_state(vlm_state, results):
    # Entry point is copy.deepcopy(env.state)
    changed = []
    for key, result in results.items():
        if key == 'all_correct' or key == 'all_state_correct' or key == 'updated_non_visible_preds':
            continue
        pred, args = key.split(" ")
        assert result[0] == 'yes' or result[0] == 'no', f"VLM gave unexpected answer {result[0]}"
        new_value = True if result[0] == 'yes' else False
        if vlm_state[pred][args] != new_value:
            vlm_state[pred][args] = new_value
            changed_str = f"{pred} {args} to {new_value}" # Str for better logging, later we can use the key
            changed.append(changed_str)

    return vlm_state, changed


def get_question_preds(predicates, visible_preds):
    question_preds = []
    non_visible_preds = []
    for predicate_dict in predicates:
        key = list(predicate_dict.keys())[0]
        predicate = key.split(" ")[0]
        args = key.split(" ")[1:]
        if predicate in visible_preds and ",".join(args) in visible_preds[predicate]:
            question_preds.append(predicate_dict)
        else:
            non_visible_preds.append(predicate_dict)

    return question_preds, non_visible_preds


def check_preconditions(env, preconditions, grounded_args, model, base_prompt=DEFAULT_PLAN_PROMPT, logger=None,
                        text_only=False, img_log_info=None):
    precondition_preds = get_preconditions_predicates(env, preconditions, grounded_args)
    logger.debug(f"Precondition predicates: {precondition_preds}")
    visible_preds = env.visible_predicates
    logger.debug(f"Visible predicates: {visible_preds}")
    question_preds, non_visible_preds = get_question_preds(precondition_preds, visible_preds)
    logger.debug(f"Non visible predicates: {non_visible_preds}")
    logger.debug(f"Question predicates: {question_preds}")

    questions = get_questions(question_preds)
    if text_only:
        raise NotImplementedError("Text only mode not implemented")
    else:
        if len(questions) == 0:
            logger.warning("No questions to ask VLM")
            results = {}
        else:
            results = ask_vlm(questions, env.render(), model, base_prompt, logger, img_log_info,
                              check_type='precondition')
            logger.debug(f"Precondition VLM results: {results}")

    # Check non visible predicates against vlm_state
    non_visible_results = {}
    for predicate in non_visible_preds:
        key = list(predicate.keys())[0]
        pddl_expected_value = predicate[key]
        predicate = key.split(" ")[0]
        args = ",".join(key.split(" ")[1:])
        vlm_state_value = env.state[predicate][args]
        non_visible_results[key] = (vlm_state_value == pddl_expected_value, vlm_state_value, pddl_expected_value)
        if vlm_state_value != pddl_expected_value:
            logger.warning(f"Non visible predicate {predicate} {args} does not match PDDL model: {vlm_state_value} != {pddl_expected_value}")

    non_visible_results['all_correct'] = all([non_visible_results[k][0] for k in non_visible_results])

    return results, non_visible_results


def check_effects(env, vlm_state, effects, grounded_args, model, base_prompt=DEFAULT_PLAN_PROMPT, previous_state=None, logger=None, img_log_info=None, text_only=False):
    effect_preds = get_effects_predicates(env, effects, grounded_args, previous_state)
    logger.debug(f"Effect predicates: {effect_preds}")
    visible_preds = env.visible_predicates
    logger.debug(f"Visible predicates: {visible_preds}")
    question_preds, non_visible_preds = get_question_preds(effect_preds, visible_preds)
    logger.debug(f"Non visible predicates: {non_visible_preds}")
    logger.debug(f"Question predicates: {question_preds}")

    questions = get_questions(question_preds)
    if text_only:
        raise NotImplementedError("Text only mode not implemented")
    else:
        if len(questions) == 0:
            logger.warning("No questions to ask VLM")
            results = {}
        else:
            results = ask_vlm(questions, env.render(), model, base_prompt, logger, img_log_info, check_type='effect')

    # Update vlm_state with non visible preds using the PDDL expected value
    updated_non_visible_preds = {}
    for predicate in non_visible_preds:
        key = list(predicate.keys())[0]
        pddl_expected_value = predicate[key]
        predicate = key.split(" ")[0]
        args = ",".join(key.split(" ")[1:])
        updated_non_visible_preds[f"{predicate} {args}"] = {'before': vlm_state[predicate][args] if args in vlm_state[predicate] else None, 'after': pddl_expected_value}
        logger.debug(f"Updating vlm_state for {predicate} {args} to {pddl_expected_value}")
        vlm_state[predicate][args] = pddl_expected_value

    results['updated_non_visible_preds'] = updated_non_visible_preds
    return results, vlm_state


def check_action(env, action, vlm_state, model, base_prompt=DEFAULT_PLAN_PROMPT, logger=None, img_log_info=None, text_only=False):
    preconditions = action.action.preconditions
    effects = action.action.effects
    grounded_params = {param.name: str(value) for param, value in zip(action.action.parameters, action.actual_parameters)}
    previous_state = copy.deepcopy(env.state)
    logger.info("Environment state before action\n" + str(env))

    preconditions_results, non_visible_precond_results = check_preconditions(env, preconditions, grounded_params, model,
                                                                             base_prompt, logger, text_only=text_only,
                                                                             img_log_info=img_log_info)
    if not non_visible_precond_results['all_correct']:
        logger.warning("Non visible preconditions not satisfied")
        return False, non_visible_precond_results, None, False, None # TODO return both precond results later on

    vlm_state, changed = update_vlm_state(vlm_state, preconditions_results)
    if len(changed) > 0:
        logger.debug("VLM state changed after preconditions:", changed)

    # If the preconditions are not satisfied according to the PDDL model, the action can not be taken
    if 'all_correct' in preconditions_results and not preconditions_results['all_correct']:
        logger.warning("Preconditions not satisfied")
        return False, preconditions_results, non_visible_precond_results, None, False, None

    legal, info = env.apply_action(action=action.action.name, params=[str(p) for p in action.actual_parameters])
    # VLM thought the action was legal, but it was not
    if not legal:
        logger.warning("Action was not legal")
        return False, preconditions_results, non_visible_precond_results, None, False, info

    logger.info("Environment state after action\n" + str(env))
    effects_results, vlm_state = check_effects(env, vlm_state, effects, grounded_params, model, base_prompt, previous_state, logger, img_log_info, text_only=text_only)
    vlm_state, changed = update_vlm_state(vlm_state, effects_results)
    if len(changed) > 0:
        logger.debug("VLM state changed after effects:", changed)

    precond_all_correct = preconditions_results['all_correct'] if 'all_correct' in preconditions_results else True
    effects_all_correct = effects_results['all_correct'] if 'all_correct' in effects_results else True
    all_correct = precond_all_correct and effects_all_correct

    # Effects state correct can be different from all_correct if there was a failure in the environment and the VLM detected it
    precond_state_correct = preconditions_results['all_state_correct'] if 'all_state_correct' in preconditions_results else True
    effects_state_correct = effects_results['all_state_correct'] if 'all_state_correct' in effects_results else True
    all_state_correct = precond_state_correct and effects_state_correct

    return all_correct, preconditions_results, non_visible_precond_results, effects_results, all_state_correct, info
