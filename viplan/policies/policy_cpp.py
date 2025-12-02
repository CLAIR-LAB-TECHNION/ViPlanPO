# python requirements for running this
# numpy
# scipy
# pillow
# up-cpor @ git+https://github.com/guyazran/up-cpor
# additional requirements:
# conda install -c conda-forge mono

import os
import random
from typing import Any, Dict, List, Optional
from logging import Logger

from PIL.Image import Image
import numpy as np
from unified_planning.shortcuts import FNode, OneshotPlanner
import unified_planning.environment as environment
from unified_planning.engines.results import PlanGenerationResultStatus
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.plans.plan import ActionInstance
from scipy.special import logit, expit

from .cpp_utils import (
    to_contingent_problem,
    enumerate_states_by_probability,
    set_cp_initial_state_constraints_from_belief,
    extract_conformant_plan
)
from .policy_interface import Policy, PolicyObservation, PolicyAction
from .policy_vila import DefaultVILAPolicy
from .up_utils import (
    create_up_problem,
    get_mapping_from_compiled_actions_to_original_actions,
    get_all_grounded_predicates_for_objects,
    state_dict_to_up_state,
    find_goal_relevant_fluents,
)
from ..models.custom_vqa.openai import OpenAIVQA, OPENAI_MODEL_ID_PREFIX


class PolicyCPP(Policy):
    def __init__(
        self,
        predicate_language: Dict[str, str],
        domain_file: str,
        problem_file: str,
        model_name: str,
        base_prompt: str,
        tasks_logger: Logger,
        conformant_prob: float = 0.8,
        belief_update_weight: float = 0.5,
        use_unknown_token: bool = True,
        use_fd_constraints: bool = True,
        planner_timeout: Optional[float] = None,
        **vlm_inference_kwargs: Dict[str, Any]
    ):
        super().__init__(predicate_language)

        # convert the classical problem into a ContingentProblem object that will
        # represent the conformant problem internally.
        orig_problem = create_up_problem(domain_file, problem_file)
        self.contingent_problem = to_contingent_problem(orig_problem)

        # create a mapping so we can translate actions back to the original problem.
        # the environment will expect actions from the original problem.
        self.action_mapping = get_mapping_from_compiled_actions_to_original_actions(
            self.contingent_problem, orig_problem
        )

        # set the conformant probability threshold for choosing the states in the belief.
        self.conformant_prob = conformant_prob

        # Whether to use fast-downward constraints to filter out impossible states.
        self.use_fd_constraints = use_fd_constraints

        # initialize the belief with total uncertainty over all fluents.
        # represented with maximum entropy (0.5 probability for each fluent).
        if self.use_fd_constraints:
            self.all_fluents = find_goal_relevant_fluents(
                orig_problem
            )
        else:
            self.all_fluents = get_all_grounded_predicates_for_objects(orig_problem)
        self.factored_belief: Dict[Any, float] = {
            fluent: 0.5
            for fluent in self.all_fluents
        }

        # set the belief update weight (how much to trust VLM estimates).
        # clamp to [0.0, 1.0] range.
        self.belief_update_weight = np.clip(belief_update_weight, 0.0, 1.0)

        # A container for the current belief set (list of possible states).
        self.belief_set = None

        # create a simulator for the original problem to step through states.
        self._sim = UPSequentialSimulator(orig_problem)

        # The previous chosen action in the format of the original problem.
        # Used to step the belief.
        self._prev_action = None

        # a container for the current plan
        self.current_plan = None

        # create planner to avoid recreating it at each replan
        env = environment.get_environment()
        env.factory.add_meta_engine('MetaCPORPlanning', 'up_cpor.engine', 'CPORMetaEngineImpl')
        self.planner = OneshotPlanner(name='MetaCPORPlanning[fast-downward]')
        self.planner_timeout = planner_timeout

        # initialize the VLM model
        # currently only GPT-based OpenAI models are supported
        if model_name.startswith(OPENAI_MODEL_ID_PREFIX):
            # expect the OpenAI API key to be set in the environment
            openai_api_key = os.environ["OPENAI_API_KEY"]

            # initialize the OpenAI API interface
            self.vlm_model = OpenAIVQA(
                model_id=model_name,
                system_prompt=base_prompt,
                api_key=openai_api_key,
                max_new_tokens=1,  # we only need the first token's logprobs
                **vlm_inference_kwargs
            )
        else:
            raise NotImplementedError(
                f"VLM model {model_name} not supported in PolicyCPP."
            )

        # select tokens for which we want to get probabilities
        self.use_unknown_token = use_unknown_token
        self.tokens_of_interest = [["yes", "Yes", "YES"], ["no", "No", "NO"]]
        if self.use_unknown_token:
            self.tokens_of_interest.append(
                ["unknown", "Unknown"]
            )

        self.base_prompt = base_prompt
        self.vlm_inference_kwargs = vlm_inference_kwargs
        self.task_logger = tasks_logger
        self.fallback_vila_policy = DefaultVILAPolicy(
            model=self.vlm_model,
            base_prompt=base_prompt,
            logger=self.task_logger,
            predicate_language=predicate_language,
        )

    def next_action(self, observation: PolicyObservation, log_extra: Dict[str, Any]) -> PolicyAction:
        # if the previous action was executed, step action in belief space
        if log_extra is None:
            log_extra = dict()
        if (observation.previous_actions and
            self._prev_action is not None and
            observation.previous_actions[-1]['outcome'] == 'executed'):
            assert self.belief_set is not None, "Belief set is not initialized."
            self._belief_step(self._prev_action)

        log_plan_extra = log_extra.copy()
        log_plan_extra['conformant_prob'] = self.conformant_prob
        log_plan_extra['belief_update_weight'] = self.belief_update_weight

        # get probabilistic grounding from the VLM
        vlm_prob_ground = self._estimate_fluent_prob(
            images=[observation.image],
            #TODO only update fluents that are relevant
            fluent_list=self.all_fluents
        )

        # update the factored belief based on VLM outputs
        self._update_belief(vlm_prob_ground)

        # check causes for replanning
        replan = False
        if self.current_plan is None:  # does plan exist?
            replan = True
        else:
            # check if the probability of the current belief set
            # still meets the conformant probability threshold
            total_belief_prob = self._get_current_belief_set_probability()
            if total_belief_prob < self.conformant_prob:
                replan = True

            # check if next action is safe
            next_action = self.current_plan[0]
            if not self._is_safe_action(next_action):
                replan = True

        log_plan_extra['replan'] = replan
        if replan:
            # set new belief set and plan
            self._set_belief_set_and_plan(log_plan_extra)
            
            self.task_logger.info(
                "Replanning executed.", extra=log_plan_extra
            )

        if self.current_plan is None:
            self.task_logger.info(
                "No plan. Exploring", extra=log_plan_extra
            )
            return self.fallback_vila_policy.next_action(
                observation, log_extra
            )

        # get the next action from the current plan
        next_action = self.current_plan.pop(0)

        # map the action back to the original problem's action
        next_action_orig = self.action_mapping(next_action)

        # format the action to be returned
        action = PolicyAction(
            name=next_action_orig.action.name,
            parameters=list(map(str, next_action_orig.actual_parameters)),
            raw_response=[
                str(a) for a in [next_action] + self.current_plan
            ]
        )

        # set previous action for next belief step.
        # the step will happen only if the action was executed.
        # note that this is in the original problem's format.
        self._prev_action = next_action_orig

        return action

    def _get_current_belief_set_probability(self) -> float:
        if not self.belief_set:
            return 0.0

        total_prob = 0.0
        for state in self.belief_set:
            per_fluent_probs = [
                self.factored_belief[fluent]
                if state[fluent] else
                (1 - self.factored_belief[fluent])
                for fluent in self.factored_belief.keys()
            ]
            state_prob = float(np.prod(per_fluent_probs))
            total_prob += state_prob

        return total_prob

    def _set_belief_set_and_plan(self, log_plan_extra) -> None:
        self.belief_set = []
        self.current_plan = None

        # accumulate probability mass until reaching threshold
        total_prob = 0.0
        for state_str, prob in enumerate_states_by_probability(self.factored_belief):
            # turn state string into a dict representation
            state = {
                fluent: (state_str[i] == '1')
                for i, fluent in enumerate(self.all_fluents)
            }

            # set initial state constraints in the contingent problem
            # based on all states selected so far.
            set_cp_initial_state_constraints_from_belief(
                self.contingent_problem,
                self.belief_set + [state]
            )

            # try to find a conformant plan for the current belief set
            try:
                plan_res = self.planner.solve(
                    self.contingent_problem,
                    timeout=self.planner_timeout
                )
            except Exception as e:
                self.task_logger.error(
                    "Error during planning",
                    extra=log_plan_extra | {"exception": str(e)}
                )
                break  # planning failed

            if plan_res.status != PlanGenerationResultStatus.SOLVED_SATISFICING:
                break  # no plan found for this belief set
            
            # found a conformant plan, add this state to the belief set
            # and save plan
            self.belief_set.append(state)
            self.current_plan = extract_conformant_plan(plan_res.plan.root_node)
            log_plan_extra['has_plan'] = self.current_plan is not None

            # check if we reached the conformant probability threshold
            total_prob += prob
            if total_prob >= self.conformant_prob:
                break
        log_plan_extra['conformant_plan_success_probability'] = total_prob
        log_plan_extra['num_selected_states'] = len(self.belief_set)

        # log info about new belief set and plan
        if self.current_plan is None:
            self.task_logger.warning(
                "No conformant plan found for the current belief.",
                extra=log_plan_extra
            )

        selected_states = [
            {
                str(fluent): state[fluent]
                for fluent in self.factored_belief.keys()
            }
            for state in self.belief_set
        ]
        self.task_logger.info(
            "New belief set",
            extra=log_plan_extra | {
                "selected_states": selected_states
            }
        )
        
    
    def _update_belief(self, fluent_probs: Dict[FNode, Optional[float]]) -> None:
        for fluent, prob in fluent_probs.items():
            if prob is None:
                continue  # no info about this fluent

            # update the belief with weighted logit-pooling
            prior_logit = logit(self.factored_belief[fluent])
            vlm_logit = logit(prob)
            updated_logit = (
                prior_logit * (1 - self.belief_update_weight) +
                vlm_logit * self.belief_update_weight
            )
            
            self.factored_belief[fluent] = float(expit(updated_logit))

    def _belief_step(self, action: ActionInstance) -> None:
        new_belief_set = []
        for state in self.belief_set:
            # create a UPState object from the state dict
            up_state = state_dict_to_up_state(
                self._sim._problem,
                {
                    str(fluent): v
                    for fluent, v in state.items()
                }
            )
                
            # step the simulator to get the next state
            next_state = self._sim.apply(up_state, action)

            # convert the UPState back to a dict representation
            next_state_dict = {
                fluent: next_state.get_value(fluent)
                for fluent in self.factored_belief.keys()
            }

            new_belief_set.append(next_state_dict)
        
        # merge duplicate states in the new belief set
        merged_belief_set = []
        seen_states = set()
        for state in new_belief_set:
            state_tuple = tuple(sorted(state.items()))
            if state_tuple not in seen_states:
                seen_states.add(state_tuple)
                merged_belief_set.append(state)

        # set belief to the new belief set
        self.belief_set = merged_belief_set
    
    def _is_safe_action(self, action: ActionInstance) -> bool:
        for state in self.belief_set:
            # create a UPState object from the state dict
            up_state = state_dict_to_up_state(
                self._sim._problem,
                {
                    str(fluent): v
                    for fluent, v in state.items()
                }
            )

            # check action applicability            
            if not self._sim.is_applicable(up_state, action):
                # action not applicable in this state
                return False
        
        # action applicable in all states
        return True
        
    def _format_prompt(self, fluent: FNode) -> str:
        fluent_name = fluent.fluent().name
        fluent_prompt_template = self.predicate_language[fluent_name]
        fluent_prompt = fluent_prompt_template.format(*fluent.args)
        return fluent_prompt
    
    def _all_fluent_prompts(self, fluents: List[FNode]):
        return [self._format_prompt(fluent) for fluent in fluents]

    def _estimate_fluent_prob(self, images: Image, fluent_list: List[FNode]) -> Dict[Any, Optional[float]]:
        # get a query for each fluent
        fluent_queries = self._all_fluent_prompts(fluent_list)

        # get VLM outputs for all fluent queries
        vlm_outputs = self.vlm_model(
            images,
            fluent_queries,
            *self.tokens_of_interest
        )
        
        # process VLM outputs into fluent probabilities
        fluent_probs = {}
        for fluent, vlm_output in zip(fluent_list, vlm_outputs):
            # unpack vlm output probabilities according to selected tokens of interest
            if self.use_unknown_token:
                yes_prob, no_prob, unknown_prob = vlm_output
            else:
                yes_prob, no_prob = vlm_output
                unknown_prob = 0.0    

            # determine fluent probability based on token probabilities
            if unknown_prob >= max(yes_prob, no_prob) or (yes_prob + no_prob == 0.0):
                fluent_probs[fluent] = None  # no info
            else:
                # normalized yes probability
                fluent_probs[fluent] = float(yes_prob / (yes_prob + no_prob))

        return fluent_probs

    def _sample_random_exploratory_action(self) -> Optional[ActionInstance]:
        """Sample a random action that is biased toward achieving the goal."""

        problem = self._sim._problem
        if not problem.actions:
            return None

        goal_fluent_names = set()
        goal_objects = set()

        for goal in problem.goals:
            try:
                if goal.is_fluent_exp():
                    fluent = goal.fluent()
                    goal_fluent_names.add(fluent.name)
                    goal_objects.update(map(str, fluent.args))
                goal_objects.update(map(str, goal.args))
            except Exception:
                continue

        def is_relevant(action) -> bool:
            for effect in getattr(action, "effects", []):
                try:
                    effect_fluent = effect.fluent
                    effect_symbol = effect_fluent.fluent()
                    if effect_symbol.name in goal_fluent_names:
                        return True
                    if any(str(arg) in goal_objects for arg in effect_fluent.args):
                        return True
                except Exception:
                    continue

            for param in action.parameters:
                objects_of_type = list(problem.objects(param.type))
                if any(str(obj) in goal_objects for obj in objects_of_type):
                    return True

            return False

        candidate_actions = [a for a in problem.actions if is_relevant(a)]
        if not candidate_actions:
            candidate_actions = list(problem.actions)

        tried_actions = set()
        while len(tried_actions) < len(candidate_actions):
            action = random.choice(candidate_actions)
            tried_actions.add(id(action))

            try:
                parameters = [
                    random.choice(list(problem.objects(param.type)))
                    for param in action.parameters
                ]
            except IndexError:
                # Action cannot be grounded due to missing objects of a given type.
                continue

            try:
                return action(*parameters)
            except Exception:
                continue

        return None

    def __del__(self):
        self.planner.destroy()