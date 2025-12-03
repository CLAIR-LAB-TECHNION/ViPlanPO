import json
from typing import Optional, Dict, Any

from viplan.code_helpers import get_logger
from viplan.policies.natural_language_utils import PREDICATE_QUESTIONS, load_prompt
from viplan.policies.policy_interface import Policy, PolicyAction, PolicyObservation


def parse_json_output(output):
    json_start = output.find('{')
    json_end = output.rfind('}')
    try:
        vlm_plan = json.loads(output[json_start:json_end + 1])
    except json.decoder.JSONDecodeError:
        print(f'Could not parse JSON output: \n{output}')
        print('Tried parsing:\n', output[json_start:json_end + 1])
        raise
    return vlm_plan


def get_priviledged_predicates_str(predicates):
    priviledged_string = ""
    for pred, values in predicates.items():
        for args in values:
            if args:
                name = pred
                args = args
                descr = PREDICATE_QUESTIONS[str(name)].format(*args)
                priviledged_string += descr + "\n"
    priviledged_string = priviledged_string.strip()
    return priviledged_string


class DefaultVILAPolicy(Policy):
    """Default policy that reproduces the original VILA planning loop."""

    def __init__(self, model, goal_string: str, logger=None, **kwargs):
        super().__init__()
        self.model = model
        self.base_prompt = load_prompt("planning/vila_igibson_json.md").replace(
            "{goal_string}", goal_string
        )
        self.logger = logger or get_logger()

    def _format_prompt(self, observation: PolicyObservation) -> str:
        prompt = self.base_prompt.replace(
            "{previous_actions}", json.dumps(list(observation.previous_actions))
        )
        priviledged_preds = observation.predicate_groundings
        if priviledged_preds is not None:
            if all(
                priviledged_preds[predicate] == {}
                for predicate in priviledged_preds
            ):
                prompt = prompt.replace("## Additional information", "")
                prompt = prompt.replace("{priviledged_info}", "")
            else:
                priviledged_str = get_priviledged_predicates_str(priviledged_preds)
                prompt = prompt.replace("{priviledged_info}", priviledged_str)
        else:
            prompt = prompt.replace("## Additional information", "")
            prompt = prompt.replace("{priviledged_info}", "")
        return prompt

    def next_action(self, observation: PolicyObservation, log_extra: Dict[str, Any]) -> Optional[PolicyAction]:
        prompt = self._format_prompt(observation)
        self.logger.debug(f"Prompt:\n{prompt}")
        if observation.image is None:
            raise ValueError("VILA policy requires an RGB observation")
        outputs = self.model.generate(
            prompts=[prompt], images=[observation.image], return_probs=False
        )
        self.logger.info("VLM output: " + outputs[0])
        vlm_plan = parse_json_output(outputs[0])
        if 'plan' not in vlm_plan or not vlm_plan['plan']:
            return None
        first_action = vlm_plan['plan'][0]
        action_params = [str(p) for p in first_action['parameters']]
        return PolicyAction(
            name=first_action['action'],
            parameters=action_params,
            raw_response=vlm_plan,
            metadata={'vlm_plan': vlm_plan},
        )
