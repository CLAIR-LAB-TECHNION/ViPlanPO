import json
from typing import Optional

from viplan.code_helpers import get_logger
from viplan.experiments.policy_interface import Policy, PolicyAction, PolicyObservation


def parse_json_output(output):
    new = output.replace('\n', '')
    if new.startswith('```json'):
        new = new[len('```json'):-3]
    new = new.strip()
    new = json.loads(new)
    return new


class DefaultVILAPolicy(Policy):
    """Default policy that reproduces the original VILA planning loop."""

    def __init__(
        self,
        model,
        base_prompt,
        logger=None,
        predicate_language=None,
        priviledged_info_formatter=None,
        **kwargs,
    ):
        super().__init__(predicate_language=predicate_language)
        self.model = model
        self.base_prompt = base_prompt
        self.logger = logger or get_logger()
        self.priviledged_info_formatter = priviledged_info_formatter

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
                priviledged_str = ""
                if self.priviledged_info_formatter:
                    priviledged_str = self.priviledged_info_formatter(priviledged_preds)
                prompt = prompt.replace("{priviledged_info}", priviledged_str)
        else:
            prompt = prompt.replace("## Additional information", "")
            prompt = prompt.replace("{priviledged_info}", "")
        return prompt

    def next_action(self, observation: PolicyObservation) -> Optional[PolicyAction]:
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
