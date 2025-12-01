import json
from typing import Optional, Dict, Any

from viplan.code_helpers import get_logger
from viplan.policies.policy_interface import Policy, PolicyAction, PolicyObservation

preds_templates = {
    'reachable': "the {0} is reachable by the agent",
    'holding':   "the agent is holding the {0}",
    'open':      "the {0} is open",
    'ontop':     "the {0} is on top of the {1}",
    'inside':    "the {0} is inside the {1}",
    'nextto':    "the {0} is next to the {1}",
}

def parse_json_output(output):
    json_start = output.find('{')
    json_end = output.rfind('}')
    vlm_plan = json.loads(output[json_start:json_end + 1])
    return vlm_plan


def get_priviledged_predicates_str(predicates):
    priviledged_string = ""
    for pred, values in predicates.items():
        for args in values:
            if args:
                name = pred
                args = args
                descr = preds_templates[str(name)].format(*args)
                priviledged_string += descr + "\n"
    priviledged_string = priviledged_string.strip()
    return priviledged_string


class DefaultVILAPolicy(Policy):
    """Default policy that reproduces the original VILA planning loop."""

    def __init__(self, model, base_prompt, logger=None, **kwargs):
        predicate_language = kwargs.get('predicate_language', preds_templates)
        super().__init__(predicate_language=predicate_language)
        self.model = model
        self.base_prompt = base_prompt
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
