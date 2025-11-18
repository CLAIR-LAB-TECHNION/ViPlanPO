from typing import Optional

from viplan.code_helpers import get_logger
from viplan.policies.policy_interface import Policy, PolicyAction, PolicyObservation


class DefaultPlanningPolicy(Policy):
    """Default policy that sequentially executes planner actions."""

    def __init__(self, action_queue, predicate_language, logger=None, **kwargs):
        super().__init__(predicate_language=predicate_language)
        self.action_queue = action_queue
        self.logger = logger or get_logger()

    def next_action(self, observation: PolicyObservation) -> Optional[PolicyAction]:
        if not self.action_queue:
            return None
        action = self.action_queue.popleft()
        if self.logger:
            self.logger.debug(f"Policy selecting action {action}")
        return PolicyAction(
            name=action.action.name,
            parameters=[str(p) for p in action.actual_parameters],
            metadata={'plan_action': action},
        )