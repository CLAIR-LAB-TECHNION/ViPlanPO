"""Reusable policy interface for ViPlan experiments.

This module defines a light-weight abstraction that allows experiments to plug
in custom policies. A policy can combine classical planning, VLM reasoning or
any other mechanism to decide the next action to execute inside a simulator.

The interface is intentionally minimal so that existing experiments (e.g.
``benchmark_igibson_plan.py`` and ``benchmark_igibson_vila.py``) can adopt it
without restructuring their evaluation loops.  User-defined policies can also
reuse the same interface to integrate with other experiments.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Type

from unified_planning.model import Problem

from viplan.planning.igibson_client_env import iGibsonClient

try:  # PIL is an optional dependency at import time.
    from PIL import Image
except Exception:  # pragma: no cover - used only when PIL is not available.
    Image = Any  # type: ignore

@dataclass
class PolicyObservation:
    """Container describing the inputs that a policy can reason about."""

    image: Optional[Image.Image]
    problem: Problem
    env: iGibsonClient = field()
    predicate_groundings: Optional[Mapping[str, Any]] = None
    previous_actions: Iterable[Mapping[str, Any]] = field(default_factory=list)
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PolicyAction:
    """Structured description of the next action to execute."""

    name: str
    parameters: Sequence[str]
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_env_command(self) -> Dict[str, Any]:
        """Return a serialisable view of the action for logging purposes."""

        return {
            "action": self.name,
            "parameters": list(self.parameters),
            "metadata": self.metadata,
        }


class Policy:
    """Base class for reusable ViPlan policies.

    Policies must implement :meth:`next_action` and may optionally override
    :meth:`observe_outcome` to receive feedback from the environment.
    """

    def __init__(self):
        pass

    def reset(self, **_: Any) -> None:
        """Hook invoked whenever the environment is (re)initialised."""

    def observe_outcome(
        self,
        action: PolicyAction,
        success: bool,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Optional hook to receive action outcomes from the simulator."""

    def next_action(self, observation: PolicyObservation, log_extra: Dict[str, Any]) -> Optional[PolicyAction]:
        """Return the next action to execute inside the simulator."""

        raise NotImplementedError

# Source - https://stackoverflow.com/a/5883218
# Posted by Duncan, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-01, License - CC BY-SA 3.0

def inheritors(klass):
    subclasses = dict()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses[child.__name__] = child
                work.append(child)
    return subclasses


def resolve_policy_class(
    policy_path: Optional[str],
    default_cls: Type[Policy],
) -> Type[Policy]:
    """Resolve a dotted ``module:Class`` path into a :class:`Policy` subclass.

    Parameters
    ----------
    policy_path:
        String referencing the policy class (``"pkg.module:MyPolicy"``).  When
        ``None`` the ``default_cls`` is returned.
    default_cls:
        Policy to use when ``policy_path`` is not provided.
    """

    if not policy_path:
        return default_cls

    return inheritors(Policy)[policy_path]

    # if ":" in policy_path:
    #     module_name, class_name = policy_path.split(":", maxsplit=1)
    # else:
    #     parts = policy_path.rsplit(".", maxsplit=1)
    #     if len(parts) != 2:
    #         raise ValueError(
    #             "policy_path must be in the form 'module:Class' or 'module.Class'"
    #         )
    #     module_name, class_name = parts
    #
    # module = importlib.import_module(module_name)
    # policy_cls = getattr(module, class_name)
    #
    # if not issubclass(policy_cls, Policy):  # pragma: no cover - defensive.
    #     raise TypeError(
    #         f"Resolved class {policy_cls.__name__} is not a Policy subclass"
    #     )
    #
    # return policy_cls

