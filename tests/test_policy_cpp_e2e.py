import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pytest

pytest.importorskip("PIL", reason="Pillow is required for PolicyCPP e2e test")
from PIL import Image
from unified_planning.exceptions import (
    UPNoRequestedEngineAvailableException,
    UPNoSuitableEngineAvailableException,
)
from unified_planning import environment as up_environment
from unified_planning import shortcuts as up_shortcuts

from viplan.policies.policy_cpp import PolicyCPP
from viplan.policies.policy_interface import PolicyAction, PolicyObservation
from viplan.policies.up_utils import create_up_problem


class DummyVQA:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, images, query_batch, *token_groups_of_interest):
        # Return a simple distribution favouring "yes" for every query.
        return [[0.7, 0.2, 0.1] for _ in query_batch]


def test_policy_cpp_end_to_end(monkeypatch):
    predicate_language = {
        "reachable": "Is {0} reachable by the agent?",
        "holding": "Is the agent holding {0}?",
        "open": "Is {0} open?",
        "ontop": "Is {0} on top of {1}?",
        "inside": "Is {0} inside {1}?",
        "nextto": "Is {0} next to {1}?",
    }

    monkeypatch.setattr(
        up_environment.get_environment().factory,
        "add_meta_engine",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "viplan.policies.policy_cpp.OneshotPlanner",
        lambda *args, **kwargs: up_shortcuts.OneshotPlanner(name="pyperplan"),
    )

    logger = logging.getLogger("policy_cpp_e2e")
    logger.addHandler(logging.NullHandler())

    domain_path = Path("data/planning/igibson/domain.pddl")
    problem_path = Path("data/planning/igibson/simple/locking_every_door_simple.pddl")

    # Avoid external API calls by swapping in a lightweight VQA stub while
    # keeping all other components real.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("viplan.policies.policy_cpp.OpenAIVQA", DummyVQA)

    try:
        policy = PolicyCPP(
            predicate_language=predicate_language,
            domain_file=str(domain_path),
            problem_file=str(problem_path),
            model_name="gpt-test",
            tasks_logger=logger,
            goal_string="Answer yes or no.",
        )
    except (
        RuntimeError,
        UPNoRequestedEngineAvailableException,
        UPNoSuitableEngineAvailableException,
    ) as exc:
        pytest.skip(f"Planner backend not available: {exc}")

    observation_image = Image.new("RGB", (640, 480), color=(255, 255, 255))
    observation = PolicyObservation(
        image=observation_image,
        problem=create_up_problem(str(domain_path), str(problem_path)),
        predicate_language=predicate_language,
        previous_actions=[],
    )

    action = policy.next_action(observation, log_extra={})

    assert isinstance(action, PolicyAction)
    assert action.name
    assert list(action.parameters)
