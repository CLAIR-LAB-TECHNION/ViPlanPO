"""Utilities for natural language prompts used by ViPlan policies."""

from __future__ import annotations

import os
from os import PathLike


PREDICATE_QUESTIONS = {
    'reachable': "Can the robot reach the {0} with its arm without moving its base?",
    'holding':   "Is the robot currently holding the {0} in its gripper?",
    'open':      "Is the {0} currently open?",
    'ontop':     "Is the {0} resting on top of the {1}?",
    'inside':    "Is the {0} located inside the {1}?",
    'nextto':    "Is the {0} positioned next to the {1}?",
}


def load_prompt(prompt_path: PathLike) -> str:
    """Return the contents of a prompt stored under ``data/prompts``.

    Parameters
    ----------
    prompt_path:
        Relative path (within ``data/prompts``) of the prompt to load.
    """

    prompt_path_str = os.fspath(prompt_path)
    if os.path.isabs(prompt_path_str) or prompt_path_str.startswith("data/prompts"):
        full_path = prompt_path_str
    else:
        full_path = os.path.join("data", "prompts", prompt_path_str)
    with open(full_path, "r") as prompt_file:
        return prompt_file.read()
