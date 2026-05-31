from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from analysis.execution_log import (
    PLANNING_DIR,
    extract_difficulty,
    extract_timestamp,
    find_execution_files,
    iter_log_records,
)

DOMAIN_FILE = "data/planning/igibson/domain-cond.pddl"

# Maps policy_cls → log message that carries the initial plan.
_PLAN_MSG = {
    "DefaultVILAPolicy": "Got VLM plan",
    "PolicyCPP": "New conformant plan",
    "PolicyPlan": "New conformant plan",
}

# Maps policy_cls → log message that signals no plan was found.
# Only policies listed here trigger the completeness check.
_NO_PLAN_MSG = {
    "PolicyCPP": "No conformant plan found for the current belief.",
}

# Key that uniquely identifies a task instance within a run.
_InstanceKey = Tuple[str, str, int]  # (task, scene_id, instance_id)


def _parse_action_string(s: str) -> Dict:
    """Parse a UP ActionInstance str like 'navigate-to(hardback_1, shelf_1)' into a dict."""
    s = s.strip()
    paren = s.index("(")
    action = s[:paren].lower().strip()
    action = action.replace("_0", "")
    params_str = s[paren + 1 : s.rindex(")")].strip()
    parameters = [p.lower().strip() for p in params_str.split(",")] if params_str else []
    return {"action": action, "parameters": parameters}


def _normalize_plan(raw_plan: list) -> List[Dict]:
    """Return the plan as a list of {action, parameters} dicts ready for UP validation.

    Handles both formats produced by the two policies:
    - DefaultVILAPolicy: list of {"action": ..., "parameters": [...]} dicts
    - PolicyCPP: list of str(ActionInstance) strings, e.g. "navigate-to(hardback_1)"
    """
    steps = []
    for step in raw_plan:
        if isinstance(step, str):
            steps.append(_parse_action_string(step))
        elif isinstance(step, dict):
            action = str(step.get("action", "")).lower().strip()
            raw_params = step.get("parameters", [])
            if isinstance(raw_params, (str, bytes)):
                parameters = [str(raw_params).lower().strip()]
            else:
                parameters = [str(p).lower().strip() for p in raw_params]
            steps.append({"action": action, "parameters": parameters})
    return steps


def _load_initial_plans(path: Path) -> List[dict]:
    """Read the first plan for each task instance from the execution log.

    For PolicyCPP instances that never found a conformant plan, records are
    emitted with plan=None.  Raises ValueError if a PolicyCPP instance finishes
    without either a plan message or a no-plan message.
    """
    run_id = extract_timestamp(path)
    difficulty = extract_difficulty(path)

    plan_records: Dict[_InstanceKey, dict] = {}
    no_plan_keys: Set[_InstanceKey] = set()
    finished_meta: Dict[_InstanceKey, dict] = {}

    for log in iter_log_records(path):
        msg = log.get("msg")

        # DefaultVILAPolicy nests plan data (incl. policy_cls) under args;
        # PolicyCPP puts everything at the top level.
        args = log.get("args") if isinstance(log, dict) else None
        src = args if isinstance(args, dict) else log
        policy_cls = src.get("policy_cls") or log.get("policy_cls")

        # Track every instance that completed the planning loop.
        if msg == "Finished planning loop":
            task, scene_id, instance_id = (
                log.get("task"), log.get("scene_id"), log.get("instance_id"),
            )
            if None not in (task, scene_id, instance_id, policy_cls):
                key: _InstanceKey = (task, scene_id, instance_id)
                finished_meta[key] = {
                    "run_id": run_id,
                    "policy_cls": policy_cls,
                    "task": task,
                    "scene_id": scene_id,
                    "instance_id": instance_id,
                    "difficulty": difficulty,
                    "problem_file": log.get("problem_file"),
                    "domain_file": DOMAIN_FILE,
                }
            continue

        # Track "no plan found" events.
        if policy_cls and msg == _NO_PLAN_MSG.get(policy_cls):
            task, scene_id, instance_id = (
                src.get("task") or log.get("task"),
                src.get("scene_id") or log.get("scene_id"),
                src.get("instance_id") or log.get("instance_id"),
            )
            no_plan_keys.add((task, scene_id, instance_id))
            continue

        # Extract the initial plan (first occurrence per instance).
        expected_msg = _PLAN_MSG.get(policy_cls)
        if expected_msg is None or msg != expected_msg:
            continue

        raw_plan = src.get("plan")
        if not isinstance(raw_plan, list):
            continue

        task = src.get("task")
        scene_id = src.get("scene_id")
        instance_id = src.get("instance_id")
        key = (task, scene_id, instance_id)

        if key in plan_records:
            continue

        plan_records[key] = {
            "run_id": run_id,
            "policy_cls": policy_cls,
            "task": task,
            "scene_id": scene_id,
            "instance_id": instance_id,
            "difficulty": difficulty,
            "problem_file": src.get("problem_file"),
            "domain_file": DOMAIN_FILE,
            "plan": _normalize_plan(raw_plan),
        }

    # Build output: one record per finished instance.
    records: List[dict] = []
    for key, meta in finished_meta.items():
        if key in plan_records:
            records.append(plan_records[key])
        elif key in no_plan_keys:
            records.append({**meta, "plan": None})
        elif meta["policy_cls"] in _NO_PLAN_MSG:
            raise ValueError(
                f"Instance {key} (policy={meta['policy_cls']}) in {path} "
                f"finished without a plan or a no-plan message."
            )
        # Policies without a defined _NO_PLAN_MSG (e.g. DefaultVILAPolicy) are
        # included only when a plan was found; absent otherwise.

    return records


def main() -> None:
    execution_files = find_execution_files()
    if not execution_files:
        print(f"No execution.jsonl files found under {PLANNING_DIR}.")
        return

    for execution_file in execution_files:
        records = _load_initial_plans(execution_file)
        if not records:
            print(f"No plans found in {execution_file}.")
            continue

        output_path = execution_file.parent / "initial_plans.jsonl"

        with output_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        print(f"Saved {len(records)} initial plan(s) to {output_path}")


if __name__ == "__main__":
    main()
