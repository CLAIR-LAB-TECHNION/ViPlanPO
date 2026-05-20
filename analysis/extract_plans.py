from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from analysis.execution_log import (
    PLANNING_DIR,
    extract_timestamp,
    find_execution_files,
    iter_log_records,
)

DOMAIN_FILE = "data/planning/igibson/domain.pddl"

# Key that uniquely identifies a task instance within a run.
_InstanceKey = Tuple[str, str, int]  # (task, scene_id, instance_id)


def _normalize_plan(raw_plan: list) -> List[Dict]:
    """Return the plan as a list of {action, parameters} dicts ready for UP validation.

    Action names and object parameters are lower-cased to match PDDL conventions.
    Parameters are always a list of strings, even when the log encodes them otherwise.
    """
    steps = []
    for step in raw_plan:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).lower().strip()
        raw_params = step.get("parameters", [])
        if isinstance(raw_params, (str, bytes)):
            parameters = [str(raw_params).lower().strip()]
        else:
            parameters = [str(p).lower().strip() for p in raw_params]
        steps.append({"action": action, "parameters": parameters})
    return steps


def _load_initial_plans(path: Path) -> List[dict]:
    """Read the first VLM plan for each task instance from the execution log.

    Returns a list of records, one per unique (task, scene_id, instance_id),
    in the order they first appear in the log.
    """
    seen: Dict[_InstanceKey, bool] = {}
    records: List[dict] = []

    for log in iter_log_records(path):
        if log.get("msg") != "Got VLM plan":
            continue

        args = log.get("args") if isinstance(log, dict) else None
        src = args if isinstance(args, dict) else log

        raw_plan = src.get("plan")
        if not isinstance(raw_plan, list):
            continue

        task = src.get("task")
        scene_id = src.get("scene_id")
        instance_id = src.get("instance_id")
        key: _InstanceKey = (task, scene_id, instance_id)

        if key in seen:
            continue
        seen[key] = True

        records.append({
            "run_id": extract_timestamp(path),
            "policy_cls": src.get("policy_cls"),
            "task": task,
            "scene_id": scene_id,
            "instance_id": instance_id,
            "problem_file": src.get("problem_file"),
            "domain_file": DOMAIN_FILE,
            "plan": _normalize_plan(raw_plan),
        })

    return records


def main() -> None:
    execution_files = find_execution_files()
    if not execution_files:
        print(f"No execution.jsonl files found under {PLANNING_DIR}.")
        return

    for execution_file in execution_files:
        records = _load_initial_plans(execution_file)
        if not records:
            print(f"No VLM plans found in {execution_file}.")
            continue

        timestamp = extract_timestamp(execution_file)
        output_path = execution_file.parent / f"initial_plans_{timestamp}.jsonl"

        with output_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        print(f"Saved {len(records)} initial plan(s) to {output_path}")


if __name__ == "__main__":
    main()
