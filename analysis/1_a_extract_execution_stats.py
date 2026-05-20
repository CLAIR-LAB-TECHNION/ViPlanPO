from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from analysis.execution_log import (
    PLANNING_DIR,
    extract_timestamp,
    find_execution_files,
    is_success,
    iter_log_records,
)

# CSV columns — the first five form the unique key matching validate_plans output.
COLUMNS = ["run_id", "policy_cls", "task", "scene_id", "instance_id", "success", "action_count", "planning_time"]

# Per-instance key: (run_id, policy_cls, task, scene_id, instance_id)
_InstanceKey = Tuple[str, str, str, str, int]

# Messages that carry explicit planner timing (conformant / symbolic planners).
_PLANNING_TIME_MSGS = frozenset({"New conformant plan", "Replanning executed."})


def _extract_instance_stats(path: Path) -> List[dict]:
    """Parse one execution log and return per-instance stats records."""
    run_id = extract_timestamp(path)

    action_counts: Dict[_InstanceKey, int] = {}
    # Prefer planning_time_seconds (explicit planner events) over compute_time (VLM calls).
    planner_times: Dict[_InstanceKey, float] = {}   # from planning events
    compute_times: Dict[_InstanceKey, float] = {}   # from Next action decided
    finish_entries: Dict[_InstanceKey, dict] = {}

    for entry in iter_log_records(path):
        msg = entry.get("msg")
        task = entry.get("task")
        scene_id = entry.get("scene_id")
        instance_id = entry.get("instance_id")
        policy_cls = entry.get("policy_cls")

        if None in (task, scene_id, instance_id, policy_cls):
            continue

        key: _InstanceKey = (run_id, policy_cls, task, scene_id, instance_id)

        if msg == "Action completed":
            action_counts[key] = action_counts.get(key, 0) + 1

        elif msg in _PLANNING_TIME_MSGS:
            pt = entry.get("planning_time_seconds")
            if pt is not None:
                planner_times[key] = planner_times.get(key, 0.0) + float(pt)

        elif msg == "Next action decided":
            ct = entry.get("compute_time")
            if ct is not None:
                compute_times[key] = compute_times.get(key, 0.0) + float(ct)

        elif msg == "Finished planning loop":
            finish_entries[key] = entry

    records = []
    for key, finish_entry in finish_entries.items():
        run_id_k, policy_cls_k, task_k, scene_id_k, instance_id_k = key
        # Use explicit planner time if available; fall back to VLM compute time.
        planning_time = planner_times.get(key) if key in planner_times else compute_times.get(key, 0.0)
        records.append({
            "run_id": run_id_k,
            "policy_cls": policy_cls_k,
            "task": task_k,
            "scene_id": scene_id_k,
            "instance_id": instance_id_k,
            "success": is_success(finish_entry),
            "action_count": action_counts.get(key, 0),
            "planning_time": planning_time,
        })

    return records


def main() -> None:
    execution_files = find_execution_files()
    if not execution_files:
        print(f"No execution.jsonl files found under {PLANNING_DIR}.", file=sys.stderr)
        sys.exit(1)

    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()

    for execution_file in execution_files:
        for record in _extract_instance_stats(execution_file):
            writer.writerow(record)


if __name__ == "__main__":
    main()
