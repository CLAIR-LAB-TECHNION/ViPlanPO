from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from unified_planning.engines import ValidationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance, SequentialPlan
from unified_planning.shortcuts import PlanValidator

PLANNING_DIR = Path("results/planning")

# Suppress the UP PDDL-reader warning about :universal-preconditions.
warnings.filterwarnings("ignore", category=UserWarning, module="unified_planning")

# CSV columns — the first five form the unique key for a plan instance.
COLUMNS = ["run_id", "policy_cls", "task", "scene_id", "instance_id", "valid"]

_ProblemCache = Dict[Tuple[str, str], object]  # (domain_file, problem_file) -> UP Problem


def _load_problem(domain_file: str, problem_file: str, cache: _ProblemCache):
    key = (domain_file, problem_file)
    if key not in cache:
        reader = PDDLReader()
        cache[key] = reader.parse_problem(domain_file, problem_file)
    return cache[key]


def _build_plan(problem, plan_steps: List[dict]) -> Optional[SequentialPlan]:
    obj_map = {o.name: o for o in problem.all_objects}
    act_map = {a.name: a for a in problem.actions}

    instances = []
    for step in plan_steps:
        action_name = step.get("action", "")
        params = step.get("parameters", [])
        if action_name not in act_map:
            return None
        try:
            grounded = tuple(obj_map[p] for p in params)
        except KeyError:
            return None
        instances.append(ActionInstance(act_map[action_name], grounded))
    return SequentialPlan(instances)


def _validate_record(record: dict, cache: _ProblemCache) -> bool:
    domain_file = record.get("domain_file", "")
    problem_file = record.get("problem_file", "")
    plan_steps = record.get("plan", [])

    if not domain_file or not problem_file:
        return False

    try:
        problem = _load_problem(domain_file, problem_file, cache)
    except Exception:
        return False

    plan = _build_plan(problem, plan_steps)
    if plan is None:
        return False

    try:
        with PlanValidator(name="sequential_plan_validator") as validator:
            result = validator.validate(problem, plan)
        return result.status == ValidationResultStatus.VALID
    except Exception:
        return False


def _find_plan_files() -> List[Path]:
    if not PLANNING_DIR.exists():
        return []
    files = list(PLANNING_DIR.rglob("initial_plans.jsonl"))
    files.sort(key=lambda p: p.as_posix())
    return files


def main() -> None:
    plan_files = _find_plan_files()
    if not plan_files:
        print(f"No initial_plans_*.jsonl files found under {PLANNING_DIR}.", file=sys.stderr)
        sys.exit(1)

    problem_cache: _ProblemCache = {}
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()

    for plan_file in plan_files:
        with plan_file.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                valid = _validate_record(record, problem_cache)
                writer.writerow({
                    "run_id": record.get("run_id", ""),
                    "policy_cls": record.get("policy_cls", ""),
                    "task": record.get("task", ""),
                    "scene_id": record.get("scene_id", ""),
                    "instance_id": record.get("instance_id", ""),
                    "valid": valid,
                })


if __name__ == "__main__":
    main()
