from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

PLANNING_DIR = Path("results/planning")
OUTPUT_DIR = PLANNING_DIR / "vila_plans"
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def _find_latest_execution_file() -> Path | None:
    """Return the most recently modified ``execution.jsonl`` under planning results."""
    if not PLANNING_DIR.exists():
        return None

    candidates = list(PLANNING_DIR.rglob("execution.jsonl"))
    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.stat().st_mtime, path.as_posix()))
    return candidates[-1]


def _extract_timestamp(path: Path) -> str:
    """Extract the run timestamp component from the execution log path."""
    for part in path.parts:
        if TIMESTAMP_PATTERN.fullmatch(part):
            return part
    return path.stem


def _format_plan(plan: Sequence[dict]) -> List[str]:
    """Convert a plan list into numbered human-readable strings."""
    formatted_steps: List[str] = []
    for index, step in enumerate(plan):
        action = step.get("action", "")
        parameters = step.get("parameters", [])
        if isinstance(parameters, Iterable) and not isinstance(parameters, (str, bytes)):
            parameters_str = ", ".join(map(str, parameters))
        else:
            parameters_str = str(parameters)
        formatted_steps.append(f"{index}) {action}({parameters_str})")
    return formatted_steps


def _load_plans(path: Path) -> List[List[str]]:
    """Read all VLM plans from the execution log."""
    plans: List[List[str]] = []
    with path.open("r") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("msg") != "Got VLM plan":
                continue

            args = record.get("args") if isinstance(record, dict) else None
            plan = None
            if isinstance(args, dict):
                plan = args.get("plan")
            if plan is None:
                plan = record.get("plan")

            if not isinstance(plan, list):
                continue

            plans.append(_format_plan(plan))
    return plans


def main() -> None:
    latest_execution = _find_latest_execution_file()
    if latest_execution is None:
        print(f"No execution.jsonl files found under {PLANNING_DIR}.")
        return

    plans = _load_plans(latest_execution)
    if not plans:
        print(f"No VLM plans found in {latest_execution}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _extract_timestamp(latest_execution)
    output_path = OUTPUT_DIR / f"{timestamp}.txt"

    with output_path.open("w") as handle:
        for index, plan in enumerate(plans, start=1):
            handle.write(f"Plan {index}:\n")
            handle.write("\n".join(plan))
            if index < len(plans):
                handle.write("\n\n")

    print(f"Saved {len(plans)} plan(s) to {output_path}")


if __name__ == "__main__":
    main()
