from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

PLANNING_DIR = Path("results/planning")
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def find_execution_files() -> List[Path]:
    """Return all ``execution.jsonl`` files under planning results sorted by path."""
    if not PLANNING_DIR.exists():
        return []
    candidates = [
        p for p in PLANNING_DIR.rglob("execution.jsonl")
        if "backup" not in p.parts
    ]
    candidates.sort(key=lambda path: path.as_posix())
    return candidates


def extract_timestamp(path: Path) -> str:
    """Extract the run timestamp component from the execution log path."""
    for part in path.parts:
        if TIMESTAMP_PATTERN.fullmatch(part):
            return part
    return path.stem


def iter_log_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON records from a JSONL execution log, skipping invalid lines.

    Also reads rotated siblings (execution.jsonl.1, .2, …) from oldest to newest
    so that log rotation is transparent to callers.
    """
    # Collect rotated files in descending numeric order (highest = oldest).
    rotated: List[Path] = sorted(
        path.parent.glob(path.name + ".*"),
        key=lambda p: int(p.suffix.lstrip(".")) if p.suffix.lstrip(".").isdigit() else 0,
        reverse=True,
    )
    for src in [*rotated, path]:
        with src.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def is_success(entry: Dict[str, Any]) -> bool:
    """Return whether a ``Finished planning loop`` log entry represents a success."""
    return bool(entry.get("completed", False))
