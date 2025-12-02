import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def is_jsonl_file(path: Path, sample_lines: int = 20) -> bool:
    """Check whether the file appears to contain JSON Lines data."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _ in range(sample_lines):
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return False


def find_latest_log_file(results_dir: Path) -> Path | None:
    """Return the newest JSONL-like log file within the results directory."""
    candidates: List[Path] = []
    for path in results_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".log", ".jsonl"}:
            continue
        try:
            _ = path.stat().st_mtime
        except OSError:
            continue
        candidates.append(path)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if is_jsonl_file(path):
            return path
    return None


def derive_reason(entry: Dict[str, Any]) -> str:
    """Extract a human readable failure reason from a log entry."""
    for key in ("policy_error", "reason", "error", "failure_reason"):
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def parse_lines(lines: List[str], source: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line in {source}: {stripped[:120]}")
            continue
        payload["_source_file"] = str(source)
        payload["_derived_reason"] = derive_reason(payload)
        payload["_completed_flag"] = bool(payload.get("completed"))
        records.append(payload)
    return records


def _render(lines: List[str]) -> None:
    """Render output in-place so the display remains stable."""

    content = "\n".join(lines)
    sys.stdout.write("\033[2J\033[H" + content + "\n")
    sys.stdout.flush()


def build_stats_lines(df: pd.DataFrame) -> List[str]:
    loops = df[df.get("msg") == "Finished planning loop"]
    total = len(loops)
    if total == 0:
        return ["No \"Finished planning loop\" entries yet."]

    successes = loops["completed"].fillna(False).astype(bool).sum()
    success_rate = successes / total if total else 0
    failures = total - successes
    lines = [
        (
            f"Planning loops: {total} | Completed: {successes} "
            f"({success_rate:.1%}) | Failures: {failures}"
        )
    ]

    if failures:
        failure_reasons = (
            loops.loc[~loops["completed"].fillna(False).astype(bool), "_derived_reason"]
            .fillna("unknown")
            .value_counts()
        )
        lines.append("Failure reasons:")
        for reason, count in failure_reasons.items():
            share = count / failures if failures else 0
            lines.append(f"  - {reason}: {count} ({share:.1%})")

    return lines


def monitor_logs(results_dir: Path, poll_interval: float = 10.0, stale_seconds: float = 60.0) -> None:
    current_file: Path | None = None
    handle = None
    position = 0
    buffer: List[Dict[str, Any]] = []

    while True:
        if current_file is None:
            current_file = find_latest_log_file(results_dir)
            if current_file is None:
                _render([f"No JSONL log files found in {results_dir}. Retrying..."])
                time.sleep(poll_interval)
                continue
            _render([f"Watching log file: {current_file}"])
            handle = current_file.open("r", encoding="utf-8")
            position = 0
            buffer = []

        handle.seek(position)
        new_lines = handle.readlines()
        position = handle.tell()

        new_records = parse_lines(new_lines, current_file)
        if new_records:
            buffer.extend(new_records)
            df = pd.DataFrame(buffer)
            _render(build_stats_lines(df) + [f"\nWatching log file:\n  {current_file.relative_to(results_dir)}"])
        else:
            file_age = time.time() - current_file.stat().st_mtime
            if file_age > stale_seconds:
                _render(
                    [
                        (
                            "No updates detected for "
                            f"{int(stale_seconds)} seconds. Searching for a new log file..."
                        )
                    ]
                )
                handle.close()
                current_file = None
                continue

        time.sleep(poll_interval)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    monitor_logs(results_dir)


if __name__ == "__main__":
    main()
