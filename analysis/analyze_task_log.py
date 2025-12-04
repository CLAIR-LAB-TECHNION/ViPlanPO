"""
Generate summary tables from JSONL task logs.

This script scans the files in ``results/for_summary`` and builds two CSV reports:
1. ``analysis/policy_instance_summary.csv``: per policy/problem/scene/instance metrics.
2. ``analysis/first_problem_simple_summary.csv``: aggregate metrics for the first
   problem in the dataset, grouped by policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

LOG_DIR = Path("results/for_summary")
OUTPUT_DIR = Path("analysis")


def _load_logs(paths: Iterable[Path]) -> pd.DataFrame:
    """Load all JSONL records into a DataFrame."""
    records = []
    for path in sorted(paths):
        with path.open("r") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _add_problem_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Attach problem name and difficulty derived from the problem_file path."""
    df = df.copy()
    df["problem"] = df["problem_file"].apply(
        lambda path: Path(path).name if pd.notna(path) else None
    )
    df["difficulty"] = df["problem_file"].apply(
        lambda path: Path(path).parent.name if pd.notna(path) else None
    )
    return df


def _count_events(df: pd.DataFrame, message: str, index_cols: list[str]) -> pd.Series:
    """Count how many times a specific message appears for each group."""
    matching = df[df["msg"] == message].copy()
    return matching.groupby(index_cols)["msg"].count().rename(message)


def build_policy_instance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the per-instance summary table."""
    enriched = _add_problem_fields(df)
    index_cols = ["policy_cls", "problem", "scene_id", "instance_id", "difficulty"]

    finished = enriched[enriched["msg"] == "Finished planning loop"].copy()
    finished = finished[index_cols + ["completed", "elapsed_time"]]
    finished = finished.rename(columns={"completed": "success", "elapsed_time": "time"})
    finished["success"] = finished["success"].astype(bool)
    finished = finished.set_index(index_cols)

    for column, message in [
        ("steps", "Action completed"),
        ("replans", "Replanning executed."),
        ("explores", "No plan. Exploring"),
    ]:
        finished[column] = _count_events(enriched, message, index_cols)

    finished[["steps", "replans", "explores"]] = (
        finished[["steps", "replans", "explores"]].fillna(0).astype(int)
    )
    finished = finished.reset_index()
    return finished


def summarize_first_problem(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics for the first problem grouped by policy."""
    if table.empty:
        return pd.DataFrame()

    # Use the first problem encountered in the table to mirror the input order instead
    # of sorting alphabetically.
    first_problem = table.iloc[0]["problem"]
    problem_subset = table[table["problem"] == first_problem]
    if problem_subset.empty:
        return pd.DataFrame()

    # Use the difficulty value from this problem for the column header.
    difficulty_label = problem_subset["difficulty"].iloc[0]

    grouped = problem_subset.groupby("policy_cls").agg(
        success_rate=("success", "mean"),
        avg_steps=("steps", "mean"),
        avg_replans=("replans", "mean"),
        avg_time=("time", "mean"),
    )

    grouped = grouped.rename(index={"DefaultVILAPolicy": "ViLaPolicy"})
    grouped.columns = pd.MultiIndex.from_product(
        [
            [difficulty_label.capitalize() if isinstance(difficulty_label, str) else difficulty_label],
            ["Success rate", "Avg. Steps", "Avg. Replans", "Avg. time"],
        ]
    )
    return grouped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_paths = list(LOG_DIR.glob("*.log"))
    df = _load_logs(log_paths)
    if df.empty:
        print("No log files found in", LOG_DIR)
        return

    policy_table = build_policy_instance_table(df)
    policy_path = OUTPUT_DIR / "policy_instance_summary.csv"
    policy_table.to_csv(policy_path, index=False)
    print(f"Wrote per-instance table to {policy_path}")

    first_problem_table = summarize_first_problem(policy_table)
    summary_path = OUTPUT_DIR / "first_problem_simple_summary.csv"
    first_problem_table.to_csv(summary_path)
    print(f"Wrote first-problem summary to {summary_path}")


if __name__ == "__main__":
    main()
