from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="unified_planning")

from analysis.execution_log import PLANNING_DIR, find_execution_files
from analysis._1_a_extract_execution_stats import _extract_instance_stats
from analysis._2_b_validate_plans import _find_plan_files, _validate_record

# Display ordering — entries absent from the data are silently skipped.
DIFFICULTY_ORDER = ["simple", "medium", "hard"]
POLICY_ORDER     = ["vila", "plan", "cpp"]
POLICY_LABEL: Dict[str, str] = {
    "vila": "VLM-P (vila)",
    "plan": "VLM-G (policy-plan)",
    "cpp":  "VLM-PG (ours)",
}

METRICS = [
    "Execution success rate",
    "First-plan satisficing",
    "Num actions taken",
    "Planning time",
]

OUTPUT_PATH = Path(__file__).parent.parent / "../RoVLaP-NeuS-2026-/content/results_table.tex"


def _path_meta(path: Path) -> Tuple[str, str]:
    """Return (difficulty, policy_dir) from a path rooted under PLANNING_DIR."""
    # PLANNING_DIR / igibson / {difficulty} / {policy_dir} / ...
    rel = path.relative_to(PLANNING_DIR)
    return rel.parts[1], rel.parts[2]


def load_stats() -> pd.DataFrame:
    rows: List[dict] = []
    for path in find_execution_files():
        try:
            difficulty, policy_dir = _path_meta(path)
        except (ValueError, IndexError):
            continue
        for record in _extract_instance_stats(path):
            rows.append({"difficulty": difficulty, "policy_dir": policy_dir, **record})
    return pd.DataFrame(rows)


def load_validation() -> pd.DataFrame:
    """Return one row per instance in initial_plans.jsonl files.

    ``initial_plan_valid`` is 1.0 (valid), 0.0 (invalid or no plan found), or
    NaN when the instance was never in any initial_plans file (e.g. plan policy).
    """
    cache: Dict[Tuple[str, str], object] = {}
    rows: List[dict] = []
    for plan_file in _find_plan_files():
        try:
            difficulty, policy_dir = _path_meta(plan_file)
        except (ValueError, IndexError):
            continue
        with plan_file.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                valid = _validate_record(record, cache)
                # None means no plan was found — counts as not satisficing (0).
                rows.append({
                    "difficulty":          difficulty,
                    "policy_dir":          policy_dir,
                    "run_id":              record.get("run_id", ""),
                    "policy_cls":          record.get("policy_cls", ""),
                    "task":                record.get("task", ""),
                    "scene_id":            record.get("scene_id", ""),
                    "instance_id":         record.get("instance_id"),
                    "initial_plan_valid":  0.0 if valid is None else float(valid),
                })
    return pd.DataFrame(rows)


def build_table(stats: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    KEY = ["run_id", "policy_cls", "task", "scene_id", "instance_id",
           "difficulty", "policy_dir"]
    df = stats.merge(valid[KEY + ["initial_plan_valid"]], on=KEY, how="left")

    difficulties = [d for d in DIFFICULTY_ORDER if d in df["difficulty"].unique()]
    policies     = [p for p in POLICY_ORDER     if p in df["policy_dir"].unique()]

    col_index = pd.MultiIndex.from_tuples([
        (d.capitalize(), POLICY_LABEL.get(p, p))
        for d in difficulties
        for p in policies
    ])

    data: Dict[str, List] = {m: [] for m in METRICS}
    for diff in difficulties:
        for pol in policies:
            grp = df[(df["difficulty"] == diff) & (df["policy_dir"] == pol)]
            n = len(grp)
            data["Execution success rate"].append(
                grp["success"].mean() * 100 if n else float("nan")
            )
            # NaN in initial_plan_valid (instances absent from validation) are
            # excluded from mean() automatically — distinct from 0.0 (no plan).
            data["First-plan satisficing"].append(
                grp["initial_plan_valid"].mean() * 100 if n else float("nan")
            )
            data["Num actions taken"].append(
                grp["action_count"].mean() if n else float("nan")
            )
            data["Planning time"].append(
                grp["planning_time"].median() if n else float("nan")
            )

    return pd.DataFrame(data, index=col_index).T


TASKS = {"sorting_books", "cleaning_out_drawers"}


def main() -> None:
    stats = load_stats()
    valid = load_validation()
    stats = stats[stats["task"].isin(TASKS)]
    valid = valid[valid["task"].isin(TASKS)]
    table = build_table(stats, valid)

    latex = table.to_latex(
        float_format="%.1f",
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        na_rep="--",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(latex)
    print(table.to_string(float_format="%.1f"))
    print(f"\nLaTeX saved to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
