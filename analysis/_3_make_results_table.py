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


def _iqm(series: pd.Series) -> float:
    """Compute the interquartile mean of a pandas Series, ignoring NaNs."""
    from scipy import stats
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(stats.trim_mean(s, 0.25))


def _warn_instance_mismatches(stats: pd.DataFrame) -> None:
    """Warn about duplicate or mismatched instance sets across policies.

    Runs on the full (unfiltered) stats so that extra tasks present in only
    some policies are detected before any task filter silently drops them.
    """
    _WARN = "\033[33m⚠️  WARNING"
    _RST  = "\033[0m"

    difficulties = [d for d in DIFFICULTY_ORDER if d in stats["difficulty"].unique()]
    policies     = [p for p in POLICY_ORDER     if p in stats["policy_dir"].unique()]
    INSTANCE_KEY = ["task", "scene_id", "instance_id"]

    for diff in difficulties:
        instance_sets: Dict[str, set] = {}
        for pol in policies:
            grp = stats[(stats["difficulty"] == diff) & (stats["policy_dir"] == pol)]
            keys = list(map(tuple, grp[INSTANCE_KEY].values))
            duplicates = {k for k in keys if keys.count(k) > 1}
            if duplicates:
                print(
                    f"{_WARN} [{diff}] '{pol}': {len(duplicates)} instance(s) appear more than once: {duplicates}.{_RST}",
                    file=sys.stderr,
                )
            instance_sets[pol] = set(keys)

        reference = next(iter(instance_sets.values()))
        for pol, inst_set in instance_sets.items():
            if inst_set != reference:
                only_in_ref   = reference - inst_set
                only_in_other = inst_set - reference
                ref_pol = policies[0]
                print(
                    f"{_WARN} [{diff}]: instance set mismatch between '{ref_pol}' and '{pol}'. "
                    f"Only in '{ref_pol}': {len(only_in_ref)}, only in '{pol}': {len(only_in_other)}.{_RST}",
                    file=sys.stderr,
                )


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
                _iqm(grp["initial_plan_valid"]) * 100 if n else float("nan")
            )
            data["Num actions taken"].append(
                _iqm(grp["action_count"]) if n else float("nan")
            )
            data["Planning time"].append(
                grp["planning_time"].median() if n else float("nan")
            )

    return pd.DataFrame(data, index=col_index).T


TASKS = {"sorting_books", "cleaning_out_drawers"}


def main() -> None:
    stats = load_stats()
    valid = load_validation()
    dropped = sorted(set(stats["task"].unique()) - TASKS)
    if dropped:
        _WARN = "\033[33m⚠️  WARNING"
        _RST  = "\033[0m"
        print(f"{_WARN}: tasks present in data but excluded from table: {dropped}.{_RST}", file=sys.stderr)
    stats = stats[stats["task"].isin(TASKS)]
    valid = valid[valid["task"].isin(TASKS)]
    _warn_instance_mismatches(stats)
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
