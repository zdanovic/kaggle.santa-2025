#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.metric import ParticipantVisibleError, score_detailed


def _score_group(df_group: pd.DataFrame) -> float:
    _, per_group = score_detailed(df_group)
    return list(per_group.values())[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    submission_files = list(results_dir.rglob("submission_refined.csv"))
    if not submission_files:
        submission_files = list(results_dir.rglob("submission.csv"))

    best_groups: Dict[int, pd.DataFrame] = {}
    best_scores: Dict[int, float] = {}
    best_sources: Dict[int, str] = {}

    for sub_path in submission_files:
        df = pd.read_csv(sub_path)
        df["group"] = df["id"].astype(str).str.split("_").str[0].astype(int)
        for n, df_group in df.groupby("group"):
            df_group = df_group[["id", "x", "y", "deg"]].reset_index(drop=True)
            try:
                score = _score_group(df_group)
            except ParticipantVisibleError:
                continue
            if n not in best_scores or score < best_scores[n]:
                best_scores[n] = score
                best_groups[n] = df_group
                best_sources[n] = str(sub_path)

    if not best_groups:
        raise SystemExit("No submissions found under results-dir.")

    expected = set(range(1, 201))
    missing = sorted(expected - set(best_groups.keys()))
    if missing:
        raise SystemExit(f"Missing groups in pool: {missing}")

    combined = pd.concat(best_groups.values(), ignore_index=True)
    combined["group"] = combined["id"].astype(str).str.split("_").str[0].astype(int)
    combined["item"] = combined["id"].astype(str).str.split("_").str[1].astype(int)
    combined = combined.sort_values(["group", "item"]).drop(columns=["group", "item"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(Path(args.output), index=False)

    meta_path = Path(args.output).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"sources": best_sources, "scores": best_scores}, indent=2))


if __name__ == "__main__":
    main()
