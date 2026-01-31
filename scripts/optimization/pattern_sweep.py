#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.io import build_submission
from santa2025.metric import score_detailed
from santa2025.solver.pattern import PatternConfig, PatternSolver


def _angle_sets() -> List[List[Tuple[float, float]]]:
    return [
        [(0.0, 180.0)],
        [(0.0, 180.0), (90.0, 270.0)],
        [(0.0, 180.0), (45.0, 225.0)],
        [(0.0, 180.0), (60.0, 240.0)],
        [(0.0, 180.0), (30.0, 210.0)],
    ]


def _offset_sets() -> List[List[float]]:
    return [
        [0.55, 0.6, 0.65, 0.7, 0.75],
        [0.6, 0.7],
        [0.5, 0.6, 0.7],
        [0.65, 0.7, 0.75],
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-max", type=int, default=200)
    parser.add_argument("--out", default="results/pattern_sweep.csv")
    parser.add_argument("--global-squeeze", action="store_true", default=False)
    args = parser.parse_args()

    records = []
    for angle_pairs in _angle_sets():
        for offsets in _offset_sets():
            cfg = PatternConfig(
                angle_pairs=angle_pairs,
                offset_ratios=offsets,
                global_squeeze=args.global_squeeze,
            )
            solver = PatternSolver(cfg)
            groups = solver.solve(n_max=args.n_max, seed=0)
            submission = build_submission(groups, decimals=6)
            total_score, _ = score_detailed(submission)
            records.append(
                {
                    "angle_pairs": angle_pairs,
                    "offsets": offsets,
                    "global_squeeze": args.global_squeeze,
                    "total_score": total_score,
                }
            )
            print(f"{angle_pairs} offsets={offsets} score={total_score:.6f}")

    records.sort(key=lambda r: r["total_score"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["total_score", "angle_pairs", "offsets", "global_squeeze"]
        )
        writer.writeheader()
        writer.writerows(records)

    best = records[0]
    print(f"best score={best['total_score']:.6f} pairs={best['angle_pairs']} offsets={best['offsets']}")


if __name__ == "__main__":
    main()
