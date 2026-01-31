#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.metric import score_detailed
from santa2025.scoring import per_group_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=False, help="Optional per-n CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    total, per_group = score_detailed(df)
    print(f"total_score: {total}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_group_dataframe(per_group).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
