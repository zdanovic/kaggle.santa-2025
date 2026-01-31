#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-n", required=True)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.per_n).sort_values("group_score", ascending=False)
    targets = df.head(args.top_k)["n"].tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(str(n) for n in targets))


if __name__ == "__main__":
    main()
