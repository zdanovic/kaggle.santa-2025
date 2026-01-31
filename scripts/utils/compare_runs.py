#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.analysis import compare_per_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Path to per_n.csv")
    parser.add_argument("--b", required=True, help="Path to per_n.csv")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = compare_per_n(Path(args.a), Path(args.b))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
