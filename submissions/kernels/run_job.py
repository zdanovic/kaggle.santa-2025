#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from santa2025.pipeline import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "kaggle.yaml"))
    parser.add_argument("--output", default="/kaggle/working/exp")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.seed is not None:
        config["seed"] = args.seed
    run_experiment(config, Path(args.output))


if __name__ == "__main__":
    main()
