#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.pipeline import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_config = load_config(Path(args.config))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)

    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        run_experiment(config, output_dir / f"exp_seed_{seed}")


if __name__ == "__main__":
    main()
