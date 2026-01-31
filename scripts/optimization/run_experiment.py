#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.pipeline import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_experiment(config, Path(args.output))


if __name__ == "__main__":
    main()
