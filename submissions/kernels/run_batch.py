#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _detect_dataset_dir() -> Path | None:
    env_dir = os.environ.get("DATASET_DIR") or os.environ.get("KAGGLE_DATASET_DIR")
    if env_dir:
        return Path(env_dir)
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    for cand in candidates:
        if (cand / "kaggle" / "batches.json").exists():
            return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--results-dir", default="/kaggle/working/exp_batches")
    args = parser.parse_args()

    dataset_dir = _detect_dataset_dir()
    if not dataset_dir:
        raise SystemExit("Dataset dir not found. Set DATASET_DIR or attach dataset.")

    batches_path = dataset_dir / "kaggle" / "batches.json"
    data = json.loads(batches_path.read_text())

    if args.batch not in data["batches"]:
        raise SystemExit(f"Unknown batch: {args.batch}")

    seeds = data["batches"][args.batch]
    config_rel = data.get("config", "configs/kaggle_independent.yaml")

    config_path = dataset_dir / config_rel
    output_dir = Path(args.results_dir) / args.batch

    cmd = [
        sys.executable,
        str(dataset_dir / "kaggle" / "run_all.py"),
        "--config",
        str(config_path),
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--results-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
