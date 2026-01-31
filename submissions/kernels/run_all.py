#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import os
import shutil
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
        if (cand / "configs" / "kaggle_independent.yaml").exists():
            return cand
    return None


def _copy_dataset(dataset_dir: Path, workdir: Path) -> None:
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")
    print(f"copying dataset -> {workdir}", flush=True)
    shutil.copytree(dataset_dir, workdir, dirs_exist_ok=True)


def _install_requirements(workdir: Path) -> None:
    req = workdir / "requirements.txt"
    if not req.exists():
        print("requirements.txt not found, skipping install")
        return
    print("installing requirements", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)


def _run_job(config: Path, seed: int, output: Path) -> None:
    print(f"seed {seed}: start output={output}", flush=True)
    start = time.perf_counter()
    cmd = [
        sys.executable,
        "/kaggle/working/kaggle/run_job.py",
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start
    print(f"seed {seed}: done time_s={elapsed:.1f}", flush=True)


def _pool_results(results_dir: Path, output_path: Path) -> None:
    print(f"pooling results in {results_dir}", flush=True)
    cmd = [
        sys.executable,
        "/kaggle/working/scripts/pool_best_of.py",
        "--results-dir",
        str(results_dir),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _score_submission(submission: Path, out_path: Path) -> None:
    print(f"scoring submission {submission}", flush=True)
    cmd = [
        sys.executable,
        "/kaggle/working/scripts/score_submission.py",
        "--submission",
        str(submission),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/kaggle/working/configs/kaggle_independent.yaml")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--results-dir", default="/kaggle/working/exp_runs")
    parser.add_argument("--no-pool", action="store_true", default=False)
    parser.add_argument("--no-install", action="store_true", default=False)
    args = parser.parse_args()

    workdir = Path("/kaggle/working")
    dataset_dir = _detect_dataset_dir()
    if dataset_dir:
        print(f"dataset detected: {dataset_dir}", flush=True)
        _copy_dataset(dataset_dir, workdir)

    if not args.no_install:
        _install_requirements(workdir)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"seeds: {seeds}", flush=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        output = results_dir / f"exp_seed_{seed}"
        _run_job(Path(args.config), seed, output)

    if not args.no_pool:
        pool_dir = results_dir / "pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        submission_path = pool_dir / "best_of_submission.csv"
        _pool_results(results_dir, submission_path)
        _score_submission(submission_path, pool_dir / "per_n_best_of.csv")
        print("pooling done", flush=True)


if __name__ == "__main__":
    main()
