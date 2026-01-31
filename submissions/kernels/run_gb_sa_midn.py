#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        if (cand / "kaggle" / "baselines" / "gb_sa_best_submission.csv").exists():
            return cand
    return None


def _copy_dataset(dataset_dir: Path, workdir: Path) -> None:
    print(f"copying dataset -> {workdir}", flush=True)
    shutil.copytree(dataset_dir, workdir, dirs_exist_ok=True)


def _install_requirements(workdir: Path) -> None:
    req = workdir / "requirements.txt"
    if not req.exists():
        print("requirements.txt not found, skipping install", flush=True)
        return
    print("installing requirements", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="/kaggle/working/kaggle/baselines/gb_sa_best_submission.csv",
    )
    parser.add_argument("--sa-range", default="81-160")
    parser.add_argument("--sa-iter", type=int, default=1800000)
    parser.add_argument("--sa-hours", type=float, default=11.0)
    parser.add_argument("--sa-tstart", type=float, default=12.0)
    parser.add_argument("--sa-tend", type=float, default=0.01)
    parser.add_argument("--sa-seed", type=int, default=4242)
    parser.add_argument("--sa-save-every", type=int, default=10)
    parser.add_argument("--sa-processes", default="auto")
    parser.add_argument("--skip-gb", action="store_true", default=True)
    parser.add_argument("--out-dir", default="/kaggle/working/gb_sa_midn")
    args = parser.parse_args()

    workdir = Path("/kaggle/working")
    dataset_dir = _detect_dataset_dir()
    if not dataset_dir:
        raise SystemExit("Dataset dir not found. Attach solver dataset.")
    _copy_dataset(dataset_dir, workdir)
    _install_requirements(workdir)

    cmd = [
        sys.executable,
        str(workdir / "kaggle" / "run_gb_sa.py"),
        "--baseline",
        args.baseline,
        "--out-dir",
        args.out_dir,
        "--sa-range",
        args.sa_range,
        "--sa-iter",
        str(args.sa_iter),
        "--sa-hours",
        str(args.sa_hours),
        "--sa-tstart",
        str(args.sa_tstart),
        "--sa-tend",
        str(args.sa_tend),
        "--sa-seed",
        str(args.sa_seed),
        "--sa-processes",
        str(args.sa_processes),
        "--sa-save-every",
        str(args.sa_save_every),
    ]
    if args.skip_gb:
        cmd.append("--skip-gb")

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
