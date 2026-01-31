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
        if (cand / "configs" / "kaggle_independent.yaml").exists():
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
        default="/kaggle/input/santa-2025-bbox3-baseline/best_submission.csv",
    )
    parser.add_argument("--n-list", default="1-20")
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--move-radius", type=float, default=0.15)
    parser.add_argument("--angle-radius", type=float, default=30.0)
    parser.add_argument("--swap-prob", type=float, default=0.05)
    parser.add_argument("--scale-prob", type=float, default=0.1)
    parser.add_argument("--scale-radius", type=float, default=0.02)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    workdir = Path("/kaggle/working")
    dataset_dir = _detect_dataset_dir()
    if not dataset_dir:
        raise SystemExit("Dataset dir not found. Attach solver dataset.")
    _copy_dataset(dataset_dir, workdir)
    _install_requirements(workdir)

    baseline = Path(args.baseline)
    if not baseline.exists():
        raise SystemExit(f"baseline not found: {baseline}")

    out_dir = workdir / "refine_smalln"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "best_submission.csv"

    cmd = [
        sys.executable,
        "/kaggle/working/scripts/refine_submission.py",
        "--submission",
        str(baseline),
        "--out",
        str(output),
        "--per-n-out",
        str(out_dir / "per_n.csv"),
        "--n-list",
        args.n_list,
        "--steps",
        str(args.steps),
        "--restarts",
        str(args.restarts),
        "--move-radius",
        str(args.move_radius),
        "--angle-radius",
        str(args.angle_radius),
        "--swap-prob",
        str(args.swap_prob),
        "--scale-prob",
        str(args.scale_prob),
        "--scale-radius",
        str(args.scale_radius),
        "--temp-start",
        "1.0",
        "--temp-end",
        "0.01",
        "--max-workers",
        str(args.max_workers),
        "--decimals",
        "15",
        "--log-every-steps",
        "20000",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
