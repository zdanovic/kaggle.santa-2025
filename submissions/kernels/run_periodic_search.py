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
        if (cand / "scripts" / "periodic_search.py").exists():
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
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument("--k-list", default="2,3")
    parser.add_argument("--angle-set", default="0,60,120,180,240,300")
    parser.add_argument("--dx-range", default="0.55,1.15")
    parser.add_argument("--dy-range", default="0.55,1.15")
    parser.add_argument("--offset-range", default="0.0,0.9")
    parser.add_argument("--lattice-angle-range", default="0.0,30.0")
    parser.add_argument("--basis-attempts", type=int, default=80)
    parser.add_argument("--search-pad", type=int, default=6)
    parser.add_argument("--center-steps", type=int, default=8)
    parser.add_argument("--score-n-list", default="60,80,100,120,150,200")
    parser.add_argument("--final-n-max", type=int, default=200)
    parser.add_argument("--neighbor-range", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--out-dir", default="/kaggle/working/periodic_search")
    parser.add_argument("--emit-decimals", type=int, default=6)
    parser.add_argument("--final-global-squeeze", action="store_true")
    parser.add_argument("--final-squeeze-factor", type=float, default=0.985)
    parser.add_argument("--final-squeeze-steps", type=int, default=20)
    parser.add_argument("--final-squeeze-iters", type=int, default=8)
    args, extra_args = parser.parse_known_args()

    workdir = Path("/kaggle/working")
    dataset_dir = _detect_dataset_dir()
    if not dataset_dir:
        raise SystemExit("Dataset dir not found. Attach solver dataset.")
    _copy_dataset(dataset_dir, workdir)
    _install_requirements(workdir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "periodic_search.json"
    out_csv = out_dir / "best_submission.csv"

    cmd = [
        sys.executable,
        str(workdir / "scripts" / "periodic_search.py"),
        "--k-list",
        args.k_list,
        "--trials",
        str(args.trials),
        "--keep",
        str(args.keep),
        "--angle-set",
        args.angle_set,
        "--dx-range",
        args.dx_range,
        "--dy-range",
        args.dy_range,
        "--offset-range",
        args.offset_range,
        f"--lattice-angle-range={args.lattice_angle_range}",
        "--basis-attempts",
        str(args.basis_attempts),
        "--search-pad",
        str(args.search_pad),
        "--center-steps",
        str(args.center_steps),
        "--score-n-list",
        args.score_n_list,
        "--final-n-max",
        str(args.final_n_max),
        "--neighbor-range",
        str(args.neighbor_range),
        "--seed",
        str(args.seed),
        "--out",
        str(out_json),
        "--emit-submission",
        str(out_csv),
        "--emit-decimals",
        str(args.emit_decimals),
        "--final-squeeze-factor",
        str(args.final_squeeze_factor),
        "--final-squeeze-steps",
        str(args.final_squeeze_steps),
        "--final-squeeze-iters",
        str(args.final_squeeze_iters),
    ]
    if args.final_global_squeeze:
        cmd.append("--final-global-squeeze")
    if extra_args:
        cmd.extend(extra_args)

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
