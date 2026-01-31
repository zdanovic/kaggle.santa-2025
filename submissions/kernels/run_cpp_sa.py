#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_cpp_source() -> Path:
    candidates = [
        Path("/kaggle/working/kaggle/sa_v1_parallel.cpp"),
        Path("/kaggle/input/santa-2025-solver/kaggle/sa_v1_parallel.cpp"),
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            maybe = entry / "kaggle" / "sa_v1_parallel.cpp"
            candidates.append(maybe)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"sa_v1_parallel.cpp not found; tried: {tried}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="/kaggle/input/santa-2025-solver/kaggle/baselines/gb_sa_best_submission.csv",
        help="Path to baseline submission",
    )
    parser.add_argument("--work-dir", default="/kaggle/working")
    parser.add_argument("--out-dir", default="/kaggle/working/cpp_sa")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--restarts", type=int, default=80)
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=50)
    parser.add_argument("--max-gens", type=int, default=3)
    parser.add_argument("--max-noimprove", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--random-inits", type=int, default=0)
    parser.add_argument("--random-init-max-n", type=int, default=12)
    parser.add_argument("--random-init-scale", type=float, default=1.2)
    parser.add_argument("--random-init-tries", type=int, default=4)
    parser.add_argument("--random-init-max-attempts", type=int, default=2000)
    parser.add_argument("--compress-steps", type=int, default=0)
    parser.add_argument("--compress-factor", type=float, default=0.99)
    parser.add_argument("--compress-relax-iters", type=int, default=60)
    parser.add_argument("--compress-relax-step", type=float, default=0.02)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = Path(args.baseline)
    if not baseline.exists():
        raise SystemExit(f"baseline not found: {baseline}")

    cpp_source = _find_cpp_source()
    binary = work_dir / "sa_v1_parallel"

    compile_cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++17",
        "-fopenmp",
        "-o",
        str(binary),
        str(cpp_source),
    ]
    print(" ".join(compile_cmd), flush=True)
    subprocess.run(compile_cmd, check=True)

    shutil.copy(baseline, out_dir / "baseline.csv")

    out_csv = out_dir / "best_submission.csv"
    threads = args.threads or (os.cpu_count() or 1)

    run_cmd = [
        str(binary),
        "-i",
        str(baseline),
        "-o",
        str(out_csv),
        "-n",
        str(args.iterations),
        "-r",
        str(args.restarts),
        "--min-n",
        str(args.min_n),
        "--max-n",
        str(args.max_n),
        "--max-gens",
        str(args.max_gens),
        "--max-noimprove",
        str(args.max_noimprove),
        "--threads",
        str(threads),
        "--seed-base",
        str(args.seed_base),
        "--random-inits",
        str(args.random_inits),
        "--random-init-max-n",
        str(args.random_init_max_n),
        "--random-init-scale",
        str(args.random_init_scale),
        "--random-init-tries",
        str(args.random_init_tries),
        "--random-init-max-attempts",
        str(args.random_init_max_attempts),
        "--compress-steps",
        str(args.compress_steps),
        "--compress-factor",
        str(args.compress_factor),
        "--compress-relax-iters",
        str(args.compress_relax_iters),
        "--compress-relax-step",
        str(args.compress_relax_step),
    ]
    print(" ".join(run_cmd), flush=True)
    subprocess.run(run_cmd, check=True)

    print(f"final submission: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
