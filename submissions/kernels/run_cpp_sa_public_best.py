#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _candidate_paths(relative_path: str) -> list[Path]:
    candidates = [
        Path("/kaggle/working") / relative_path,
        Path("/kaggle/input/santa-2025-solver") / relative_path,
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            candidates.append(entry / relative_path)
    return candidates


def _find_repo_file(relative_path: str) -> Path:
    for candidate in _candidate_paths(relative_path):
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in _candidate_paths(relative_path))
    raise SystemExit(f"Missing {relative_path}; tried: {tried}")


def _ensure_requirements() -> None:
    try:
        import pandas  # noqa: F401
        import shapely  # noqa: F401
    except Exception:
        requirements = _find_repo_file("requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="kaggle/baselines/public_best_submission_70_836.csv",
    )
    parser.add_argument("--out-dir", default="/kaggle/working/cpp_sa_public_best")
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--restarts", type=int, default=96)
    parser.add_argument("--max-gens", type=int, default=3)
    parser.add_argument("--max-noimprove", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=20250108)
    parser.add_argument("--random-inits", type=int, default=8)
    parser.add_argument("--random-init-max-n", type=int, default=20)
    parser.add_argument("--random-init-scale", type=float, default=1.25)
    parser.add_argument("--random-init-tries", type=int, default=6)
    parser.add_argument("--random-init-max-attempts", type=int, default=4000)
    parser.add_argument("--compress-steps", type=int, default=2)
    parser.add_argument("--compress-factor", type=float, default=0.985)
    parser.add_argument("--compress-relax-iters", type=int, default=80)
    parser.add_argument("--compress-relax-step", type=float, default=0.02)
    args = parser.parse_args()

    _ensure_requirements()

    baseline_path = _find_repo_file(args.baseline)
    run_script = _find_repo_file("kaggle/run_cpp_sa.py")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(run_script),
        "--baseline",
        str(baseline_path),
        "--out-dir",
        str(out_dir),
        "--iterations",
        str(args.iterations),
        "--restarts",
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
        str(args.threads),
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
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"final submission: {out_dir / 'best_submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
