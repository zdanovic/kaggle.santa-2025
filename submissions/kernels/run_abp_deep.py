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
    parser.add_argument("--out-dir", default="/kaggle/working/abp_deep")
    parser.add_argument("--groups", default="1-40")
    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument("--restarts", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-minutes", type=int, default=200)
    args = parser.parse_args()

    _ensure_requirements()

    baseline_path = _find_repo_file(args.baseline)
    run_script = _find_repo_file("scripts/run_a_bit_better_public.py")
    cpp_path = _find_repo_file("scripts/single_group_optimizer.cpp")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "best_submission.csv"

    cmd = [
        sys.executable,
        str(run_script),
        "--submission",
        str(baseline_path),
        "--out",
        str(out_csv),
        "--work-dir",
        str(out_dir),
        "--groups",
        args.groups,
        "--iters",
        str(args.iters),
        "--restarts",
        str(args.restarts),
        "--timeout",
        str(args.timeout),
        "--max-minutes",
        str(args.max_minutes),
        "--cpp",
        str(cpp_path),
        "--recompile",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"final submission: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
