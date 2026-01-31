#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _candidate_paths(relative_path: str) -> List[Path]:
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
    candidates = _candidate_paths(relative_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"Missing {relative_path}; tried: {tried}")


def _resolve_baseline(requested: Path) -> Path:
    if requested.exists():
        return requested
    fallback_rel_paths = [
        "results/submissions/best_submission.csv",
        "results/submissions/best_submission_pool.csv",
        "kaggle/baselines/gb_sa_best_submission.csv",
    ]
    for rel in fallback_rel_paths:
        for candidate in _candidate_paths(rel):
            if candidate.exists():
                return candidate
    raise SystemExit(f"baseline not found: {requested}")


def _ensure_requirements() -> None:
    try:
        import pandas  # noqa: F401
        import shapely  # noqa: F401
    except Exception:
        requirements = _find_repo_file("requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)


def _run_score(script_path: Path, submission: Path) -> float:
    result = subprocess.run(
        [sys.executable, str(script_path), "--submission", str(submission)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"total_score:\s*([0-9.]+)", result.stdout)
    if not match:
        raise RuntimeError(f"Score not found in output:\n{result.stdout}")
    return float(match.group(1))


def _maybe_update(best_path: Path, best_score: float, candidate_path: Path, candidate_score: float) -> tuple[Path, float]:
    if candidate_score < best_score:
        return candidate_path, candidate_score
    return best_path, best_score


def _run_cpp_sa(
    script_path: Path,
    baseline: Path,
    out_dir: Path,
    sa_args: List[str],
) -> Path:
    cmd = [
        sys.executable,
        str(script_path),
        "--baseline",
        str(baseline),
        "--out-dir",
        str(out_dir),
    ] + sa_args
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / "best_submission.csv"


def _run_bbox3(
    script_path: Path,
    baseline: Path,
    bbox3: Path,
    out_dir: Path,
    bbox3_args: List[str],
) -> Path:
    cmd = [
        sys.executable,
        str(script_path),
        "--baseline",
        str(baseline),
        "--bbox3",
        str(bbox3),
        "--out-dir",
        str(out_dir),
        "--log-file",
        str(out_dir / "bbox3_run.log"),
    ] + bbox3_args
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / "best_submission.csv"


def _prepare_pool_dir(pool_dir: Path, work_root: Path) -> Path:
    if pool_dir.exists():
        return pool_dir
    pool_zip = pool_dir.with_suffix(".zip")
    if not pool_zip.exists():
        pool_zip = pool_dir.parent / "submissions.zip"
    if not pool_zip.exists():
        raise SystemExit(f"pool submissions not found: {pool_dir}")
    extract_root = work_root / "public_pool_dataset"
    extract_dir = extract_root / "submissions"
    if extract_dir.exists():
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(pool_zip, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-dir",
        default="/kaggle/input/santa-2025-public-pool/submissions",
        help="Directory with subfolders containing submission.csv files.",
    )
    parser.add_argument(
        "--baseline",
        default="/kaggle/input/santa-2025-solver/results/submissions/best_submission.csv",
        help="Baseline submission CSV.",
    )
    parser.add_argument("--out-dir", default="/kaggle/working/public_pool_pipeline")
    parser.add_argument("--bbox3", default="/kaggle/input/santa-2025-csv/bbox3")

    parser.add_argument("--sa-min-n", type=int, default=60)
    parser.add_argument("--sa-max-n", type=int, default=200)
    parser.add_argument("--sa-iterations", type=int, default=20000)
    parser.add_argument("--sa-restarts", type=int, default=60)
    parser.add_argument("--sa-max-gens", type=int, default=3)
    parser.add_argument("--sa-max-noimprove", type=int, default=10)
    parser.add_argument("--sa-random-inits", type=int, default=0)
    parser.add_argument("--sa-seed-base", type=int, default=0)

    parser.add_argument("--bbox3-budget-sec", type=int, default=36000)
    parser.add_argument("--bbox3-buffer-sec", type=int, default=1200)
    parser.add_argument("--bbox3-decimals", type=int, default=15)
    parser.add_argument(
        "--bbox3-aggressive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-sa", action="store_true")
    parser.add_argument("--skip-bbox3", action="store_true")
    args = parser.parse_args()

    _ensure_requirements()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_script = _find_repo_file("scripts/pool_best_of.py")
    score_script = _find_repo_file("scripts/score_submission.py")
    cpp_sa_script = _find_repo_file("kaggle/run_cpp_sa.py")
    bbox3_script = _find_repo_file("kaggle/run_bbox3.py")

    baseline_path = _resolve_baseline(Path(args.baseline))

    best_path = baseline_path
    best_score = _run_score(score_script, baseline_path)
    print(f"baseline score: {best_score:.12f}", flush=True)

    pool_out_dir = out_dir / "pool"
    pool_out_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = pool_out_dir / "best_submission.csv"
    pool_dir = _prepare_pool_dir(Path(args.pool_dir), out_dir)
    subprocess.run(
        [
            sys.executable,
            str(pool_script),
            "--results-dir",
            str(pool_dir),
            "--output",
            str(pooled_path),
        ],
        check=True,
    )
    pooled_score = _run_score(score_script, pooled_path)
    print(f"pool score: {pooled_score:.12f}", flush=True)
    best_path, best_score = _maybe_update(best_path, best_score, pooled_path, pooled_score)

    if not args.skip_sa:
        sa_out_dir = out_dir / "cpp_sa"
        sa_out_dir.mkdir(parents=True, exist_ok=True)
        sa_args = [
            "--min-n",
            str(args.sa_min_n),
            "--max-n",
            str(args.sa_max_n),
            "--iterations",
            str(args.sa_iterations),
            "--restarts",
            str(args.sa_restarts),
            "--max-gens",
            str(args.sa_max_gens),
            "--max-noimprove",
            str(args.sa_max_noimprove),
            "--seed-base",
            str(args.sa_seed_base),
            "--random-inits",
            str(args.sa_random_inits),
        ]
        sa_path = _run_cpp_sa(cpp_sa_script, best_path, sa_out_dir, sa_args)
        sa_score = _run_score(score_script, sa_path)
        print(f"cpp_sa score: {sa_score:.12f}", flush=True)
        best_path, best_score = _maybe_update(best_path, best_score, sa_path, sa_score)

    if not args.skip_bbox3:
        bbox3_path = Path(args.bbox3)
        if not bbox3_path.exists():
            raise SystemExit(f"bbox3 not found: {bbox3_path}")
        bbox_out_dir = out_dir / "bbox3"
        bbox_out_dir.mkdir(parents=True, exist_ok=True)
        bbox3_args: List[str] = [
            "--budget-sec",
            str(args.bbox3_budget_sec),
            "--buffer-sec",
            str(args.bbox3_buffer_sec),
            "--decimals",
            str(args.bbox3_decimals),
        ]
        if args.bbox3_aggressive:
            bbox3_args += [
                "--phase-a-timeout",
                "600",
                "--phase-a-n",
                "1200,1500,1800,2100,2400",
                "--phase-a-r",
                "30,45,60,75,90",
                "--phase-a-top-k",
                "5",
                "--phase-b-timeout",
                "1800",
                "--phase-b-top-k",
                "3",
                "--phase-b-fix-passes",
                "2",
                "--phase-c-timeout",
                "3600",
                "--phase-c-top-k",
                "2",
                "--phase-c-fix-passes",
                "3",
                "--fallback-n",
                "2000",
                "--fallback-r",
                "60",
                "--fallback-timeout",
                "3600",
            ]
        bbox_path = _run_bbox3(bbox3_script, best_path, bbox3_path, bbox_out_dir, bbox3_args)
        bbox_score = _run_score(score_script, bbox_path)
        print(f"bbox3 score: {bbox_score:.12f}", flush=True)
        best_path, best_score = _maybe_update(best_path, best_score, bbox_path, bbox_score)

    final_path = out_dir / "best_submission.csv"
    shutil.copy(best_path, final_path)
    print(f"final score: {best_score:.12f}", flush=True)
    print(f"final submission: {final_path}", flush=True)


if __name__ == "__main__":
    main()
