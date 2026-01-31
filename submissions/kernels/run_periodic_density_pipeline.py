#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


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
    for candidate in _candidate_paths(relative_path):
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in _candidate_paths(relative_path))
    raise SystemExit(f"Missing {relative_path}; tried: {tried}")


def _find_pool_dir(pool_dir: Path) -> Path | None:
    if pool_dir.exists():
        return pool_dir
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            candidate = entry / "public_pool_dataset" / "submissions"
            if candidate.exists():
                return candidate
            candidate = entry / "submissions"
            if candidate.exists() and (entry / "sources.json").exists():
                return candidate
    return None


def _find_pool_zip(pool_zip: Path) -> Path | None:
    if pool_zip.exists():
        return pool_zip
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            candidate = entry / "public_pool_dataset.zip"
            if candidate.exists():
                return candidate
    return None


def _resolve_baseline(requested: Path) -> Path:
    if requested.exists():
        return requested
    fallback_rel_paths = [
        "kaggle/baselines/best_submission.csv",
        "results/submissions/best_submission.csv",
        "results/submissions/best_submission_public_pool.csv",
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


def _try_score(script_path: Path, submission: Path, label: str) -> float:
    try:
        return _run_score(script_path, submission)
    except Exception as exc:
        print(f"{label} score failed: {exc}", flush=True)
        return float("inf")


def _prepare_pool_dir(pool_dir: Path, pool_zip: Path, work_root: Path) -> Path:
    if pool_dir.exists():
        return pool_dir
    if not pool_zip.exists():
        raise SystemExit(f"pool submissions not found: {pool_zip}")
    extract_root = work_root / "public_pool_dataset"
    extract_dir = extract_root / "submissions"
    if extract_dir.exists():
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(pool_zip, "r") as zf:
        zf.extractall(extract_root)
    return extract_dir


def _prepare_merge_dir(root: Path, baseline: Path, candidate: Path) -> Path:
    merge_dir = root / "merge"
    baseline_dir = merge_dir / "baseline"
    candidate_dir = merge_dir / "candidate"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(baseline, baseline_dir / "submission.csv")
    shutil.copy(candidate, candidate_dir / "submission.csv")
    return merge_dir


def _maybe_update(best_path: Path, best_score: float, candidate_path: Path, candidate_score: float) -> tuple[Path, float]:
    if candidate_score < best_score:
        return candidate_path, candidate_score
    return best_path, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-dir",
        default="/kaggle/input/santa-2025-solver/public_pool_dataset/submissions",
    )
    parser.add_argument(
        "--pool-zip",
        default="/kaggle/input/santa-2025-solver/public_pool_dataset.zip",
    )
    parser.add_argument(
        "--baseline",
        default="/kaggle/input/santa-2025-solver/results/submissions/best_submission_public_pool.csv",
    )
    parser.add_argument("--out-dir", default="/kaggle/working/periodic_density_pipeline")
    parser.add_argument("--skip-pool", action="store_true")

    parser.add_argument("--k-list", default="2,3,4")
    parser.add_argument("--trials", type=int, default=8000)
    parser.add_argument("--keep-density", type=int, default=10)
    parser.add_argument("--keep-proxy", type=int, default=6)
    parser.add_argument("--angle-set", default="0,30,45,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--angle-jitter", type=float, default=8.0)
    parser.add_argument("--dx-range", default="0.28,0.9")
    parser.add_argument("--dy-range", default="0.35,1.1")
    parser.add_argument("--offset-range", default="0.0,1.0")
    parser.add_argument("--lattice-angle-range", default="-15.0,15.0")
    parser.add_argument("--basis-attempts", type=int, default=140)
    parser.add_argument("--search-pad", type=int, default=8)
    parser.add_argument("--center-steps", type=int, default=8)
    parser.add_argument("--score-n-list", default="80,100,120,150,200")
    parser.add_argument("--final-n-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8821)
    parser.add_argument("--final-global-squeeze", action="store_true")
    parser.add_argument("--final-squeeze-factor", type=float, default=0.985)
    parser.add_argument("--final-squeeze-steps", type=int, default=24)
    parser.add_argument("--final-squeeze-iters", type=int, default=10)
    parser.add_argument("--refine-steps", type=int, default=260)
    parser.add_argument("--refine-restarts", type=int, default=3)
    parser.add_argument("--refine-dx-scale", type=float, default=0.06)
    parser.add_argument("--refine-dy-scale", type=float, default=0.06)
    parser.add_argument("--refine-offset-scale", type=float, default=0.07)
    parser.add_argument("--refine-angle-scale", type=float, default=6.0)
    parser.add_argument("--refine-basis-scale", type=float, default=0.12)
    parser.add_argument("--refine-deg-scale", type=float, default=10.0)
    parser.add_argument("--refine-decay", type=float, default=0.985)
    parser.add_argument("--refine-accept-temp", type=float, default=0.0004)
    args = parser.parse_args()

    _ensure_requirements()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_script = _find_repo_file("scripts/pool_best_of.py")
    score_script = _find_repo_file("scripts/score_submission.py")
    periodic_script = _find_repo_file("scripts/periodic_density_search.py")

    baseline_path = _resolve_baseline(Path(args.baseline))
    if not args.skip_pool:
        pool_dir = _find_pool_dir(Path(args.pool_dir))
        pool_zip = _find_pool_zip(Path(args.pool_zip))
        if pool_dir is None and pool_zip is None:
            print("pool submissions not found; skipping pool merge", flush=True)
        else:
            if pool_dir is None:
                pool_dir = _prepare_pool_dir(Path(args.pool_dir), pool_zip, out_dir)
            pool_out = out_dir / "pool_baseline.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(pool_script),
                    "--results-dir",
                    str(pool_dir),
                    "--output",
                    str(pool_out),
                ],
                check=True,
            )
            baseline_path = pool_out

    baseline_score = _run_score(score_script, baseline_path)
    print(f"baseline score: {baseline_score:.12f}", flush=True)

    periodic_dir = out_dir / "periodic_density"
    periodic_dir.mkdir(parents=True, exist_ok=True)
    periodic_json = periodic_dir / "periodic_density.json"
    periodic_cfg = periodic_dir / "best_config.yaml"
    periodic_submission = periodic_dir / "periodic_submission.csv"

    cmd = [
        sys.executable,
        str(periodic_script),
        "--k-list",
        args.k_list,
        "--trials",
        str(args.trials),
        "--keep-density",
        str(args.keep_density),
        "--keep-proxy",
        str(args.keep_proxy),
        "--angle-set",
        args.angle_set,
        "--angle-jitter",
        str(args.angle_jitter),
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
        "--seed",
        str(args.seed),
        "--out",
        str(periodic_json),
        "--emit-config",
        str(periodic_cfg),
        "--emit-submission",
        str(periodic_submission),
        "--emit-decimals",
        "6",
        "--final-squeeze-factor",
        str(args.final_squeeze_factor),
        "--final-squeeze-steps",
        str(args.final_squeeze_steps),
        "--final-squeeze-iters",
        str(args.final_squeeze_iters),
        "--refine-steps",
        str(args.refine_steps),
        "--refine-restarts",
        str(args.refine_restarts),
        "--refine-dx-scale",
        str(args.refine_dx_scale),
        "--refine-dy-scale",
        str(args.refine_dy_scale),
        "--refine-offset-scale",
        str(args.refine_offset_scale),
        "--refine-angle-scale",
        str(args.refine_angle_scale),
        "--refine-basis-scale",
        str(args.refine_basis_scale),
        "--refine-deg-scale",
        str(args.refine_deg_scale),
        "--refine-decay",
        str(args.refine_decay),
        "--refine-accept-temp",
        str(args.refine_accept_temp),
    ]
    if args.final_global_squeeze:
        cmd.append("--final-global-squeeze")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    periodic_score = _try_score(score_script, periodic_submission, "periodic")
    if periodic_score < float("inf"):
        print(f"periodic score: {periodic_score:.12f}", flush=True)

    merge_dir = _prepare_merge_dir(out_dir, baseline_path, periodic_submission)
    merged_path = out_dir / "best_submission.csv"
    subprocess.run(
        [
            sys.executable,
            str(pool_script),
            "--results-dir",
            str(merge_dir),
            "--output",
            str(merged_path),
        ],
        check=True,
    )
    merged_score = _try_score(score_script, merged_path, "merged")
    if merged_score < float("inf"):
        print(f"merged score: {merged_score:.12f}", flush=True)

    best_path = baseline_path
    best_score = baseline_score
    best_path, best_score = _maybe_update(best_path, best_score, periodic_submission, periodic_score)
    best_path, best_score = _maybe_update(best_path, best_score, merged_path, merged_score)

    final_path = out_dir / "final_best_submission.csv"
    shutil.copy(best_path, final_path)
    print(f"final score: {best_score:.12f}", flush=True)
    print(f"final submission: {final_path}", flush=True)


if __name__ == "__main__":
    main()
