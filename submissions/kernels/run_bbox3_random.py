#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import time
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


def _resolve_src_dir() -> Path:
    for candidate in _candidate_paths("src"):
        if (candidate / "santa2025" / "metric.py").exists():
            return candidate
    raise SystemExit("src directory not found for scoring.")


def _score_submission(path: Path) -> float:
    import pandas as pd
    from santa2025.metric import score_detailed

    df = pd.read_csv(path)
    total, _ = score_detailed(df)
    return float(total)


def _log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="/kaggle/input/santa-2025-solver/results/submissions/best_submission_public_pool.csv",
    )
    parser.add_argument("--bbox3", default="/kaggle/input/santa-2025-csv/bbox3")
    parser.add_argument("--work-dir", default="/kaggle/working")
    parser.add_argument("--out-dir", default="/kaggle/working/bbox3_random")
    parser.add_argument("--budget-sec", type=int, default=36000)
    parser.add_argument("--buffer-sec", type=int, default=1200)
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--max-n", type=int, default=2400)
    parser.add_argument("--min-r", type=int, default=20)
    parser.add_argument("--max-r", type=int, default=90)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--min-improvement", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _ensure_requirements()
    src_dir = _resolve_src_dir()
    sys.path.append(str(src_dir))
    from santa2025.metric import ParticipantVisibleError  # noqa: E402

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bbox3_random.log"

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    work_submission = work_dir / "submission.csv"

    baseline = _resolve_baseline(Path(args.baseline))
    bbox3_src = Path(args.bbox3)
    if not bbox3_src.exists():
        raise SystemExit(f"bbox3 binary not found: {bbox3_src}")

    bbox3_bin = work_dir / "bbox3"
    shutil.copy(bbox3_src, bbox3_bin)
    subprocess.run(["chmod", "+x", str(bbox3_bin)], check=False)

    shutil.copy(baseline, work_submission)
    best_path = out_dir / "best_submission.csv"
    shutil.copy(work_submission, best_path)

    best_score = _score_submission(best_path)
    _log(f"baseline score: {best_score:.12f}", log_path)

    rng = random.Random(args.seed or int(time.time()))
    start = time.time()

    def time_left() -> float:
        return args.budget_sec - (time.time() - start)

    attempt = 0
    while time_left() > args.buffer_sec:
        attempt += 1
        n_value = rng.randint(args.min_n, args.max_n)
        r_value = rng.randint(args.min_r, args.max_r)
        _log(
            f"[{attempt:04d}] run bbox3 n={n_value} r={r_value} time_left={time_left():.0f}s",
            log_path,
        )
        try:
            proc = subprocess.run(
                [str(bbox3_bin), "-n", str(n_value), "-r", str(r_value)],
                capture_output=True,
                text=True,
                timeout=int(args.timeout_sec),
                cwd=str(work_dir),
            )
        except subprocess.TimeoutExpired:
            _log(f"[{attempt:04d}] timeout n={n_value} r={r_value}", log_path)
            shutil.copy(best_path, work_submission)
            continue

        log_file = out_dir / "bbox3_logs" / f"bbox3_n{n_value}_r{r_value}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_parts = []
        if proc.stdout:
            log_parts.append("=== STDOUT ===\n" + proc.stdout)
        if proc.stderr:
            log_parts.append("=== STDERR ===\n" + proc.stderr)
        log_file.write_text("\n\n".join(log_parts))

        try:
            current_score = _score_submission(work_submission)
        except ParticipantVisibleError:
            _log(f"[{attempt:04d}] invalid (overlap) -> revert", log_path)
            shutil.copy(best_path, work_submission)
            continue

        if current_score < best_score - args.min_improvement:
            _log(
                f"[{attempt:04d}] improved {best_score:.12f} -> {current_score:.12f}",
                log_path,
            )
            best_score = current_score
            shutil.copy(work_submission, best_path)
        else:
            _log(f"[{attempt:04d}] not better score={current_score:.12f}", log_path)
            shutil.copy(best_path, work_submission)

    final_path = out_dir / "best_submission.csv"
    _log(f"final score: {best_score:.12f}", log_path)
    _log(f"final submission: {final_path}", log_path)


if __name__ == "__main__":
    main()
