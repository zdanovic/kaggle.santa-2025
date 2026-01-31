#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.metric import ParticipantVisibleError, score_detailed


def _parse_target_ns(value: str | None) -> List[int] | None:
    if value is None:
        return None
    targets: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            targets.extend(range(start, end + 1))
        else:
            targets.append(int(part))
    return sorted(set(targets))


def _score_submission(path: Path) -> tuple[float, dict[int, float]]:
    df = pd.read_csv(path)
    return score_detailed(df)


def _compile_cpp(cpp_path: Path, bin_path: Path) -> None:
    base_cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++17",
        "-o",
        str(bin_path),
        str(cpp_path),
    ]
    try:
        cmd = [
            "g++",
            "-O3",
            "-march=native",
            "-std=c++17",
            "-fopenmp",
            "-o",
            str(bin_path),
            str(cpp_path),
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(base_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", default="results/a_bit_better_public_run")
    parser.add_argument("--groups", default=None, help="Comma/range list, e.g. 1-50,80")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=200)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-minutes", type=int, default=0, help="Stop after N minutes (0 = no limit)")
    parser.add_argument("--cpp", default="scripts/single_group_optimizer.cpp")
    parser.add_argument("--recompile", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    work_dir = (repo_root / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.submission).resolve()
    working_csv = work_dir / "submission.csv"
    temp_csv = work_dir / "submission_temp.csv"
    output_path = Path(args.out).resolve()

    cpp_path = (repo_root / args.cpp).resolve()
    bin_path = work_dir / "single_group_optimizer"

    if args.recompile or not bin_path.exists():
        _compile_cpp(cpp_path, bin_path)

    subprocess.run(["cp", str(input_path), str(working_csv)], check=True)

    base_score, per_group = _score_submission(working_csv)
    print(f"base_score: {base_score}", flush=True)

    targets = _parse_target_ns(args.groups)
    if targets is None:
        filtered = {n: s for n, s in per_group.items() if args.min_n <= n <= args.max_n}
        targets = [n for n, _ in sorted(filtered.items(), key=lambda x: x[1], reverse=True)[: args.top_k]]

    start_time = time.time()
    improved_groups: List[int] = []
    for n in targets:
        if args.max_minutes > 0 and (time.time() - start_time) > args.max_minutes * 60:
            break
        env = os.environ.copy()
        env["GROUP_NUMBER"] = str(n)
        cmd = [
            str(bin_path),
            "-i",
            str(working_csv),
            "-o",
            str(temp_csv),
            "-n",
            str(args.iters),
            "-r",
            str(args.restarts),
        ]
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            print(f"group {n}: timeout", flush=True)
            continue
        except subprocess.CalledProcessError as exc:
            print(f"group {n}: error {exc}", flush=True)
            continue

        if ">>> IMPROVED" in result.stdout and temp_csv.exists():
            temp_csv.replace(working_csv)
            improved_groups.append(n)
            print(result.stdout.strip().splitlines()[-1], flush=True)

    try:
        final_score, _ = _score_submission(working_csv)
    except ParticipantVisibleError as exc:
        raise SystemExit(f"Invalid submission after optimization: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", str(working_csv), str(output_path)], check=True)
    print(f"final_score: {final_score}", flush=True)

    summary = {
        "base_score": base_score,
        "final_score": final_score,
        "improved_groups": improved_groups,
        "targets": targets,
        "iters": args.iters,
        "restarts": args.restarts,
        "timeout": args.timeout,
        "max_minutes": args.max_minutes,
        "cpp": str(cpp_path),
        "input": str(input_path),
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
