#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.io import build_submission, write_submission_csv
from santa2025.metric import score_detailed
from santa2025.solver.pattern import PatternConfig, PatternSolver, PatternSpec


def _parse_range(value: str) -> Tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected range as 'min,max', got {value!r}")
    return float(parts[0]), float(parts[1])


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _score_candidate(
    solver: PatternSolver,
    spec: PatternSpec,
    n_list: Iterable[int],
    seed: int,
) -> float:
    rng = random.Random(seed)
    total = 0.0
    for n in n_list:
        layout = solver._best_layout(n, spec, rng)
        score, _ = solver._score_and_bounds(layout)
        total += score
    return total


def _valid_candidate(solver: PatternSolver, spec: PatternSpec) -> bool:
    offset = spec.dx * spec.offset_ratio
    return not solver._grid_collision(spec.dx, spec.dy, offset, spec.angle_a, spec.angle_b)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _refine_candidate(
    solver: PatternSolver,
    spec: PatternSpec,
    n_list: List[int],
    dx_min: float,
    dx_max: float,
    dy_min: float,
    dy_max: float,
    off_min: float,
    off_max: float,
    rng: random.Random,
    steps: int,
    decay: float,
    step_dx_scale: float,
    step_dy_scale: float,
    step_off_scale: float,
) -> PatternSpec:
    best = spec
    best_score = _score_candidate(solver, spec, n_list, seed=rng.randrange(1_000_000))

    step_dx = (dx_max - dx_min) * step_dx_scale
    step_dy = (dy_max - dy_min) * step_dy_scale

    for i in range(steps):
        scale = decay**i
        ndx = _clamp(best.dx + rng.uniform(-1.0, 1.0) * step_dx * scale, dx_min, dx_max)
        ndy = _clamp(best.dy + rng.uniform(-1.0, 1.0) * step_dy * scale, dy_min, dy_max)
        noff = _clamp(
            best.offset_ratio + rng.uniform(-1.0, 1.0) * step_off_scale * scale,
            off_min,
            off_max,
        )
        candidate = PatternSpec(
            angle_a=best.angle_a,
            angle_b=best.angle_b,
            offset_ratio=noff,
            dx=ndx,
            dy=ndy,
        )
        if not _valid_candidate(solver, candidate):
            continue
        score = _score_candidate(solver, candidate, n_list, seed=rng.randrange(1_000_000))
        if score < best_score:
            best_score = score
            best = candidate

    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--angle-set", default="0,180,45,225,60,240,90,270")
    parser.add_argument("--angle-jitter", type=float, default=0.0)
    parser.add_argument("--dx-range", default="0.3,1.2")
    parser.add_argument("--dy-range", default="0.3,1.2")
    parser.add_argument("--offset-range", default="0.45,0.85")
    parser.add_argument("--score-n-list", default="60,80,100,120,150,200")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--out", default="results/pattern_opt.json")
    parser.add_argument("--emit-submission", default="")
    parser.add_argument("--emit-decimals", type=int, default=6)
    parser.add_argument("--final-n-max", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--global-squeeze", action="store_true")
    parser.add_argument("--squeeze-factor", type=float, default=0.985)
    parser.add_argument("--squeeze-steps", type=int, default=20)
    parser.add_argument("--squeeze-iters", type=int, default=8)
    parser.add_argument("--selection-mode", default="square_search")
    parser.add_argument("--center-steps", type=int, default=6)
    parser.add_argument("--search-pad", type=int, default=2)
    parser.add_argument("--refine-steps", type=int, default=120)
    parser.add_argument("--refine-decay", type=float, default=0.985)
    parser.add_argument("--refine-dx-scale", type=float, default=0.10)
    parser.add_argument("--refine-dy-scale", type=float, default=0.10)
    parser.add_argument("--refine-offset-scale", type=float, default=0.12)
    args = parser.parse_args()

    angle_set = _parse_float_list(args.angle_set)
    dx_min, dx_max = _parse_range(args.dx_range)
    dy_min, dy_max = _parse_range(args.dy_range)
    off_min, off_max = _parse_range(args.offset_range)
    score_n_list = [int(x.strip()) for x in args.score_n_list.split(",") if x.strip()]

    cfg = PatternConfig(
        angle_pairs=[(0.0, 180.0)],
        offset_ratios=[0.6],
        dx_min=dx_min,
        dx_max=dx_max,
        dy_min=dy_min,
        dy_max=dy_max,
        grid_size=args.grid_size,
        selection_mode=args.selection_mode,
        center_steps=args.center_steps,
        search_pad=args.search_pad,
        global_squeeze=args.global_squeeze,
        squeeze_factor=args.squeeze_factor,
        squeeze_steps=args.squeeze_steps,
        squeeze_iters=args.squeeze_iters,
    )
    solver = PatternSolver(cfg)

    rng = random.Random(args.seed)
    best: List[dict] = []

    for _ in range(args.trials):
        angle_a = rng.choice(angle_set)
        angle_b = rng.choice(angle_set)
        if args.angle_jitter:
            angle_a = (angle_a + rng.uniform(-args.angle_jitter, args.angle_jitter)) % 360.0
            angle_b = (angle_b + rng.uniform(-args.angle_jitter, args.angle_jitter)) % 360.0
        offset_ratio = rng.uniform(off_min, off_max)
        min_dx = solver._min_dx(angle_a, angle_b)
        if min_dx is None:
            continue
        dx_mul = rng.uniform(1.0, 1.18)
        dx = _clamp(min_dx * dx_mul, dx_min, dx_max)
        offset = dx * offset_ratio
        min_dy = solver._min_dy(dx, offset, angle_a, angle_b)
        if min_dy is None:
            continue
        dy_mul = rng.uniform(1.0, 1.18)
        dy = _clamp(min_dy * dy_mul, dy_min, dy_max)
        spec = PatternSpec(
            angle_a=angle_a,
            angle_b=angle_b,
            offset_ratio=offset_ratio,
            dx=dx,
            dy=dy,
        )
        if not _valid_candidate(solver, spec):
            continue
        proxy = _score_candidate(solver, spec, score_n_list, seed=rng.randrange(1_000_000))
        record = {
            "angle_a": spec.angle_a,
            "angle_b": spec.angle_b,
            "offset_ratio": spec.offset_ratio,
            "dx": spec.dx,
            "dy": spec.dy,
            "proxy_score": proxy,
        }
        best.append(record)
        best.sort(key=lambda r: r["proxy_score"])
        best = best[: args.keep]

    refined: List[dict] = []
    for record in best:
        spec = PatternSpec(
            angle_a=record["angle_a"],
            angle_b=record["angle_b"],
            offset_ratio=record["offset_ratio"],
            dx=record["dx"],
            dy=record["dy"],
        )
        if args.refine_steps > 0:
            spec = _refine_candidate(
                solver,
                spec,
                score_n_list,
                dx_min,
                dx_max,
                dy_min,
                dy_max,
                off_min,
                off_max,
                rng,
                args.refine_steps,
                args.refine_decay,
                args.refine_dx_scale,
                args.refine_dy_scale,
                args.refine_offset_scale,
            )
        proxy = _score_candidate(solver, spec, score_n_list, seed=rng.randrange(1_000_000))
        refined.append(
            {
                "angle_a": spec.angle_a,
                "angle_b": spec.angle_b,
                "offset_ratio": spec.offset_ratio,
                "dx": spec.dx,
                "dy": spec.dy,
                "proxy_score": proxy,
            }
        )

    refined.sort(key=lambda r: r["proxy_score"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refined, indent=2))

    if not refined:
        raise SystemExit("No valid patterns found.")

    best_spec = refined[0]
    final_spec = PatternSpec(
        angle_a=best_spec["angle_a"],
        angle_b=best_spec["angle_b"],
        offset_ratio=best_spec["offset_ratio"],
        dx=best_spec["dx"],
        dy=best_spec["dy"],
    )

    if args.emit_submission:
        groups = {}
        rng = random.Random(args.seed)
        for n in range(1, args.final_n_max + 1):
            groups[n] = solver._best_layout(n, final_spec, rng)
        submission = build_submission(groups, decimals=args.emit_decimals)
        total_score, _ = score_detailed(submission)
        write_submission_csv(submission, Path(args.emit_submission))
        print(f"final score={total_score:.6f} -> {args.emit_submission}")

    print("best:", json.dumps(best_spec, indent=2))


if __name__ == "__main__":
    main()
