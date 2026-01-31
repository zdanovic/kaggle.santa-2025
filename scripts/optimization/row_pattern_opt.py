#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.io import build_submission, write_submission_csv
from santa2025.metric import score_detailed
from santa2025.solver.row_pattern import RowPatternConfig, RowPatternSolver, RowPatternSpec


def _parse_range(value: str) -> Tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected range as 'min,max', got {value!r}")
    return float(parts[0]), float(parts[1])


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _score_candidate(
    solver: RowPatternSolver,
    spec: RowPatternSpec,
    n_list: Iterable[int],
) -> float:
    total = 0.0
    for n in n_list:
        layout = solver.best_layout(n, spec)
        score, _ = solver._score_and_bounds(layout)
        total += score
    return total


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _refine(
    solver: RowPatternSolver,
    spec: RowPatternSpec,
    n_list: List[int],
    dx_min: float,
    dx_max: float,
    dy_min: float,
    dy_max: float,
    offset_min: float,
    offset_max: float,
    rng: random.Random,
    steps: int,
    decay: float,
    step_dx_scale: float,
    step_dy_scale: float,
    step_off_scale: float,
) -> RowPatternSpec:
    best = spec
    best_score = _score_candidate(solver, spec, n_list)
    step_dx = (dx_max - dx_min) * step_dx_scale
    step_dy = (dy_max - dy_min) * step_dy_scale
    period = len(spec.angles)

    for i in range(steps):
        scale = decay**i
        ndx = _clamp(best.dx + rng.uniform(-1.0, 1.0) * step_dx * scale, dx_min, dx_max)
        ndy = _clamp(best.dy + rng.uniform(-1.0, 1.0) * step_dy * scale, dy_min, dy_max)
        offsets = []
        for off in best.offsets:
            offsets.append(_clamp(off + rng.uniform(-1.0, 1.0) * step_off_scale * scale, offset_min, offset_max))
        if len(offsets) != period:
            offsets = offsets[:period]
        candidate = RowPatternSpec(
            angles=best.angles,
            offsets=offsets,
            dx=ndx,
            dy=ndy,
        )
        if solver._grid_collision(candidate):
            continue
        score = _score_candidate(solver, candidate, n_list)
        if score < best_score:
            best_score = score
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=3)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--angle-set", default="0,180,30,210,45,225,60,240,90,270,120,300")
    parser.add_argument("--angle-jitter", type=float, default=0.0)
    parser.add_argument("--dx-range", default="0.3,1.2")
    parser.add_argument("--dy-range", default="0.3,1.2")
    parser.add_argument("--offset-range", default="0.0,1.0")
    parser.add_argument("--score-n-list", default="60,80,100,120,150,200")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--out", default="results/row_pattern_opt.json")
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
    parser.add_argument("--search-pad", type=int, default=3)
    parser.add_argument("--refine-steps", type=int, default=120)
    parser.add_argument("--refine-decay", type=float, default=0.985)
    parser.add_argument("--refine-dx-scale", type=float, default=0.10)
    parser.add_argument("--refine-dy-scale", type=float, default=0.10)
    parser.add_argument("--refine-offset-scale", type=float, default=0.12)
    args = parser.parse_args()

    period = max(2, int(args.period))
    angle_set = _parse_float_list(args.angle_set)
    dx_min, dx_max = _parse_range(args.dx_range)
    dy_min, dy_max = _parse_range(args.dy_range)
    off_min, off_max = _parse_range(args.offset_range)
    score_n_list = [int(x.strip()) for x in args.score_n_list.split(",") if x.strip()]

    cfg = RowPatternConfig(
        period=period,
        grid_size=args.grid_size,
        selection_mode=args.selection_mode,
        center_steps=args.center_steps,
        search_pad=args.search_pad,
        global_squeeze=args.global_squeeze,
        squeeze_factor=args.squeeze_factor,
        squeeze_steps=args.squeeze_steps,
        squeeze_iters=args.squeeze_iters,
    )
    solver = RowPatternSolver(cfg)
    rng = random.Random(args.seed)

    best: List[dict] = []

    for _ in range(args.trials):
        angles = [rng.choice(angle_set) for _ in range(period)]
        if args.angle_jitter:
            angles = [(a + rng.uniform(-args.angle_jitter, args.angle_jitter)) % 360.0 for a in angles]
        offsets = [rng.uniform(off_min, off_max) for _ in range(period)]

        min_dx = None
        for a in angles:
            m = solver.min_dx_for_angle(a, dx_min, dx_max)
            if m is None:
                min_dx = None
                break
            min_dx = m if min_dx is None else max(min_dx, m)
        if min_dx is None:
            continue
        if min_dx > dx_max:
            continue

        dx = _clamp(min_dx * rng.uniform(1.0, 1.15), dx_min, dx_max)

        min_dy = None
        for i in range(period):
            j = (i + 1) % period
            offset_a = offsets[i] * dx
            offset_b = offsets[j] * dx
            m = solver.min_dy_for_pair(
                dx,
                offset_a,
                angles[i],
                offset_b,
                angles[j],
                dy_min,
                dy_max,
            )
            if m is None:
                min_dy = None
                break
            min_dy = m if min_dy is None else max(min_dy, m)
        if min_dy is None:
            continue
        if min_dy > dy_max:
            continue

        dy = _clamp(min_dy * rng.uniform(1.0, 1.15), dy_min, dy_max)
        spec = RowPatternSpec(angles=angles, offsets=offsets, dx=dx, dy=dy)
        if solver._grid_collision(spec):
            continue

        proxy = _score_candidate(solver, spec, score_n_list)
        record = {
            "angles": angles,
            "offsets": offsets,
            "dx": dx,
            "dy": dy,
            "proxy_score": proxy,
        }
        best.append(record)
        best.sort(key=lambda r: r["proxy_score"])
        best = best[: args.keep]

    refined: List[dict] = []
    for record in best:
        spec = RowPatternSpec(
            angles=record["angles"],
            offsets=record["offsets"],
            dx=record["dx"],
            dy=record["dy"],
        )
        if args.refine_steps > 0:
            spec = _refine(
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
        proxy = _score_candidate(solver, spec, score_n_list)
        refined.append(
            {
                "angles": spec.angles,
                "offsets": spec.offsets,
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
    final_spec = RowPatternSpec(
        angles=best_spec["angles"],
        offsets=best_spec["offsets"],
        dx=best_spec["dx"],
        dy=best_spec["dy"],
    )

    if args.emit_submission:
        groups = solver.solve(n_max=args.final_n_max, spec=final_spec)
        submission = build_submission(groups, decimals=args.emit_decimals)
        total_score, _ = score_detailed(submission)
        write_submission_csv(submission, Path(args.emit_submission))
        print(f"final score={total_score:.6f} -> {args.emit_submission}")

    print("best:", json.dumps(best_spec, indent=2))


if __name__ == "__main__":
    main()
