#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.geometry import build_tree_polygon
from santa2025.io import build_submission, write_submission_csv
from santa2025.metric import ParticipantVisibleError, score_detailed

from ortools.sat.python import cp_model
from shapely.strtree import STRtree


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    deg: float
    bounds: Tuple[float, float, float, float]


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _bbox_for_group(df_group: pd.DataFrame) -> Tuple[float, float, float, float]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for _, row in df_group.iterrows():
        poly = build_tree_polygon(row["x"], row["y"], row["deg"], 1.0)
        bx0, by0, bx1, by1 = poly.bounds
        minx = min(minx, bx0)
        miny = min(miny, by0)
        maxx = max(maxx, bx1)
        maxy = max(maxy, by1)
    return minx, miny, maxx, maxy


def _generate_candidates(
    df_group: pd.DataFrame,
    angle_set: List[float],
    angle_jitter: float,
    jitter_span: float,
    jitter_steps: int,
    random_points: int,
    seed: int,
    max_candidates: int,
) -> List[Candidate]:
    rng = random.Random(seed)

    baseline = df_group[["x", "y", "deg"]].copy()
    baseline["deg"] = baseline["deg"].astype(float)

    minx, miny, maxx, maxy = _bbox_for_group(baseline)
    side = max(maxx - minx, maxy - miny)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    pad = side * 0.25 + 0.2

    jitter_steps = max(0, jitter_steps)
    if jitter_steps > 0:
        step = jitter_span / jitter_steps
        deltas = [step * i for i in range(-jitter_steps, jitter_steps + 1)]
    else:
        deltas = [0.0]

    candidates: List[Candidate] = []
    seen = set()

    for _, row in baseline.iterrows():
        base_x = float(row["x"])
        base_y = float(row["y"])
        base_deg = float(row["deg"])
        local_angles = set(angle_set)
        local_angles.add(base_deg % 360.0)
        if angle_jitter > 0:
            local_angles.add((base_deg + angle_jitter) % 360.0)
            local_angles.add((base_deg - angle_jitter) % 360.0)
        for dx in deltas:
            for dy in deltas:
                x = base_x + dx
                y = base_y + dy
                for deg in local_angles:
                    key = (round(x, 3), round(y, 3), round(deg, 2))
                    if key in seen:
                        continue
                    poly = build_tree_polygon(x, y, deg, 1.0)
                    candidates.append(Candidate(x=x, y=y, deg=deg, bounds=poly.bounds))
                    seen.add(key)

    for _ in range(random_points):
        x = rng.uniform(cx - side / 2 - pad, cx + side / 2 + pad)
        y = rng.uniform(cy - side / 2 - pad, cy + side / 2 + pad)
        deg = rng.choice(angle_set)
        key = (round(x, 3), round(y, 3), round(deg, 2))
        if key in seen:
            continue
        poly = build_tree_polygon(x, y, deg, 1.0)
        candidates.append(Candidate(x=x, y=y, deg=deg, bounds=poly.bounds))
        seen.add(key)

    if len(candidates) > max_candidates:
        candidates = rng.sample(candidates, max_candidates)

    return candidates


def _build_overlap_pairs(candidates: List[Candidate]) -> List[Tuple[int, int]]:
    polys = [
        build_tree_polygon(c.x, c.y, c.deg, 1e6)
        for c in candidates
    ]
    tree = STRtree(polys)
    pairs: List[Tuple[int, int]] = []
    for i, poly in enumerate(polys):
        for j in tree.query(poly):
            if j <= i:
                continue
            other = polys[j]
            if poly.intersects(other) and not poly.touches(other):
                pairs.append((i, j))
    return pairs


def _solve_exact(
    n: int,
    candidates: List[Candidate],
    overlap_pairs: List[Tuple[int, int]],
    time_limit: int,
    threads: int,
    scale: int,
) -> List[Candidate] | None:
    model = cp_model.CpModel()
    x_vars = [model.NewBoolVar(f"x_{i}") for i in range(len(candidates))]

    model.Add(sum(x_vars) == n)

    for i, j in overlap_pairs:
        model.Add(x_vars[i] + x_vars[j] <= 1)

    xs = [c.bounds[0] for c in candidates] + [c.bounds[2] for c in candidates]
    ys = [c.bounds[1] for c in candidates] + [c.bounds[3] for c in candidates]
    min_x = int(math.floor(min(xs) * scale))
    max_x = int(math.ceil(max(xs) * scale))
    min_y = int(math.floor(min(ys) * scale))
    max_y = int(math.ceil(max(ys) * scale))

    minX = model.NewIntVar(min_x, max_x, "minX")
    maxX = model.NewIntVar(min_x, max_x, "maxX")
    minY = model.NewIntVar(min_y, max_y, "minY")
    maxY = model.NewIntVar(min_y, max_y, "maxY")

    Mx = max_x - min_x + 1
    My = max_y - min_y + 1
    M = max(Mx, My) + 10 * scale

    for i, c in enumerate(candidates):
        x0 = int(math.floor(c.bounds[0] * scale))
        x1 = int(math.ceil(c.bounds[2] * scale))
        y0 = int(math.floor(c.bounds[1] * scale))
        y1 = int(math.ceil(c.bounds[3] * scale))
        model.Add(minX <= x0 + M * (1 - x_vars[i]))
        model.Add(maxX >= x1 - M * (1 - x_vars[i]))
        model.Add(minY <= y0 + M * (1 - x_vars[i]))
        model.Add(maxY >= y1 - M * (1 - x_vars[i]))

    side = model.NewIntVar(0, max(Mx, My) * 2, "side")
    model.Add(side >= maxX - minX)
    model.Add(side >= maxY - minY)
    model.Minimize(side)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1, time_limit)
    solver.parameters.num_search_workers = max(1, threads)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    selected: List[Candidate] = []
    for i, c in enumerate(candidates):
        if solver.Value(x_vars[i]) == 1:
            selected.append(c)
    if len(selected) != n:
        return None
    return selected


def _score_group(df_group: pd.DataFrame) -> float:
    _, per_group = score_detailed(df_group)
    return list(per_group.values())[0]


def _load_baseline_groups(path: Path) -> Dict[int, pd.DataFrame]:
    df = pd.read_csv(path)
    groups: Dict[int, pd.DataFrame] = {}
    df["group"] = df["id"].astype(str).str.split("_").str[0].astype(int)
    for n, df_group in df.groupby("group"):
        sub = df_group[["id", "x", "y", "deg"]].copy()
        for c in ["x", "y", "deg"]:
            sub[c] = sub[c].astype(str).str[1:].astype(float)
        groups[n] = sub.reset_index(drop=True)
    return groups


def _to_submission_rows(n: int, candidates: List[Candidate]) -> pd.DataFrame:
    rows = []
    for idx, c in enumerate(candidates):
        rows.append(
            {
                "id": f"{n:03d}_{idx}",
                "x": f"s{c.x}",
                "y": f"s{c.y}",
                "deg": f"s{c.deg}",
            }
        )
    return pd.DataFrame(rows)


def _baseline_rows(n: int, group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in group.iterrows():
        rows.append(
            {
                "id": f"{n:03d}_{idx}",
                "x": f"s{row['x']}",
                "y": f"s{row['y']}",
                "deg": f"s{row['deg']}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/submissions/best_submission.csv")
    parser.add_argument("--n-list", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--out-dir", default="results/exact_smalln")
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--angle-set", default="0,30,45,60,90,120,180,240,300")
    parser.add_argument("--angle-jitter", type=float, default=8.0)
    parser.add_argument("--jitter-span", type=float, default=0.12)
    parser.add_argument("--jitter-steps", type=int, default=2)
    parser.add_argument("--random-points", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7777)
    args = parser.parse_args()

    n_list = _parse_int_list(args.n_list)
    angle_set = _parse_float_list(args.angle_set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_groups = _load_baseline_groups(Path(args.baseline))
    updated_groups: Dict[int, pd.DataFrame] = {}
    log_rows = []

    for n in n_list:
        if n not in baseline_groups:
            continue
        start = time.time()
        base_group = baseline_groups[n]
        candidates = _generate_candidates(
            base_group,
            angle_set,
            args.angle_jitter,
            args.jitter_span,
            args.jitter_steps,
            args.random_points,
            args.seed + n,
            args.max_candidates,
        )
        if len(candidates) < n:
            updated_groups[n] = _baseline_rows(n, base_group)
            continue

        overlap_pairs = _build_overlap_pairs(candidates)
        selected = _solve_exact(
            n,
            candidates,
            overlap_pairs,
            args.time_limit,
            args.threads,
            args.scale,
        )
        if selected is None:
            updated_groups[n] = _baseline_rows(n, base_group)
            continue

        candidate_rows = _to_submission_rows(n, selected)
        base_rows = _baseline_rows(n, base_group)
        try:
            base_score = _score_group(base_rows)
            cand_score = _score_group(candidate_rows)
        except ParticipantVisibleError:
            updated_groups[n] = _baseline_rows(n, base_group)
            continue

        if cand_score < base_score:
            updated_groups[n] = candidate_rows
        else:
            updated_groups[n] = base_rows
        elapsed = time.time() - start
        log_rows.append({"n": n, "candidates": len(candidates), "pairs": len(overlap_pairs), "time_s": round(elapsed, 2)})

    # fill remaining groups from baseline
    for n, group in baseline_groups.items():
        if n not in updated_groups:
            updated_groups[n] = _baseline_rows(n, group)

    combined = pd.concat(updated_groups.values(), ignore_index=True)
    combined["group"] = combined["id"].astype(str).str.split("_").str[0].astype(int)
    combined["item"] = combined["id"].astype(str).str.split("_").str[1].astype(int)
    combined = combined.sort_values(["group", "item"]).drop(columns=["group", "item"])

    out_csv = out_dir / "best_submission.csv"
    combined.to_csv(out_csv, index=False)

    try:
        total_score, _ = score_detailed(combined)
        print(f"total_score={total_score:.6f}")
    except ParticipantVisibleError as exc:
        print(f"score_error: {exc}")

    if log_rows:
        pd.DataFrame(log_rows).to_csv(out_dir / "run_log.csv", index=False)


if __name__ == "__main__":
    main()
