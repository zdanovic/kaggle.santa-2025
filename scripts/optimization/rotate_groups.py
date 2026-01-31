#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.geometry import TREE_POINTS
from santa2025.geometry import build_tree_polygon
from santa2025.io import (
    TreePlacement,
    build_submission,
    groups_from_submission,
    load_submission_csv,
    write_submission_csv,
)
from santa2025.metric import ParticipantVisibleError, score_detailed
from shapely.strtree import STRtree


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


def _rotation_matrix(angle_deg: float) -> tuple[float, float, float, float]:
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return c, -s, s, c


def _rotated_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    c, n_s, s, c2 = _rotation_matrix(angle_deg)
    x = points[:, 0]
    y = points[:, 1]
    rx = c * x + n_s * y
    ry = s * x + c2 * y
    return np.column_stack([rx, ry])


def _bounding_side(points: np.ndarray) -> float:
    minx = float(points[:, 0].min())
    maxx = float(points[:, 0].max())
    miny = float(points[:, 1].min())
    maxy = float(points[:, 1].max())
    return max(maxx - minx, maxy - miny)


def _build_group_points(placements: List[TreePlacement]) -> np.ndarray:
    base = np.array(TREE_POINTS, dtype=np.float64)
    all_points = []
    for p in placements:
        c, n_s, s, c2 = _rotation_matrix(p.deg)
        x = base[:, 0]
        y = base[:, 1]
        rx = c * x + n_s * y + p.x
        ry = s * x + c2 * y + p.y
        all_points.append(np.column_stack([rx, ry]))
    return np.vstack(all_points)


def _search_best_angle(
    points: np.ndarray,
    coarse_step: float,
    refine_step: float,
    fine_step: float,
    refine_radius: float,
    fine_radius: float,
) -> tuple[float, float]:
    best_angle = 0.0
    best_side = _bounding_side(_rotated_points(points, 0.0))

    for angle in np.arange(0.0, 90.0 + 1e-9, coarse_step):
        side = _bounding_side(_rotated_points(points, float(angle)))
        if side < best_side:
            best_side = side
            best_angle = float(angle)

    def _refine(step: float, radius: float, angle_center: float, side_best: float) -> tuple[float, float]:
        lo = max(0.0, angle_center - radius)
        hi = min(90.0, angle_center + radius)
        best_a = angle_center
        best_s = side_best
        for angle in np.arange(lo, hi + 1e-9, step):
            side = _bounding_side(_rotated_points(points, float(angle)))
            if side < best_s:
                best_s = side
                best_a = float(angle)
        return best_a, best_s

    best_angle, best_side = _refine(refine_step, refine_radius, best_angle, best_side)
    best_angle, best_side = _refine(fine_step, fine_radius, best_angle, best_side)
    return best_angle, best_side


def _rotate_group(
    placements: List[TreePlacement],
    angle_deg: float,
    points: np.ndarray,
) -> List[TreePlacement]:
    rotated_points = _rotated_points(points, angle_deg)
    minx = float(rotated_points[:, 0].min())
    maxx = float(rotated_points[:, 0].max())
    miny = float(rotated_points[:, 1].min())
    maxy = float(rotated_points[:, 1].max())
    shift_x = -0.5 * (minx + maxx)
    shift_y = -0.5 * (miny + maxy)

    c, n_s, s, c2 = _rotation_matrix(angle_deg)
    updated: List[TreePlacement] = []
    for p in placements:
        nx = c * p.x + n_s * p.y + shift_x
        ny = s * p.x + c2 * p.y + shift_y
        ndeg = (p.deg + angle_deg) % 360.0
        updated.append(TreePlacement(x=nx, y=ny, deg=ndeg))
    return updated


def _quantize(value: float, decimals: int) -> float:
    return float(f"{value:.{decimals}f}")


def _round_placements(placements: List[TreePlacement], decimals: int) -> List[TreePlacement]:
    return [
        TreePlacement(
            x=_quantize(p.x, decimals),
            y=_quantize(p.y, decimals),
            deg=_quantize(p.deg, decimals),
        )
        for p in placements
    ]


def _has_overlap(placements: List[TreePlacement], scale_factor: float = 1e18) -> bool:
    if len(placements) <= 1:
        return False
    polygons = [
        build_tree_polygon(p.x, p.y, p.deg, scale_factor)
        for p in placements
    ]
    tree = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for cand in tree.query(poly):
            if hasattr(cand, "geom_type"):
                other = cand
                if other is poly:
                    continue
            else:
                j = int(cand)
                if j == i:
                    continue
                other = polygons[j]
            if poly.intersects(other) and not poly.touches(other):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--n-list", default=None, help="Comma/range list, e.g. 1-50,80")
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    parser.add_argument("--coarse-step", type=float, default=1.0)
    parser.add_argument("--refine-step", type=float, default=0.1)
    parser.add_argument("--fine-step", type=float, default=0.02)
    parser.add_argument("--refine-radius", type=float, default=1.0)
    parser.add_argument("--fine-radius", type=float, default=0.2)
    args = parser.parse_args()

    df = load_submission_csv(Path(args.submission))
    groups = groups_from_submission(df)
    orig_groups = {k: [TreePlacement(**asdict(p)) for p in v] for k, v in groups.items()}

    targets = _parse_target_ns(args.n_list)
    if targets is None:
        targets = sorted(groups.keys())

    summary: Dict[int, dict] = {}
    reverted: List[int] = []
    for n in targets:
        placements = groups[n]
        points = _build_group_points(placements)
        base_side = _bounding_side(points)
        best_angle, best_side = _search_best_angle(
            points,
            args.coarse_step,
            args.refine_step,
            args.fine_step,
            args.refine_radius,
            args.fine_radius,
        )
        base_score = (base_side * base_side) / float(n)
        best_score = (best_side * best_side) / float(n)
        improvement = base_score - best_score

        summary[n] = {
            "base_side": base_side,
            "best_side": best_side,
            "base_score": base_score,
            "best_score": best_score,
            "angle_deg": best_angle,
            "improvement": improvement,
        }

        if improvement > args.min_improvement:
            rotated = _rotate_group(placements, best_angle, points)
            rounded = _round_placements(rotated, args.decimals)
            if _has_overlap(rounded):
                groups[n] = orig_groups[n]
                reverted.append(n)
            else:
                groups[n] = rounded

    submission = build_submission(groups, decimals=args.decimals)

    total_score = float("inf")
    while True:
        try:
            total_score, _ = score_detailed(submission)
            break
        except ParticipantVisibleError as exc:
            msg = str(exc)
            match = re.search(r"group (\d+)", msg)
            if not match:
                break
            group_id = int(match.group(1))
            if group_id in orig_groups:
                groups[group_id] = orig_groups[group_id]
                reverted.append(group_id)
                submission = build_submission(groups, decimals=args.decimals)
                continue
            break

    write_submission_csv(submission, Path(args.out))

    summary_payload = {
        "total_score": total_score,
        "reverted_groups": sorted(set(reverted)),
        "config": {
            "decimals": args.decimals,
            "min_improvement": args.min_improvement,
            "coarse_step": args.coarse_step,
            "refine_step": args.refine_step,
            "fine_step": args.fine_step,
            "refine_radius": args.refine_radius,
            "fine_radius": args.fine_radius,
            "n_list": args.n_list,
        },
        "per_group": summary,
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(f"total_score: {total_score}", flush=True)


if __name__ == "__main__":
    main()
