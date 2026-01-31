#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from shapely.strtree import STRtree

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.geometry import build_tree_polygon
from santa2025.io import TreePlacement, groups_from_submission, write_submission_csv
from santa2025.metric import ParticipantVisibleError, score_detailed
from santa2025.scoring import per_group_dataframe
from santa2025.solver.local_search import LocalSearchConfig, LocalSearchRefiner
from concurrent.futures import ProcessPoolExecutor


def _parse_target_ns(value: str) -> List[int]:
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


def _refine_group(args: Tuple[int, List[TreePlacement], dict, int]) -> Tuple[int, List[TreePlacement], float]:
    n, placements, config_dict, seed = args
    config = LocalSearchConfig(**config_dict)
    refiner = LocalSearchRefiner(config)
    print(f"refine group {n}: start seed={seed}", flush=True)
    refined, score = refiner.refine(placements, seed=seed)
    print(f"refine group {n}: done score={score:.6f}", flush=True)
    return n, refined, score


def _quantize(value: float, decimals: int) -> float:
    return float(f"{value:.{decimals}f}")


def _round_placements(
    placements: List[TreePlacement],
    decimals: int,
) -> List[TreePlacement]:
    return [
        TreePlacement(
            x=_quantize(p.x, decimals),
            y=_quantize(p.y, decimals),
            deg=_quantize(p.deg, decimals),
        )
        for p in placements
    ]


def _has_overlap(placements: List[TreePlacement], scale_factor: float) -> bool:
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


def _build_group_frame(
    n: int,
    placements: List[TreePlacement],
    decimals: int,
) -> pd.DataFrame:
    rows = []
    for idx, placement in enumerate(placements):
        rows.append(
            {
                "id": f"{n:03d}_{idx}",
                "x": f"s{placement.x:.{decimals}f}",
                "y": f"s{placement.y:.{decimals}f}",
                "deg": f"s{placement.deg:.{decimals}f}",
            }
        )
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def _combine_frames(frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames.values(), ignore_index=True)
    combined["group"] = combined["id"].astype(str).str.split("_").str[0].astype(int)
    combined["item"] = combined["id"].astype(str).str.split("_").str[1].astype(int)
    combined = combined.sort_values(["group", "item"]).drop(columns=["group", "item"])
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-n-out", required=False)
    parser.add_argument("--n-list", default="1-12")
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--move-radius", type=float, default=0.12)
    parser.add_argument("--angle-radius", type=float, default=25.0)
    parser.add_argument("--swap-prob", type=float, default=0.0)
    parser.add_argument("--scale-prob", type=float, default=0.0)
    parser.add_argument("--scale-radius", type=float, default=0.02)
    parser.add_argument("--temp-start", type=float, default=1.0)
    parser.add_argument("--temp-end", type=float, default=0.01)
    parser.add_argument("--scale-factor", type=float, default=1e18)
    parser.add_argument("--bounds", type=float, default=100.0)
    parser.add_argument("--log-every-steps", type=int, default=10000)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    df["group"] = df["id"].astype(str).str.split("_").str[0].astype(int)
    orig_frames: Dict[int, pd.DataFrame] = {
        n: grp[["id", "x", "y", "deg"]].copy() for n, grp in df.groupby("group")
    }
    groups = groups_from_submission(df)
    targets = _parse_target_ns(args.n_list)
    targets = [n for n in targets if n in groups]
    if not targets:
        raise SystemExit("No matching groups found for n-list.")

    ls_config = LocalSearchConfig(
        steps=args.steps,
        restarts=args.restarts,
        move_radius=args.move_radius,
        angle_radius=args.angle_radius,
        swap_prob=args.swap_prob,
        scale_prob=args.scale_prob,
        scale_radius=args.scale_radius,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        scale_factor=args.scale_factor,
        bounds=args.bounds,
        log_every_steps=args.log_every_steps,
    )

    tasks = [(n, groups[n], asdict(ls_config), args.seed + n) for n in targets]
    results: List[Tuple[int, List[TreePlacement], float]] = []
    if args.max_workers > 1:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for result in executor.map(_refine_group, tasks):
                results.append(result)
    else:
        for task in tasks:
            results.append(_refine_group(task))

    for n, refined, _ in results:
        groups[n] = refined

    overlap_groups: List[int] = []
    refined_frames: Dict[int, pd.DataFrame] = {}
    for n in targets:
        rounded = _round_placements(groups[n], args.decimals)
        if _has_overlap(rounded, ls_config.scale_factor):
            print(f"overlap after rounding in group {n:03d}; reverting", flush=True)
            overlap_groups.append(n)
            continue
        refined_frames[n] = _build_group_frame(n, rounded, args.decimals)

    final_frames: Dict[int, pd.DataFrame] = {}
    for n in sorted(orig_frames.keys()):
        if n in refined_frames:
            final_frames[n] = refined_frames[n]
        else:
            final_frames[n] = orig_frames[n]

    submission = _combine_frames(final_frames)
    write_submission_csv(submission, Path(args.out))

    reverted_after_score: List[int] = []
    while True:
        try:
            total_score, per_group = score_detailed(submission)
            break
        except ParticipantVisibleError as exc:
            msg = str(exc)
            print(f"score_detailed failed: {msg}", flush=True)
            match = re.search(r"group (\d+)", msg)
            if not match:
                print("reverting all refined groups due to unknown overlap", flush=True)
                submission = _combine_frames(orig_frames)
                write_submission_csv(submission, Path(args.out))
                total_score, per_group = score_detailed(submission)
                reverted_after_score = targets[:]
                break
            group_id = int(match.group(1))
            if group_id in refined_frames:
                print(f"reverting group {group_id:03d} due to overlap", flush=True)
                final_frames[group_id] = orig_frames[group_id]
                refined_frames.pop(group_id, None)
                reverted_after_score.append(group_id)
                submission = _combine_frames(final_frames)
                write_submission_csv(submission, Path(args.out))
                continue
            print(
                f"overlap in non-refined group {group_id:03d}; reverting all refined groups",
                flush=True,
            )
            submission = _combine_frames(orig_frames)
            write_submission_csv(submission, Path(args.out))
            total_score, per_group = score_detailed(submission)
            reverted_after_score = targets[:]
            break

    summary = {
        "total_score": total_score,
        "refined_groups": targets,
        "overlap_fallback_groups": overlap_groups,
        "score_fallback_groups": reverted_after_score,
        "config": asdict(ls_config),
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"total_score: {total_score}", flush=True)

    if args.per_n_out:
        out_path = Path(args.per_n_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_group_dataframe(per_group).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
