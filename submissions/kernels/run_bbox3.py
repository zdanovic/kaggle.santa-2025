#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import pandas as pd
    from shapely import affinity
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
except ModuleNotFoundError as exc:
    missing = exc.name
    if missing not in {"shapely", "pandas"}:
        raise
    if missing == "pandas":
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "shapely"], check=True)
    import pandas as pd
    from shapely import affinity
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

try:
    from scipy.optimize import minimize_scalar
    from scipy.spatial import ConvexHull

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


TREE_POINTS = (
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
)
SCALE_FACTOR = 1e18


@dataclass
class Tree:
    x: float
    y: float
    deg: float
    polygon: Polygon

    def clone(self) -> "Tree":
        return Tree(self.x, self.y, self.deg, build_tree_polygon(self.x, self.y, self.deg))


def build_tree_polygon(x: float, y: float, deg: float) -> Polygon:
    base = Polygon([(px * SCALE_FACTOR, py * SCALE_FACTOR) for px, py in TREE_POINTS])
    rotated = affinity.rotate(base, deg, origin=(0.0, 0.0))
    return affinity.translate(
        rotated,
        xoff=x * SCALE_FACTOR,
        yoff=y * SCALE_FACTOR,
    )


def _format_value(value: float, decimals: int) -> str:
    return f"s{value:.{decimals}f}"


def parse_csv(csv_path: str) -> Tuple[Dict[str, List[Tree]], Dict[str, float]]:
    df = pd.read_csv(csv_path)
    df["x"] = df["x"].astype(str).str.strip().str.lstrip("s")
    df["y"] = df["y"].astype(str).str.strip().str.lstrip("s")
    df["deg"] = df["deg"].astype(str).str.strip().str.lstrip("s")
    df[["group_id", "item_id"]] = df["id"].str.split("_", n=2, expand=True)

    groups: Dict[str, List[Tree]] = {}
    side_lengths: Dict[str, float] = {}
    for group_id, group_data in df.groupby("group_id"):
        trees = [
            Tree(
                x=float(row["x"]),
                y=float(row["y"]),
                deg=float(row["deg"]),
                polygon=build_tree_polygon(float(row["x"]), float(row["y"]), float(row["deg"])),
            )
            for _, row in group_data.iterrows()
        ]
        groups[group_id] = trees
        side_lengths[group_id] = side_length(trees)
    return groups, side_lengths


def side_length(trees: Iterable[Tree]) -> float:
    all_polygons = [t.polygon for t in trees]
    bounds = unary_union(all_polygons).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / SCALE_FACTOR


def total_score(side_lengths: Dict[str, float]) -> float:
    score = 0.0
    for group_id, value in side_lengths.items():
        score += (value * value) / float(int(group_id))
    return score


def calculate_bbox_side_at_angle(angle_deg: float, points: np.ndarray) -> float:
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated = points.dot(rot_matrix_T)
    min_xy = np.min(rotated, axis=0)
    max_xy = np.max(rotated, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])


def _hull_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    if SCIPY_AVAILABLE:
        hull = ConvexHull(points)
        return points[hull.vertices]
    hull = MultiPoint(points).convex_hull
    coords = np.array(hull.exterior.coords)
    return coords[:-1]


def _edge_angles(points: np.ndarray) -> List[float]:
    angles: set[float] = set()
    if points.shape[0] < 2:
        return [0.0]
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 90.0
        angles.add(angle)
    return sorted(angles) if angles else [0.0]


def optimize_rotation(trees: List[Tree], angle_max: float, epsilon: float) -> Tuple[float, float]:
    points = []
    for tree in trees:
        points.extend(list(tree.polygon.exterior.coords))
    points_np = np.array(points)
    if points_np.size == 0:
        return 0.0, 0.0

    hull_points = _hull_points(points_np)
    initial_side = calculate_bbox_side_at_angle(0.0, hull_points)

    if SCIPY_AVAILABLE and hull_points.shape[0] >= 3:
        res = minimize_scalar(
            lambda a: calculate_bbox_side_at_angle(a, hull_points),
            bounds=(0.001, float(angle_max)),
            method="bounded",
        )
        best_angle = float(res.x)
        best_side = float(res.fun)
    else:
        best_angle = 0.0
        best_side = initial_side
        for angle in _edge_angles(hull_points):
            if angle > angle_max:
                continue
            cand = calculate_bbox_side_at_angle(angle, hull_points)
            if cand < best_side:
                best_side = cand
                best_angle = angle

    epsilon_scaled = epsilon * SCALE_FACTOR
    if initial_side - best_side <= epsilon_scaled:
        return initial_side / SCALE_FACTOR, 0.0
    return best_side / SCALE_FACTOR, best_angle


def apply_rotation(trees: List[Tree], angle_deg: float) -> List[Tree]:
    if not trees or abs(angle_deg) < 1e-12:
        return [t.clone() for t in trees]

    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds) / SCALE_FACTOR
    min_y = min(b[1] for b in bounds) / SCALE_FACTOR
    max_x = max(b[2] for b in bounds) / SCALE_FACTOR
    max_y = max(b[3] for b in bounds) / SCALE_FACTOR
    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])

    points = np.array([[t.x, t.y] for t in trees])
    rotated = (points - center).dot(rot_matrix.T) + center

    rotated_trees = []
    for i, tree in enumerate(trees):
        nx = float(rotated[i, 0])
        ny = float(rotated[i, 1])
        ndeg = (tree.deg + angle_deg) % 360.0
        rotated_trees.append(Tree(nx, ny, ndeg, build_tree_polygon(nx, ny, ndeg)))
    return rotated_trees


def write_submission(groups: Dict[str, List[Tree]], out_file: str, decimals: int) -> None:
    rows = []
    for group_id in sorted(groups.keys(), key=lambda x: int(x)):
        for item_id, tree in enumerate(groups[group_id]):
            rows.append(
                {
                    "id": f"{group_id}_{item_id}",
                    "x": _format_value(tree.x, decimals),
                    "y": _format_value(tree.y, decimals),
                    "deg": _format_value(tree.deg, decimals),
                }
            )
    pd.DataFrame(rows).to_csv(out_file, index=False)


def fix_direction(
    in_csv: str,
    out_csv: str,
    passes: int,
    angle_max: float,
    epsilon: float,
    group_max: int,
    decimals: int,
) -> float:
    groups, side_lengths = parse_csv(in_csv)
    current_score = total_score(side_lengths)

    for _ in range(int(passes)):
        changed = False
        for group_id_main in range(group_max, 2, -1):
            gid = f"{group_id_main:03d}"
            if gid not in groups:
                continue
            trees = groups[gid]
            best_side, best_angle = optimize_rotation(trees, angle_max, epsilon)
            if best_side < side_lengths[gid]:
                groups[gid] = apply_rotation(trees, best_angle)
                side_lengths[gid] = best_side
                changed = True
        new_score = total_score(side_lengths)
        if new_score >= current_score or not changed:
            current_score = new_score
            break
        current_score = new_score

    write_submission(groups, out_csv, decimals)
    return current_score


def has_overlap(trees: List[Tree]) -> bool:
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for cand in tree_index.query(poly):
            if isinstance(cand, (int, np.integer)):
                j = int(cand)
                if j == i:
                    continue
                other = polygons[j]
            else:
                if cand is poly:
                    continue
                other = cand
            if poly.intersects(other) and not poly.touches(other):
                return True
    return False


def score_and_validate_submission(path: str, group_max: int) -> Dict[str, object]:
    groups, side_lengths = parse_csv(path)
    failed = []
    for n in range(1, group_max + 1):
        gid = f"{n:03d}"
        trees = groups.get(gid)
        if not trees:
            continue
        if has_overlap(trees):
            failed.append(n)
    return {
        "total_score": total_score(side_lengths),
        "failed_overlap_n": failed,
        "ok": len(failed) == 0,
    }


def _load_groups(filename: str) -> Tuple[List[str], Dict[str, List[List[str]]]]:
    groups: Dict[str, List[List[str]]] = {}
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            group = row[0].split("_")[0]
            groups.setdefault(group, []).append(row)
    return header, groups


def replace_group(target_file: str, donor_file: str, group_id: str) -> None:
    header_t, groups_t = _load_groups(target_file)
    _, groups_d = _load_groups(donor_file)
    if group_id not in groups_d:
        raise ValueError(f"Donor file has no group {group_id}")
    groups_t[group_id] = groups_d[group_id]
    with open(target_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_t)
        for g in sorted(groups_t.keys(), key=lambda x: int(x)):
            for row in groups_t[g]:
                writer.writerow(row)


def repair_overlaps_in_place(
    submission_path: str,
    donor_path: str,
    angle_max: float,
    epsilon: float,
    group_max: int,
    decimals: int,
) -> Dict[str, object]:
    res = score_and_validate_submission(submission_path, group_max)
    if res["ok"]:
        return res
    for n in res["failed_overlap_n"]:
        replace_group(submission_path, donor_path, f"{n:03d}")
    fix_direction(
        submission_path,
        submission_path,
        passes=1,
        angle_max=angle_max,
        epsilon=epsilon,
        group_max=group_max,
        decimals=decimals,
    )
    return score_and_validate_submission(submission_path, group_max)


FINAL_SCORE_RE = re.compile(
    r"Final\s+(?:Total\s+)?Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def parse_bbox3_final_score(output: str) -> float | None:
    match = FINAL_SCORE_RE.search(output or "")
    return float(match.group(1)) if match else None


def run_bbox3(
    binary: str,
    timeout_sec: int,
    n_value: int,
    r_value: int,
    work_dir: Path,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [binary, "-n", str(n_value), "-r", str(r_value)],
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
        cwd=str(work_dir),
    )


def _write_bbox3_log(out_dir: Path, n_value: int, r_value: int, proc: subprocess.CompletedProcess) -> Path:
    log_dir = out_dir / "bbox3_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"bbox3_n{n_value}_r{r_value}.log"
    parts = []
    if proc.stdout:
        parts.append("=== STDOUT ===\n" + proc.stdout)
    if proc.stderr:
        parts.append("=== STDERR ===\n" + proc.stderr)
    log_path.write_text("\n\n".join(parts))
    return log_path


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _log(msg: str, log_file: str | None) -> None:
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="/kaggle/input/santa-2025-csv/santa-2025.csv")
    parser.add_argument("--bbox3", default="/kaggle/input/santa-2025-csv/bbox3")
    parser.add_argument("--work-dir", default="/kaggle/working")
    parser.add_argument("--out-dir", default="/kaggle/working/bbox3_run")
    parser.add_argument("--budget-sec", type=int, default=36000)
    parser.add_argument("--buffer-sec", type=int, default=1200)
    parser.add_argument("--decimals", type=int, default=15)
    parser.add_argument("--min-improvement", type=float, default=1e-10)
    parser.add_argument("--angle-max", type=float, default=89.999)
    parser.add_argument("--angle-epsilon", type=float, default=1e-7)
    parser.add_argument("--group-max", type=int, default=200)
    parser.add_argument("--initial-fix-passes", type=int, default=2)
    parser.add_argument("--phase-a-timeout", type=int, default=900)
    parser.add_argument(
        "--phase-a-n",
        default="1200,1500,1800",
    )
    parser.add_argument("--phase-a-r", default="40,60")
    parser.add_argument("--phase-a-top-k", type=int, default=4)
    parser.add_argument("--phase-a-fix-passes", type=int, default=1)
    parser.add_argument("--phase-b-timeout", type=int, default=1800)
    parser.add_argument("--phase-b-top-k", type=int, default=3)
    parser.add_argument("--phase-b-fix-passes", type=int, default=2)
    parser.add_argument("--phase-c-timeout", type=int, default=3600)
    parser.add_argument("--phase-c-top-k", type=int, default=2)
    parser.add_argument("--fallback-n", type=int, default=1500)
    parser.add_argument("--fallback-r", type=int, default=60)
    parser.add_argument("--fallback-timeout", type=int, default=3600)
    parser.add_argument("--phase-c-fix-passes", type=int, default=3)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or str(out_dir / "bbox3_run.log")

    baseline = Path(args.baseline)
    bbox3_src = Path(args.bbox3)
    if not baseline.exists():
        raise SystemExit(f"baseline not found: {baseline}")
    if not bbox3_src.exists():
        raise SystemExit(f"bbox3 binary not found: {bbox3_src}")

    work_submission = work_dir / "submission.csv"
    bbox3_bin = work_dir / "bbox3"
    shutil.copy(baseline, work_submission)
    shutil.copy(bbox3_src, bbox3_bin)
    subprocess.run(["chmod", "+x", str(bbox3_bin)], check=False)

    start = time.time()

    def time_left() -> float:
        return args.budget_sec - (time.time() - start)

    _log("=" * 68, log_file)
    _log(f"START {datetime.now().isoformat(timespec='seconds')}", log_file)
    _log(f"BUDGET {args.budget_sec}s", log_file)
    _log(f"SCIPY_AVAILABLE={SCIPY_AVAILABLE}", log_file)
    _log("=" * 68, log_file)

    _log("initial fix_direction", log_file)
    best_score = fix_direction(
        str(work_submission),
        str(work_submission),
        passes=args.initial_fix_passes,
        angle_max=args.angle_max,
        epsilon=args.angle_epsilon,
        group_max=args.group_max,
        decimals=args.decimals,
    )
    val0 = repair_overlaps_in_place(
        str(work_submission),
        str(baseline),
        angle_max=args.angle_max,
        epsilon=args.angle_epsilon,
        group_max=args.group_max,
        decimals=args.decimals,
    )
    best_score = min(best_score, float(val0["total_score"]))
    best_path = out_dir / "best_submission.csv"
    shutil.copy(work_submission, best_path)
    _log(f"initial best_score={best_score:.14f} overlap_ok={val0['ok']}", log_file)

    candidates: List[Dict[str, float]] = []
    phase_a_n = _parse_int_list(args.phase_a_n)
    phase_a_r = _parse_int_list(args.phase_a_r)

    _log("phase A: short scan", log_file)
    for r_value in phase_a_r:
        for n_value in phase_a_n:
            if time_left() < args.buffer_sec:
                _log("phase A: budget low, stopping", log_file)
                break
            _log(
                f"[A] timeout={args.phase_a_timeout}s n={n_value} r={r_value} "
                f"time_left={time_left():.0f}s",
                log_file,
            )
            try:
                res = run_bbox3(
                    str(bbox3_bin),
                    args.phase_a_timeout,
                    n_value,
                    r_value,
                    work_dir,
                )
            except subprocess.TimeoutExpired:
                _log(f"[A] TIMEOUT n={n_value} r={r_value}", log_file)
                continue
            _write_bbox3_log(out_dir, n_value, r_value, res)
            bbox_final = parse_bbox3_final_score(res.stdout + "\n" + res.stderr)
            if bbox_final is None:
                _log(f"[A] no score parsed n={n_value} r={r_value}", log_file)
                raw = score_and_validate_submission(str(work_submission), args.group_max)
                bbox_final = float(raw["total_score"])
                _log(
                    f"[A] computed score={bbox_final:.14f} overlap_ok={raw['ok']}",
                    log_file,
                )
            if bbox_final < best_score - args.min_improvement:
                candidates.append({"n": n_value, "r": r_value, "score": bbox_final})
                _log(f"[A] candidate score={bbox_final:.14f}", log_file)
            else:
                _log(f"[A] not better score={bbox_final:.14f}", log_file)

    candidates.sort(key=lambda x: x["score"])
    candidates = candidates[: args.phase_a_top_k]
    _log(f"[A] top candidates={candidates}", log_file)

    for cand in list(candidates):
        if time_left() < args.buffer_sec:
            _log("[A] budget low for processing winners", log_file)
            break
        _log(f"[A->PROC] n={cand['n']} r={cand['r']}", log_file)
        try:
            run_bbox3(
                str(bbox3_bin),
                args.phase_a_timeout,
                int(cand["n"]),
                int(cand["r"]),
                work_dir,
            )
        except subprocess.TimeoutExpired:
            continue
        fix_direction(
            str(work_submission),
            str(work_submission),
            passes=args.phase_a_fix_passes,
            angle_max=args.angle_max,
            epsilon=args.angle_epsilon,
            group_max=args.group_max,
            decimals=args.decimals,
        )
        val = repair_overlaps_in_place(
            str(work_submission),
            str(baseline),
            angle_max=args.angle_max,
            epsilon=args.angle_epsilon,
            group_max=args.group_max,
            decimals=args.decimals,
        )
        cur = float(val["total_score"])
        snap = out_dir / f"A_n{int(cand['n'])}_r{int(cand['r'])}_score{cur:.12f}.csv"
        shutil.copy(work_submission, snap)
        if val["ok"] and cur < best_score - args.min_improvement:
            best_score = cur
            shutil.copy(work_submission, best_path)
            _log(f"[A->PROC] NEW BEST={best_score:.14f}", log_file)
        else:
            shutil.copy(best_path, work_submission)

    if not candidates:
        _log(
            f"[A] no candidates, running fallback n={args.fallback_n} r={args.fallback_r}",
            log_file,
        )
        try:
            res = run_bbox3(
                str(bbox3_bin),
                args.fallback_timeout,
                args.fallback_n,
                args.fallback_r,
                work_dir,
            )
            _write_bbox3_log(out_dir, args.fallback_n, args.fallback_r, res)
            bbox_final = parse_bbox3_final_score(res.stdout + "\n" + res.stderr)
            if bbox_final is None:
                _log("[A] fallback produced no final score, validating output", log_file)
            fix_direction(
                str(work_submission),
                str(work_submission),
                passes=args.phase_a_fix_passes,
                angle_max=args.angle_max,
                epsilon=args.angle_epsilon,
                group_max=args.group_max,
                decimals=args.decimals,
            )
            val = repair_overlaps_in_place(
                str(work_submission),
                str(baseline),
                angle_max=args.angle_max,
                epsilon=args.angle_epsilon,
                group_max=args.group_max,
                decimals=args.decimals,
            )
            cur = float(val["total_score"])
            snap = out_dir / f"A_fallback_n{args.fallback_n}_r{args.fallback_r}_score{cur:.12f}.csv"
            shutil.copy(work_submission, snap)
            if val["ok"] and cur < best_score - args.min_improvement:
                best_score = cur
                shutil.copy(work_submission, best_path)
                _log(f"[A->PROC] NEW BEST={best_score:.14f}", log_file)
            else:
                shutil.copy(best_path, work_submission)
            score_for_candidate = bbox_final if bbox_final is not None else cur
            candidates.append(
                {"n": args.fallback_n, "r": args.fallback_r, "score": float(score_for_candidate)}
            )
        except subprocess.TimeoutExpired:
            _log("[A] fallback TIMEOUT", log_file)

    _log("phase B: medium runs", log_file)
    candidates.sort(key=lambda x: x["score"])
    candidates = candidates[: args.phase_b_top_k]
    for cand in candidates:
        if time_left() < (args.phase_b_timeout + args.buffer_sec):
            _log("[B] budget low, stopping", log_file)
            break
        _log(f"[B] timeout={args.phase_b_timeout}s n={cand['n']} r={cand['r']}", log_file)
        try:
            res = run_bbox3(
                str(bbox3_bin),
                args.phase_b_timeout,
                int(cand["n"]),
                int(cand["r"]),
                work_dir,
            )
        except subprocess.TimeoutExpired:
            _log(f"[B] TIMEOUT n={cand['n']} r={cand['r']}", log_file)
            continue
        _write_bbox3_log(out_dir, int(cand["n"]), int(cand["r"]), res)
        bbox_final = parse_bbox3_final_score(res.stdout + "\n" + res.stderr)
        if bbox_final is None:
            _log(f"[B] no score parsed n={cand['n']} r={cand['r']}", log_file)
            raw = score_and_validate_submission(str(work_submission), args.group_max)
            bbox_final = float(raw["total_score"])
            _log(
                f"[B] computed score={bbox_final:.14f} overlap_ok={raw['ok']}",
                log_file,
            )
        if bbox_final >= best_score - args.min_improvement:
            _log(f"[B] not better score={bbox_final:.14f}", log_file)
            shutil.copy(best_path, work_submission)
            continue
        fix_direction(
            str(work_submission),
            str(work_submission),
            passes=args.phase_b_fix_passes,
            angle_max=args.angle_max,
            epsilon=args.angle_epsilon,
            group_max=args.group_max,
            decimals=args.decimals,
        )
        val = repair_overlaps_in_place(
            str(work_submission),
            str(baseline),
            angle_max=args.angle_max,
            epsilon=args.angle_epsilon,
            group_max=args.group_max,
            decimals=args.decimals,
        )
        cur = float(val["total_score"])
        snap = out_dir / f"B_n{int(cand['n'])}_r{int(cand['r'])}_score{cur:.12f}.csv"
        shutil.copy(work_submission, snap)
        if val["ok"] and cur < best_score - args.min_improvement:
            best_score = cur
            shutil.copy(work_submission, best_path)
            _log(f"[B] NEW BEST={best_score:.14f}", log_file)
        else:
            shutil.copy(best_path, work_submission)

    _log("phase C: long runs", log_file)
    phase_c_list: List[Dict[str, float]] = []
    if candidates:
        best_c = candidates[0]
        n0, r0 = int(best_c["n"]), int(best_c["r"])
        for dn in (-100, -50, 50, 100):
            phase_c_list.append({"n": max(1, n0 + dn), "r": r0, "score": best_c["score"]})
        for dr in (-10, -5, 5, 10):
            phase_c_list.append({"n": n0, "r": max(1, r0 + dr), "score": best_c["score"]})
    seen: set[Tuple[int, int]] = set()
    deduped = []
    for cand in phase_c_list:
        key = (int(cand["n"]), int(cand["r"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    for cand in deduped[: args.phase_c_top_k]:
        if time_left() < (args.phase_c_timeout + args.buffer_sec):
            _log("[C] budget low, stopping", log_file)
            break
        _log(f"[C] timeout={args.phase_c_timeout}s n={cand['n']} r={cand['r']}", log_file)
        try:
            res = run_bbox3(
                str(bbox3_bin),
                args.phase_c_timeout,
                int(cand["n"]),
                int(cand["r"]),
                work_dir,
            )
        except subprocess.TimeoutExpired:
            _log(f"[C] TIMEOUT n={cand['n']} r={cand['r']}", log_file)
            continue
        _write_bbox3_log(out_dir, int(cand["n"]), int(cand["r"]), res)
        bbox_final = parse_bbox3_final_score(res.stdout + "\n" + res.stderr)
        if bbox_final is None:
            _log(f"[C] no score parsed n={cand['n']} r={cand['r']}", log_file)
            raw = score_and_validate_submission(str(work_submission), args.group_max)
            bbox_final = float(raw["total_score"])
            _log(
                f"[C] computed score={bbox_final:.14f} overlap_ok={raw['ok']}",
                log_file,
            )
        if bbox_final >= best_score - args.min_improvement:
            _log(f"[C] not better score={bbox_final:.14f}", log_file)
            shutil.copy(best_path, work_submission)
            continue
        fix_direction(
            str(work_submission),
            str(work_submission),
            passes=args.phase_c_fix_passes,
            angle_max=args.angle_max,
            epsilon=args.angle_epsilon,
            group_max=args.group_max,
            decimals=args.decimals,
        )
        val = repair_overlaps_in_place(
            str(work_submission),
            str(baseline),
            angle_max=args.angle_max,
            epsilon=args.angle_epsilon,
            group_max=args.group_max,
            decimals=args.decimals,
        )
        cur = float(val["total_score"])
        snap = out_dir / f"C_n{int(cand['n'])}_r{int(cand['r'])}_score{cur:.12f}.csv"
        shutil.copy(work_submission, snap)
        if val["ok"] and cur < best_score - args.min_improvement:
            best_score = cur
            shutil.copy(work_submission, best_path)
            _log(f"[C] NEW BEST={best_score:.14f}", log_file)
        else:
            shutil.copy(best_path, work_submission)

    shutil.copy(best_path, work_submission)
    final_val = score_and_validate_submission(str(work_submission), args.group_max)
    _log("=" * 68, log_file)
    _log(f"END {datetime.now().isoformat(timespec='seconds')}", log_file)
    _log(f"BEST SCORE {best_score:.14f}", log_file)
    _log(f"FINAL overlap_ok={final_val['ok']} failed={final_val['failed_overlap_n']}", log_file)
    _log("=" * 68, log_file)


if __name__ == "__main__":
    main()
