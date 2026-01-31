#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.geometry import build_tree_polygon
from santa2025.io import build_submission, write_submission_csv
from santa2025.solver.periodic import PeriodicBasis, PeriodicConfig, PeriodicSolver


def _overlaps(poly_a, poly_b) -> bool:
    return poly_a.intersects(poly_b) and not poly_a.touches(poly_b)


def _valid_basis(
    basis: List[PeriodicBasis],
    dx: float,
    dy: float,
    offset: float,
    lattice_angle_deg: float,
    scale: float = 1e6,
    neighbor_range: int = 2,
) -> bool:
    if not basis:
        return False
    shifts = range(-neighbor_range, neighbor_range + 1)
    angle_rad = float(lattice_angle_deg) * (3.141592653589793 / 180.0)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def rotate(x: float, y: float) -> tuple[float, float]:
        if lattice_angle_deg:
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        return (x, y)

    polys = []
    for b in basis:
        rx, ry = rotate(b.x, b.y)
        polys.append(build_tree_polygon(rx, ry, b.deg, scale))
    for i, poly_i in enumerate(polys):
        for j, base_b in enumerate(basis):
            for sx in shifts:
                for sy in shifts:
                    if i == j and sx == 0 and sy == 0:
                        continue
                    tx = base_b.x + sx * dx + sy * offset
                    ty = base_b.y + sy * dy
                    rx, ry = rotate(tx, ty)
                    poly_j = build_tree_polygon(rx, ry, base_b.deg, scale)
                    if _overlaps(poly_i, poly_j):
                        return False
    return True


def _random_basis(
    k: int,
    dx: float,
    dy: float,
    offset: float,
    lattice_angle_deg: float,
    angle_set: List[float],
    angle_jitter: float,
    rng: random.Random,
    neighbor_range: int,
    attempts: int,
) -> List[PeriodicBasis] | None:
    basis: List[PeriodicBasis] = []
    for _ in range(k):
        for _ in range(attempts):
            x = rng.uniform(0.0, dx)
            y = rng.uniform(0.0, dy)
            deg = rng.choice(angle_set)
            if angle_jitter > 0.0:
                deg = (deg + rng.uniform(-angle_jitter, angle_jitter)) % 360.0
            candidate = basis + [PeriodicBasis(x=x, y=y, deg=deg)]
            if _valid_basis(candidate, dx, dy, offset, lattice_angle_deg, neighbor_range=neighbor_range):
                basis = candidate
                break
        else:
            return None
    return basis


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_basis(basis: List[PeriodicBasis], dx: float, dy: float) -> List[PeriodicBasis]:
    normalized: List[PeriodicBasis] = []
    for b in basis:
        x = b.x % dx if dx else b.x
        y = b.y % dy if dy else b.y
        deg = b.deg % 360.0
        normalized.append(PeriodicBasis(x=x, y=y, deg=deg))
    return normalized


def _basis_from_record(record: dict) -> List[PeriodicBasis]:
    return [PeriodicBasis(x=b["x"], y=b["y"], deg=b["deg"]) for b in record["basis"]]


def _record_from_params(
    record: dict,
    dx: float,
    dy: float,
    offset: float,
    lattice_angle_deg: float,
    basis: List[PeriodicBasis],
    proxy_score: float,
) -> dict:
    updated = dict(record)
    updated["dx"] = dx
    updated["dy"] = dy
    updated["offset"] = offset
    updated["lattice_angle_deg"] = lattice_angle_deg
    updated["basis"] = [asdict(b) for b in basis]
    updated["proxy_score"] = proxy_score
    return updated


def _density_score(dx: float, dy: float, k: int) -> float:
    return (dx * dy) / float(k)


def _score_proxy(cfg: PeriodicConfig, score_n_list: Iterable[int]) -> float:
    solver = PeriodicSolver(cfg)
    return _score_subset(solver, score_n_list)


def _refine_candidate(
    record: dict,
    rng: random.Random,
    score_n_list: List[int],
    args: argparse.Namespace,
    dx_min: float,
    dx_max: float,
    dy_min: float,
    dy_max: float,
    off_min: float,
    off_max: float,
) -> dict:
    best = dict(record)
    best_proxy = float(record["proxy_score"])

    steps = int(args.refine_steps)
    restarts = int(args.refine_restarts)
    if steps <= 0:
        return best

    for _ in range(max(1, restarts)):
        cur = dict(best)
        cur_proxy = best_proxy
        dx = float(cur["dx"])
        dy = float(cur["dy"])
        offset = float(cur["offset"])
        lattice_angle = float(cur.get("lattice_angle_deg", 0.0))
        basis = _basis_from_record(cur)

        offset_ratio = offset / dx if dx > 0 else 0.0

        step_dx = (dx_max - dx_min) * float(args.refine_dx_scale)
        step_dy = (dy_max - dy_min) * float(args.refine_dy_scale)
        step_off = float(args.refine_offset_scale)
        step_angle = float(args.refine_angle_scale)
        step_basis = float(args.refine_basis_scale)
        step_deg = float(args.refine_deg_scale)
        decay = float(args.refine_decay)
        accept_temp = float(args.refine_accept_temp)

        for i in range(steps):
            scale = decay ** i
            ndx, ndy = dx, dy
            noffset = offset
            nangle = lattice_angle
            nbasis = [PeriodicBasis(x=b.x, y=b.y, deg=b.deg) for b in basis]

            pick = rng.random()
            if pick < 0.2:
                ndx = _clamp(dx + rng.uniform(-1.0, 1.0) * step_dx * scale, dx_min, dx_max)
                noffset = offset_ratio * ndx
            elif pick < 0.4:
                ndy = _clamp(dy + rng.uniform(-1.0, 1.0) * step_dy * scale, dy_min, dy_max)
            elif pick < 0.55:
                noffset = offset + rng.uniform(-1.0, 1.0) * step_off * scale * dx
            elif pick < 0.7:
                nangle = _clamp(
                    lattice_angle + rng.uniform(-1.0, 1.0) * step_angle * scale,
                    float(args.lattice_angle_min),
                    float(args.lattice_angle_max),
                )
            elif pick < 0.9 and nbasis:
                idx = rng.randrange(len(nbasis))
                nx = nbasis[idx].x + rng.uniform(-1.0, 1.0) * step_basis * scale * dx
                ny = nbasis[idx].y + rng.uniform(-1.0, 1.0) * step_basis * scale * dy
                nbasis[idx] = PeriodicBasis(x=nx, y=ny, deg=nbasis[idx].deg)
            elif nbasis:
                idx = rng.randrange(len(nbasis))
                ndeg = nbasis[idx].deg + rng.uniform(-1.0, 1.0) * step_deg * scale
                nbasis[idx] = PeriodicBasis(x=nbasis[idx].x, y=nbasis[idx].y, deg=ndeg)

            if ndx <= 0 or ndy <= 0:
                continue

            noffset = _clamp(noffset, off_min * ndx, off_max * ndx)
            nbasis = _normalize_basis(nbasis, ndx, ndy)

            if not _valid_basis(
                nbasis,
                ndx,
                ndy,
                noffset,
                nangle,
                neighbor_range=args.neighbor_range,
            ):
                continue

            cfg = PeriodicConfig(
                dx=ndx,
                dy=ndy,
                offset=noffset,
                basis=nbasis,
                search_pad=args.search_pad,
                center_steps=args.center_steps,
                selection_mode="square_search",
                lattice_angle_deg=nangle,
                global_squeeze=args.final_global_squeeze,
                squeeze_factor=args.final_squeeze_factor,
                squeeze_steps=args.final_squeeze_steps,
                squeeze_iters=args.final_squeeze_iters,
            )
            proxy = _score_proxy(cfg, score_n_list)

            accept = proxy < cur_proxy
            if not accept and accept_temp > 0.0:
                delta = proxy - cur_proxy
                accept = rng.random() < pow(2.718281828, -delta / accept_temp)

            if accept:
                dx, dy, offset, lattice_angle = ndx, ndy, noffset, nangle
                basis = nbasis
                cur_proxy = proxy
                offset_ratio = offset / dx if dx > 0 else 0.0
                if proxy < best_proxy:
                    best_proxy = proxy
                    best = _record_from_params(best, dx, dy, offset, lattice_angle, basis, proxy)

    return best


def _parse_range(value: str) -> Tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected range as 'min,max', got {value!r}")
    return float(parts[0]), float(parts[1])


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _score_subset(solver: PeriodicSolver, n_list: Iterable[int]) -> float:
    total = 0.0
    for n in n_list:
        layout = solver._best_layout(n)
        score, _ = solver._score_and_bounds(layout)
        total += score
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-list", default="2,3,4")
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--keep-density", type=int, default=0)
    parser.add_argument("--keep-proxy", type=int, default=0)
    parser.add_argument("--density-threshold", type=float, default=0.0)
    parser.add_argument("--angle-set", default="0,30,45,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--angle-jitter", type=float, default=8.0)
    parser.add_argument("--dx-range", default="0.28,0.9")
    parser.add_argument("--dy-range", default="0.35,1.1")
    parser.add_argument("--offset-range", default="0.0,1.0")
    parser.add_argument("--lattice-angle-range", default="-15.0,15.0")
    parser.add_argument("--basis-attempts", type=int, default=120)
    parser.add_argument("--search-pad", type=int, default=8)
    parser.add_argument("--center-steps", type=int, default=8)
    parser.add_argument("--score-n-list", default="80,100,120,150,200")
    parser.add_argument("--final-n-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--out", default="results/periodic_density.json")
    parser.add_argument("--neighbor-range", type=int, default=2)
    parser.add_argument("--emit-submission", default="")
    parser.add_argument("--emit-decimals", type=int, default=6)
    parser.add_argument("--final-global-squeeze", action="store_true")
    parser.add_argument("--final-squeeze-factor", type=float, default=0.985)
    parser.add_argument("--final-squeeze-steps", type=int, default=20)
    parser.add_argument("--final-squeeze-iters", type=int, default=8)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--refine-restarts", type=int, default=1)
    parser.add_argument("--refine-dx-scale", type=float, default=0.08)
    parser.add_argument("--refine-dy-scale", type=float, default=0.08)
    parser.add_argument("--refine-offset-scale", type=float, default=0.08)
    parser.add_argument("--refine-angle-scale", type=float, default=6.0)
    parser.add_argument("--refine-basis-scale", type=float, default=0.12)
    parser.add_argument("--refine-deg-scale", type=float, default=10.0)
    parser.add_argument("--refine-decay", type=float, default=0.985)
    parser.add_argument("--refine-accept-temp", type=float, default=0.0)
    parser.add_argument("--emit-config", default="")
    args = parser.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    angle_set = [float(x) for x in args.angle_set.split(",") if x.strip()]
    dx_min, dx_max = _parse_range(args.dx_range)
    dy_min, dy_max = _parse_range(args.dy_range)
    off_min, off_max = _parse_range(args.offset_range)
    angle_min, angle_max = _parse_range(args.lattice_angle_range)
    args.lattice_angle_min = angle_min
    args.lattice_angle_max = angle_max
    score_n_list = _parse_int_list(args.score_n_list)

    rng = random.Random(args.seed)
    results: List[dict] = []

    keep_density = int(args.keep_density) if args.keep_density > 0 else int(args.keep)
    keep_proxy = int(args.keep_proxy) if args.keep_proxy > 0 else int(args.keep)

    for k in k_list:
        density_best: List[dict] = []
        for _ in range(args.trials):
            dx = rng.uniform(dx_min, dx_max)
            dy = rng.uniform(dy_min, dy_max)
            offset = rng.uniform(off_min, off_max) * dx
            lattice_angle_deg = rng.uniform(angle_min, angle_max)
            basis = _random_basis(
                k,
                dx,
                dy,
                offset,
                lattice_angle_deg,
                angle_set,
                args.angle_jitter,
                rng,
                neighbor_range=args.neighbor_range,
                attempts=args.basis_attempts,
            )
            if basis is None:
                continue
            density = _density_score(dx, dy, k)
            if args.density_threshold > 0 and density > args.density_threshold:
                continue
            record = {
                "k": k,
                "dx": dx,
                "dy": dy,
                "offset": offset,
                "lattice_angle_deg": lattice_angle_deg,
                "basis": [asdict(b) for b in basis],
                "density": density,
            }
            density_best.append(record)
            density_best.sort(key=lambda r: r["density"])
            density_best = density_best[:keep_density]

        proxy_best: List[dict] = []
        for record in density_best:
            basis = [PeriodicBasis(x=b["x"], y=b["y"], deg=b["deg"]) for b in record["basis"]]
            cfg = PeriodicConfig(
                dx=record["dx"],
                dy=record["dy"],
                offset=record["offset"],
                basis=basis,
                search_pad=args.search_pad,
                center_steps=args.center_steps,
                selection_mode="square_search",
                lattice_angle_deg=record.get("lattice_angle_deg", 0.0),
                global_squeeze=False,
            )
            proxy_score = _score_proxy(cfg, score_n_list)
            record["proxy_score"] = proxy_score

            if args.refine_steps > 0:
                record = _refine_candidate(
                    record,
                    rng,
                    score_n_list,
                    args,
                    dx_min,
                    dx_max,
                    dy_min,
                    dy_max,
                    off_min,
                    off_max,
                )
            proxy_best.append(record)
            proxy_best.sort(key=lambda r: r["proxy_score"])
            proxy_best = proxy_best[:keep_proxy]

        for record in proxy_best:
            basis = [
                PeriodicBasis(x=b["x"], y=b["y"], deg=b["deg"])
                for b in record["basis"]
            ]
            cfg = PeriodicConfig(
                dx=record["dx"],
                dy=record["dy"],
                offset=record["offset"],
                basis=basis,
                search_pad=args.search_pad,
                center_steps=args.center_steps,
                selection_mode="square_search",
                lattice_angle_deg=record.get("lattice_angle_deg", 0.0),
                global_squeeze=args.final_global_squeeze,
                squeeze_factor=args.final_squeeze_factor,
                squeeze_steps=args.final_squeeze_steps,
                squeeze_iters=args.final_squeeze_iters,
            )
            solver = PeriodicSolver(cfg)
            total_score = solver.score_total(args.final_n_max)
            record["total_score"] = total_score
            results.append(record)
            print(
                f"k={k} density={record['density']:.6f} proxy={record['proxy_score']:.6f} total={total_score:.6f}",
                flush=True,
            )

    results.sort(key=lambda r: r.get("total_score", float("inf")))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} candidates to {out_path}")

    if args.emit_config and results:
        best_cfg = results[0]
        cfg_payload = {
            "seed": args.seed,
            "n_max": args.final_n_max,
            "output_decimals": args.emit_decimals,
            "baseline": {
                "mode": "periodic",
                "dx": float(best_cfg["dx"]),
                "dy": float(best_cfg["dy"]),
                "offset": float(best_cfg["offset"]),
                "basis": best_cfg["basis"],
                "search_pad": args.search_pad,
                "center_steps": args.center_steps,
                "selection_mode": "square_search",
                "lattice_angle_deg": float(best_cfg.get("lattice_angle_deg", 0.0)),
                "global_squeeze": args.final_global_squeeze,
                "squeeze_factor": args.final_squeeze_factor,
                "squeeze_steps": args.final_squeeze_steps,
                "squeeze_iters": args.final_squeeze_iters,
            },
            "refine": {"enabled": False},
        }
        emit_path = Path(args.emit_config)
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        emit_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
        print(f"Wrote config to {emit_path}", flush=True)

    if args.emit_submission and results:
        best = results[0]
        basis = [
            PeriodicBasis(x=b["x"], y=b["y"], deg=b["deg"])
            for b in best["basis"]
        ]
        cfg = PeriodicConfig(
            dx=best["dx"],
            dy=best["dy"],
            offset=best["offset"],
            basis=basis,
            search_pad=args.search_pad,
            center_steps=args.center_steps,
            selection_mode="square_search",
            lattice_angle_deg=best.get("lattice_angle_deg", 0.0),
            global_squeeze=args.final_global_squeeze,
            squeeze_factor=args.final_squeeze_factor,
            squeeze_steps=args.final_squeeze_steps,
            squeeze_iters=args.final_squeeze_iters,
        )
        solver = PeriodicSolver(cfg)
        groups = solver.solve(n_max=args.final_n_max, seed=args.seed)
        submission = build_submission(groups, decimals=args.emit_decimals)
        write_submission_csv(submission, Path(args.emit_submission))
        print(f"Wrote submission to {args.emit_submission}", flush=True)


if __name__ == "__main__":
    main()
