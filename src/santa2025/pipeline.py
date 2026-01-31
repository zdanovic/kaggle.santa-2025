from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from santa2025.io import TreePlacement, build_submission, write_submission_csv
from santa2025.metric import score_detailed
from santa2025.scoring import per_group_dataframe, top_groups
from santa2025.solver.greedy import GreedyIncrementalSolver
from santa2025.solver.independent import IndependentConfig, IndependentSolver
from santa2025.solver.local_search import LocalSearchConfig, LocalSearchRefiner
from santa2025.solver.pattern import PatternConfig, PatternSolver
from santa2025.solver.periodic import PeriodicBasis, PeriodicConfig, PeriodicSolver


def _refine_group(args: Tuple[int, List[TreePlacement], dict, int]) -> Tuple[int, List[TreePlacement], float]:
    n, placements, config_dict, seed = args
    config = LocalSearchConfig(**config_dict)
    refiner = LocalSearchRefiner(config)
    print(f"refine group {n}: start seed={seed}", flush=True)
    refined, score = refiner.refine(placements, seed=seed)
    print(f"refine group {n}: done score={score:.6f}", flush=True)
    return n, refined, score


def _serialize_groups(groups: Dict[int, List[TreePlacement]]) -> dict:
    payload = {}
    for n, placements in groups.items():
        payload[str(n)] = [asdict(p) for p in placements]
    return payload


def _parse_target_ns(value) -> List[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
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
    if isinstance(value, list):
        return sorted({int(x) for x in value})
    return None


def run_experiment(config: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    seed = int(config.get("seed", 1337))
    n_max = int(config.get("n_max", 200))
    decimals = int(config.get("output_decimals", 6))

    baseline_cfg = config.get("baseline", {})
    mode = baseline_cfg.get("mode", "incremental")

    print(
        f"run_experiment start: seed={seed} n_max={n_max} decimals={decimals} mode={mode}",
        flush=True,
    )

    if mode == "pattern":
        raw_pairs = baseline_cfg.get("angle_pairs", [[0.0, 180.0]])
        angle_pairs = [(float(a), float(b)) for a, b in raw_pairs]
        pattern_cfg = PatternConfig(
            angle_pairs=angle_pairs,
            offset_ratios=[float(x) for x in baseline_cfg.get("offset_ratios", [0.6, 0.7])],
            dx_min=float(baseline_cfg.get("dx_min", 0.1)),
            dx_max=float(baseline_cfg.get("dx_max", 2.0)),
            dy_min=float(baseline_cfg.get("dy_min", 0.05)),
            dy_max=float(baseline_cfg.get("dy_max", 2.0)),
            grid_size=int(baseline_cfg.get("grid_size", 4)),
            spacing_margin=float(baseline_cfg.get("spacing_margin", 1.001)),
            rows_pad=int(baseline_cfg.get("rows_pad", 2)),
            scale_factor=float(baseline_cfg.get("scale_factor", 1e18)),
            bounds=float(baseline_cfg.get("bounds", 100.0)),
            global_squeeze=bool(baseline_cfg.get("global_squeeze", False)),
            squeeze_factor=float(baseline_cfg.get("squeeze_factor", 0.985)),
            squeeze_steps=int(baseline_cfg.get("squeeze_steps", 20)),
            squeeze_iters=int(baseline_cfg.get("squeeze_iters", 8)),
            jitter=float(baseline_cfg.get("jitter", 0.0)),
            angle_jitter=float(baseline_cfg.get("angle_jitter", 0.0)),
            selection_mode=str(baseline_cfg.get("selection_mode", "row_major")),
            search_pad=int(baseline_cfg.get("search_pad", 2)),
            center_steps=int(baseline_cfg.get("center_steps", 4)),
        )
        solver = PatternSolver(pattern_cfg)
        print(
            "baseline pattern: "
            f"pairs={pattern_cfg.angle_pairs} "
            f"offsets={pattern_cfg.offset_ratios} "
            f"grid={pattern_cfg.grid_size}",
            flush=True,
        )
        groups = solver.solve(n_max=n_max, seed=seed)
    elif mode == "periodic":
        basis_raw = baseline_cfg.get("basis", [])
        if not basis_raw:
            raise ValueError("periodic mode requires baseline.basis entries.")
        basis: List[PeriodicBasis] = []
        for entry in basis_raw:
            if isinstance(entry, dict):
                bx = float(entry["x"])
                by = float(entry["y"])
                bdeg = float(entry["deg"])
            else:
                bx, by, bdeg = entry
                bx = float(bx)
                by = float(by)
                bdeg = float(bdeg)
            basis.append(PeriodicBasis(x=bx, y=by, deg=bdeg))

        periodic_cfg = PeriodicConfig(
            dx=float(baseline_cfg.get("dx", 0.5)),
            dy=float(baseline_cfg.get("dy", 0.9)),
            offset=float(baseline_cfg.get("offset", 0.3)),
            basis=basis,
            search_pad=int(baseline_cfg.get("search_pad", 4)),
            center_steps=int(baseline_cfg.get("center_steps", 4)),
            selection_mode=str(baseline_cfg.get("selection_mode", "square_search")),
            lattice_angle_deg=float(baseline_cfg.get("lattice_angle_deg", 0.0)),
            scale_factor=float(baseline_cfg.get("scale_factor", 1e18)),
            bounds=float(baseline_cfg.get("bounds", 100.0)),
            global_squeeze=bool(baseline_cfg.get("global_squeeze", False)),
            squeeze_factor=float(baseline_cfg.get("squeeze_factor", 0.985)),
            squeeze_steps=int(baseline_cfg.get("squeeze_steps", 20)),
            squeeze_iters=int(baseline_cfg.get("squeeze_iters", 8)),
        )
        solver = PeriodicSolver(periodic_cfg)
        print(
            "baseline periodic: "
            f"dx={periodic_cfg.dx} dy={periodic_cfg.dy} "
            f"offset={periodic_cfg.offset} basis={len(periodic_cfg.basis)}",
            flush=True,
        )
        groups = solver.solve(n_max=n_max, seed=seed)
    elif mode == "independent":
        independent_cfg = IndependentConfig(
            init_method=str(baseline_cfg.get("init_method", "hex")),
            init_restarts=int(baseline_cfg.get("init_restarts", 6)),
            spacing_scale=float(baseline_cfg.get("spacing_scale", 1.08)),
            jitter=float(baseline_cfg.get("jitter", 0.04)),
            squeeze_factor=float(baseline_cfg.get("squeeze_factor", 0.985)),
            squeeze_steps=int(baseline_cfg.get("squeeze_steps", 25)),
            angle_mode=str(baseline_cfg.get("angle_mode", "random")),
            scale_factor=float(baseline_cfg.get("scale_factor", 1e18)),
            bounds=float(baseline_cfg.get("bounds", 100.0)),
            fallback_attempts=int(baseline_cfg.get("fallback_attempts", 2)),
        )
        solver = IndependentSolver(independent_cfg)
        print(
            "baseline independent: "
            f"init_method={independent_cfg.init_method} "
            f"restarts={independent_cfg.init_restarts} "
            f"spacing_scale={independent_cfg.spacing_scale} "
            f"squeeze_steps={independent_cfg.squeeze_steps}",
            flush=True,
        )
        groups = solver.solve(n_max=n_max, seed=seed)
    else:
        solver = GreedyIncrementalSolver(
            attempts_per_tree=int(baseline_cfg.get("attempts_per_tree", 10)),
            start_radius=float(baseline_cfg.get("start_radius", 20.0)),
            step_in=float(baseline_cfg.get("step_in", 0.5)),
            step_out=float(baseline_cfg.get("step_out", 0.05)),
            scale_factor=float(baseline_cfg.get("scale_factor", 1e18)),
        )
        print(
            "baseline incremental: "
            f"attempts_per_tree={solver.attempts_per_tree} "
            f"start_radius={solver.start_radius} "
            f"step_in={solver.step_in} "
            f"step_out={solver.step_out}",
            flush=True,
        )
        groups = solver.solve(n_max=n_max, seed=seed)

    baseline_time = time.perf_counter() - start_time
    submission = build_submission(groups, decimals=decimals)
    total_score, per_group = score_detailed(submission)
    print(f"baseline score={total_score:.6f} time_s={baseline_time:.1f}", flush=True)

    df_groups = per_group_dataframe(per_group)
    df_groups.to_csv(output_dir / "per_n.csv", index=False)
    write_submission_csv(submission, output_dir / "submission.csv")
    (output_dir / "groups_baseline.json").write_text(
        json.dumps(_serialize_groups(groups), indent=2)
    )

    summary = {
        "stage": "baseline",
        "total_score": total_score,
        "config": config,
    }
    (output_dir / "summary_baseline.json").write_text(json.dumps(summary, indent=2))

    refine_cfg = config.get("refine", {})
    if not refine_cfg.get("enabled", True):
        total_time = time.perf_counter() - start_time
        print(f"run_experiment done (no refine) time_s={total_time:.1f}", flush=True)
        return

    target_ns = _parse_target_ns(refine_cfg.get("target_ns"))
    if target_ns:
        targets = [n for n in target_ns if n in per_group]
        print(
            f"refine start: targets={len(targets)} explicit={targets}",
            flush=True,
        )
    else:
        target_k = int(refine_cfg.get("top_k", 40))
        targets = top_groups(per_group, target_k)
        print(f"refine start: targets={len(targets)} top_k={target_k}", flush=True)

    ls_config = LocalSearchConfig(
        steps=int(refine_cfg.get("steps", 5000)),
        restarts=int(refine_cfg.get("restarts", 4)),
        move_radius=float(refine_cfg.get("move_radius", 0.08)),
        angle_radius=float(refine_cfg.get("angle_radius", 12.0)),
        temp_start=float(refine_cfg.get("temp_start", 1.0)),
        temp_end=float(refine_cfg.get("temp_end", 0.05)),
        scale_factor=float(refine_cfg.get("scale_factor", 1e18)),
        bounds=float(refine_cfg.get("bounds", 100.0)),
        log_every_steps=int(refine_cfg.get("log_every_steps", 0)),
    )

    max_workers = int(refine_cfg.get("max_workers", 1))
    print(
        "refine config: "
        f"steps={ls_config.steps} "
        f"restarts={ls_config.restarts} "
        f"max_workers={max_workers} "
        f"log_every_steps={ls_config.log_every_steps}",
        flush=True,
    )
    tasks = [(n, groups[n], asdict(ls_config), seed + n) for n in targets]

    results: List[Tuple[int, List[TreePlacement], float]] = []
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_refine_group, tasks):
                results.append(result)
    else:
        for task in tasks:
            results.append(_refine_group(task))

    for n, refined, _ in results:
        groups[n] = refined

    submission_refined = build_submission(groups, decimals=decimals)
    total_score_refined, per_group_refined = score_detailed(submission_refined)
    df_groups_refined = per_group_dataframe(per_group_refined)

    df_groups_refined.to_csv(output_dir / "per_n_refined.csv", index=False)
    write_submission_csv(submission_refined, output_dir / "submission_refined.csv")
    (output_dir / "groups_refined.json").write_text(
        json.dumps(_serialize_groups(groups), indent=2)
    )

    summary_refined = {
        "stage": "refined",
        "total_score": total_score_refined,
        "baseline_score": total_score,
        "targets": targets,
        "config": config,
    }
    (output_dir / "summary_refined.json").write_text(json.dumps(summary_refined, indent=2))
    total_time = time.perf_counter() - start_time
    print(
        f"refine score={total_score_refined:.6f} total_time_s={total_time:.1f}",
        flush=True,
    )


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)
