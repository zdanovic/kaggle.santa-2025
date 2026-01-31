from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from santa2025.geometry import build_tree_polygon, polygons_bounds
from santa2025.io import TreePlacement


@dataclass
class TreeState:
    x: float
    y: float
    deg: float
    polygon: Polygon


@dataclass
class LocalSearchConfig:
    steps: int = 5000
    restarts: int = 4
    move_radius: float = 0.08
    angle_radius: float = 12.0
    swap_prob: float = 0.0
    scale_prob: float = 0.0
    scale_radius: float = 0.02
    temp_start: float = 1.0
    temp_end: float = 0.05
    scale_factor: float = 1e18
    bounds: float = 100.0
    log_every_steps: int = 0
    gravity_weight: float = 0.0  # Compactness term: 1e-4 for small n, 0 to disable


def _side_length(states: List[TreeState], scale_factor: float) -> float:
    bounds = polygons_bounds([s.polygon for s in states])
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / scale_factor


def _group_score(side_length: float, n: int) -> float:
    return (side_length * side_length) / n


def _dist_sum(states: List[TreeState]) -> float:
    """Sum of squared distances from origin (for gravity term)."""
    return sum(s.x * s.x + s.y * s.y for s in states)


def _gravity_energy(side_length: float, n: int, dist_sum: float, gravity_weight: float) -> float:
    """Energy with optional gravity compactness term."""
    base_score = (side_length * side_length) / n
    if gravity_weight <= 0 or n == 0:
        return base_score
    # Effective gravity weight is reduced for large n
    effective_gw = gravity_weight if n <= 50 else gravity_weight * 0.01
    normalized_dist = dist_sum / n
    return base_score + effective_gw * normalized_dist


def _collides(candidate: Polygon, states: List[TreeState], skip_index: int) -> bool:
    cb = candidate.bounds
    for i, state in enumerate(states):
        if i == skip_index:
            continue
        sb = state.polygon.bounds
        if cb[2] < sb[0] or cb[0] > sb[2] or cb[3] < sb[1] or cb[1] > sb[3]:
            continue
        if candidate.intersects(state.polygon) and not candidate.touches(state.polygon):
            return True
    return False


def _has_collision(polygons: List[Polygon]) -> bool:
    tree = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for idx in tree.query(poly):
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


class LocalSearchRefiner:
    def __init__(self, config: LocalSearchConfig) -> None:
        self.config = config

    def _init_states(self, placements: List[TreePlacement]) -> List[TreeState]:
        states: List[TreeState] = []
        for p in placements:
            poly = build_tree_polygon(p.x, p.y, p.deg, self.config.scale_factor)
            states.append(TreeState(x=p.x, y=p.y, deg=p.deg, polygon=poly))
        return states

    def refine(self, placements: List[TreePlacement], seed: int) -> Tuple[List[TreePlacement], float]:
        rng = random.Random(seed)
        best_placements = placements
        best_score = float("inf")
        start_time = time.perf_counter()

        for restart in range(self.config.restarts):
            states = self._init_states(placements)
            current_side = _side_length(states, self.config.scale_factor)
            n = len(states)
            gw = self.config.gravity_weight
            if gw > 0:
                current_dist = _dist_sum(states)
                current_score = _gravity_energy(current_side, n, current_dist, gw)
            else:
                current_dist = 0.0
                current_score = _group_score(current_side, n)
            print(
                f"refine restart {restart + 1}/{self.config.restarts} n={len(states)} "
                f"score={current_score:.6f}",
                flush=True,
            )

            if current_score < best_score:
                best_score = current_score
                best_placements = [
                    TreePlacement(x=s.x, y=s.y, deg=s.deg) for s in states
                ]

            for step in range(self.config.steps):
                t = step / max(1, self.config.steps - 1)
                temp = self.config.temp_start * (1.0 - t) + self.config.temp_end * t

                move_pick = rng.random()
                if self.config.scale_prob > 0.0 and move_pick < self.config.scale_prob:
                    bounds = polygons_bounds([s.polygon for s in states])
                    cx = (bounds[0] + bounds[2]) / 2.0 / self.config.scale_factor
                    cy = (bounds[1] + bounds[3]) / 2.0 / self.config.scale_factor
                    scale = 1.0 - rng.uniform(0.0, self.config.scale_radius) * temp
                    candidate_states: List[TreeState] = []
                    for s in states:
                        nx = _clamp(cx + (s.x - cx) * scale, -self.config.bounds, self.config.bounds)
                        ny = _clamp(cy + (s.y - cy) * scale, -self.config.bounds, self.config.bounds)
                        candidate_states.append(
                            TreeState(
                                x=nx,
                                y=ny,
                                deg=s.deg,
                                polygon=build_tree_polygon(nx, ny, s.deg, self.config.scale_factor),
                            )
                        )
                    if _has_collision([s.polygon for s in candidate_states]):
                        continue
                    new_side = _side_length(candidate_states, self.config.scale_factor)
                    if gw > 0:
                        new_dist = _dist_sum(candidate_states)
                        new_score = _gravity_energy(new_side, n, new_dist, gw)
                    else:
                        new_score = _group_score(new_side, n)
                    accept = new_score <= current_score
                    if not accept:
                        delta = current_score - new_score
                        accept = rng.random() < math.exp(delta / max(temp, 1e-6))
                    if accept:
                        states = candidate_states
                        current_score = new_score
                        current_side = new_side
                        if gw > 0:
                            current_dist = new_dist
                        real_score = _group_score(current_side, n)
                        if real_score < best_score:
                            best_score = real_score
                            best_placements = [
                                TreePlacement(x=s.x, y=s.y, deg=s.deg) for s in states
                            ]
                    continue

                if (
                    self.config.swap_prob > 0.0
                    and len(states) > 1
                    and move_pick < self.config.scale_prob + self.config.swap_prob
                ):
                    i, j = rng.sample(range(len(states)), 2)
                    si = states[i]
                    sj = states[j]
                    cand_i = build_tree_polygon(sj.x, sj.y, si.deg, self.config.scale_factor)
                    cand_j = build_tree_polygon(si.x, si.y, sj.deg, self.config.scale_factor)
                    if _collides(cand_i, states, i) or _collides(cand_j, states, j):
                        continue
                    old_i = states[i]
                    old_j = states[j]
                    states[i] = TreeState(x=sj.x, y=sj.y, deg=si.deg, polygon=cand_i)
                    states[j] = TreeState(x=si.x, y=si.y, deg=sj.deg, polygon=cand_j)
                    new_side = _side_length(states, self.config.scale_factor)
                    if gw > 0:
                        new_dist = _dist_sum(states)
                        new_score = _gravity_energy(new_side, n, new_dist, gw)
                    else:
                        new_score = _group_score(new_side, n)
                    accept = new_score <= current_score
                    if not accept:
                        delta = current_score - new_score
                        accept = rng.random() < math.exp(delta / max(temp, 1e-6))
                    if accept:
                        current_score = new_score
                        current_side = new_side
                        if gw > 0:
                            current_dist = new_dist
                        real_score = _group_score(current_side, n)
                        if real_score < best_score:
                            best_score = real_score
                            best_placements = [
                                TreePlacement(x=s.x, y=s.y, deg=s.deg) for s in states
                            ]
                    else:
                        states[i] = old_i
                        states[j] = old_j
                    continue

                idx = rng.randrange(len(states))
                state = states[idx]

                move_scale = self.config.move_radius * temp
                angle_scale = self.config.angle_radius * temp

                dx = rng.uniform(-move_scale, move_scale)
                dy = rng.uniform(-move_scale, move_scale)
                ddeg = rng.uniform(-angle_scale, angle_scale)

                nx = _clamp(state.x + dx, -self.config.bounds, self.config.bounds)
                ny = _clamp(state.y + dy, -self.config.bounds, self.config.bounds)
                ndeg = (state.deg + ddeg) % 360.0

                candidate = build_tree_polygon(nx, ny, ndeg, self.config.scale_factor)
                if _collides(candidate, states, idx):
                    continue

                old_state = states[idx]
                states[idx] = TreeState(x=nx, y=ny, deg=ndeg, polygon=candidate)
                new_side = _side_length(states, self.config.scale_factor)
                if gw > 0:
                    new_dist = _dist_sum(states)
                    new_score = _gravity_energy(new_side, n, new_dist, gw)
                else:
                    new_score = _group_score(new_side, n)

                accept = new_score <= current_score
                if not accept:
                    delta = current_score - new_score
                    accept = rng.random() < math.exp(delta / max(temp, 1e-6))

                if accept:
                    current_score = new_score
                    current_side = new_side
                    if gw > 0:
                        current_dist = new_dist
                    real_score = _group_score(current_side, n)
                    if real_score < best_score:
                        best_score = real_score
                        best_placements = [
                            TreePlacement(x=s.x, y=s.y, deg=s.deg) for s in states
                        ]
                else:
                    states[idx] = old_state

                if self.config.log_every_steps and step > 0:
                    if step % self.config.log_every_steps == 0:
                        elapsed = time.perf_counter() - start_time
                        print(
                            f"refine step={step} restart={restart + 1} "
                            f"current={current_score:.6f} best={best_score:.6f} "
                            f"time_s={elapsed:.1f}",
                            flush=True,
                        )

            elapsed = time.perf_counter() - start_time
            print(
                f"refine restart {restart + 1} done best={best_score:.6f} "
                f"time_s={elapsed:.1f}",
                flush=True,
            )

        return best_placements, best_score
