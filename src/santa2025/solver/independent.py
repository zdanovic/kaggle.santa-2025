from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from shapely.strtree import STRtree

from santa2025.geometry import build_tree_polygon, polygons_bounds, tree_max_radius
from santa2025.io import TreePlacement
from santa2025.solver.greedy import GreedyIncrementalSolver


@dataclass
class IndependentConfig:
    init_method: str = "hex"
    init_restarts: int = 6
    spacing_scale: float = 1.08
    jitter: float = 0.04
    squeeze_factor: float = 0.985
    squeeze_steps: int = 25
    angle_mode: str = "random"
    scale_factor: float = 1e18
    bounds: float = 100.0
    fallback_attempts: int = 2


def _hex_points(n: int, spacing: float) -> List[Tuple[float, float]]:
    side = int(math.ceil(math.sqrt(n))) + 3
    dx = spacing
    dy = spacing * math.sqrt(3.0) / 2.0

    points: List[Tuple[float, float]] = []
    for row in range(-side, side + 1):
        y = row * dy
        offset = (row & 1) * (dx / 2.0)
        for col in range(-side, side + 1):
            x = col * dx + offset
            points.append((x, y))

    points.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return points[:n]


def _build_polygons(placements: List[TreePlacement], scale_factor: float):
    return [build_tree_polygon(p.x, p.y, p.deg, scale_factor) for p in placements]


def _has_collision(polygons) -> bool:
    tree = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for idx in tree.query(poly):
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False


def _bounding_side(polygons, scale_factor: float) -> float:
    minx, miny, maxx, maxy = polygons_bounds(polygons)
    return max(maxx - minx, maxy - miny) / scale_factor


def _group_score(polygons, scale_factor: float) -> float:
    side = _bounding_side(polygons, scale_factor)
    return (side * side) / len(polygons)


def _apply_squeeze(
    placements: List[TreePlacement],
    factor: float,
    steps: int,
    scale_factor: float,
) -> List[TreePlacement]:
    current = placements
    for _ in range(steps):
        scaled = [
            TreePlacement(x=p.x * factor, y=p.y * factor, deg=p.deg) for p in current
        ]
        polygons = _build_polygons(scaled, scale_factor)
        if _has_collision(polygons):
            break
        current = scaled
    return current


class IndependentSolver:
    def __init__(self, config: IndependentConfig) -> None:
        self.config = config

    def _angles(self, rng: random.Random, n: int) -> List[float]:
        if self.config.angle_mode == "alternating":
            return [0.0 if (i % 2 == 0) else 180.0 for i in range(n)]
        return [rng.uniform(0.0, 360.0) for _ in range(n)]

    def _build_initial(self, rng: random.Random, n: int) -> List[TreePlacement]:
        base_spacing = 2.0 * tree_max_radius() * self.config.spacing_scale
        points = _hex_points(n, base_spacing)

        jitter = self.config.jitter
        angles = self._angles(rng, n)
        placements = []
        for (x, y), angle in zip(points, angles):
            jx = rng.uniform(-jitter, jitter)
            jy = rng.uniform(-jitter, jitter)
            placements.append(TreePlacement(x=x + jx, y=y + jy, deg=angle))

        placements = _apply_squeeze(
            placements,
            factor=self.config.squeeze_factor,
            steps=self.config.squeeze_steps,
            scale_factor=self.config.scale_factor,
        )
        return placements

    def _build_candidate(self, rng: random.Random, n: int) -> List[TreePlacement] | None:
        placements = self._build_initial(rng, n)
        polygons = _build_polygons(placements, self.config.scale_factor)
        if _has_collision(polygons):
            return None
        return placements

    def _fallback(self, seed: int, n: int) -> List[TreePlacement]:
        solver = GreedyIncrementalSolver(scale_factor=self.config.scale_factor)
        groups = solver.solve(n_max=n, seed=seed)
        return groups[n]

    def solve(self, n_max: int, seed: int) -> Dict[int, List[TreePlacement]]:
        rng = random.Random(seed)
        groups: Dict[int, List[TreePlacement]] = {}

        for n in range(1, n_max + 1):
            best = None
            best_score = float("inf")
            for _ in range(self.config.init_restarts):
                candidate = self._build_candidate(rng, n)
                if candidate is None:
                    continue
                polygons = _build_polygons(candidate, self.config.scale_factor)
                score = _group_score(polygons, self.config.scale_factor)
                if score < best_score:
                    best_score = score
                    best = candidate

            if best is None:
                best = self._fallback(seed + n, n)

            groups[n] = best

        return groups
