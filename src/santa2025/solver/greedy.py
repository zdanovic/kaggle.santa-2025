from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from shapely.strtree import STRtree

from santa2025.geometry import build_tree_polygon, polygons_bounds
from santa2025.io import TreePlacement


@dataclass
class PlacedTree:
    x: float
    y: float
    deg: float
    polygon: object


def _weighted_angle(rng: random.Random) -> float:
    while True:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        if rng.uniform(0.0, 1.0) < abs(math.sin(2.0 * angle)):
            return angle


def _random_rotation(rng: random.Random) -> float:
    return rng.uniform(0.0, 360.0)


def _collides(candidate, polygons, tree_index: STRtree) -> bool:
    indices = tree_index.query(candidate)
    for idx in indices:
        if candidate.intersects(polygons[idx]) and not candidate.touches(polygons[idx]):
            return True
    return False


def _bounding_square_side(polygons, scale_factor: float) -> float:
    minx, miny, maxx, maxy = polygons_bounds(polygons)
    return max(maxx - minx, maxy - miny) / scale_factor


class GreedyIncrementalSolver:
    def __init__(
        self,
        attempts_per_tree: int = 10,
        start_radius: float = 20.0,
        step_in: float = 0.5,
        step_out: float = 0.05,
        scale_factor: float = 1e18,
    ) -> None:
        self.attempts_per_tree = attempts_per_tree
        self.start_radius = start_radius
        self.step_in = step_in
        self.step_out = step_out
        self.scale_factor = scale_factor

    def _place_one(
        self,
        rng: random.Random,
        placed: List[PlacedTree],
        angle_deg: float,
    ) -> PlacedTree:
        if not placed:
            poly = build_tree_polygon(0.0, 0.0, angle_deg, self.scale_factor)
            return PlacedTree(x=0.0, y=0.0, deg=angle_deg, polygon=poly)

        polygons = [p.polygon for p in placed]
        tree_index = STRtree(polygons)

        best_x = 0.0
        best_y = 0.0
        best_radius = float("inf")

        for _ in range(self.attempts_per_tree):
            vec_angle = _weighted_angle(rng)
            vx = math.cos(vec_angle)
            vy = math.sin(vec_angle)

            radius = self.start_radius
            collision_found = False

            while radius >= 0.0:
                px = radius * vx
                py = radius * vy
                candidate = build_tree_polygon(px, py, angle_deg, self.scale_factor)

                if _collides(candidate, polygons, tree_index):
                    collision_found = True
                    break
                radius -= self.step_in

            if collision_found:
                while True:
                    radius += self.step_out
                    px = radius * vx
                    py = radius * vy
                    candidate = build_tree_polygon(px, py, angle_deg, self.scale_factor)
                    if not _collides(candidate, polygons, tree_index):
                        break
            else:
                radius = 0.0
                px = 0.0
                py = 0.0
                candidate = build_tree_polygon(px, py, angle_deg, self.scale_factor)

            if radius < best_radius:
                best_radius = radius
                best_x = px
                best_y = py

        poly = build_tree_polygon(best_x, best_y, angle_deg, self.scale_factor)
        return PlacedTree(x=best_x, y=best_y, deg=angle_deg, polygon=poly)

    def solve(
        self,
        n_max: int = 200,
        seed: int = 1337,
        existing: Optional[Dict[int, List[TreePlacement]]] = None,
    ) -> Dict[int, List[TreePlacement]]:
        rng = random.Random(seed)

        groups: Dict[int, List[TreePlacement]] = {}
        placed: List[PlacedTree] = []

        if existing:
            for n in sorted(existing.keys()):
                groups[n] = existing[n]
            last_n = max(existing.keys())
            placed = []
            for p in existing[last_n]:
                poly = build_tree_polygon(p.x, p.y, p.deg, self.scale_factor)
                placed.append(PlacedTree(x=p.x, y=p.y, deg=p.deg, polygon=poly))

        start_n = max(groups.keys()) + 1 if groups else 1
        for n in range(start_n, n_max + 1):
            angle_deg = _random_rotation(rng)
            new_tree = self._place_one(rng, placed, angle_deg)
            placed.append(new_tree)
            groups[n] = [TreePlacement(x=p.x, y=p.y, deg=p.deg) for p in placed]

        return groups

    def score_side_lengths(self, groups: Dict[int, List[TreePlacement]]) -> Dict[int, float]:
        side_lengths: Dict[int, float] = {}
        for n, placements in groups.items():
            polygons = [
                build_tree_polygon(p.x, p.y, p.deg, self.scale_factor) for p in placements
            ]
            side_lengths[n] = _bounding_square_side(polygons, self.scale_factor)
        return side_lengths
