from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from shapely.strtree import STRtree

from santa2025.geometry import build_tree_polygon
from santa2025.io import TreePlacement


def _overlaps(poly_a, poly_b) -> bool:
    return poly_a.intersects(poly_b) and not poly_a.touches(poly_b)


@dataclass(frozen=True)
class PeriodicBasis:
    x: float
    y: float
    deg: float


@dataclass
class PeriodicConfig:
    dx: float = 0.5
    dy: float = 0.9
    offset: float = 0.3
    basis: List[PeriodicBasis] = field(default_factory=list)
    search_pad: int = 4
    center_steps: int = 4
    selection_mode: str = "square_search"
    lattice_angle_deg: float = 0.0
    scale_factor: float = 1e18
    collision_scale: float = 1e6
    bounds: float = 100.0
    global_squeeze: bool = False
    squeeze_factor: float = 0.985
    squeeze_steps: int = 20
    squeeze_iters: int = 8


class PeriodicSolver:
    def __init__(self, config: PeriodicConfig) -> None:
        if not config.basis:
            raise ValueError("PeriodicConfig requires a non-empty basis.")
        self.config = config
        self._angle_bounds = self._build_angle_bounds()

    def _build_angle_bounds(self) -> dict[float, Tuple[float, float, float, float]]:
        bounds: dict[float, Tuple[float, float, float, float]] = {}
        for b in self.config.basis:
            angle = float(b.deg) % 360.0
            if angle in bounds:
                continue
            poly = build_tree_polygon(0.0, 0.0, angle, 1.0)
            bounds[angle] = poly.bounds
        return bounds

    def _bounds_for_angle(self, angle: float) -> Tuple[float, float, float, float]:
        angle = float(angle) % 360.0
        cached = self._angle_bounds.get(angle)
        if cached is not None:
            return cached
        poly = build_tree_polygon(0.0, 0.0, angle, 1.0)
        bounds = poly.bounds
        self._angle_bounds[angle] = bounds
        return bounds

    def _score_and_bounds(
        self, placements: List[TreePlacement]
    ) -> Tuple[float, Tuple[float, float, float, float]]:
        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for p in placements:
            bx0, by0, bx1, by1 = self._bounds_for_angle(p.deg)
            minx = min(minx, p.x + bx0)
            miny = min(miny, p.y + by0)
            maxx = max(maxx, p.x + bx1)
            maxy = max(maxy, p.y + by1)
        side = max(maxx - minx, maxy - miny)
        score = (side * side) / len(placements)
        return score, (minx, miny, maxx, maxy)

    def _center(
        self, placements: List[TreePlacement], bounds: Tuple[float, float, float, float]
    ) -> List[TreePlacement]:
        minx, miny, maxx, maxy = bounds
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        return [TreePlacement(x=p.x - cx, y=p.y - cy, deg=p.deg) for p in placements]

    def _tile_points(self, n: int) -> List[TreePlacement]:
        dx = self.config.dx
        dy = self.config.dy
        offset = self.config.offset
        basis = self.config.basis
        angle_deg = float(self.config.lattice_angle_deg)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        per_cell = len(basis)
        side_cells = int(math.ceil(math.sqrt(max(1, n / per_cell)))) + self.config.search_pad

        placements: List[TreePlacement] = []
        for j in range(-side_cells, side_cells + 1):
            for i in range(-side_cells, side_cells + 1):
                tx = i * dx + j * offset
                ty = j * dy
                for b in basis:
                    x = b.x + tx
                    y = b.y + ty
                    if angle_deg:
                        xr = x * cos_a - y * sin_a
                        yr = x * sin_a + y * cos_a
                        x, y = xr, yr
                    placements.append(TreePlacement(x=x, y=y, deg=b.deg))
        return placements

    def _square_search(self, placements: List[TreePlacement], n: int) -> List[TreePlacement]:
        steps = max(1, int(self.config.center_steps))
        dx = self.config.dx
        dy = self.config.dy
        offsets_x = [dx * (i / steps) for i in range(steps)]
        offsets_y = [dy * (i / steps) for i in range(steps)]

        best_layout: List[TreePlacement] = []
        best_score = float("inf")

        for cx in offsets_x:
            for cy in offsets_y:
                ranked: List[Tuple[float, TreePlacement]] = []
                for p in placements:
                    bx0, by0, bx1, by1 = self._bounds_for_angle(p.deg)
                    hx = max(abs(p.x + bx0 - cx), abs(p.x + bx1 - cx))
                    hy = max(abs(p.y + by0 - cy), abs(p.y + by1 - cy))
                    ranked.append((max(hx, hy), p))
                ranked.sort(key=lambda t: t[0])
                chosen = [p for _, p in ranked[:n]]
                score, _ = self._score_and_bounds(chosen)
                if score < best_score:
                    best_score = score
                    best_layout = chosen

        return best_layout

    def _scale(self, placements: List[TreePlacement], factor: float) -> List[TreePlacement]:
        return [TreePlacement(x=p.x * factor, y=p.y * factor, deg=p.deg) for p in placements]

    def _has_collision(self, placements: List[TreePlacement]) -> bool:
        polys = [
            build_tree_polygon(p.x, p.y, p.deg, self.config.collision_scale)
            for p in placements
        ]
        tree = STRtree(polys)
        for i, poly in enumerate(polys):
            for idx in tree.query(poly):
                if idx == i:
                    continue
                if _overlaps(poly, polys[idx]):
                    return True
        return False

    def _global_squeeze(self, placements: List[TreePlacement]) -> List[TreePlacement]:
        if not self.config.global_squeeze:
            return placements
        factor = self.config.squeeze_factor
        current_scale = 1.0
        current = placements
        low = None
        high = 1.0

        for _ in range(self.config.squeeze_steps):
            next_scale = current_scale * factor
            scaled = self._scale(placements, next_scale)
            if self._has_collision(scaled):
                low = next_scale
                high = current_scale
                break
            current_scale = next_scale
            current = scaled

        if low is None:
            return current

        for _ in range(self.config.squeeze_iters):
            mid = (low + high) / 2.0
            scaled = self._scale(placements, mid)
            if self._has_collision(scaled):
                low = mid
            else:
                high = mid
        return self._scale(placements, high)

    def _best_layout(self, n: int) -> List[TreePlacement]:
        placements = self._tile_points(n)
        if self.config.selection_mode == "square_search":
            best = self._square_search(placements, n)
        else:
            def key(p: TreePlacement) -> Tuple[float, float]:
                return (max(abs(p.x), abs(p.y)), abs(p.x) + abs(p.y))

            best = sorted(placements, key=key)[:n]

        best = self._global_squeeze(best)
        _, bounds = self._score_and_bounds(best)
        return self._center(best, bounds)

    def solve(self, n_max: int, seed: int = 0) -> dict[int, List[TreePlacement]]:
        groups: dict[int, List[TreePlacement]] = {}
        for n in range(1, n_max + 1):
            groups[n] = self._best_layout(n)
        return groups

    def score_total(self, n_max: int) -> float:
        total = 0.0
        for n in range(1, n_max + 1):
            layout = self._best_layout(n)
            score, _ = self._score_and_bounds(layout)
            total += score
        return total
