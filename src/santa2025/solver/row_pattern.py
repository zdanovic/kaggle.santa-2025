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
class RowPatternSpec:
    angles: List[float]
    offsets: List[float]  # ratios of dx in [0, 1]
    dx: float
    dy: float


@dataclass
class RowPatternConfig:
    period: int = 3
    grid_size: int = 4
    selection_mode: str = "square_search"
    center_steps: int = 6
    search_pad: int = 3
    scale_factor: float = 1e18
    collision_scale: float = 1e6
    global_squeeze: bool = False
    squeeze_factor: float = 0.985
    squeeze_steps: int = 20
    squeeze_iters: int = 8


class RowPatternSolver:
    def __init__(self, config: RowPatternConfig) -> None:
        self.config = config
        self._angle_bounds = {}

    def _bounds_for_angle(self, angle: float) -> Tuple[float, float, float, float]:
        angle = float(angle) % 360.0
        cached = self._angle_bounds.get(angle)
        if cached is not None:
            return cached
        poly = build_tree_polygon(0.0, 0.0, angle, 1.0)
        bounds = poly.bounds
        self._angle_bounds[angle] = bounds
        return bounds

    def _row_collision(self, angle: float, dx: float) -> bool:
        poly_a = build_tree_polygon(0.0, 0.0, angle, self.config.collision_scale)
        poly_b = build_tree_polygon(dx, 0.0, angle, self.config.collision_scale)
        return _overlaps(poly_a, poly_b)

    def _pair_collision(
        self,
        dx: float,
        dy: float,
        offset_a: float,
        angle_a: float,
        offset_b: float,
        angle_b: float,
    ) -> bool:
        polys = []
        cols = self.config.grid_size
        for r in range(2):
            offset = offset_a if r == 0 else offset_b
            angle = angle_a if r == 0 else angle_b
            y = r * dy
            for c in range(cols):
                x = c * dx + offset
                polys.append(build_tree_polygon(x, y, angle, self.config.collision_scale))
        tree = STRtree(polys)
        for i, poly in enumerate(polys):
            for idx in tree.query(poly):
                if idx == i:
                    continue
                if _overlaps(poly, polys[idx]):
                    return True
        return False

    def _grid_collision(self, spec: RowPatternSpec) -> bool:
        polys = []
        rows = self.config.grid_size
        cols = self.config.grid_size
        period = len(spec.angles)
        for r in range(rows):
            row_idx = r % period
            angle = spec.angles[row_idx]
            offset = spec.offsets[row_idx] * spec.dx
            y = r * spec.dy
            for c in range(cols):
                x = c * spec.dx + offset
                polys.append(build_tree_polygon(x, y, angle, self.config.collision_scale))
        tree = STRtree(polys)
        for i, poly in enumerate(polys):
            for idx in tree.query(poly):
                if idx == i:
                    continue
                if _overlaps(poly, polys[idx]):
                    return True
        return False

    def min_dx_for_angle(self, angle: float, dx_min: float, dx_max: float) -> float | None:
        if self._row_collision(angle, dx_max):
            return None
        lo = dx_min
        hi = dx_max
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if self._row_collision(angle, mid):
                lo = mid
            else:
                hi = mid
        return hi

    def min_dy_for_pair(
        self,
        dx: float,
        offset_a: float,
        angle_a: float,
        offset_b: float,
        angle_b: float,
        dy_min: float,
        dy_max: float,
    ) -> float | None:
        if self._pair_collision(dx, dy_max, offset_a, angle_a, offset_b, angle_b):
            return None
        lo = dy_min
        hi = dy_max
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if self._pair_collision(dx, mid, offset_a, angle_a, offset_b, angle_b):
                lo = mid
            else:
                hi = mid
        return hi

    def _tile_points(self, n: int, spec: RowPatternSpec) -> List[TreePlacement]:
        dx = spec.dx
        dy = spec.dy
        period = len(spec.angles)
        side_cells = int(math.ceil(math.sqrt(max(1, n)))) + self.config.search_pad
        placements: List[TreePlacement] = []
        for j in range(-side_cells, side_cells + 1):
            angle = spec.angles[j % period]
            offset = spec.offsets[j % period] * dx
            for i in range(-side_cells, side_cells + 1):
                x = i * dx + offset
                y = j * dy
                placements.append(TreePlacement(x=x, y=y, deg=angle))
        return placements

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

    def _square_search(self, placements: List[TreePlacement], n: int, dx: float, dy: float) -> List[TreePlacement]:
        steps = max(1, int(self.config.center_steps))
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

    def best_layout(self, n: int, spec: RowPatternSpec) -> List[TreePlacement]:
        placements = self._tile_points(n, spec)
        if self.config.selection_mode == "square_search":
            best = self._square_search(placements, n, spec.dx, spec.dy)
        else:
            def key(p: TreePlacement) -> Tuple[float, float]:
                return (max(abs(p.x), abs(p.y)), abs(p.x) + abs(p.y))

            best = sorted(placements, key=key)[:n]

        best = self._global_squeeze(best)
        _, bounds = self._score_and_bounds(best)
        return self._center(best, bounds)

    def solve(self, n_max: int, spec: RowPatternSpec) -> dict[int, List[TreePlacement]]:
        groups: dict[int, List[TreePlacement]] = {}
        for n in range(1, n_max + 1):
            groups[n] = self.best_layout(n, spec)
        return groups
