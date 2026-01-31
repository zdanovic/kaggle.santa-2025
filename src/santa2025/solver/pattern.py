from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from shapely.strtree import STRtree

from santa2025.geometry import build_tree_polygon
from santa2025.io import TreePlacement


def _overlaps(poly_a, poly_b) -> bool:
    return poly_a.intersects(poly_b) and not poly_a.touches(poly_b)


@dataclass(frozen=True)
class PatternSpec:
    angle_a: float
    angle_b: float
    offset_ratio: float
    dx: float
    dy: float


@dataclass
class PatternConfig:
    angle_pairs: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 180.0)]
    )
    offset_ratios: List[float] = field(default_factory=lambda: [0.6, 0.7])
    dx_min: float = 0.1
    dx_max: float = 2.0
    dy_min: float = 0.05
    dy_max: float = 2.0
    grid_size: int = 4
    spacing_margin: float = 1.001
    rows_pad: int = 2
    scale_factor: float = 1e18
    collision_scale: float = 1e6
    bounds: float = 100.0
    global_squeeze: bool = False
    squeeze_factor: float = 0.985
    squeeze_steps: int = 20
    squeeze_iters: int = 8
    jitter: float = 0.0
    angle_jitter: float = 0.0
    selection_mode: str = "row_major"
    search_pad: int = 2
    center_steps: int = 4


class PatternSolver:
    def __init__(self, config: PatternConfig) -> None:
        self.config = config
        self._angle_bounds = self._build_angle_bounds()
        self._patterns = self._build_patterns()

    def _build_angle_bounds(self) -> Dict[float, Tuple[float, float, float, float]]:
        angles = set()
        for a, b in self.config.angle_pairs:
            angles.add(float(a))
            angles.add(float(b))
        bounds: Dict[float, Tuple[float, float, float, float]] = {}
        for angle in angles:
            poly = build_tree_polygon(0.0, 0.0, angle, 1.0)
            bounds[angle] = poly.bounds
        return bounds

    def _bounds_for_angle(self, angle: float) -> Tuple[float, float, float, float]:
        cached = self._angle_bounds.get(angle)
        if cached is not None:
            return cached
        poly = build_tree_polygon(0.0, 0.0, angle, 1.0)
        bounds = poly.bounds
        self._angle_bounds[angle] = bounds
        return bounds

    def _min_dx(self, angle_a: float, angle_b: float) -> float | None:
        lo = self.config.dx_min
        hi = self.config.dx_max
        base = build_tree_polygon(0.0, 0.0, angle_a, self.config.collision_scale)
        if _overlaps(
            base,
            build_tree_polygon(hi, 0.0, angle_b, self.config.collision_scale),
        ):
            return None
        for _ in range(40):
            mid = (lo + hi) / 2.0
            other = build_tree_polygon(mid, 0.0, angle_b, self.config.collision_scale)
            if _overlaps(base, other):
                lo = mid
            else:
                hi = mid
        return hi

    def _grid_collision(
        self,
        dx: float,
        dy: float,
        offset: float,
        angle_a: float,
        angle_b: float,
    ) -> bool:
        polys = []
        for r in range(self.config.grid_size):
            y = r * dy
            row_offset = (r % 2) * offset
            for c in range(self.config.grid_size):
                x = c * dx + row_offset
                angle = angle_a if ((r + c) % 2 == 0) else angle_b
                polys.append(
                    build_tree_polygon(x, y, angle, self.config.collision_scale)
                )
        tree = STRtree(polys)
        for i, poly in enumerate(polys):
            for idx in tree.query(poly):
                if idx == i:
                    continue
                if _overlaps(poly, polys[idx]):
                    return True
        return False

    def _min_dy(
        self, dx: float, offset: float, angle_a: float, angle_b: float
    ) -> float | None:
        lo = self.config.dy_min
        hi = self.config.dy_max
        if self._grid_collision(dx, hi, offset, angle_a, angle_b):
            return None
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if self._grid_collision(dx, mid, offset, angle_a, angle_b):
                lo = mid
            else:
                hi = mid
        return hi

    def _build_patterns(self) -> List[PatternSpec]:
        patterns: List[PatternSpec] = []
        for angle_a, angle_b in self.config.angle_pairs:
            angle_a = float(angle_a)
            angle_b = float(angle_b)
            dx = self._min_dx(angle_a, angle_b)
            if dx is None:
                continue
            for offset_ratio in self.config.offset_ratios:
                offset = dx * float(offset_ratio)
                dy = self._min_dy(dx, offset, angle_a, angle_b)
                if dy is None:
                    continue
                patterns.append(
                    PatternSpec(
                        angle_a=angle_a,
                        angle_b=angle_b,
                        offset_ratio=float(offset_ratio),
                        dx=dx * self.config.spacing_margin,
                        dy=dy * self.config.spacing_margin,
                    )
                )
        return patterns

    def _grid_placements(
        self,
        rows: int,
        cols: int,
        spec: PatternSpec,
        rng: random.Random | None = None,
    ) -> List[TreePlacement]:
        placements: List[TreePlacement] = []
        offset = spec.dx * spec.offset_ratio
        jitter = self.config.jitter if rng else 0.0
        angle_jitter = self.config.angle_jitter if rng else 0.0
        for r in range(rows):
            y = r * spec.dy
            row_offset = (r % 2) * offset
            for c in range(cols):
                x = c * spec.dx + row_offset
                if rng:
                    x += rng.uniform(-jitter, jitter)
                    yj = rng.uniform(-jitter, jitter)
                else:
                    yj = 0.0
                angle = spec.angle_a if ((r + c) % 2 == 0) else spec.angle_b
                if rng and angle_jitter:
                    angle = (angle + rng.uniform(-angle_jitter, angle_jitter)) % 360.0
                placements.append(TreePlacement(x=x, y=y + yj, deg=angle))
        return placements

    def _grid_placements_centered(
        self,
        rows: int,
        cols: int,
        spec: PatternSpec,
        rng: random.Random | None = None,
    ) -> List[TreePlacement]:
        placements: List[TreePlacement] = []
        offset = spec.dx * spec.offset_ratio
        jitter = self.config.jitter if rng else 0.0
        angle_jitter = self.config.angle_jitter if rng else 0.0
        y0 = -((rows - 1) / 2.0) * spec.dy
        x0 = -((cols - 1) / 2.0) * spec.dx
        for r in range(rows):
            y = y0 + r * spec.dy
            row_offset = (r % 2) * offset
            for c in range(cols):
                x = x0 + c * spec.dx + row_offset
                if rng:
                    x += rng.uniform(-jitter, jitter)
                    yj = rng.uniform(-jitter, jitter)
                else:
                    yj = 0.0
                angle = spec.angle_a if ((r + c) % 2 == 0) else spec.angle_b
                if rng and angle_jitter:
                    angle = (angle + rng.uniform(-angle_jitter, angle_jitter)) % 360.0
                placements.append(TreePlacement(x=x, y=y + yj, deg=angle))
        return placements

    def _select_centered(self, placements: List[TreePlacement], n: int) -> List[TreePlacement]:
        def key(p: TreePlacement) -> tuple[float, float]:
            return (max(abs(p.x), abs(p.y)), abs(p.x) + abs(p.y))

        ordered = sorted(placements, key=key)
        return ordered[:n]

    def _scale(self, placements: List[TreePlacement], factor: float) -> List[TreePlacement]:
        return [
            TreePlacement(x=p.x * factor, y=p.y * factor, deg=p.deg)
            for p in placements
        ]

    def _build_polygons(self, placements: List[TreePlacement]):
        return [
            build_tree_polygon(p.x, p.y, p.deg, self.config.scale_factor)
            for p in placements
        ]

    def _has_collision(self, placements: List[TreePlacement]) -> bool:
        polys = self._build_polygons(placements)
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

    def _score_and_bounds(self, placements: List[TreePlacement]) -> Tuple[float, Tuple[float, float, float, float]]:
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

    def _center(self, placements: List[TreePlacement], bounds: Tuple[float, float, float, float]) -> List[TreePlacement]:
        minx, miny, maxx, maxy = bounds
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        centered = [
            TreePlacement(x=p.x - cx, y=p.y - cy, deg=p.deg) for p in placements
        ]
        return centered

    def _square_search(
        self,
        placements: List[TreePlacement],
        n: int,
        spec: PatternSpec,
    ) -> List[TreePlacement]:
        steps = max(1, int(self.config.center_steps))
        offsets_x = [spec.dx * (i / steps) for i in range(steps)]
        offsets_y = [spec.dy * (i / steps) for i in range(steps)]

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

    def _best_layout(self, n: int, spec: PatternSpec, rng: random.Random) -> List[TreePlacement]:
        max_rows = int(math.ceil(math.sqrt(n))) + self.config.rows_pad
        best_score = float("inf")
        best_layout: List[TreePlacement] = []
        best_bounds = (0.0, 0.0, 0.0, 0.0)

        if self.config.selection_mode == "square_search":
            rows = max_rows + int(self.config.search_pad)
            cols = rows
            placements = self._grid_placements_centered(rows, cols, spec, None)
            placements = self._square_search(placements, n, spec)
            score, bounds = self._score_and_bounds(placements)
            best_score = score
            best_layout = placements
            best_bounds = bounds
        else:
            for rows in range(1, max_rows + 1):
                cols = int(math.ceil(n / rows))
                if self.config.selection_mode == "center":
                    placements = self._grid_placements_centered(rows, cols, spec, rng)
                    placements = self._select_centered(placements, n)
                    if self.config.jitter and self._has_collision(placements):
                        placements = self._grid_placements_centered(rows, cols, spec, None)
                        placements = self._select_centered(placements, n)
                else:
                    placements = self._grid_placements(rows, cols, spec, rng)[:n]
                    if self.config.jitter and self._has_collision(placements):
                        placements = self._grid_placements(rows, cols, spec, None)[:n]
                score, bounds = self._score_and_bounds(placements)
                if score < best_score:
                    best_score = score
                    best_layout = placements
                    best_bounds = bounds

        best_layout = self._global_squeeze(best_layout)
        _, bounds = self._score_and_bounds(best_layout)
        return self._center(best_layout, bounds)

    def solve(self, n_max: int, seed: int = 0) -> Dict[int, List[TreePlacement]]:
        groups: Dict[int, List[TreePlacement]] = {}
        rng = random.Random(seed)
        for n in range(1, n_max + 1):
            best_score = float("inf")
            best_layout: List[TreePlacement] = []
            for spec in self._patterns:
                layout = self._best_layout(n, spec, rng)
                score, _ = self._score_and_bounds(layout)
                if score < best_score:
                    best_score = score
                    best_layout = layout
            groups[n] = best_layout
        return groups
