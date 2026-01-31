from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterable, Tuple

from shapely import affinity
from shapely.geometry import Polygon


TREE_POINTS = (
    # Tip
    (0.0, 0.8),
    # Right side - top tier
    (0.125, 0.5),
    (0.0625, 0.5),
    # Right side - middle tier
    (0.2, 0.25),
    (0.1, 0.25),
    # Right side - bottom tier
    (0.35, 0.0),
    # Right trunk
    (0.075, 0.0),
    (0.075, -0.2),
    # Left trunk
    (-0.075, -0.2),
    (-0.075, 0.0),
    # Left side - bottom tier
    (-0.35, 0.0),
    # Left side - middle tier
    (-0.1, 0.25),
    (-0.2, 0.25),
    # Left side - top tier
    (-0.0625, 0.5),
    (-0.125, 0.5),
)


@lru_cache(maxsize=4)
def base_tree_polygon(scale_factor: float = 1e18) -> Polygon:
    scaled = [(x * scale_factor, y * scale_factor) for x, y in TREE_POINTS]
    return Polygon(scaled)


def build_tree_polygon(
    center_x: float,
    center_y: float,
    angle_deg: float,
    scale_factor: float = 1e18,
) -> Polygon:
    base = base_tree_polygon(scale_factor)
    rotated = affinity.rotate(base, angle_deg, origin=(0.0, 0.0))
    return affinity.translate(rotated, xoff=center_x * scale_factor, yoff=center_y * scale_factor)


def polygons_bounds(polygons: Iterable[Polygon]) -> Tuple[float, float, float, float]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for poly in polygons:
        bx0, by0, bx1, by1 = poly.bounds
        minx = min(minx, bx0)
        miny = min(miny, by0)
        maxx = max(maxx, bx1)
        maxy = max(maxy, by1)
    return minx, miny, maxx, maxy


@lru_cache(maxsize=1)
def tree_max_radius() -> float:
    return max(math.hypot(x, y) for x, y in TREE_POINTS)
