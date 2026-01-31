"""Rotation optimization utilities for Santa 2025.

Based on techniques from top public solutions:
- Rotate entire group to minimize bounding box
- Uses ConvexHull edge angles for candidate rotations
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from santa2025.geometry import build_tree_polygon
from santa2025.io import TreePlacement

try:
    from scipy.optimize import minimize_scalar
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _bbox_side_at_angle(angle_deg: float, points: np.ndarray) -> float:
    """Calculate bounding box side length after rotating points by angle."""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated = points.dot(rot_matrix_T)
    min_xy = np.min(rotated, axis=0)
    max_xy = np.max(rotated, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])


def _hull_points(points: np.ndarray) -> np.ndarray:
    """Get convex hull points for faster rotation optimization."""
    if points.shape[0] < 3:
        return points
    if SCIPY_AVAILABLE:
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            pass
    return points


def _edge_angles(points: np.ndarray) -> List[float]:
    """Get angles from convex hull edges as rotation candidates."""
    angles: set = set()
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


def optimize_rotation(
    placements: List[TreePlacement],
    angle_max: float = 89.999,
    epsilon: float = 1e-7,
    scale_factor: float = 1e18,
) -> Tuple[float, float]:
    """Find optimal rotation angle to minimize bounding box.
    
    Returns:
        (best_side_length, best_angle_deg)
    """
    # Collect all polygon points
    points = []
    for p in placements:
        poly = build_tree_polygon(p.x, p.y, p.deg, scale_factor)
        points.extend(list(poly.exterior.coords))
    
    points_np = np.array(points)
    if points_np.size == 0:
        return 0.0, 0.0
    
    hull_pts = _hull_points(points_np)
    initial_side = _bbox_side_at_angle(0.0, hull_pts)
    
    if SCIPY_AVAILABLE and hull_pts.shape[0] >= 3:
        # Use scipy optimization
        res = minimize_scalar(
            lambda a: _bbox_side_at_angle(a, hull_pts),
            bounds=(0.001, float(angle_max)),
            method="bounded",
        )
        best_angle = float(res.x)
        best_side = float(res.fun)
    else:
        # Fallback: test edge angles
        best_angle = 0.0
        best_side = initial_side
        for angle in _edge_angles(hull_pts):
            if angle > angle_max:
                continue
            cand = _bbox_side_at_angle(angle, hull_pts)
            if cand < best_side:
                best_side = cand
                best_angle = angle
    
    epsilon_scaled = epsilon * scale_factor
    if initial_side - best_side <= epsilon_scaled:
        return initial_side / scale_factor, 0.0
    return best_side / scale_factor, best_angle


def apply_rotation(
    placements: List[TreePlacement],
    angle_deg: float,
) -> List[TreePlacement]:
    """Rotate all trees by angle_deg around the group center."""
    if not placements or abs(angle_deg) < 1e-12:
        return list(placements)
    
    # Find center
    xs = [p.x for p in placements]
    ys = [p.y for p in placements]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    # Rotate
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated = []
    for p in placements:
        dx = p.x - cx
        dy = p.y - cy
        nx = cx + dx * c - dy * s
        ny = cy + dx * s + dy * c
        ndeg = (p.deg + angle_deg) % 360.0
        rotated.append(TreePlacement(x=nx, y=ny, deg=ndeg))
    return rotated


def fix_direction(
    placements: List[TreePlacement],
    angle_max: float = 89.999,
    epsilon: float = 1e-7,
    scale_factor: float = 1e18,
) -> Tuple[List[TreePlacement], float]:
    """Optimize group rotation to minimize bounding box.
    
    This is a key technique from top public solutions that often provides
    small but consistent improvements.
    
    Returns:
        (optimized_placements, improvement_amount)
    """
    if not placements:
        return placements, 0.0
    
    # Get initial side length
    from santa2025.geometry import polygons_bounds
    initial_polys = [
        build_tree_polygon(p.x, p.y, p.deg, scale_factor)
        for p in placements
    ]
    bounds = polygons_bounds(initial_polys)
    initial_side = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / scale_factor
    
    # Find optimal rotation
    best_side, best_angle = optimize_rotation(
        placements, angle_max, epsilon, scale_factor
    )
    
    if best_side < initial_side - epsilon:
        rotated = apply_rotation(placements, best_angle)
        improvement = initial_side - best_side
        return rotated, improvement
    
    return list(placements), 0.0
