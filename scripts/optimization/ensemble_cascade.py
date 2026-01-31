#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.metric import ParticipantVisibleError, score_detailed

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*_args, **_kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap


PX = np.array(
    [0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125],
    dtype=np.float64,
)
PY = np.array(
    [0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5],
    dtype=np.float64,
)
TOTAL_LEN = 200 * 201 // 2


@njit(cache=True)
def _tree_bounds(cx: float, cy: float, angle_deg: float) -> Tuple[float, float, float, float]:
    a = angle_deg * np.pi / 180.0
    ca = np.cos(a)
    sa = np.sin(a)
    mnx = 1e30
    mny = 1e30
    mxx = -1e30
    mxy = -1e30
    for i in range(15):
        rx = ca * PX[i] - sa * PY[i]
        ry = sa * PX[i] + ca * PY[i]
        x = rx + cx
        y = ry + cy
        if x < mnx:
            mnx = x
        if y < mny:
            mny = y
        if x > mxx:
            mxx = x
        if y > mxy:
            mxy = y
    return mnx, mny, mxx, mxy


@njit(cache=True)
def _group_side(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, start: int, n: int) -> float:
    mnx = 1e30
    mny = 1e30
    mxx = -1e30
    mxy = -1e30
    for i in range(n):
        bx0, by0, bx1, by1 = _tree_bounds(xs[start + i], ys[start + i], degs[start + i])
        if bx0 < mnx:
            mnx = bx0
        if by0 < mny:
            mny = by0
        if bx1 > mxx:
            mxx = bx1
        if by1 > mxy:
            mxy = by1
    w = mxx - mnx
    h = mxy - mny
    return w if w >= h else h


@njit(cache=True)
def _group_side_skip(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, start: int, n: int, skip_local: int) -> float:
    mnx = 1e30
    mny = 1e30
    mxx = -1e30
    mxy = -1e30
    for i in range(n):
        if i == skip_local:
            continue
        bx0, by0, bx1, by1 = _tree_bounds(xs[start + i], ys[start + i], degs[start + i])
        if bx0 < mnx:
            mnx = bx0
        if by0 < mny:
            mny = by0
        if bx1 > mxx:
            mxx = bx1
        if by1 > mxy:
            mxy = by1
    w = mxx - mnx
    h = mxy - mny
    return w if w >= h else h


@njit(cache=True)
def _total_score(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    total = 0.0
    for n in range(1, 201):
        start = n * (n - 1) // 2
        side = _group_side(xs, ys, degs, start, n)
        total += (side * side) / n
    return total


def _expand_inputs(inputs: Iterable[str]) -> List[Path]:
    expanded: List[Path] = []
    for item in inputs:
        item = item.strip()
        if not item:
            continue
        matches = glob.glob(item)
        if matches:
            expanded.extend(Path(m) for m in matches)
        else:
            expanded.append(Path(item))
    return expanded


def _load_submission_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, dtype=str)
    if "id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "id"})
    df = df[["id", "x", "y", "deg"]].copy()
    for col in ["x", "y", "deg"]:
        df[col] = df[col].astype(str).str.strip().str.lstrip("sS")
    parts = df["id"].astype(str).str.split("_", n=1, expand=True)
    df["group"] = parts[0].astype(int)
    df["item"] = parts[1].astype(int)
    xs = np.zeros(TOTAL_LEN, dtype=np.float64)
    ys = np.zeros(TOTAL_LEN, dtype=np.float64)
    degs = np.zeros(TOTAL_LEN, dtype=np.float64)
    for _, row in df.iterrows():
        n = int(row["group"])
        item = int(row["item"])
        start = n * (n - 1) // 2
        idx = start + item
        xs[idx] = float(row["x"])
        ys[idx] = float(row["y"])
        degs[idx] = float(row["deg"])
    return xs, ys, degs


def _best_of_submissions(arrays: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_xs = np.zeros(TOTAL_LEN, dtype=np.float64)
    out_ys = np.zeros(TOTAL_LEN, dtype=np.float64)
    out_degs = np.zeros(TOTAL_LEN, dtype=np.float64)
    for n in range(1, 201):
        start = n * (n - 1) // 2
        best_score = 1e300
        best_idx = -1
        for i, (xs, ys, degs) in enumerate(arrays):
            side = _group_side(xs, ys, degs, start, n)
            score = (side * side) / n
            if score < best_score:
                best_score = score
                best_idx = i
        bx, by, bd = arrays[best_idx]
        out_xs[start:start + n] = bx[start:start + n]
        out_ys[start:start + n] = by[start:start + n]
        out_degs[start:start + n] = bd[start:start + n]
    return out_xs, out_ys, out_degs


def _backward_iteration(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_xs = xs.copy()
    out_ys = ys.copy()
    out_degs = degs.copy()
    best_side = 1e300
    best_xs = None
    best_ys = None
    best_degs = None
    for n in range(200, 0, -1):
        start = n * (n - 1) // 2
        side = _group_side(xs, ys, degs, start, n)
        if side < best_side:
            best_side = side
            best_xs = xs[start:start + n].copy()
            best_ys = ys[start:start + n].copy()
            best_degs = degs[start:start + n].copy()
            out_xs[start:start + n] = xs[start:start + n]
            out_ys[start:start + n] = ys[start:start + n]
            out_degs[start:start + n] = degs[start:start + n]
        else:
            if best_xs is not None and best_xs.shape[0] >= n:
                out_xs[start:start + n] = best_xs[:n]
                out_ys[start:start + n] = best_ys[:n]
                out_degs[start:start + n] = best_degs[:n]
            else:
                out_xs[start:start + n] = xs[start:start + n]
                out_ys[start:start + n] = ys[start:start + n]
                out_degs[start:start + n] = degs[start:start + n]
    return out_xs, out_ys, out_degs


@njit(cache=True)
def _score_contribution(side: float, n: int) -> float:
    return (side * side) / n


def _deletion_cascade_beam(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, beam_width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Initial state: (score, xs, ys, degs)
    # score is the total score of the submission
    current_score = _total_score(xs, ys, degs)
    
    # We maintain a list of candidates. Each candidate is a tuple:
    # (total_score, xs, ys, degs)
    candidates = [(current_score, xs.copy(), ys.copy(), degs.copy())]

    eps = 1e-15

    # Iterate from 200 down to 2
    for n in range(200, 1, -1):
        print(f"Propagating from N={n} (Branching factor: {n+1}, Beam: {len(candidates)})")
        
        next_candidates = []
        
        start_n = n * (n - 1) // 2
        start_prev = (n - 1) * (n - 2) // 2
        
        # Expand each candidate
        for cand_score, c_xs, c_ys, c_degs in candidates:
            # Option 0: Keep existing configuration for n-1
            # The score is already correct for this path
            next_candidates.append((cand_score, c_xs, c_ys, c_degs))
            
            # Identify the score contribution of the CURRENT configuration of n-1
            # We need to subtract this if we replace it.
            current_side_prev = _group_side(c_xs, c_ys, c_degs, start_prev, n - 1)
            old_contrib = _score_contribution(current_side_prev, n - 1)
            
            # Option 1..n: Replace n-1 with subset of n
            for del_idx in range(n):
                # Calculate new side for n-1 using subset of n
                new_side = _group_side_skip(c_xs, c_ys, c_degs, start_n, n, del_idx)
                
                # Check if it's an improvement (or just different, if we want diversity)
                # In strict Beam Search, we just calculate the new total score
                new_contrib = _score_contribution(new_side, n - 1)
                
                # New score = Old Total - Old Contribution (for n-1) + New Contribution (for n-1)
                new_total_score = cand_score - old_contrib + new_contrib
                
                # Beam Search allows strictly worse scores locally in hope of better scores later.
                # Do NOT filter by improvement here. Just add to candidates.
                
                # Create new arrays
                # Optimization: Only copy if we select it? 
                # For simplicity, copy now. To optimize, we could just store the "action" and materialize later,
                # but memory is cheap here.
                new_xs = c_xs.copy()
                new_ys = c_ys.copy()
                new_degs = c_degs.copy()
                
                # Apply copy from N to N-1
                out_idx = start_prev
                for i in range(n):
                    if i == del_idx:
                        continue
                    in_idx = start_n + i
                    new_xs[out_idx] = new_xs[in_idx]
                    new_ys[out_idx] = new_ys[in_idx]
                    new_degs[out_idx] = new_degs[in_idx]
                    out_idx += 1
                    
                next_candidates.append((new_total_score, new_xs, new_ys, new_degs))
        
        # Prune candidates
        # Sort by score ascending (lower is better)
        next_candidates.sort(key=lambda x: x[0])

        if next_candidates:
            best_s = next_candidates[0][0]
            gaps = [f"{c[0]-best_s:.6g}" for c in next_candidates[:min(3, len(next_candidates))]]
            print(f"  Top scores gaps: {gaps}")

        
        # Deduplication:
        # We want to keep diverse configurations.
        # Identical scores STRONGLY suggest identical configurations in this deterministic problem 
        # (unless we have hash collisions in floating point, but unlikely to happen for DIFFERENT configs).
        # However, to be safe and encourage trying "second best" deletion, we should keep them if they are distinct.
        # But fully checking state equality is expensive. 
        # Let's rely on score + simple check? Or just keep Top K regardless, assuming they might be different 
        # if the indices removed were different.
        
        # Actually, let's just keep the top K distinct SCORES to avoid flooding the beam with equivalent states.
        # BUT, if the "second best" deletion leads to a strictly worse score, we WANT to keep it in the beam 
        # solely to see if it allows a better deletion next time.
        
        # So: Don't excessively dedup. Just allow top K.
        # BUT, if we have identical scores, it's 99.9% the same state.
        
        unique_candidates = []
        seen_scores = set()
        for cand in next_candidates:
            s = round(cand[0], 12) # Use high precision
            if s not in seen_scores:
                seen_scores.add(s)
                unique_candidates.append(cand)
            
            if len(unique_candidates) >= beam_width:
                break
        
        candidates = unique_candidates
        
        # If we didn't fill the beam with unique scores, fill it with the rest of sorted candidates
        # (duplicates allowed effectively, though they usually mean redundant work, they won't hurt accuracy)
        if len(candidates) < beam_width and len(next_candidates) > len(candidates):
             remaining_needed = beam_width - len(candidates)
             # Add non-unique ones (skip if they are literally the same object identity, but here they are tuples)
             # We just take the next best ones that we skipped.
             
             # Actually, if scores are identical, it's waste of compute. 
             # Let's just stick to unique scores but simpler.
             pass

    # Return best candidate
    return candidates[0][1], candidates[0][2], candidates[0][3]


def _write_submission(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, out_path: Path, decimals: int) -> None:
    rows = []
    for n in range(1, 201):
        start = n * (n - 1) // 2
        for t in range(n):
            idx = start + t
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{xs[idx]:.{decimals}f}",
                    "y": f"s{ys[idx]:.{decimals}f}",
                    "deg": f"s{(degs[idx] % 360.0):.{decimals}f}",
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated file paths or globs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--decimals", type=int, default=18)
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--skip-cascade", action="store_true")
    parser.add_argument("--beam-width", type=int, default=1, help="Beam width for deletion cascade.")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs.split(","))
    input_paths = [p for p in input_paths if p.exists()]
    if not input_paths:
        raise SystemExit("No input CSVs found.")

    arrays = [_load_submission_arrays(p) for p in input_paths]

    xs, ys, degs = _best_of_submissions(arrays)
    base_score = _total_score(xs, ys, degs)

    if not args.skip_backward:
        xs, ys, degs = _backward_iteration(xs, ys, degs)
    backward_score = _total_score(xs, ys, degs)

    if not args.skip_cascade:
        print(f"Running deletion cascade with beam width {args.beam_width}...")
        xs, ys, degs = _deletion_cascade_beam(xs, ys, degs, args.beam_width)
    final_score = _total_score(xs, ys, degs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_submission(xs, ys, degs, out_path, args.decimals)

    official_score = None
    try:
        df = pd.read_csv(out_path)
        official_score, _ = score_detailed(df)
        print(f"official_score: {official_score}")
    except ParticipantVisibleError as exc:
        print(f"official_score_error: {exc}")

    summary = {
        "inputs": [str(p) for p in input_paths],
        "base_score": float(base_score),
        "backward_score": float(backward_score),
        "final_score": float(final_score),
        "official_score": float(official_score) if official_score is not None else None,
        "output": str(out_path),
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
