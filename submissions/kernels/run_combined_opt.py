#!/usr/bin/env python3
"""
Santa 2025 - Combined Optimization Pipeline

Combines the best techniques from public solutions:
1. bbox3 multi-phase optimization (from top kernels)
2. Gravity-weighted SA for small-n groups
3. fix_direction rotation optimization
4. GB (Greedy Backward) beam search

Run on Kaggle with:
- GPU: Not required
- Time: 12 hours recommended
"""
from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path


def _find_bbox3_script() -> Path:
    candidates = [
        Path("/kaggle/working/kaggle/run_bbox3.py"),
        Path("/kaggle/working/run_bbox3.py"),
        Path("/kaggle/input/santa-2025-solver/run_bbox3.py"),  # Dataset root
        Path("/kaggle/input/santa-2025-solver/kaggle/run_bbox3.py"),
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            # Check both root and kaggle subfolder
            candidates.append(entry / "run_bbox3.py")
            candidates.append(entry / "kaggle" / "run_bbox3.py")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"run_bbox3.py not found; tried: {tried}")



def _find_gb_sa_script() -> Path:
    candidates = [
        Path("/kaggle/working/kaggle/run_gb_sa.py"),
        Path("/kaggle/working/run_gb_sa.py"),
        Path("/kaggle/input/santa-2025-solver/run_gb_sa.py"),  # Dataset root
        Path("/kaggle/input/santa-2025-solver/kaggle/run_gb_sa.py"),
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            candidates.append(entry / "run_gb_sa.py")
            candidates.append(entry / "kaggle" / "run_gb_sa.py")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None



def _select_baseline() -> Path:
    candidates = [
        Path("/kaggle/input/santa-2025-sota-baseline/best_submission.csv"),
        Path("/kaggle/input/santa-2025-solver/best_submission.csv"),
        Path("/kaggle/input/santa-2025-bbox3-baseline/best_submission.csv"),
        Path("/kaggle/input/santa-2025-csv/santa-2025.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"No baseline found")


def main() -> None:
    out_dir = Path("/kaggle/working/combined_opt")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    baseline = _select_baseline()
    print(f"Using baseline: {baseline}")
    
    # ==== PHASE 1: bbox3 multi-phase optimization ====
    print("\n" + "=" * 60)
    print("PHASE 1: bbox3 multi-phase optimization")
    print("=" * 60)
    
    bbox3_script = _find_bbox3_script()
    bbox3_out = out_dir / "bbox3"
    bbox3_out.mkdir(parents=True, exist_ok=True)
    
    bbox3_cmd = [
        sys.executable,
        str(bbox3_script),
        "--baseline", str(baseline),
        "--bbox3", "/kaggle/input/santa-2025-csv/bbox3",
        "--out-dir", str(bbox3_out),
        "--log-file", str(bbox3_out / "bbox3.log"),
        "--budget-sec", "28800",  # 8 hours for bbox3
        "--buffer-sec", "600",
        "--decimals", "15",
        # Phase A: wide scan
        "--phase-a-timeout", "600",
        "--phase-a-n", "1200,1500,1800,2100",
        "--phase-a-r", "40,60,80",
        "--phase-a-top-k", "4",
        # Phase B: medium refinement
        "--phase-b-timeout", "1200",
        "--phase-b-top-k", "3",
        "--phase-b-fix-passes", "2",
        # Phase C: deep search on best
        "--phase-c-timeout", "2400",
        "--phase-c-top-k", "2",
        "--phase-c-fix-passes", "3",
        # Fallback
        "--fallback-n", "1800",
        "--fallback-r", "60",
        "--fallback-timeout", "2400",
    ]
    print(" ".join(bbox3_cmd))
    subprocess.run(bbox3_cmd, check=False)
    
    # Get bbox3 result
    bbox3_best = bbox3_out / "best_submission.csv"
    if not bbox3_best.exists():
        print("WARNING: bbox3 did not produce output, using baseline")
        bbox3_best = baseline
    
    # ==== PHASE 2: GB+SA optimization on small-n (if script exists) ====
    gb_sa_script = _find_gb_sa_script()
    if gb_sa_script:
        print("\n" + "=" * 60)
        print("PHASE 2: GB+SA optimization (small/mid n)")
        print("=" * 60)
        
        gb_sa_out = out_dir / "gb_sa"
        gb_sa_out.mkdir(parents=True, exist_ok=True)
        working_csv = Path("/kaggle/working/submission_.csv")
        output_csv = Path("/kaggle/working/submission.csv")
        
        shutil.copy(bbox3_best, working_csv)
        shutil.copy(bbox3_best, output_csv)
        
        gb_sa_cmd = [
            sys.executable,
            str(gb_sa_script),
            "--baseline", str(bbox3_best),
            "--out-dir", str(gb_sa_out),
            "--sa-hours", "3.5",  # ~3.5 hours for SA
            "--sa-iter", "800000",
            "--sa-range", "1-100",  # Focus on small/mid n
            "--sa-tstart", "12.0",
            "--sa-tend", "0.005",
            "--sa-seed", "42",
        ]
        print(" ".join(gb_sa_cmd))
        subprocess.run(gb_sa_cmd, check=False)
        
        final_best = gb_sa_out / "best_submission.csv"
        if not final_best.exists():
            final_best = output_csv if output_csv.exists() else bbox3_best
    else:
        print("\n[SKIP] GB+SA script not found")
        final_best = bbox3_best
    
    # ==== FINAL: Copy best result ====
    print("\n" + "=" * 60)
    print("FINAL: Copying best submission")
    print("=" * 60)
    
    final_out = out_dir / "best_submission.csv"
    shutil.copy(final_best, final_out)
    shutil.copy(final_best, Path("/kaggle/working/submission.csv"))
    
    print(f"\nFinal submission: {final_out}")
    print("Done!")


if __name__ == "__main__":
    main()
