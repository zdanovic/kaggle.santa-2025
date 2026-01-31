#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _ensure_ortools() -> None:
    if importlib.util.find_spec("ortools") is not None:
        return
    req = Path("/kaggle/input/santa-2025-solver/requirements.txt")
    if req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ortools>=9.7.0"], check=True)


def main() -> None:
    _ensure_ortools()
    runner = Path("/kaggle/input/santa-2025-solver/scripts/exact_smalln.py")
    out_dir = Path("/kaggle/working/exact_smalln_best")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "--baseline",
        "/kaggle/input/santa-2025-solver/kaggle/baselines/ensemble_cascade_v3.csv",
        "--n-list",
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        "--out-dir",
        str(out_dir),
        "--time-limit",
        "180",
        "--threads",
        "8",
        "--angle-set",
        "0,30,45,60,90,120,180,240,300",
        "--angle-jitter",
        "8",
        "--jitter-span",
        "0.14",
        "--jitter-steps",
        "3",
        "--random-points",
        "320",
        "--max-candidates",
        "2000",
        "--scale",
        "1000",
        "--seed",
        "8821",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
