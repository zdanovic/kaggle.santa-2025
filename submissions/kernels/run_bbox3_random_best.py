#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    runner = Path("/kaggle/input/santa-2025-solver/kaggle/run_bbox3_random.py")
    out_dir = Path("/kaggle/working/bbox3_random_best")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "--baseline",
        "/kaggle/input/santa-2025-solver/kaggle/baselines/ensemble_cascade_v3.csv",
        "--bbox3",
        "/kaggle/input/santa-2025-csv/bbox3",
        "--out-dir",
        str(out_dir),
        "--budget-sec",
        "36000",
        "--buffer-sec",
        "1200",
        "--min-n",
        "200",
        "--max-n",
        "2600",
        "--min-r",
        "20",
        "--max-r",
        "120",
        "--timeout-sec",
        "900",
        "--seed",
        "20250109",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
