#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    runner = Path("/kaggle/input/santa-2025-solver/kaggle/run_cpp_sa.py")
    cmd = [
        sys.executable,
        str(runner),
        "--baseline",
        "/kaggle/input/santa-2025-solver/kaggle/baselines/gb_sa_best_submission.csv",
        "--out-dir",
        "/kaggle/working/cpp_sa_midn",
        "--min-n",
        "13",
        "--max-n",
        "40",
        "--iterations",
        "25000",
        "--restarts",
        "80",
        "--max-gens",
        "2",
        "--max-noimprove",
        "6",
        "--seed-base",
        "2000",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
