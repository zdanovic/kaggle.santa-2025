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
        "/kaggle/working/cpp_sa_smalln_deep",
        "--min-n",
        "1",
        "--max-n",
        "12",
        "--iterations",
        "50000",
        "--restarts",
        "160",
        "--max-gens",
        "3",
        "--max-noimprove",
        "8",
        "--seed-base",
        "1000",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
