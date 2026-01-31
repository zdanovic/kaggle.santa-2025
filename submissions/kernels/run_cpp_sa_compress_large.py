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
        "/kaggle/working/cpp_sa_compress_large",
        "--min-n",
        "50",
        "--max-n",
        "200",
        "--iterations",
        "25000",
        "--restarts",
        "50",
        "--max-gens",
        "2",
        "--max-noimprove",
        "4",
        "--seed-base",
        "9000",
        "--compress-steps",
        "20",
        "--compress-factor",
        "0.998",
        "--compress-relax-iters",
        "80",
        "--compress-relax-step",
        "0.02",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
