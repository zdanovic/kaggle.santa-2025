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
        "/kaggle/working/cpp_sa_batch_b",
        "--min-n",
        "21",
        "--max-n",
        "50",
        "--iterations",
        "16000",
        "--restarts",
        "70",
        "--max-gens",
        "2",
        "--max-noimprove",
        "6",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
