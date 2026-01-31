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
        "/kaggle/working/cpp_sa_smalln_global",
        "--min-n",
        "1",
        "--max-n",
        "12",
        "--iterations",
        "60000",
        "--restarts",
        "200",
        "--max-gens",
        "3",
        "--max-noimprove",
        "8",
        "--seed-base",
        "3000",
        "--random-inits",
        "120",
        "--random-init-max-n",
        "12",
        "--random-init-scale",
        "1.25",
        "--random-init-tries",
        "6",
        "--random-init-max-attempts",
        "2500",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
