#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    runner = Path("/kaggle/input/santa-2025-solver/kaggle/run_periodic_search.py")
    cmd = [
        sys.executable,
        str(runner),
        "--trials",
        "5000",
        "--keep",
        "15",
        "--k-list",
        "3,4",
        "--angle-set",
        "0,60,120,180,240,300",
        "--angle-jitter",
        "10",
        "--dx-range",
        "0.45,1.2",
        "--dy-range",
        "0.45,1.2",
        "--offset-range",
        "0.0,1.0",
        "--lattice-angle-range=-12.0,12.0",
        "--basis-attempts",
        "100",
        "--search-pad",
        "8",
        "--center-steps",
        "8",
        "--score-n-list",
        "40,60,80,100,120,150,200",
        "--final-n-max",
        "200",
        "--neighbor-range",
        "2",
        "--seed",
        "8821",
        "--out-dir",
        "/kaggle/working/periodic_search_b",
        "--emit-decimals",
        "6",
        "--final-global-squeeze",
        "--final-squeeze-factor",
        "0.985",
        "--final-squeeze-steps",
        "24",
        "--final-squeeze-iters",
        "10",
        "--refine-steps",
        "160",
        "--refine-restarts",
        "2",
        "--refine-dx-scale",
        "0.06",
        "--refine-dy-scale",
        "0.06",
        "--refine-offset-scale",
        "0.08",
        "--refine-angle-scale",
        "4.5",
        "--refine-basis-scale",
        "0.12",
        "--refine-deg-scale",
        "10.0",
        "--refine-decay",
        "0.985",
        "--emit-config",
        "/kaggle/working/periodic_search_b/best_config.yaml",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
