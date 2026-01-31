#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    runner = Path("/kaggle/input/santa-2025-solver/scripts/pattern_opt.py")
    out_dir = Path("/kaggle/working/pattern_opt")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "--trials",
        "12000",
        "--keep",
        "8",
        "--angle-set",
        "0,180,30,210,45,225,60,240,90,270,120,300",
        "--angle-jitter",
        "6",
        "--dx-range",
        "0.3,1.1",
        "--dy-range",
        "0.3,1.2",
        "--offset-range",
        "0.45,0.85",
        "--score-n-list",
        "60,80,100,120,150,200",
        "--seed",
        "9321",
        "--out",
        str(out_dir / "pattern_opt.json"),
        "--emit-submission",
        str(out_dir / "best_submission.csv"),
        "--emit-decimals",
        "6",
        "--final-n-max",
        "200",
        "--grid-size",
        "5",
        "--selection-mode",
        "square_search",
        "--center-steps",
        "8",
        "--search-pad",
        "3",
        "--global-squeeze",
        "--squeeze-factor",
        "0.985",
        "--squeeze-steps",
        "24",
        "--squeeze-iters",
        "10",
        "--refine-steps",
        "180",
        "--refine-decay",
        "0.985",
        "--refine-dx-scale",
        "0.12",
        "--refine-dy-scale",
        "0.12",
        "--refine-offset-scale",
        "0.14",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
