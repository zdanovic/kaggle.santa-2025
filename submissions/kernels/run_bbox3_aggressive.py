#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_bbox3_script() -> Path:
    candidates = [
        Path("/kaggle/working/kaggle/run_bbox3.py"),
        Path("/kaggle/input/santa-2025-solver/kaggle/run_bbox3.py"),
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            maybe = entry / "kaggle" / "run_bbox3.py"
            candidates.append(maybe)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"run_bbox3.py not found; tried: {tried}")


def main() -> None:
    root = Path(__file__).resolve().parent
    bbox3_script = _find_bbox3_script()
    out_dir = Path("/kaggle/working/bbox3_aggressive")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(bbox3_script),
        "--baseline",
        "/kaggle/input/santa-2025-bbox3-baseline/best_submission.csv",
        "--bbox3",
        "/kaggle/input/santa-2025-csv/bbox3",
        "--out-dir",
        str(out_dir),
        "--log-file",
        str(out_dir / "bbox3_run.log"),
        "--budget-sec",
        "36000",
        "--buffer-sec",
        "1200",
        "--decimals",
        "15",
        "--phase-a-timeout",
        "600",
        "--phase-a-n",
        "1200,1500,1800,2100,2400",
        "--phase-a-r",
        "30,45,60,75,90",
        "--phase-a-top-k",
        "5",
        "--phase-b-timeout",
        "1800",
        "--phase-b-top-k",
        "3",
        "--phase-b-fix-passes",
        "2",
        "--phase-c-timeout",
        "3600",
        "--phase-c-top-k",
        "2",
        "--phase-c-fix-passes",
        "3",
        "--fallback-n",
        "2000",
        "--fallback-r",
        "60",
        "--fallback-timeout",
        "3600",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
