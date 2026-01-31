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
    bbox3_script = _find_bbox3_script()
    out_dir = Path("/kaggle/working/bbox3_gb_sa_aggressive")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(bbox3_script),
        "--baseline",
        "/kaggle/input/santa-2025-solver/kaggle/baselines/gb_sa_best_submission.csv",
        "--bbox3",
        "/kaggle/input/santa-2025-csv/bbox3",
        "--out-dir",
        str(out_dir),
        "--log-file",
        str(out_dir / "bbox3_run.log"),
        "--budget-sec",
        "36000",
        "--buffer-sec",
        "1800",
        "--decimals",
        "15",
        "--initial-fix-passes",
        "2",
        "--phase-a-timeout",
        "900",
        "--phase-a-n",
        "900,1100,1300,1500,1700",
        "--phase-a-r",
        "50,60,70,80",
        "--phase-a-top-k",
        "5",
        "--phase-a-fix-passes",
        "2",
        "--phase-b-timeout",
        "2100",
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
        "1500",
        "--fallback-r",
        "60",
        "--fallback-timeout",
        "3600",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    best_csv = out_dir / "best_submission.csv"
    log_path = out_dir / "bbox3_run.log"
    root_best = Path("/kaggle/working/best_submission.csv")
    root_log = Path("/kaggle/working/bbox3_run.log")
    if best_csv.exists():
        root_best.write_bytes(best_csv.read_bytes())
    if log_path.exists():
        root_log.write_bytes(log_path.read_bytes())
    summary = out_dir / "summary.txt"
    if log_path.exists():
        summary.write_text(log_path.read_text(), encoding="utf-8")
    elif best_csv.exists():
        summary.write_text("bbox3_gb_sa_aggressive: best_submission.csv saved", encoding="utf-8")


if __name__ == "__main__":
    main()
