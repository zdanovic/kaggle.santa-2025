#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_run_all() -> Path:
    candidates = [
        Path("/kaggle/working/kaggle/run_all.py"),
        Path("/kaggle/input/santa-2025-solver/kaggle/run_all.py"),
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            maybe = entry / "kaggle" / "run_all.py"
            candidates.append(maybe)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"run_all.py not found; tried: {tried}")


def main() -> None:
    root = Path(__file__).resolve().parent
    run_all = _find_run_all()
    cmd = [
        sys.executable,
        str(run_all),
        "--config",
        "/kaggle/working/configs/kaggle_pattern_jitter.yaml",
        "--seeds",
        "101,202,303",
        "--results-dir",
        "/kaggle/working/exp_pattern_jitter",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
