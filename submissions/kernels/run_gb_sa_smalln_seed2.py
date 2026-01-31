#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_script() -> Path:
    candidates = [
        Path("/kaggle/working/kaggle/run_gb_sa_smalln.py"),
        Path("/kaggle/input/santa-2025-solver/kaggle/run_gb_sa_smalln.py"),
    ]
    base = Path("/kaggle/input")
    if base.exists():
        for entry in base.iterdir():
            maybe = entry / "kaggle" / "run_gb_sa_smalln.py"
            candidates.append(maybe)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"run_gb_sa_smalln.py not found; tried: {tried}")


def main() -> None:
    script = _find_script()
    out_dir = Path("/kaggle/working/gb_sa_smalln_seed2")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--sa-seed",
        "2021",
        "--out-dir",
        str(out_dir),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
