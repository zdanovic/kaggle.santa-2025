#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    # Path to C++ SA runner in the solver dataset
    runner = Path("/kaggle/input/santa-2025-solver/run_cpp_sa.py")
    
    # Fix: Copy C++ source to where runner expects it
    cpp_src = Path("/kaggle/input/santa-2025-solver/sa_v1_parallel.cpp")
    work_cpp_dir = Path("/kaggle/working/kaggle")
    work_cpp_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(cpp_src, work_cpp_dir / "sa_v1_parallel.cpp")
    
    # SOTA Baseline
    baseline = Path("/kaggle/input/santa-2025-sota-baseline/best_submission.csv")
    
    cmd = [
        sys.executable,
        str(runner),
        "--baseline", str(baseline),
        "--out-dir", "/kaggle/working/attack_small",
        # Attack Range: N=1 to 50
        "--min-n", "1",
        "--max-n", "50",
        # Deep Search Params
        "--iterations", "500000",
        "--restarts", "20",
        "--max-gens", "3",
        "--max-noimprove", "8",
        "--seed-base", "1000",
        # Random Inits to escape local optima
        "--random-inits", "50",
        "--random-init-max-n", "20",
        "--random-init-scale", "1.2",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
