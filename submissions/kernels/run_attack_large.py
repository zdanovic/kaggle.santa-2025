#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    runner = Path("/kaggle/input/santa-2025-solver/run_cpp_sa.py")
    
    # Fix: Copy C++ source to where runner expects it
    cpp_src = Path("/kaggle/input/santa-2025-solver/sa_v1_parallel.cpp")
    work_cpp_dir = Path("/kaggle/working/kaggle")
    work_cpp_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(cpp_src, work_cpp_dir / "sa_v1_parallel.cpp")

    baseline = Path("/kaggle/input/santa-2025-sota-baseline/best_submission.csv")
    
    cmd = [
        sys.executable,
        str(runner),
        "--baseline", str(baseline),
        "--out-dir", "/kaggle/working/attack_large",
        # Attack Range: N=50 to 200
        "--min-n", "50",
        "--max-n", "200",
        # Compression SA params
        "--iterations", "50000",
        "--restarts", "50",
        "--max-gens", "3",
        "--max-noimprove", "6",
        "--seed-base", "5000",
        # Compression specific
        "--compress-steps", "30",
        "--compress-factor", "0.995",
        "--compress-relax-iters", "100",
        "--compress-relax-step", "0.01",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
