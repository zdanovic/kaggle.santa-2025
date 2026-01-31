# Repository Audit & Transformation Plan

## 1. Audit Analysis
The current repository (`santa-2025`) contains high-quality optimization code but suffers from "Competition Sprawl"—a mix of core logic, temporary experiments, Kaggle artifacts, and redundant copies of data.

### Classification

#### 🟢 KEEP (Core Logic)
- **`src/santa2025/`**: The core Python package.
- **`scripts/`**: Essential tools (`merge_improvements.py`, `ensemble_cascade.py`, `exact_smalln.py`).
- **`configs/`**: Configuration files.
- **`docs/`**: Documentation.
- **`experiments/`**: Valid experiment tracking.

#### 🟡 MOVE (Refactor)
- **`kaggle/`**: Currently a mix of runners and dataset files.
    - -> `submissions/kernels/`: Clean kernel launch scripts.
    - -> `submissions/datasets/`: Dataset definitions.
- **`scripts/fetch_*.py`**: Scrapers.
    - -> `tools/scrapers/` or `scripts/utils/`.
- **`*.cpp`** (`scripts/single_group_optimizer.cpp`, `kaggle/sa_v1_parallel.cpp`):
    - -> `src/cpp/`: Centralize C++ extensions.

#### 🔴 DELETE (Junk/Temp)
- **`temp_check_solver/`**: Temporary extraction folder.
- **`temp_nctuan_code/`**: Temporary code inspection.
- **`logs_small/`**: Logs from debugging.
- **`kaggle_datasets/`, `kaggle_kernels/`, `kaggle_output/`**: Local Kaggle stages; should be gitignored or cleaned.
- **`public_datasets_extra*`**: Large binary data/CSVs. Add to `.gitignore` or move to `data/external`.
- **`results/`**: Output artifacts. Add to `.gitignore`.

---

## 2. Proposed Portfolio Structure

This structure emphasizes **Engineering Rigor** over Competition Speed.

```text
santa-2025-optimizer/
├── assets/                 # Images, diagrams for README
├── configs/                # Hydra/YAML configs
├── data/                   # Data directory (Gitignored)
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/                   # Documentation (Architecture, Algorithms)
├── notebooks/              # Jupyter notebooks for analysis (not code)
├── src/
│   └── santa2025/          # Core Python package
│       ├── optimization/   # SA, GA, Beam Search logic
│       ├── geometry/       # Polygon/Placement logic
│       └── utils/
├── src_cpp/                # C++ source code for performance critical parts
│   ├── sa_solver.cpp
│   └── ...
├── scripts/                # CLI entry points
│   ├── optimize.py
│   ├── merge.py
│   └── validate.py
├── submissions/            # Kaggle specific deployment
│   ├── kernels/            # Kernel scripts (e.g. combined_opt)
│   └── datasets/           # Dataset metadata
├── tests/                  # Unit tests (Critical for portfolio!)
├── .gitignore
├── README.md
├── pyproject.toml          # Modern packaging
└── setup.py                # Legacy packaging (optional)
```

## 3. Action Plan

1.  **Clean Root**: Delete `temp_*` and `logs_*`.
2.  **Centralize C++**: Move `.cpp` files to `src_cpp/`.
3.  **Organize Scripts**: Group `scripts/*.py` into logical subfolders (`optimization`, `utils`).
4.  **Isolate Kaggle Code**: Move `kaggle/` content to `submissions/`.
5.  **Standardize Inputs**: Ensure all data loading goes through `src/` or `data/`.
6.  **Create README**: Rewrite README to focus on the "2D Irregular Packing Problem" and the "Simulated Annealing + Beam Search" solution.

**Specific Question for User:**
- The directory `kaggle/` contains `run_bbox3.py` (28KB). Is this a third-party script or your code? If third-party, we should place it in `src/external/` or `vendor/`.
