# Refactoring Strategy: Santa 2025 Optimization

## 1. Architectural Smells & Analysis

### 🔻 Code Duplication ("Script Cloning")
**Observation**: The repository contains multiple similar scripts (`run_bbox3_aggressive.py`, `run_bbox3_focused.py`, `run_bbox3_random.py`).
**Smell**: "Copy-Paste Programming". Changes to the core logic (e.g., how the metric is calculated) must be manually propagated to 5+ files.
**Solution**: Unify into a single `scripts/optimization/run_solver.py` with a strategy pattern or configuration injection.

### 🔻 Hardcoded Environments ("The Kaggle Path Problem")
**Observation**: Scripts frequently check for `/kaggle/input/...` or fallback to relative local paths with `if/else` blocks scattered throughout the code.
**Smell**: "Environment Coupling". The code "knows" too much about where it's running.
**Solution**: Introduce a `PathsConfig` object or environment variable handling (e.g., `DATA_ROOT`) processed once at entry point.

### 🔻 Implicit C++ Compilation
**Observation**: C++ sources (`src_cpp/`) are compiled via ad-hoc subprocess calls or presumed to exist as binaries.
**Smell**: "Flaky Build System". Reproducibility depends on the user manually running the right compile command.
**Solution**: Use a standard `setup.py` with `Extension` modules or a `Makefile` that handles compilation dependencies automatically.

## 2. Refactoring Plan

### Phase A: Modularize the Solver
**Goal**: converting scripts into reusable modules.

**Current**:
```python
# run_bbox3_aggressive.py
def optimize(...):
    # ... lots of logic ...
if __name__ == "__main__":
    optimize()
```

**Proposed**:
```python
# src/santa2025/solvers/bbox3.py
class BBox3Solver:
    def __init__(self, config: Dict): ...
    def solve(self, data): ...

# scripts/optimization/run_solver.py
def main():
    solver = SolverFactory.create(args.strategy)
    solver.solve()
```

### Phase B: Configuration Management
**Goal**: Move from hardcoded parameters to YAML/Hydra.

**Current**:
```python
# Inside script
ITERATIONS = 500000
RESTARTS = 20
```

**Proposed**:
```yaml
# configs/experiments/aggressive.yaml
bbox3:
  iterations: 500000
  restarts: 20
  strategy: "aggressive"
```
Use `hydra` or `argparse` to load these configurations dynamically.

### Phase C: Unified Data Interface
**Goal**: Abstract file I/O.

Create a `DataManager` class in `src/santa2025/io.py` that handles:
- Loading the correct baseline (SOTA vs fresh).
- Parsing raw CSVs into internal geometry objects.
- Saving submissions with standardized naming conventions (`algorithm_N_timestamp.csv`).

## 3. Trade-offs

| Improvement | Pros | Cons |
|:---|:---|:---|
| **Class-based Solvers** | Testability, Reusability, Clean Imports | Slight boilerplate overhead |
| **Hydra Configs** | Experiment tracking, reproducibility | Learning curve for simple scripts |
| **C++ Python Bindings** | High performance, seamless integration | Complex build capability requirements |

## 4. Immediate Next Steps (Low Effort, High Rewards)
1.  **Consolidate `run_bbox3_*.py`**: Create one master script with `--mode [aggressive|focused|random]` flag.
2.  **Extract `Classes`**: Ensure `ChristmasTree` and `Metric` are pure library code (already mostly done in `src/`), but verify no logic leaks into scripts.
3.  **Add Unit Tests**: Add a simple test suite for the `Metric` calculation to prevent regressions during refactoring.
