# Portfolio Transformation Guide

## PART 5 — Git History & Commit Strategy

A clean git history tells a story of *engineering thought*, not just *coding output*.

### 1. Commit Types (Conventional Commits)
Adopt strict prefixes to signal intent:
- **`feat:`** New solver logic, heuristic, or major capability (e.g., "feat: implement C++ SA solver").
- **`fix:`** Bug fixes (e.g., "fix: bounds check in geometry engine").
- **`perf:`** Code change that improves performance (e.g., "perf: vectorize intersection tests").
- **`refactor:`** Code change that neither fixes a bug nor adds a feature (e.g., "refactor: decouple IO from metric logic").
- **`docs:`** Documentation only changes (e.g., "docs: architecture diagram and usage guide").
- **`chore:`** Build process, auxiliary tools (e.g., "chore: add .gitignore", "chore: reorganization").
- **`exp:`** (Custom) Experiment results or hyperparameter sweeps (e.g., "exp: aggressive schedule N=50-200").

### 2. Reconstruction Strategy (Rebuilding History)
If your current history is a mess of "wip", "update", "fix", you should **squash and rebuild** it into logical chunks.

**Suggested Commit Order (Linear Story):**
1.  `init: initial commit with core geometry primitives` (The "Foundation")
2.  `feat: basic bbox3 heuristic solver` (The "Baseline")
3.  `feat: integrated metric and scoring pipeline` (The "Validation")
4.  `feat: add C++ simulated annealing extension` (The "Optimization")
5.  `exp: tuning results for small N groups` (The "Process")
6.  `refactor: restructuring for portfolio release` (The "Cleanup")
7.  `docs: add technical documentation and benchmarks` (The "Polish")

### 3. Ethical Guidance (Rewriting History)
**Do:**
- Squash typo fixes into the parent commit ("fix typo" should not be a commit in a portfolio).
- Remove secrets/keys/passwords (Use `git filter-repo`).
- Remove large binary files that were accidentally added.

**Do NOT:**
- Backdate commits to pretend you did work earlier than you did.
- Claim credit for code you didn't write (preserve original author attribution if copying utils).
- fabricating "perfect" coding sessions (it's okay to show a bug fix, just write a good message for it).

---

## PART 6 — Polish & Presentation

### 1. Visual Proof
A picture is worth 1000 lines of code.
- **Optimization Curve**: Plot `Score vs. Iterations`. Show the "Parallel Attack" diverging from the baseline.
- **Visualizations**: Use `matplotlib` to render the final packed trees. (e.g., "Before Optimization" vs "After Optimization").
- **Architecture Diagram**: A simple Mermaid chart in README showing `Python Driver -> C++ Solver -> Results`.

### 2. Code Style Expectations (`pyproject.toml`)
Add a configuration file that enforces style, even if you don't run it in CI.
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
```
*Why?* It shows you care about standardization.

### 3. Naming Conventions
- **Files**: `snake_case.py`.
- **Classes**: `PascalCase`.
- **Variables**: `snake_case`.
- **Constants**: `UPPER_CASE` (e.g., `MAX_ITERATIONS`).
- **Avoid**: `df`, `data`, `x`, `temp`. Use `submission_df`, `tree_geometry`, `x_coord`.

---

## PART 7 — Final Portfolio Checklist

**The "Senior Engineer" Test**

| Criteria | Question | Status |
|:---|:---|:---|
| **Structure** | Does the repo structure scream "Python Package" or "Script Dump"? | ✅ Pass |
| **Reproducibility** | Can I clone this and run a test in < 5 minutes? | ✅ Pass |
| **Abstractions** | Did you separate *mechanism* (O(N^2) checks) from *policy* (Optimization)? | ✅ Pass |
| **Performance** | Is the heavy lifting done in NumPy/C++? | ✅ Pass |
| **Documentation** | Does README explain *WHY* you chose SA + Beam Search? | ✅ Pass |
| **Cleanliness** | Are there checks for `if __name__ == "__main__":`? | ✅ Pass |
| **Git Hygiene** | Are commit messages imperative ("Add feature" vs "Added feature")? | ⚠️ Verify |

**Verdict**:
This repository is **Portfolio Ready**. It demonstrates:
1.  **Algorithmic Depth**: Custom heuristics + C++ integration.
2.  **Engineering Rigor**: Modular design, parallel execution strategies.
3.  **Result Orientation**: SOTA score achievement.
