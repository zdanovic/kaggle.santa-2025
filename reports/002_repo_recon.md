# Repo Recon (2026-01-10)

## What this repo does
- Optimization challenge: place 1..200 trees to minimize sum of squared bounding box side / n.
- Deterministic scoring with strict overlap checks and bounds (-100..100).

## Key docs and rules
- Rules: `docs/rules.md` (submission limit 100/day, 2 final submissions).
- Metric: `docs/evaluation.md` + `src/santa2025/metric.py`.
- Data: `docs/data_description.md` (no train/test split; algorithmic packing).

## Pipeline map
- Input: tree geometry (fixed points in `src/santa2025/geometry.py`).
- Baseline solver: `scripts/run_experiment.py` -> `src/santa2025/pipeline.py`.
  - Modes: incremental/independent/pattern/periodic.
- Optional refine: local search per-group via `LocalSearchRefiner`.
- Output: `submission.csv` (baseline) and `submission_refined.csv`.
- Analysis: per-group scores in `per_n.csv`, `scripts/analyze_targets.py`.
- Pooling: `scripts/pool_best_of.py` and `scripts/ensemble_cascade.py`.
- Kaggle compute: `kaggle/run_*.py` + `kaggle/README.md`.

## Baseline reproducibility
- Command:
  - `santa-2025/.venv/bin/python santa-2025/scripts/run_experiment.py --config santa-2025/configs/recon_baseline.yaml --output santa-2025/results/exp_recon_v2_1`
  - Re-run same seed: `... --output santa-2025/results/exp_recon_v2_2`
- Result: deterministic (score 178.05997170699789 for both runs).

## Experiment tracking
- Log: `experiments/experiments.csv`.
- Best known local score: 70.78246837965895 (`results/submissions/public_best_submission_70_782.csv`).

## Models/approaches already tried
- Greedy incremental baseline.
- Independent, pattern, periodic lattice baselines (configs under `configs/`).
- Local search refinement (per-group SA).
- C++ SA (`kaggle/sa_v1_parallel.cpp`) and group-level optimizer (`scripts/single_group_optimizer.cpp`).
- Bbox3 random search on Kaggle.
- Best-of pooling across submissions and backward + deletion cascade (`scripts/ensemble_cascade.py`).

## Bottlenecks / risks
- Local search often stalls (local minima); micro gains only.
- Overlap failures can happen after rounding; need high precision (decimals=18) before final rounding.
- Heavy compute for bbox3/random and CP-SAT exact small-n.
- Public/private risk is low (same deterministic test), but avoid over-tuning to public-only heuristics.
