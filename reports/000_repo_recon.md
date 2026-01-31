# Repo Recon (Santa 2025)

Date: 2026-01-08

## Key Docs / Rules
- Rules: `docs/rules.md` (submission limit 100/day, external data allowed if reasonable).
- Evaluation: `docs/evaluation.md` (sum of per-n square box area / n; overlaps invalid; x,y in [-100,100]; `s` prefix required).
- Competition summary: `docs/competition_summary.md` (deadline 2026-01-30, 100/day, 2 final submissions).

## Repository Structure
- Core code: `src/santa2025/` (metric clone, geometry, solvers, pipeline).
- Scripts: `scripts/` (run_experiment, seed_sweep, pool_best_of, refine_submission, scoring).
- Configs: `configs/` (baseline/independent/pattern/periodic configs).
- Results: `results/` (per-run outputs, pooled best-of, submissions).
- Kaggle: `kaggle/` (kernels + dataset runner scripts for Kaggle compute).
- Logs/artifacts: `kaggle_output/`.

## Pipeline Map (Local)
1) `scripts/run_experiment.py` loads YAML config via `pipeline.load_config`.
2) Baseline solver (incremental / independent / pattern / periodic) generates placements for n=1..200.
3) Optional local search refine via `LocalSearchRefiner` for selected groups (top_k or explicit).
4) Outputs:
   - `submission.csv` (baseline) and `submission_refined.csv` (if enabled)
   - `per_n.csv` / `per_n_refined.csv`
   - `summary_baseline.json` / `summary_refined.json`
5) Pooling: `scripts/pool_best_of.py` merges per-n best from multiple runs into a single submission.

## Kaggle Compute
- Runner scripts in `kaggle/` (e.g., `run_all.py`, `run_bbox3_random.py`).
- Kernel metadata in `kaggle/kernel-metadata*.json`.
- Datasets and artifacts in `kaggle_output/`.

## Baseline Reproducibility
Command (baseline only, refine disabled):
```bash
./.venv/bin/python scripts/run_experiment.py --config configs/recon_baseline.yaml --output results/exp_recon_base_1
./.venv/bin/python scripts/run_experiment.py --config configs/recon_baseline.yaml --output results/exp_recon_base_2
```
Result: deterministic; both runs score `178.05997170699789` with identical settings.

## Experiment Tracking
- New log: `experiments/experiments.csv`.
- Per-run metadata: `results/*/summary_*.json` and `per_n*.csv`.

## Current Best Known Submission (Local Score)
- `kaggle_output/bbox3_random_long/bbox3_random/best_submission.csv`
  - Local score: `70.97093966199184`.
- `results/submissions/best_submission_public_pool.csv`
  - Local score: `70.97093974407339`.

## Bottlenecks / Risks
- Local search runtime grows quickly with higher steps/restarts and larger n.
- Rounding to 6 decimals can introduce overlaps; needs validation and fallback.
- Multi-process refinement can be nondeterministic if random seeds are not fixed per group.

