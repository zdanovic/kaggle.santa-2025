# Diagnostics (Santa 2025)

Date: 2026-01-08

## 1) Metric + Constraints
- Metric: sum over n=1..200 of `(s_n^2 / n)` where `s_n` is the side length of the minimal axis-aligned square bounding box.
- Hard constraints:
  - Trees may touch but must not overlap.
  - Coordinates must be within [-100, 100].
  - Submission values must be strings prefixed with `s` (e.g., `s0.123456`).
- Implication: improvements are deterministic geometry optimizations per n; no train/test leakage risk.

## 2) What Drives Score Today
Top contributors in pooled best-per-n (`results/best_per_n.csv`) are **small n**:
`n = 1..15` dominate the total.
This implies:
- Small-n improvements have disproportionate impact on total score.
- Large-n improvements still matter, but marginal gains are smaller per n.

## 3) Stability + Determinism
- Baseline deterministic with fixed seed (verified).
- Local search relies on RNG; to be reproducible, fix seed per group.
- Rounding to 6 decimals can introduce overlaps; must re-check collisions after rounding and fallback if needed.

## 4) Leakage / Split / CV
Not applicable: this is an optimization challenge with a fixed geometry and no labels.
Public/private split affects leaderboard score only; improvements should generalize by construction.

## 5) Error Analysis (Practical)
- Use `per_n.csv` to identify the worst groups and focus search there.
- Prioritize small-n exact or near-exact solutions (n<=20).

## 6) Observed Regimes (Current Artifacts)
- Baseline (incremental greedy): ~178.06.
- Best pooled submission: ~70.97094.
- Periodic/pattern baselines are weaker than the current best.

## 7) Prioritized Improvement Levers
1) **Small-n exact / high-precision search** (n<=20), with overlap-safe rounding.
2) **Diverse per-n pooling** from different solvers (bbox3/SA/periodic), then best-of.
3) **Targeted local search** only where `per_n` contributions are largest (top-k).

