# Anti-Plateau Diagnostics (2026-01-10)

## A) Validation / leakage
- No train/test split; this is a deterministic packing optimization over fixed geometry.
- Leakage risk is mainly format-related: any overlap or out-of-bounds yields invalid submission.
- Precision risk: rounding can introduce overlaps; prefer high precision during search, round only for final CSV.

## B) Stability
- Baseline with fixed seed is deterministic.
- Heuristic refiners (local search / SA) are seed-sensitive but show small gains; most runs plateau.
- Correlation to public is effectively 1.0 (single deterministic test), so overfitting risk is low.

## C) Error / drift analysis
- Per-group scoring shows broad contribution: top 40 groups ~21.8% of total score.
- Improvements must be spread across many n to move the needle.
- Small-n exact search helps but limited ceiling; large-n improvements dominate.

## Immediate bottlenecks
- Local search moves are too local; need global resets or new baselines.
- Pooling/cascade gains depend on diverse input submissions; homogeneous pool yields no improvement.
