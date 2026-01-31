# Strategy Notes

This competition is a sum over independent n-tree problems. That means:

- Each n can be optimized independently.
- The best final submission is the per-n best across all runs.
- Focus on n with the largest group_score first (s_n^2 / n).

Practical plan:

1) Build a stable baseline for all n (incremental greedy).
2) Run local search on the top K groups by contribution.
3) Repeat with different seeds and pool per-n best.
4) Track deltas per n to focus effort on the worst groups.
5) Move heavy search to Kaggle; keep local fast iteration.

Keep a pool of best per-n solutions and rebuild submissions using pool_best_of.py.
