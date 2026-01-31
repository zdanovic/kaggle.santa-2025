from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from santa2025.metric import score_detailed


def per_group_scores(submission: pd.DataFrame) -> Dict[int, float]:
    _, per_group = score_detailed(submission)
    return per_group


def per_group_dataframe(per_group: Dict[int, float]) -> pd.DataFrame:
    rows = [(n, score) for n, score in per_group.items()]
    df = pd.DataFrame(rows, columns=["n", "group_score"])
    return df.sort_values("group_score", ascending=False).reset_index(drop=True)


def top_groups(per_group: Dict[int, float], top_k: int) -> List[int]:
    df = per_group_dataframe(per_group)
    return df.head(top_k)["n"].tolist()
