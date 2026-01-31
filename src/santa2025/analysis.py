from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_per_n(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("n").reset_index(drop=True)


def compare_per_n(path_a: Path, path_b: Path) -> pd.DataFrame:
    df_a = load_per_n(path_a).rename(columns={"group_score": "score_a"})
    df_b = load_per_n(path_b).rename(columns={"group_score": "score_b"})
    df = df_a.merge(df_b, on="n", how="inner")
    df["delta"] = df["score_b"] - df["score_a"]
    return df.sort_values("delta", ascending=False).reset_index(drop=True)
