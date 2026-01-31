from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class TreePlacement:
    x: float
    y: float
    deg: float


def _format_value(value: float, decimals: int) -> str:
    return f"s{value:.{decimals}f}"


def build_submission(
    groups: Dict[int, List[TreePlacement]],
    decimals: int = 6,
) -> pd.DataFrame:
    rows: List[Tuple[str, str, str, str]] = []
    for n in sorted(groups.keys()):
        placements = groups[n]
        for idx, placement in enumerate(placements):
            row_id = f"{n:03d}_{idx}"
            rows.append(
                (
                    row_id,
                    _format_value(placement.x, decimals),
                    _format_value(placement.y, decimals),
                    _format_value(placement.deg, decimals),
                )
            )
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def write_submission_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_submission_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def groups_from_submission(df: pd.DataFrame) -> Dict[int, List[TreePlacement]]:
    df = df.copy()
    df["group"] = df["id"].astype(str).str.split("_").str[0].astype(int)
    df["idx"] = df["id"].astype(str).str.split("_").str[1].astype(int)
    df = df.sort_values(["group", "idx"]).reset_index(drop=True)

    groups: Dict[int, List[TreePlacement]] = {}
    for n, df_group in df.groupby("group"):
        placements: List[TreePlacement] = []
        for _, row in df_group.iterrows():
            x = float(str(row["x"]).lstrip("s"))
            y = float(str(row["y"]).lstrip("s"))
            deg = float(str(row["deg"]).lstrip("s"))
            placements.append(TreePlacement(x=x, y=y, deg=deg))
        groups[int(n)] = placements
    return groups
