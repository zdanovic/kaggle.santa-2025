#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List


REQUIRED_COLUMNS = {"id", "x", "y", "deg"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "submission"


def _has_required_header(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
    except Exception:
        return False
    normalized = {h.strip().lower() for h in header}
    return REQUIRED_COLUMNS.issubset(normalized)


def _collect_csvs(paths: Iterable[Path]) -> List[Path]:
    candidates: List[Path] = []
    for base in paths:
        if base.is_file() and base.suffix.lower() == ".csv":
            candidates.append(base)
            continue
        if not base.exists():
            continue
        candidates.extend(p for p in base.rglob("*.csv") if p.is_file())
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        default="public_datasets,kaggle_output/public_kernels",
        help="Comma-separated list of directories or CSVs to scan (relative to repo root).",
    )
    parser.add_argument(
        "--include",
        default="results/submissions/best_submission.csv",
        help="Comma-separated list of extra CSVs to force-include (relative to repo root).",
    )
    parser.add_argument(
        "--out-dir",
        default="public_pool_dataset",
        help="Output directory (relative to repo root).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    inputs = [
        (root / p.strip()).resolve()
        for p in args.inputs.split(",")
        if p.strip()
    ]
    includes = [
        (root / p.strip()).resolve()
        for p in args.include.split(",")
        if p.strip()
    ]
    out_dir = (root / args.out_dir).resolve()
    submissions_dir = out_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    candidates = _collect_csvs(inputs + includes)
    sources: dict[str, str] = {}
    used: dict[str, int] = {}
    kept = 0

    for csv_path in candidates:
        if not _has_required_header(csv_path):
            continue
        try:
            rel = csv_path.resolve().relative_to(root)
        except ValueError:
            rel = Path(csv_path.name)
        slug = _slugify(rel.as_posix().rsplit(".", 1)[0])
        if slug in used:
            used[slug] += 1
            slug = f"{slug}_{used[slug]}"
        else:
            used[slug] = 1

        target_dir = submissions_dir / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "submission.csv"
        shutil.copy2(csv_path, target_path)
        sources[slug] = str(csv_path)
        kept += 1

    sources_path = out_dir / "sources.json"
    sources_path.write_text(json.dumps(sources, indent=2))
    print(f"Collected {kept} submissions into {out_dir}")


if __name__ == "__main__":
    main()
