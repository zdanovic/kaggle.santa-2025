#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPTS = [
    "fetch_competition_docs.py",
    "fetch_metric_notebook.py",
    "fetch_getting_started.py",
    "fetch_leaderboard.py",
    "fetch_forum_topics.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        subprocess.run(["python3", str(root / script)], check=True)


if __name__ == "__main__":
    main()
