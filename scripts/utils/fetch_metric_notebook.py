#!/usr/bin/env python3
from __future__ import annotations

import json
import http.cookiejar
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.constants import METRIC_NOTEBOOK_REF

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _get_session():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.open("https://www.kaggle.com").read()
    return opener


def _sanitize(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    opener = _get_session()

    url = f"https://www.kaggle.com/api/v1/kernels/pull/{METRIC_NOTEBOOK_REF}"
    payload = json.loads(opener.open(url).read().decode("utf-8"))
    source = payload["blob"]["source"]

    nb = json.loads(source)
    (DOCS / "metric_notebook.ipynb").write_text(source)

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    code = "\n\n".join("".join(c.get("source", "")) for c in code_cells)
    (DOCS / "metric_raw.py").write_text(_sanitize(code))


if __name__ == "__main__":
    main()
