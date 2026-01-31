#!/usr/bin/env python3
from __future__ import annotations

import json
import http.cookiejar
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.constants import FORUM_ID

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _get_session():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.open("https://www.kaggle.com").read()
    xsrf = None
    for c in cj:
        if c.name == "XSRF-TOKEN":
            xsrf = c.value
            break
    return opener, xsrf


def _api_post(opener, xsrf, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(path, data=body)
    req.add_header("Content-Type", "application/json")
    if xsrf:
        req.add_header("X-XSRF-TOKEN", xsrf)
    return json.loads(opener.open(req).read().decode("utf-8"))


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    opener, xsrf = _get_session()

    data = _api_post(
        opener,
        xsrf,
        "https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId",
        {"forumId": FORUM_ID},
    )

    (DOCS / "forum_topics.json").write_text(json.dumps(data, indent=2))

    notes_path = DOCS / "community_notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Community Notes\n\n"
            "Use this file to summarize forum insights and top-kernel ideas.\n"
        )


if __name__ == "__main__":
    main()
