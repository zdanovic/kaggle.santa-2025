#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import http.cookiejar
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.constants import COMPETITION_ID

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


def _sanitize(text: str | None) -> str | None:
    if text is None:
        return None
    return text.encode("ascii", errors="ignore").decode("ascii")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    opener, xsrf = _get_session()

    data = _api_post(
        opener,
        xsrf,
        "https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard",
        {"competitionId": COMPETITION_ID},
    )

    (DOCS / "leaderboard_full.json").write_text(json.dumps(data, indent=2))

    teams_by_id = {t["teamId"]: t for t in data.get("teams", [])}

    rows = []
    for row in data.get("publicLeaderboard", []):
        team = teams_by_id.get(row["teamId"], {})
        leader = None
        if team.get("teamUpInfo") and team["teamUpInfo"].get("teamLeader"):
            leader = team["teamUpInfo"]["teamLeader"].get("userName")
        rows.append(
            {
                "rank": row.get("rank"),
                "score": row.get("displayScore"),
                "team_id": row.get("teamId"),
                "team_name": _sanitize(team.get("teamName")),
                "leader": _sanitize(leader),
                "submission_id": row.get("submissionId"),
            }
        )

    top_rows = rows[:200]
    with (DOCS / "leaderboard_top.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "score", "team_id", "team_name", "leader", "submission_id"],
        )
        writer.writeheader()
        writer.writerows(top_rows)


if __name__ == "__main__":
    main()
