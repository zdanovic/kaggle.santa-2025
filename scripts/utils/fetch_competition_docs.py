#!/usr/bin/env python3
from __future__ import annotations

import json
import http.cookiejar
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from santa2025.constants import COMPETITION_ID, COMPETITION_SLUG


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


def _sanitize(text: str) -> str:
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2014": "-",
        "\u2013": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", errors="ignore").decode("ascii")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    opener, xsrf = _get_session()

    comp = _api_post(
        opener,
        xsrf,
        "https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition",
        {"competitionName": COMPETITION_SLUG},
    )

    pages = _api_post(
        opener,
        xsrf,
        "https://www.kaggle.com/api/i/competitions.PageService/ListPages",
        {"competitionId": COMPETITION_ID},
    )

    (DOCS / "competition_context.json").write_text(
        json.dumps({"competition": comp, "pages": pages}, indent=2)
    )

    page_map = {p["name"]: p["content"] for p in pages.get("pages", [])}

    def write_page(name: str, filename: str) -> None:
        content = _sanitize(page_map.get(name, ""))
        (DOCS / filename).write_text(content)

    write_page("Description", "competition_overview.md")
    write_page("Evaluation", "evaluation.md")
    write_page("data-description", "data_description.md")
    write_page("Timeline", "timeline.md")
    write_page("Prizes", "prizes.md")
    write_page("rules", "rules.md")

    summary_lines = [
        f"id: {comp.get('id')}",
        f"slug: {comp.get('competitionName')}",
        f"title: {comp.get('title')}",
        f"brief: {comp.get('briefDescription')}",
        f"deadline: {comp.get('deadline')}",
        f"max_daily_submissions: {comp.get('maxDailySubmissions')}",
        f"num_scored_submissions: {comp.get('numScoredSubmissions')}",
        f"max_team_size: {comp.get('maxTeamSize')}",
        f"requires_identity_verification: {comp.get('requiresIdentityVerification')}",
        f"row_id_column: {comp.get('rowIdColumnName')}",
        f"total_solution_rows: {comp.get('totalSolutionRows')}",
    ]
    (DOCS / "competition_summary.md").write_text("\n".join(summary_lines))


if __name__ == "__main__":
    main()
