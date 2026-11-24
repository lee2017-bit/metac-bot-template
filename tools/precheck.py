"""
Cheap pre-check for the one-shot workflow: is there an open question in the
bot's tournaments that the bot has not forecast yet?

Standard library only (no dependency install), one or two API calls per
tournament. Writes run=true/false to $GITHUB_OUTPUT. Fails open: on any error
it answers run=true, so a broken pre-check never makes the bot miss questions.

Env: METACULUS_TOKEN, PRECHECK_TOURNAMENTS (comma-separated ids or slugs,
default "33121,minibench" = Fall 2026 + the current MiniBench).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://www.metaculus.com/api/posts/"


def open_unforecast(tournament: str, token: str) -> list[str]:
    missing: list[str] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {"tournaments": tournament, "statuses": "open", "limit": 100, "offset": offset}
        )
        req = urllib.request.Request(
            f"{API}?{params}",
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        for post in page.get("results", []):
            q = post.get("question")
            if not q:  # group or conditional posts: let the bot decide
                missing.append(f"{post.get('id')} (non-simple post)")
                continue
            latest = (q.get("my_forecasts") or {}).get("latest")
            if not latest:
                missing.append(str(post.get("id")))
        if not page.get("next"):
            return missing
        offset += 100


def main() -> int:
    out = os.getenv("GITHUB_OUTPUT")
    decision = "true"
    try:
        token = os.environ["METACULUS_TOKEN"]
        tournaments = [t.strip() for t in os.getenv("PRECHECK_TOURNAMENTS", "33121,minibench").split(",") if t.strip()]
        missing = {t: open_unforecast(t, token) for t in tournaments}
        n = sum(len(v) for v in missing.values())
        print(f"open questions without our forecast: {n} " + json.dumps({k: v[:10] for k, v in missing.items()}))
        decision = "true" if n else "false"
    except Exception as e:  # fail open
        print(f"pre-check failed ({type(e).__name__}: {e}); running the bot anyway")
    if out:
        with open(out, "a") as f:
            f.write(f"run={decision}\n")
    print(f"run={decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
