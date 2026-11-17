"""Read-only: count questions in the bot tournaments by status (no LLM calls)."""
import os
import requests

H = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
BASE = "https://www.metaculus.com/api/posts/"
for t in ["33121", "fall-futureeval-2026", "minibench"]:
    for st in ["upcoming", "open", "closed", "resolved"]:
        r = requests.get(BASE, headers=H, params={"tournaments": t, "statuses": st, "limit": 100, "order_by": "-published_at"}, timeout=60)
        if r.status_code != 200:
            print(t, st, "HTTP", r.status_code, r.text[:150]); continue
        js = r.json(); res = js.get("results", [])
        print(f"{t:22s} {st:9s} count={js.get('count', len(res))}")
        for p in res[:3]:
            q = p.get("question") or {}
            print("    ", p.get("id"), (p.get("title") or "")[:70], "| open", q.get("open_time"), "| close", q.get("scheduled_close_time"))
