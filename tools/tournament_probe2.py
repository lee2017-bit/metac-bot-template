"""Read-only follow-up pull: past MiniBench rounds and revealed aggregates on
closed questions we forecast. No LLM calls, nothing posted."""
import json
import os
import re
import time

import requests

H = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
API = "https://www.metaculus.com/api"
out = {"errors": [], "past_minibench": [], "leaderboards": {}, "details": []}


def get(path, params=None, tries=6):
    r = None
    for i in range(tries):
        try:
            r = requests.get(API + path, headers=H, params=params, timeout=90)
        except Exception as e:
            out["errors"].append(f"{path}: {e!r}")
            time.sleep(5)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(8 * (i + 1))
            continue
        time.sleep(1.0)
        return r
    return r


cur = get("/projects/tournaments/minibench/")
cj = cur.json() if cur is not None and cur.status_code == 200 else {}
desc = cj.get("description") or ""
out["minibench_description_links"] = re.findall(r"https?://[^\s\"')<>]+|/tournament/[^\s\"')<>]+|/notebooks/[^\s\"')<>]+", desc)
out["minibench_keys"] = sorted(cj.keys())
print("links in MiniBench description:", out["minibench_description_links"])

# Past rounds are unlisted, so look at the project ids created between the
# Summer season project and the current MiniBench project.
keep = ("id", "name", "slug", "type", "prize_pool", "start_date", "close_date", "forecasting_end_date",
        "questions_count", "forecasts_count", "forecasters_count", "score_type")
for pid in range(33168, 33022, -1):
    r = get(f"/projects/tournaments/{pid}/")
    if r is None or r.status_code != 200:
        continue
    t = r.json()
    name = (t.get("name") or "") + " " + (t.get("slug") or "")
    if "minibench" in name.lower() or "mini bench" in name.lower():
        out["past_minibench"].append({k: t.get(k) for k in keep})
        print("found", {k: t.get(k) for k in ("id", "name", "slug", "start_date", "questions_count", "forecasters_count")})
print("past MiniBench rounds found:", len(out["past_minibench"]))

for t in out["past_minibench"]:
    r = get(f"/leaderboards/project/{t['id']}/", {"with_entries": "true"})
    if r is None or r.status_code != 200:
        out["errors"].append(f"leaderboard {t['id']}: {getattr(r, 'status_code', None)}")
        continue
    lbs = r.json()
    if isinstance(lbs, dict):
        lbs = [lbs]
    for lb in lbs:
        if not lb.get("is_primary_leaderboard"):
            continue
        entries = [{
            "username": (e.get("user") or {}).get("username"), "is_bot": (e.get("user") or {}).get("is_bot"),
            "aggregation_method": e.get("aggregation_method"), "rank": e.get("rank"), "score": e.get("score"),
            "coverage": e.get("coverage"), "contribution_count": e.get("contribution_count"),
            "prize": e.get("prize"), "excluded": e.get("excluded"), "take": e.get("take"),
            "percent_prize": e.get("percent_prize"),
        } for e in lb.get("entries") or []]
        out["leaderboards"][str(t["id"])] = {
            "meta": {k: v for k, v in lb.items() if k not in ("entries", "userEntry")}, "entries": entries}
        print("leaderboard", t["id"], t.get("start_date"), "entries", len(entries), "finalized", lb.get("finalized"),
              "prize winners", sum(1 for e in entries if e["prize"]))

# Revealed aggregates for closed questions we forecast (to compare with our own forecast).
probe1 = json.load(open("probe.json")) if os.path.exists("probe.json") else None
ids = [int(x) for x in os.environ.get("POST_IDS", "").split(",") if x.strip()]
for pid in ids:
    r = get(f"/posts/{pid}/", {"with_cp": "true"})
    if r is None or r.status_code != 200:
        out["errors"].append(f"post {pid}: {getattr(r, 'status_code', None)}")
        continue
    p = r.json()
    q = p.get("question") or {}
    if q.get("status") == "open":
        continue
    aggs = q.get("aggregations") or {}
    def latest(m):
        a = (aggs.get(m) or {}).get("latest") or {}
        return {k: a.get(k) for k in ("centers", "interval_lower_bounds", "interval_upper_bounds", "forecast_values",
                                       "forecaster_count", "means")} if a else None
    my = (q.get("my_forecasts") or {}).get("latest") or {}
    out["details"].append({
        "post_id": pid, "title": q.get("title"), "type": q.get("type"), "status": q.get("status"),
        "resolution": q.get("resolution"), "options": q.get("options"), "scaling": q.get("scaling"),
        "nr_forecasters": p.get("nr_forecasters"),
        "my_forecast_values": my.get("forecast_values"),
        "agg": {m: latest(m) for m in aggs},
    })
print("details:", len(out["details"]), "errors:", len(out["errors"]), out["errors"][:6])
json.dump(out, open("probe2.json", "w"))
print("bytes:", os.path.getsize("probe2.json"))
