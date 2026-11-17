"""Read-only data pull for strategy review (no LLM calls, no forecasts posted).

Collects: tournament metadata, leaderboards (current + past MiniBench rounds +
recent seasons), and per-question timing / our own forecast / revealed
community aggregate for the two bot tournaments. Writes probe.json.
"""
import json
import os
import time

import requests

H = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
API = "https://www.metaculus.com/api"
out = {"errors": []}


def get(path, params=None, tries=6):
    r = None
    for i in range(tries):
        try:
            r = requests.get(API + path, headers=H, params=params, timeout=90)
        except Exception as e:  # network hiccup
            out["errors"].append(f"{path} {params}: {e!r}")
            time.sleep(5)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(8 * (i + 1))
            continue
        time.sleep(1.5)
        return r
    return r


def js(path, params=None):
    r = get(path, params)
    if r is None or r.status_code != 200:
        out["errors"].append(f"{path} {params}: HTTP {getattr(r, 'status_code', None)} {getattr(r, 'text', '')[:120]}")
        return None
    return r.json()


me = js("/users/me") or {}
out["me"] = {k: me.get(k) for k in ("id", "username", "is_bot")}

# ---- tournaments ----------------------------------------------------------
tlist = js("/projects/tournaments/") or []
if isinstance(tlist, dict):
    tlist = tlist.get("results", [])
keep_keys = ("id", "name", "slug", "type", "prize_pool", "start_date", "close_date",
             "forecasting_end_date", "questions_count", "forecasts_count", "is_ongoing",
             "score_type", "forecasters_count")
rel = [t for t in tlist if any(s in ((t.get("slug") or "") + " " + (t.get("name") or "")).lower()
                               for s in ("minibench", "futureeval", "aib", "ai benchmark", "bot"))]
out["tournaments"] = [{k: t.get(k) for k in keep_keys} for t in rel]
print("tournaments listed:", len(tlist), "relevant:", len(rel))

cur_mb = js("/projects/tournaments/minibench/") or {}
out["current_minibench"] = {k: cur_mb.get(k) for k in keep_keys}
fall = js("/projects/tournaments/fall-futureeval-2026/") or {}
out["fall"] = {k: fall.get(k) for k in keep_keys}

mb_rounds = sorted([t for t in rel if "minibench" in ((t.get("slug") or "") + (t.get("name") or "")).lower()],
                   key=lambda t: t.get("start_date") or "")
ids = [33121, 33022, 32916, 32813]
if cur_mb.get("id"):
    ids.append(cur_mb["id"])
ids += [t["id"] for t in mb_rounds[-12:]]
ids = list(dict.fromkeys(ids))

# ---- leaderboards ---------------------------------------------------------
out["leaderboards"] = {}
for pid in ids:
    lbs = js(f"/leaderboards/project/{pid}/", {"with_entries": "true"})
    if not lbs:
        continue
    if isinstance(lbs, dict):
        lbs = [lbs]
    keep = []
    for lb in lbs:
        entries = []
        for e in lb.get("entries") or []:
            u = e.get("user") or {}
            entries.append({
                "username": u.get("username"), "is_bot": u.get("is_bot"), "user_id": u.get("id"),
                "aggregation_method": e.get("aggregation_method"),
                "rank": e.get("rank"), "score": e.get("score"), "coverage": e.get("coverage"),
                "contribution_count": e.get("contribution_count"), "prize": e.get("prize"),
                "medal": e.get("medal"), "excluded": e.get("excluded"),
                "take": e.get("take"), "percent_prize": e.get("percent_prize"),
            })
        meta = {k: v for k, v in lb.items() if k not in ("entries", "userEntry")}
        ue = lb.get("userEntry")
        keep.append({"meta": meta, "entries": entries, "user_entry": ue})
    out["leaderboards"][str(pid)] = keep
    print("leaderboard", pid, [(k["meta"].get("score_type"), len(k["entries"]), k["meta"].get("finalized")) for k in keep])

# ---- questions ------------------------------------------------------------
def trim_latest(d):
    if not d:
        return None
    return {k: d.get(k) for k in ("start_time", "end_time", "forecast_values", "probability_yes",
                                   "probability_yes_per_category", "continuous_cdf", "centers",
                                   "interval_lower_bounds", "interval_upper_bounds", "forecaster_count")}


out["questions"] = {}
for tname in ("33121", "minibench"):
    rows = []
    for st in ("upcoming", "open", "closed", "resolved"):
        offset = 0
        while True:
            page = js("/posts/", {"tournaments": tname, "statuses": st, "limit": 100, "offset": offset,
                                  "with_cp": "true", "order_by": "-published_at"})
            if not page:
                break
            res = page.get("results", [])
            for p in res:
                q = p.get("question")
                if not q:
                    rows.append({"post_id": p.get("id"), "title": p.get("title"), "status": st, "kind": "non-question"})
                    continue
                my = q.get("my_forecasts") or {}
                aggs = q.get("aggregations") or {}
                rows.append({
                    "post_id": p.get("id"), "question_id": q.get("id"), "title": q.get("title") or p.get("title"),
                    "status": st, "type": q.get("type"), "open_time": q.get("open_time"),
                    "scheduled_close_time": q.get("scheduled_close_time"), "actual_close_time": q.get("actual_close_time"),
                    "scheduled_resolve_time": q.get("scheduled_resolve_time"), "actual_resolve_time": q.get("actual_resolve_time"),
                    "cp_reveal_time": q.get("cp_reveal_time"), "resolution": q.get("resolution"),
                    "question_weight": q.get("question_weight"), "scaling": q.get("scaling"),
                    "options": q.get("options"), "nr_forecasters": p.get("nr_forecasters"),
                    # Our own forecast is left out for questions that are still open.
                    "my_latest": None if st == "open" else trim_latest(my.get("latest")),
                    "my_forecast_count": len(my.get("history") or []),
                    "my_score_data": my.get("score_data"),
                    "aggregations_latest": None if st == "open" else {
                        m: trim_latest((aggs.get(m) or {}).get("latest")) for m in aggs
                    },
                })
            offset += len(res)
            if len(res) < 100 or offset >= 1000:
                break
    out["questions"][tname] = rows
    print(tname, "questions:", len(rows), {s: sum(1 for r in rows if r["status"] == s) for s in ("upcoming", "open", "closed", "resolved")},
          "forecast by us:", sum(1 for r in rows if r.get("my_forecast_count")))

json.dump(out, open("probe.json", "w"))
print("errors:", len(out["errors"]), out["errors"][:8])
print("bytes:", os.path.getsize("probe.json"))
