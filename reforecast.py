"""
Post-close re-forecasting experiment.

For questions the live bot has already forecast and that have CLOSED, run the
forecasting step again with other models, on the research the bot saved when it
first forecast the question. Nothing is posted. Once a question resolves, every
variant is scored on it, so forecasting set-ups can be compared on the same
questions and the same evidence.

Why this design:
- No leakage: the research is the text saved at the time (the forecasters do
  not browse), and the prompts carry that original date.
- No rule issues: only closed questions are used (their forecasts can no
  longer change) and nothing is published.
- Cheap: research, the expensive part, is not repeated.

Commands:
  reforecast.py run    [--reports DIR] [--max-questions N]
  reforecast.py score  (fetch resolutions, write the summary)

State lives in REFORECAST_STATE (default reforecast/results.jsonl): one JSON
line per sample, plus resolution lines. Spend uses spend_guard with its own
ledger and cap (set SPEND_LEDGER / SPEND_CAP_USD / RUN_CAP_USD in the workflow).
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import io
import json
import logging
import math
import os
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger("reforecast")

STATE = Path(os.getenv("REFORECAST_STATE", "reforecast/results.jsonl"))
SUMMARY = Path(os.getenv("REFORECAST_SUMMARY", "reforecast/summary.md"))
# name=model:samples, comma separated
CONFIGS = os.getenv(
    "REFORECAST_CONFIGS",
    "kimi-k3=moonshotai/Kimi-K3:3,kimi-k3.1-pro=moonshotai/Kimi-K3.1-Pro:3,"
    "deepseek-flash=deepseek-ai/DeepSeek-V4.1-Flash-1017:3",
)
TEST_SLUGS = {"bot-testing-area"}


PROVIDERS = ("oc", "claude", "claude-noresearch")
NO_RESEARCH = (
    "No research report is available for this question. Forecast from the question text, its "
    "background and resolution criteria, and your own knowledge up to today's date, and say that "
    "you had no research."
)


def parse_configs(text: str) -> dict[str, tuple[str, str, int]]:
    """name=[provider|]model:samples, comma separated. provider: oc (default,
    OpenAI-compatible/Nebius), claude, or claude-noresearch (Claude without the
    saved research)."""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, rest = part.split("=", 1)
        provider, _, rest = rest.rpartition("|") if "|" in rest else ("oc", "", rest)
        model, _, n = rest.rpartition(":")
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r} in {part!r}")
        out[name.strip()] = (provider, model.strip(), int(n))
    return out


# ------------------------------------------------------------------ state


def load_state() -> list[dict]:
    if not STATE.exists():
        return []
    return [json.loads(line) for line in STATE.read_text().splitlines() if line.strip()]


def append_state(rows: list[dict]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ reports


def download_report_artifacts(dest: Path) -> None:
    """Fetch every reports-* artifact of this repository (needs GITHUB_TOKEN
    with actions: read). Test Bot artifacts are included but their questions
    are skipped later (bot-testing-area)."""
    repo = os.environ["GITHUB_REPOSITORY"]
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}
    page = 1
    while True:
        r = requests.get(
            f"https://api.github.com/repos/{repo}/actions/artifacts",
            params={"per_page": 100, "page": page},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        arts = r.json().get("artifacts", [])
        if not arts:
            break
        for a in arts:
            if a.get("expired") or not a["name"].startswith("reports-"):
                continue
            out = dest / a["name"]
            if out.exists():
                continue
            z = requests.get(a["archive_download_url"], headers=headers, timeout=60)
            if z.status_code != 200:
                logger.warning(f"could not download {a['name']}: {z.status_code}")
                continue
            out.mkdir(parents=True)
            zipfile.ZipFile(io.BytesIO(z.content)).extractall(out)
        page += 1


def collect_reports(folder: Path) -> dict[int, dict]:
    """One entry per question: the earliest saved report (the one that was posted)."""
    from forecasting_tools import DataOrganizer

    by_q: dict[int, dict] = {}
    for f in sorted(glob.glob(str(folder / "**" / "*.json"), recursive=True)):
        try:
            reports = DataOrganizer.load_reports_from_file_path(f)
        except Exception as e:
            logger.info(f"skip {f}: {type(e).__name__}")
            continue
        for rep in reports:
            q = rep.question
            if set(q.tournament_slugs or []) & TEST_SLUGS:
                continue
            qid = q.id_of_question
            accessed = q.date_accessed or datetime.now(timezone.utc)
            if qid in by_q and by_q[qid]["accessed"] <= accessed:
                continue
            research = extract_research(rep.explanation)
            if not research:
                continue
            by_q[qid] = {"report": rep, "accessed": accessed, "research": research}
    return by_q


def extract_research(explanation: str) -> str:
    start = explanation.find("## Report 1 Research")
    end = explanation.find("# FORECASTS")
    if start < 0:
        return ""
    return explanation[start + len("## Report 1 Research") : end if end > start else None].strip()


def posted_model(explanation: str) -> str:
    """The forecasting model named in the report's LLM line ('' if unknown)."""
    marker = "'default': {'original_model': '"
    i = explanation.find(marker)
    if i < 0:
        return ""
    return explanation[i + len(marker) :].split("'", 1)[0].removeprefix("openai/")


# ------------------------------------------------------------------ predictions


def prediction_to_json(pred) -> object:
    if isinstance(pred, (int, float)):
        return float(pred)
    if hasattr(pred, "predicted_options"):
        return {o.option_name: float(o.probability) for o in pred.predicted_options}
    if hasattr(pred, "get_cdf"):
        return {"cdf": [float(p.percentile) for p in pred.get_cdf()]}
    return str(pred)


class _FrozenDatetime(datetime):
    as_of: datetime | None = None

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        t = cls.as_of or datetime.now(timezone.utc)
        return t.astimezone(tz) if tz else t.replace(tzinfo=None)


async def run(reports_dir: Path, max_questions: int) -> None:
    os.environ.setdefault("LLM_PROVIDER", "openai_compat")
    import main  # the live bot's prompts and parsing, unchanged
    from claude_llm import current_model
    from forecasting_tools import MetaculusApi, MonetaryCostManager
    from forecasting_tools.ai_models.resource_managers.hard_limit_manager import HardLimitExceededError
    from forecasting_tools import GeneralLlm
    from claude_llm import ClaudeForecastLlm
    from openai_compat import OpenAICompatForecastLlm, parser_llm
    from spend_guard import record_run, remaining_budget

    # Nothing in this experiment may publish anything.
    def _refuse(*a, **k):
        raise RuntimeError("reforecast must never post to Metaculus")

    for name in dir(MetaculusApi):
        if name.startswith("post_"):
            setattr(MetaculusApi, name, staticmethod(_refuse))

    budget = remaining_budget()
    configs = parse_configs(CONFIGS)
    # a config counts as done for a question once it has a usable sample
    done = {(r["qid"], r["config"]) for r in load_state()
            if r.get("kind") == "sample" and (r.get("prediction") is not None or r.get("reused_posted"))}
    by_q = collect_reports(reports_dir)
    now = datetime.now(timezone.utc)
    # REFORECAST_SCOPE=state: only questions already in the experiment (so a new
    # set of configs runs on exactly the same questions). REFORECAST_INCLUDE_OPEN=1:
    # also open non-MiniBench questions; their samples are stored for scoring
    # later and never posted or used by the live bot.
    scope_state = os.getenv("REFORECAST_SCOPE", "") == "state"
    include_open = os.getenv("REFORECAST_INCLUDE_OPEN", "0") == "1"
    in_state = {r["qid"] for r in load_state() if r.get("kind") == "sample"}
    stop_at = os.getenv("REFORECAST_STOP_AT", "").strip()
    stop_at = datetime.fromisoformat(stop_at.replace("Z", "+00:00")) if stop_at else None
    todo = []
    for qid, item in by_q.items():
        q = item["report"].question
        is_open = q.close_time is None or q.close_time > now
        if is_open:
            if not include_open or "minibench" in " ".join(q.tournament_slugs or []).lower():
                continue  # still open: not touched unless asked for
        elif scope_state and qid not in in_state:
            continue
        missing = [c for c in configs if (qid, c) not in done]
        if missing:
            todo.append((is_open, q.close_time or now, qid, missing))
    # closed first (they resolve soonest), then binary and multiple choice, oldest first
    order = {"binary": 0, "multiple_choice": 1}
    todo.sort(key=lambda t: (t[0], order.get(by_q[t[2]]["report"].question.question_type, 2), t[1]))
    n_open = sum(1 for t in todo if t[0])
    print(f"{len(by_q)} questions with saved research, {len(todo)} with work left "
          f"({n_open} still open); budget ${budget:.2f}")

    def make_bot(llm, parser):
        b = main.FallTemplateBot2026(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            publish_reports_to_metaculus=False,
            llms={"default": llm, "summarizer": parser, "researcher": "claude-web-search", "parser": parser},
        )
        b.ensemble_models = []  # the model is set per sample below
        return b

    bots = {}
    kinds = {p for p, _, _ in configs.values()}
    if "oc" in kinds:
        first = next(m for p, m, _ in configs.values() if p == "oc")
        bots["oc"] = make_bot(OpenAICompatForecastLlm(first),
                              parser_llm(os.getenv("PARSER_MODEL_OC", "zai-org/GLM-5.3-Flash")))
    if kinds & {"claude", "claude-noresearch"}:
        first = next(m for p, m, _ in configs.values() if p != "oc")
        claude_bot = make_bot(
            ClaudeForecastLlm(first, effort=os.getenv("FORECAST_EFFORT", "high")),
            GeneralLlm(model=os.getenv("PARSER_MODEL", "anthropic/claude-haiku-5-5"), temperature=None),
        )
        bots["claude"] = bots["claude-noresearch"] = claude_bot

    cost_manager = MonetaryCostManager(hard_limit=budget)
    cost_manager.__enter__()
    n_questions = n_errors = 0
    try:
        for _, _, qid, missing in todo[:max_questions]:
            if stop_at and datetime.now(timezone.utc) >= stop_at:
                print(f"Stop time {stop_at.isoformat()} reached; not starting more questions.")
                break
            item = by_q[qid]
            rep, q = item["report"], item["report"].question
            main.datetime = _FrozenDatetime  # prompts say "Today is <original date>"
            _FrozenDatetime.as_of = item["accessed"]
            posted = posted_model(rep.explanation)
            for b in {id(b): b for b in bots.values()}.values():
                b._note_pads.append(await b._initialize_notepad(q))
            rows = []
            jobs = []  # (config name, provider, model, sample index)
            for cname in missing:
                provider, model, n = configs[cname]
                if posted and posted.split("/")[-1] == model.split("/")[-1] and provider != "claude-noresearch":
                    # the posted forecast already is this model on this research
                    rows.append({"kind": "sample", "qid": qid, "config": cname, "model": model,
                                 "reused_posted": True, "prediction": None})
                    continue
                jobs += [(cname, provider, model, i) for i in range(n)]

            async def one(cname: str, provider: str, model: str, i: int) -> dict:
                token = current_model.set(model)
                research = NO_RESEARCH if provider == "claude-noresearch" else item["research"]
                try:
                    # Cost managers are per task, so this one sees exactly this
                    # sample's calls (forecast and parsing).
                    with MonetaryCostManager() as own:
                        pred = await bots[provider]._make_prediction(q, research)
                    return {"prediction": prediction_to_json(pred.prediction_value),
                            "cost": round(own.current_usage, 5)}
                except HardLimitExceededError:
                    raise
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                finally:
                    current_model.reset(token)

            # every sample of every config at once (the date is per question)
            results = await asyncio.gather(*(one(*j) for j in jobs))
            for (cname, provider, model, i), res in zip(jobs, results):
                n_errors += "error" in res
                rows.append({"kind": "sample", "qid": qid, "config": cname, "provider": provider,
                             "model": model, "i": i, **res})
            for row in rows:
                row.update({"url": q.page_url, "type": q.question_type,
                            "as_of": item["accessed"].isoformat(),
                            "close_time": q.close_time.isoformat() if q.close_time else None,
                            "posted_model": posted,
                            "posted": prediction_to_json(rep.prediction),
                            "title": q.question_text[:160]})
            append_state(rows)
            n_questions += 1
            print(f"{q.page_url}: {len(rows)} samples, run cost so far ${cost_manager.current_usage:.2f}")
    except Exception as e:
        print(f"Stopped: {type(e).__name__}: {str(e)[:200]}")
    finally:
        main.datetime = datetime
        cost_manager.__exit__(None, None, None)
        record_run(cost_manager.current_usage, n_questions, n_errors, "reforecast")


# ------------------------------------------------------------------ scoring


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def aggregate_binary(ps: list[float]) -> dict[str, float]:
    mean_logit = statistics.fmean(_logit(p) for p in ps)
    return {
        "mean": statistics.fmean(ps),
        "median": statistics.median(ps),
        "geo_odds": _sigmoid(mean_logit),
        "extremized_1.3": _sigmoid(1.3 * mean_logit),
    }


def score_binary(p: float, yes: bool) -> dict[str, float]:
    p = min(max(p, 0.001), 0.999)  # Metaculus' own bounds
    return {"log": math.log(p if yes else 1 - p), "brier": (p - (1.0 if yes else 0.0)) ** 2}


def score_mc(dist: dict[str, float], outcome: str) -> float | None:
    if outcome not in dist:
        return None
    total = sum(dist.values()) or 1.0
    return math.log(max(dist[outcome] / total, 0.001))


def score_continuous(cdf: list[float], q_bounds: dict, value) -> float | None:
    """ln of the probability mass in the bin holding the resolution (open tails
    use the out-of-range mass). Comparable across variants on one question."""
    lo, hi = q_bounds["lower"], q_bounds["upper"]
    n = len(cdf) - 1
    if value == "below_lower_bound":
        mass = cdf[0]
    elif value == "above_upper_bound":
        mass = 1 - cdf[-1]
    else:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        if q_bounds.get("zero_point") is not None:
            return None  # log-scaled axis: skip rather than mis-score
        k = min(max(int((x - lo) / (hi - lo) * n), 0), n - 1)
        mass = cdf[k + 1] - cdf[k]
    return math.log(max(mass, 1e-4))


def fetch_resolutions(rows: list[dict]) -> list[dict]:
    from forecasting_tools import MetaculusApi

    known = {r["qid"] for r in rows if r.get("kind") == "resolution"}
    urls = {r["qid"]: r["url"] for r in rows if r.get("kind") == "sample"}
    new = []
    for qid, url in urls.items():
        if qid in known:
            continue
        try:
            q = MetaculusApi.get_question_by_url(url)
        except Exception as e:
            logger.info(f"{url}: {type(e).__name__}")
            continue
        if not q.resolution_string:
            continue
        new.append({
            "kind": "resolution", "qid": qid, "url": url,
            "resolution": q.resolution_string,
            "bounds": {
                "lower": getattr(q, "lower_bound", None), "upper": getattr(q, "upper_bound", None),
                "zero_point": getattr(q, "zero_point", None),
            },
        })
    return new


def summarize(rows: list[dict]) -> str:
    res = {r["qid"]: r for r in rows if r.get("kind") == "resolution"}
    samples: dict[tuple[int, str], list] = {}
    meta: dict[int, dict] = {}
    for r in rows:
        if r.get("kind") != "sample":
            continue
        meta[r["qid"]] = r
        if r.get("prediction") is not None:
            samples.setdefault((r["qid"], r["config"]), []).append(r["prediction"])
        elif r.get("reused_posted"):
            samples.setdefault((r["qid"], r["config"]), []).append(r["posted"])
    lines = ["# Post-close re-forecast experiment", "",
             f"Questions with samples: {len(meta)}; resolved: {sum(1 for q in meta if q in res and res[q]['resolution'] not in ('annulled', 'ambiguous'))}.",
             "Same research per question; only the forecasting step differs. 'posted' is what the live bot posted.", ""]
    # binary
    table: dict[str, list[tuple[int, float, float]]] = {}
    for qid, m in meta.items():
        if m["type"] != "binary" or qid not in res or res[qid]["resolution"] not in ("yes", "no"):
            continue
        yes = res[qid]["resolution"] == "yes"
        variants = {"posted": m["posted"]}
        for (q2, cname), ps in samples.items():
            if q2 != qid:
                continue
            ps = [p for p in ps if isinstance(p, (int, float))]
            if ps:
                for agg, val in aggregate_binary(ps).items():
                    variants[f"{cname}/{agg}"] = val
        for name, p in variants.items():
            if isinstance(p, (int, float)):
                s = score_binary(float(p), yes)
                table.setdefault(name, []).append((qid, s["log"], s["brier"]))
    if table:
        posted = {qid: lg for qid, lg, _ in table.get("posted", [])}
        lines += ["## Binary (log score: higher is better; Brier: lower is better)", "",
                  "| variant | n | mean log | mean Brier | log vs posted (paired, ± s.e.) |", "|---|---|---|---|---|"]
        for name in sorted(table, key=lambda k: -statistics.fmean(x[1] for x in table[k])):
            vals = table[name]
            diffs = [lg - posted[qid] for qid, lg, _ in vals if qid in posted]
            if len(diffs) > 1:
                se = statistics.stdev(diffs) / math.sqrt(len(diffs))
                d = f"{statistics.fmean(diffs):+.3f} ± {se:.3f}"
            else:
                d = "—"
            lines.append(f"| {name} | {len(vals)} | {statistics.fmean(x[1] for x in vals):.3f} | "
                         f"{statistics.fmean(x[2] for x in vals):.3f} | {d} |")
        lines.append("")
    # multiple choice and continuous: mean log score per config (mean of samples)
    for qtype, label in (("multiple_choice", "Multiple choice"), ("numeric", "Numeric"), ("discrete", "Discrete")):
        agg: dict[str, list[float]] = {}
        for qid, m in meta.items():
            if m["type"] != qtype or qid not in res or res[qid]["resolution"] in ("annulled", "ambiguous"):
                continue
            outcome = res[qid]["resolution"]
            variants = {"posted": m["posted"]}
            for (q2, cname), ps in samples.items():
                if q2 != qid:
                    continue
                if qtype == "multiple_choice":
                    dicts = [p for p in ps if isinstance(p, dict)]
                    if dicts:
                        keys = dicts[0].keys()
                        variants[cname] = {k: statistics.fmean(d.get(k, 0) for d in dicts) for k in keys}
                else:
                    cdfs = [p["cdf"] for p in ps if isinstance(p, dict) and "cdf" in p]
                    if cdfs:
                        variants[cname] = {"cdf": [statistics.fmean(c[i] for c in cdfs) for i in range(len(cdfs[0]))]}
            for name, v in variants.items():
                if qtype == "multiple_choice" and isinstance(v, dict):
                    s = score_mc(v, outcome)
                elif isinstance(v, dict) and "cdf" in v:
                    s = score_continuous(v["cdf"], res[qid]["bounds"], outcome)
                else:
                    s = None
                if s is not None:
                    agg.setdefault(name, []).append(s)
        if agg:
            lines += [f"## {label} (mean log score of the averaged forecast)", "", "| variant | n | mean log |", "|---|---|---|"]
            for name in sorted(agg, key=lambda k: -statistics.fmean(agg[k])):
                lines.append(f"| {name} | {len(agg[name])} | {statistics.fmean(agg[name]):.3f} |")
            lines.append("")
    errors = sum(1 for r in rows if r.get("kind") == "sample" and r.get("error"))
    cost = sum(r.get("cost") or 0 for r in rows if r.get("kind") == "sample")
    lines += [f"Samples: {sum(1 for r in rows if r.get('kind') == 'sample')}, errors: {errors}, "
              f"sample cost ${cost:.2f} (parser calls and timeouts are in the ledger, not here)."]
    lines += ["", "Small samples: a difference smaller than about two standard errors is noise."]
    return "\n".join(lines) + "\n"


def score() -> None:
    rows = load_state()
    new = fetch_resolutions(rows)
    if new:
        append_state(new)
        rows += new
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(summarize(rows))
    print(SUMMARY.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument("--reports", default="reforecast/reports")
    ap.add_argument("--max-questions", type=int, default=int(os.getenv("REFORECAST_MAX_QUESTIONS", "15")))
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    if args.command == "run":
        folder = Path(args.reports)
        folder.mkdir(parents=True, exist_ok=True)
        if not args.no_download:
            download_report_artifacts(folder)
        asyncio.run(run(folder, args.max_questions))
    else:
        score()
