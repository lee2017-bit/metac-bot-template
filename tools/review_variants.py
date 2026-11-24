"""
Score aggregation variants of the forecasts the bot already posted.

Each saved report lists every forecaster's prediction ("*Forecaster k*: ...").
From those this script rebuilds, for every resolved question:
  - posted        : what the bot actually posted
  - all/mean, all/median, all/geo_odds (binary) : all N forecasters
  - 3subsets/<agg>: the average score over every 3-forecaster subset (what a
                    3-forecast ensemble would have scored on average)
  - single        : the average single forecaster
  - model/<name>  : only the forecasters of one model (from the ensemble order)
and scores them: binary log score and Brier, multiple choice log score,
numeric/discrete/date mean pinball loss on the reported percentiles
(normalised by the question range; lower is better).

No LLM calls and no posting. Resolutions come from a JSONL file with
{"kind": "resolution", "qid": <question id>, "url": ..., "resolution": ...}
rows (reforecast/results.jsonl has them), or are fetched with --fetch
(needs METACULUS_TOKEN, read-only).

  python tools/review_variants.py --reports DIR [--reports DIR2] \
      --resolutions reforecast/results.jsonl [--ensemble "claude-fable-5-2:2,claude-opus-5-5:3"] \
      [--out review_variants.md]

Caveat: forecaster k is mapped to a model by the ensemble order
(k-1 mod len(ensemble)), which is the order the bot assigns models in; check
it against a few reports before relying on the per-model rows.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

_PCT = re.compile(r"^\s*-\s*([0-9.]+)%\s*chance of value below\s*(-?[0-9.eE+]+)", re.M)
_BIN = re.compile(r"^\*Forecaster (\d+)\*:\s*([0-9.]+)%\s*$", re.M)
_HEAD = re.compile(r"^\*Forecaster (\d+)\*:", re.M)


def expand_ensemble(spec: str) -> list[str]:
    out: list[str] = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, n = part.rpartition(":")
        out += [name] * int(n) if name else [part]
    return out


def forecaster_blocks(explanation: str) -> dict[int, str]:
    start = explanation.find("### Forecasts")
    end = explanation.find("### Research Summary", start)
    section = explanation[start:end if end > start else None]
    heads = list(_HEAD.finditer(section))
    blocks = {}
    for i, m in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(section)
        blocks[int(m.group(1))] = section[m.start():stop]
    return blocks


def parse_forecasters(report) -> dict[int, object]:
    """Forecaster number -> prediction (float, {option: p} or [(pct, value)])."""
    qtype = report.question.question_type
    blocks = forecaster_blocks(report.explanation)
    out: dict[int, object] = {}
    for k, text in blocks.items():
        if qtype == "binary":
            m = _BIN.search(text)
            if m:
                out[k] = float(m.group(2)) / 100
        elif qtype == "multiple_choice":
            dist = {}
            for opt in report.question.options:
                m = re.search(r"^\s*-\s*" + re.escape(opt) + r":\s*([0-9.]+)%", text, re.M)
                if m:
                    dist[opt] = float(m.group(1)) / 100
            if len(dist) == len(report.question.options):
                out[k] = dist
        else:
            pts = [(float(p) / 100, float(v)) for p, v in _PCT.findall(text)]
            if len(pts) >= 3:
                out[k] = pts
    return out


# ------------------------------------------------------------------ aggregation


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def agg_binary(ps: list[float], how: str) -> float:
    if how == "mean":
        return statistics.fmean(ps)
    if how == "median":
        return statistics.median(ps)
    m = statistics.fmean(_logit(p) for p in ps)
    return 1 / (1 + math.exp(-m))


def agg_mc(ds: list[dict], how: str) -> dict:
    keys = ds[0].keys()
    f = statistics.fmean if how == "mean" else statistics.median
    raw = {k: f(d.get(k, 0.0) for d in ds) for k in keys}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def agg_cont(qs: list[list[tuple[float, float]]], how: str) -> list[tuple[float, float]]:
    f = statistics.fmean if how == "mean" else statistics.median
    levels = sorted({round(p, 4) for q in qs for p, _ in q})
    out = []
    for lv in levels:
        vals = [v for q in qs for p, v in q if round(p, 4) == lv]
        if len(vals) == len(qs):
            out.append((lv, f(vals)))
    return out


# ------------------------------------------------------------------ scoring


def score(qtype: str, pred, resolution: str, q) -> dict | None:
    if resolution in ("annulled", "ambiguous", None):
        return None
    if qtype == "binary":
        if resolution not in ("yes", "no"):
            return None
        p = min(max(float(pred), 0.001), 0.999)
        yes = resolution == "yes"
        return {"log": math.log(p if yes else 1 - p), "brier": (p - (1.0 if yes else 0.0)) ** 2}
    if qtype == "multiple_choice":
        if resolution not in pred:
            return None
        return {"log": math.log(max(pred[resolution], 0.001))}
    # continuous: mean pinball loss over the reported percentiles, normalised by the range
    try:
        x = float(resolution)
    except (TypeError, ValueError):
        lo, hi = q.lower_bound, q.upper_bound
        if resolution == "below_lower_bound":
            x = lo
        elif resolution == "above_upper_bound":
            x = hi
        else:
            return None
    span = (q.upper_bound - q.lower_bound) or 1.0
    losses = [max(lv * (x - v), (lv - 1) * (x - v)) / span for lv, v in pred]
    return {"pinball": statistics.fmean(losses)} if losses else None


def variants(qtype: str, fc: dict[int, object], models: list[str]) -> dict[str, list]:
    """name -> list of predictions (the score of a name is the mean over its list)."""
    ks = sorted(fc)
    preds = [fc[k] for k in ks]
    if qtype == "binary":
        aggs, agg = ("mean", "median", "geo_odds"), agg_binary
    elif qtype == "multiple_choice":
        aggs, agg = ("mean", "median"), agg_mc
    else:
        aggs, agg = ("mean", "median"), agg_cont
    out: dict[str, list] = {}
    for how in aggs:
        out[f"all/{how}"] = [agg(preds, how)]
        if len(preds) >= 4:
            out[f"3subsets/{how}"] = [agg([fc[k] for k in sub], how) for sub in itertools.combinations(ks, 3)]
    out["single"] = preds
    if models:
        by: dict[str, list] = {}
        for k in ks:
            by.setdefault(models[(k - 1) % len(models)], []).append(fc[k])
        for name, ps in by.items():
            out[f"model/{name}/{aggs[0]}"] = [agg(ps, aggs[0])]
    return out


# ------------------------------------------------------------------ io


def load_reports(dirs: list[str]) -> dict[int, object]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forecasting_tools import DataOrganizer

    by_q: dict[int, object] = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                reps = DataOrganizer.load_reports_from_file_path(f)
            except Exception:
                continue
            for r in reps:
                q = r.question
                if "bot-testing-area" in (q.tournament_slugs or []):
                    continue
                old = by_q.get(q.id_of_question)
                if old is None or (q.date_accessed and old.question.date_accessed and q.date_accessed < old.question.date_accessed):
                    by_q[q.id_of_question] = r
    return by_q


def load_resolutions(path: str | None, reports: dict, fetch: bool) -> dict[int, str]:
    res: dict[int, str] = {}
    if path and os.path.exists(path):
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if row.get("kind") == "resolution":
                    res[int(row["qid"])] = row["resolution"]
    if fetch:
        from forecasting_tools import MetaculusApi

        for qid, r in reports.items():
            if qid in res:
                continue
            try:
                q = MetaculusApi.get_question_by_url(r.question.page_url)
            except Exception:
                continue
            if q.resolution_string:
                res[qid] = q.resolution_string
    return res


def prediction_of(report, levels: list[float] | None = None):
    """The posted prediction. For continuous questions it is read off the
    posted distribution at the forecasters' percentile levels, so it is scored
    on the same points as the other variants."""
    p = report.prediction
    if isinstance(p, (int, float)):
        return float(p)
    if hasattr(p, "predicted_options"):
        return {o.option_name: o.probability for o in p.predicted_options}
    if hasattr(p, "declared_percentiles") and levels:
        pts = sorted((x.percentile, x.value) for x in p.declared_percentiles)
        out = []
        for lv in levels:
            for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
                if p0 <= lv <= p1:
                    out.append((lv, v0 if p1 == p0 else v0 + (v1 - v0) * (lv - p0) / (p1 - p0)))
                    break
        return out or None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", action="append", required=True)
    ap.add_argument("--resolutions")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--ensemble", default="claude-fable-5-2:2,claude-opus-5-5:3")
    ap.add_argument("--out", default="review_variants.md")
    args = ap.parse_args()
    models = expand_ensemble(args.ensemble)
    reports = load_reports(args.reports)
    res = load_resolutions(args.resolutions, reports, args.fetch)
    table: dict[str, dict[str, list[tuple[int, float]]]] = {}
    n_used = 0
    for qid, r in reports.items():
        if qid not in res:
            continue
        qtype = r.question.question_type
        fc = parse_forecasters(r)
        if len(fc) < 2:
            continue
        group = {"binary": "binary", "multiple_choice": "multiple_choice"}.get(qtype, "continuous")
        vs = variants(qtype, fc, models)
        levels = None
        if group == "continuous":
            levels = sorted({lv for pts in fc.values() for lv, _ in pts})
        posted = prediction_of(r, levels)
        if posted is not None:
            vs["posted"] = [posted]
        used = False
        for name, preds in vs.items():
            ss = [score(qtype, p, res[qid], r.question) for p in preds]
            ss = [s for s in ss if s]
            if not ss:
                continue
            used = True
            for metric in ss[0]:
                table.setdefault(f"{group}:{metric}", {}).setdefault(name, []).append(
                    (qid, statistics.fmean(s[metric] for s in ss))
                )
        n_used += used
    lines = ["# Aggregation variants of the posted forecasts", "",
             f"Reports: {len(reports)} questions; resolved and parsed: {n_used}. Ensemble order: {', '.join(models)}.",
             "Same research and same forecasts throughout; only how the forecasts are combined differs.", ""]
    for key in sorted(table):
        group, metric = key.split(":")
        higher_better = metric == "log"
        base = dict(table[key].get("all/median", []))
        lines += [f"## {group} — {metric} ({'higher' if higher_better else 'lower'} is better)", "",
                  "| variant | n | mean | vs all/median (paired, ± s.e.) |", "|---|---|---|---|"]
        order = sorted(table[key], key=lambda v: statistics.fmean(x for _, x in table[key][v]), reverse=higher_better)
        for name in order:
            vals = table[key][name]
            diffs = [x - base[q] for q, x in vals if q in base]
            if len(diffs) > 1 and name != "all/median":
                d = f"{statistics.fmean(diffs):+.4f} ± {statistics.stdev(diffs) / math.sqrt(len(diffs)):.4f}"
            else:
                d = "—"
            lines.append(f"| {name} | {len(vals)} | {statistics.fmean(x for _, x in vals):.4f} | {d} |")
        lines.append("")
    lines.append("A difference smaller than about two standard errors is noise. Do not change the bot on one review.")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
