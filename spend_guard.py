"""
Spend cap and key-expiry guard.

- spend/ledger.json keeps the running total across runs (GitHub Actions restores
  and saves it with actions/cache). The bot refuses to start once the total
  reaches SPEND_CAP_USD, and each run gets at most the remaining amount as a
  hard limit inside forecasting-tools.
- LLM_KEY_EXPIRES (YYYY-MM-DD, UTC): from that date the run fails loudly
  instead of erroring on every question, so the failed-run e-mail tells the
  owner to switch keys.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER = Path(os.getenv("SPEND_LEDGER", "spend/ledger.json"))


def _load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"total_usd": 0.0, "runs": []}


def remaining_budget() -> float:
    cap = float(os.getenv("SPEND_CAP_USD", "0") or 0)
    if cap <= 0:
        print("SPEND_CAP_USD is not set; refusing to run without a cap.")
        sys.exit(1)
    # SPEND_OFFSET_USD: spend that is not in the ledger (local tests, test runs
    # that ran in parallel with the loop), so the cap still covers everything.
    offset = float(os.getenv("SPEND_OFFSET_USD", "0") or 0)
    spent = _load()["total_usd"] + offset
    left = cap - spent
    print(
        f"Spend so far ${spent:.2f} of cap ${cap:.2f} (ledger ${spent - offset:.2f} + "
        f"outside ledger ${offset:.2f}); ${left:.2f} left."
    )
    if left <= 0.5:
        print("Spend cap reached. Not forecasting.")
        sys.exit(0)
    run_cap = float(os.getenv("RUN_CAP_USD", "0") or 0)
    if run_cap > 0 and run_cap < left:
        print(f"This run is limited to ${run_cap:.2f}.")
        return run_cap
    return left


def check_key_expiry() -> None:
    """Stop quietly (exit 0, so no failure e-mail every 20 minutes) once the
    cutoff passes. Accepts a date (UTC midnight) or an ISO datetime."""
    expires = os.getenv("LLM_KEY_EXPIRES", "").strip()
    if not expires:
        return
    if "T" in expires:
        cutoff = datetime.fromisoformat(expires.replace("Z", "+00:00"))
    else:
        cutoff = datetime.combine(
            date.fromisoformat(expires), datetime.min.time(), tzinfo=timezone.utc
        )
    if datetime.now(timezone.utc) >= cutoff:
        print(f"LLM key cutoff {expires} has passed. Bot stopped; not forecasting.")
        sys.exit(0)


def record_run(cost_usd: float, n_questions: int, n_errors: int, mode: str) -> None:
    data = _load()
    data["total_usd"] = round(data["total_usd"] + cost_usd, 4)
    data["runs"].append(
        {
            "at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "mode": mode,
            "cost_usd": round(cost_usd, 4),
            "questions": n_questions,
            "errors": n_errors,
        }
    )
    data["runs"] = data["runs"][-500:]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(data, indent=1))
    print(f"Run cost ${cost_usd:.4f}; total ${data['total_usd']:.2f}.")
