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
    spent = _load()["total_usd"]
    left = cap - spent
    print(f"Spend so far ${spent:.2f} of cap ${cap:.2f}; ${left:.2f} left.")
    if left <= 0.5:
        print("Spend cap reached. Not forecasting.")
        sys.exit(1)
    run_cap = float(os.getenv("RUN_CAP_USD", "0") or 0)
    if run_cap > 0 and run_cap < left:
        print(f"This run is limited to ${run_cap:.2f}.")
        return run_cap
    return left


def check_key_expiry() -> None:
    expires = os.getenv("LLM_KEY_EXPIRES", "").strip()
    if expires and date.fromisoformat(expires) <= datetime.now(timezone.utc).date():
        print(
            f"LLM key marked as expiring on {expires}. Switch the model/key "
            "before running again (see NOTES)."
        )
        sys.exit(1)


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
