"""
Named setting profiles, applied before any module reads its settings.

The workflow runs main.py once per tournament. With CONFIG_PROFILE=MB, every
setting X that has an X_MB variable takes that value instead, from the time in
PROFILE_FROM_MB onwards (before then the plain settings apply). This lets
MiniBench keep Claude, with its own ledger, cap, cut-off and cheaper models,
while the Fall tournament moves to Nebius.

SWITCH_ON_CAP_MB=1: once the profile's own Claude ledger reaches its cap, the
provider is forced to openai_compat, so the questions after that go to Nebius
(with Nebius's ledger and cap, see main.py) instead of being skipped.

Imported by main.py before claude_research and the other modules that read
their settings at import time.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

KEYS = (
    "SPEND_LEDGER", "SPEND_CAP_USD", "SPEND_OFFSET_USD", "RUN_CAP_USD", "LLM_KEY_EXPIRES",
    "LLM_PROVIDER", "PROVIDER_SWITCH_AT",
    "FORECAST_ENSEMBLE", "FORECAST_EFFORT", "PARSER_MODEL",
    "RESEARCH_MODEL", "RESEARCH_EFFORT", "RESEARCH_MAX_SEARCHES", "RESEARCH_MAX_FETCHES",
    "RESEARCH_FETCH_MAX_TOKENS", "RESEARCH_MAX_MARKET_LOOKUPS", "RESEARCH_FULL_CONTEXT",
    "DISCRETE_PER_VALUE_MAX",
)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))


def _still_anthropic(now: datetime) -> bool:
    if os.getenv("LLM_PROVIDER", "anthropic") != "anthropic":
        return False
    switch = os.getenv("PROVIDER_SWITCH_AT", "").strip()
    return not switch or now < _parse(switch)


def apply(now: datetime | None = None) -> str:
    """Apply the profile named in CONFIG_PROFILE; returns a one-line note."""
    profile = os.getenv("CONFIG_PROFILE", "").strip()
    if not profile:
        return ""
    now = now or datetime.now(timezone.utc)
    start = os.getenv(f"PROFILE_FROM_{profile}", "").strip()
    if start and now < _parse(start):
        return f"profile {profile} not active before {start}"
    changed = []
    for key in KEYS:
        value = os.getenv(f"{key}_{profile}")
        if value is not None:
            os.environ[key] = value
            changed.append(key)
    note = f"profile {profile} active ({len(changed)} settings)"
    if os.getenv(f"SWITCH_ON_CAP_{profile}") == "1" and _still_anthropic(now):
        cap = float(os.getenv("SPEND_CAP_USD", "0") or 0)
        offset = float(os.getenv("SPEND_OFFSET_USD", "0") or 0)
        ledger = Path(os.getenv("SPEND_LEDGER", "spend/ledger.json"))
        spent = offset + (json.loads(ledger.read_text())["total_usd"] if ledger.exists() else 0.0)
        if cap > 0 and spent >= cap - 0.5:
            os.environ["LLM_PROVIDER"] = "openai_compat"
            os.environ["PROVIDER_SWITCH_AT"] = ""
            note += f"; Claude cap reached (${spent:.2f} of ${cap:.2f}), switching to the second provider"
    return note


NOTE = apply()
