"""
Keyless lookup of prediction-market prices (Polymarket, Kalshi, Manifold).

Used as a client-side tool by the research step: the research model passes a
short keyword query and gets back the matching open markets with their exact
titles, prices, close dates and (shortened) rules, so it can judge whether a
market really matches the question before the forecasters anchor on it.

Stand-alone on purpose (only `requests`): it does not know about Metaculus and
can be reused by any forecasting pipeline.

Markets that mirror Metaculus questions are dropped, so the tournament's
community prediction cannot leak in through a mirror.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15
_HEADERS = {"Accept": "application/json", "User-Agent": "forecast-research/1.0"}
_MAX_CHARS = 9000


def _get(url: str, params: dict) -> object:
    r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _short(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _mentions_metaculus(*texts: str | None) -> bool:
    return any("metaculus" in (t or "").lower() for t in texts)


def _polymarket(query: str) -> list[str]:
    data = _get(
        "https://gamma-api.polymarket.com/public-search",
        {"q": query, "limit_per_type": 4, "events_status": "active"},
    )
    lines: list[str] = []
    for ev in (data.get("events") or [])[:4]:
        if ev.get("closed") or _mentions_metaculus(ev.get("title"), ev.get("description")):
            continue
        markets = [m for m in ev.get("markets") or [] if not m.get("closed")]
        if not markets:
            continue
        lines.append(
            f"- Polymarket event: {_short(ev.get('title'), 140)} "
            f"(volume ${float(ev.get('volume') or 0):,.0f}; rules: {_short(ev.get('description'), 420)})"
        )
        markets.sort(key=lambda m: -float(m.get("volumeNum") or m.get("volume") or 0))
        for m in markets[:8]:
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                prices = json.loads(m.get("outcomePrices") or "[]")
            except (TypeError, ValueError):
                outcomes, prices = [], []
            priced = ", ".join(
                f"{o} {float(p) * 100:.1f}%" for o, p in zip(outcomes, prices)
            )
            lines.append(
                f"    * {_short(m.get('question'), 150)}: {priced or 'no price'} "
                f"(bid {m.get('bestBid')}, ask {m.get('bestAsk')}, last {m.get('lastTradePrice')}; "
                f"volume ${float(m.get('volumeNum') or m.get('volume') or 0):,.0f}; "
                f"ends {str(m.get('endDate'))[:10]}; updated {str(m.get('updatedAt'))[:16]})"
            )
    return lines


def _kalshi(query: str) -> list[str]:
    data = _get(
        "https://api.elections.kalshi.com/v1/search/series",
        {"query": query, "page_size": 4},
    )
    lines: list[str] = []
    for ev in (data.get("current_page") or [])[:4]:
        markets = [m for m in ev.get("markets") or [] if not m.get("result")]
        if not markets:
            continue
        lines.append(
            f"- Kalshi event: {_short(ev.get('event_title'), 140)} "
            f"{_short(ev.get('event_subtitle'), 60)} (total volume {ev.get('total_volume')} contracts)"
        )
        for m in markets[:10]:
            lines.append(
                f"    * {_short(m.get('title') or m.get('yes_subtitle'), 150)} "
                f"[{_short(m.get('yes_subtitle'), 50)}]: yes bid {m.get('yes_bid')}c, "
                f"ask {m.get('yes_ask')}c, last {m.get('last_price')}c "
                f"(closes {str(m.get('close_ts'))[:10]})"
            )
    return lines


def _manifold(query: str) -> list[str]:
    data = _get(
        "https://api.manifold.markets/v0/search-markets",
        {"term": query, "limit": 8, "filter": "open", "sort": "liquidity"},
    )
    lines: list[str] = []
    for m in data if isinstance(data, list) else []:
        if m.get("isResolved") or _mentions_metaculus(
            m.get("question"), m.get("creatorUsername"), m.get("creatorName")
        ):
            continue
        if m.get("outcomeType") != "BINARY" or m.get("probability") is None:
            continue
        if (m.get("uniqueBettorCount") or 0) < 10:
            continue
        close = m.get("closeTime")
        close_s = (
            datetime.fromtimestamp(close / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if close
            else "?"
        )
        lines.append(
            f"- Manifold (play money, {m.get('uniqueBettorCount')} traders): "
            f"{_short(m.get('question'), 160)}: {float(m['probability']) * 100:.0f}% (closes {close_s})"
        )
        if len(lines) >= 4:
            break
    return lines


def search_markets_sync(query: str) -> str:
    """Return a compact text listing of open markets matching `query`."""
    query = " ".join(str(query).split())[:120]
    if not query:
        return "Empty query."
    sections: list[str] = []
    for name, fn in (("Polymarket", _polymarket), ("Kalshi", _kalshi), ("Manifold", _manifold)):
        try:
            lines = fn(query)
        except Exception as e:  # network errors must never break research
            logger.info(f"market lookup: {name} failed for {query!r}: {type(e).__name__}")
            sections.append(f"{name}: lookup failed.")
            continue
        sections.append("\n".join(lines) if lines else f"{name}: no open market found.")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = (
        f"Prediction-market search for '{query}' at {now}. Prices are live quotes. "
        "A market is only evidence for the question if its wording, thresholds and dates match; "
        "check that before using it.\n" + "\n".join(sections)
    )
    return text[:_MAX_CHARS]


async def search_markets(query: str) -> str:
    return await asyncio.to_thread(search_markets_sync, query)


if __name__ == "__main__":
    import sys

    print(search_markets_sync(" ".join(sys.argv[1:]) or "fed decision december"))
