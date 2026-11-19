"""
Research step that needs no search API key: an OpenAI-compatible model (see
openai_compat.py) drives four client-side tools.

  news_search  GDELT DOC API (free, open), falling back to Google News RSS
  wiki_search  Wikipedia search API
  web_fetch    downloads a page or PDF and returns its text
  market_search  Polymarket / Kalshi / Manifold (market_lookup.py)

metaculus.com is refused by every tool, so the community prediction cannot be
read. Nothing here depends on Metaculus question types: it takes a prompt and
returns a research write-up, so it can be reused by other forecasting code.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import requests
from forecasting_tools import MonetaryCostManager

from market_lookup import search_markets
import openai

from openai_compat import add_cost, cost_of, get_client, price_of, worst_case_cost

logger = logging.getLogger(__name__)

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL_OC", "zai-org/GLM-5.3")
# 8000: at 16000 GLM-5.3 sometimes thought for over 10 minutes and timed out.
MAX_TOKENS = int(os.getenv("RESEARCH_MAX_TOKENS_OC", "8000"))
CALL_TIMEOUT = float(os.getenv("RESEARCH_CALL_TIMEOUT_OC", "300"))
MAX_TOOL_CALLS = int(os.getenv("RESEARCH_OC_MAX_TOOL_CALLS", "12"))
TOOL_LIMITS = {
    "news_search": int(os.getenv("RESEARCH_OC_MAX_NEWS", "4")),
    "wiki_search": int(os.getenv("RESEARCH_OC_MAX_WIKI", "2")),
    "web_fetch": int(os.getenv("RESEARCH_OC_MAX_FETCHES", "4")),
    "market_search": int(os.getenv("RESEARCH_OC_MAX_MARKETS", "2")),
}
PAGE_CHARS = int(os.getenv("RESEARCH_OC_PAGE_CHARS", "14000"))
_TIMEOUT = 20
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; forecast-research/1.0)"}
_BLOCKED = ("metaculus.com",)


def _blocked(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == b or host.endswith("." + b) for b in _BLOCKED)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ----------------------------------------------------------------- tools


# GDELT allows one request per 5 seconds. Research for several questions runs
# at once, so requests take numbered slots; one that would wait too long is
# skipped (Google News still answers).
_gdelt_lock = threading.Lock()
_gdelt_next = [0.0]


def _gdelt_slot(max_wait: float = 25.0) -> bool:
    with _gdelt_lock:
        now = time.monotonic()
        start = max(now, _gdelt_next[0])
        if start - now > max_wait:
            return False
        _gdelt_next[0] = start + 5.5
    time.sleep(start - now)
    return True


def _google_news(query: str) -> list[str]:
    r = requests.get(
        f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en",
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    root = ET.fromstring(r.content)
    items = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        src = item.findtext("source") or ""
        if "metaculus" in (title + src).lower():
            continue
        try:
            when = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            when = datetime(1970, 1, 1, tzinfo=timezone.utc)
        src_el = item.find("source")
        site = urlparse(src_el.get("url", "")).hostname if src_el is not None else None
        items.append((when, f"- {when:%Y-%m-%d} {src}{f' ({site})' if site else ''}: {title}"))
    items.sort(key=lambda t: t[0], reverse=True)  # the feed is not in date order
    return [line for _, line in items[:12]]


def _gdelt(query: str) -> list[str]:
    if not _gdelt_slot():
        return []
    r = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": query,
            "mode": "ArtList",
            "maxrecords": 8,
            "format": "json",
            "timespan": "1m",
            "sort": "HybridRel",  # relevance with recency; DateDesc was mostly noise
        },
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    if not (r.ok and r.text.strip().startswith("{")):
        # Often 429 on shared IPs (one request per 5 s per IP).
        logger.info(f"GDELT gave no JSON ({r.status_code})")
        return []
    lines = []
    for a in r.json().get("articles") or []:
        url = a.get("url") or ""
        if _blocked(url) or "metaculus" in (a.get("title") or "").lower():
            continue
        lines.append(f"- {(a.get('seendate') or '')[:8]} {a.get('domain')}: {a.get('title')} ({url})")
    return lines


def news_search_sync(query: str) -> str:
    """Google News headlines (relevant, dated, no article links) plus, when
    GDELT answers, a few articles with links that web_fetch can open."""
    query = " ".join(str(query).split())[:200]
    sections = []
    for name, fn in (("Google News", _google_news), ("GDELT", _gdelt)):
        try:
            lines = fn(query)
        except Exception as e:  # news must never break research
            logger.info(f"{name} failed for {query!r}: {type(e).__name__}")
            lines = []
        if lines:
            sections.append((name, lines))
    if not sections:
        return f"No news found for '{query}'."
    out = [f"News search for '{query}' at {_now()}."]
    for name, lines in sections:
        hint = (
            "headlines only, no links; fetch the publisher's or an official page with web_fetch if you know its URL"
            if name == "Google News"
            else "articles with links, open one with web_fetch to read it; may include loosely related items"
        )
        out.append(f"{name} ({hint}):")
        out.extend(lines)
    return "\n".join(out)


def _wiki_action_api(query: str) -> list[tuple[str, str]]:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": 6, "format": "json"},
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    r.raise_for_status()  # 429 is common on shared IPs
    return [(h.get("title", ""), h.get("snippet", "")) for h in r.json().get("query", {}).get("search", [])]


def _wiki_rest_api(query: str) -> list[tuple[str, str]]:
    r = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 6},
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return [
        (pg.get("title", ""), " ".join(x for x in (pg.get("description"), pg.get("excerpt")) if x))
        for pg in r.json().get("pages", [])
    ]


def _wiki_search_page(query: str) -> list[tuple[str, str]]:
    # The ordinary search page, which was still answering when both APIs gave 429.
    r = requests.get(
        "https://en.wikipedia.org/w/index.php",
        params={"search": query, "fulltext": "1", "ns0": "1"},
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    hits = []
    for m in re.finditer(
        r'<div class="mw-search-result-heading"><a href="/wiki/[^"]+" title="([^"]+)".*?'
        r'<div class="searchresult">(.*?)</div>',
        r.text,
        re.S,
    ):
        hits.append((unescape(m.group(1)), m.group(2)))
        if len(hits) >= 6:
            break
    return hits


def wiki_search_sync(query: str) -> str:
    hits, error = None, None
    # Next endpoint whenever one is rate-limited (429 is common on shared IPs).
    for fn in (_wiki_action_api, _wiki_rest_api, _wiki_search_page):
        try:
            hits = fn(query)
            break
        except Exception as e:
            error = e
    if hits is None:
        return (
            f"Wikipedia search failed ({type(error).__name__}). "
            "web_fetch on https://en.wikipedia.org/wiki/<Article_title> still works."
        )
    if not hits:
        return f"No Wikipedia article found for '{query}'."
    out = []
    for title, snippet in hits:
        snippet = unescape(re.sub(r"<[^>]+>", "", snippet or ""))
        out.append(
            f"- {title}: {snippet} (https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))})"
        )
    return f"Wikipedia search for '{query}':\n" + "\n".join(out)


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "table", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n", t)
        return t.strip()


def web_fetch_sync(url: str) -> str:
    url = str(url).strip()
    if not re.match(r"^https?://", url):
        return "Invalid URL: give a full http(s) address."
    if _blocked(url):
        return "This site is not allowed."
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, stream=True, allow_redirects=True)
        if _blocked(r.url):
            return "This site is not allowed."
        raw = b""
        for chunk in r.iter_content(65536):
            raw += chunk
            if len(raw) > 15_000_000:
                break
    except Exception as e:
        return f"Could not open {url} ({type(e).__name__})."
    if r.status_code >= 400:
        return f"Could not open {url} (HTTP {r.status_code})."
    ctype = (r.headers.get("content-type") or "").lower()
    title = ""
    if "pdf" in ctype or raw[:5] == b"%PDF-":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages[:20]]
            text = "\n".join(pages)
        except Exception as e:
            return f"Opened {url} but could not read the PDF ({type(e).__name__})."
    elif "json" in ctype or "csv" in ctype or ctype.startswith("text/plain"):
        text = raw.decode(r.encoding or "utf-8", errors="replace")
    else:
        parser = _TextExtractor()
        try:
            parser.feed(raw.decode(r.encoding or r.apparent_encoding or "utf-8", errors="replace"))
        except Exception:
            pass
        text, title = parser.text(), " ".join(parser.title.split())
    text = text.strip()
    if len(text) < 200:
        return (
            f"Opened {r.url} but found almost no readable text (the page may need JavaScript). "
            f"Text found: {text[:300]}"
        )
    more = f" [truncated: {len(text)} characters in total]" if len(text) > PAGE_CHARS else ""
    return (
        f"Fetched {r.url} at {_now()}" + (f" - {title}" if title else "") + f"{more}\n"
        + text[:PAGE_CHARS]
    )


_TOOL_FUNCS = {
    "news_search": news_search_sync,
    "wiki_search": wiki_search_sync,
    "web_fetch": web_fetch_sync,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": "Search news. Returns dated headlines with their publisher (Google News, no links) and, when available, a few articles with URLs (GDELT, last month). Use 2-6 keywords; you may search in the local language of the event.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Open a web page or PDF and return its text. Use it for the resolution source named in the question, official data pages, and articles found with news_search.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search Wikipedia for background and historical base rates. Returns titles, snippets and URLs to open with web_fetch.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_search",
            "description": "Search Polymarket, Kalshi and Manifold for open prediction markets on the same or a related event. Returns titles, prices, close dates and rules. Use 2-5 keywords.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM = (
    "You are a research assistant for a forecaster. You have tools: news_search, web_fetch, "
    "wiki_search and market_search. You cannot browse except through them. Work in steps: open the "
    "resolution source first if the question names one, search for the latest news, look for base "
    "rates and for a matching prediction market, then write the final report. Give the date of every "
    "fact and say which facts you could not verify. Never invent a figure, a quote or a URL. Your final "
    "message is the report and must not call a tool. Think briefly between tool calls: each reply, "
    "hidden reasoning included, is limited to about 8,000 tokens."
)


async def _run_tool(name: str, args: dict, used: dict) -> str:
    if name not in TOOL_LIMITS:
        return f"Unknown tool {name}."
    if used.get(name, 0) >= TOOL_LIMITS[name]:
        return f"Limit for {name} reached; use what you have."
    used[name] = used.get(name, 0) + 1
    try:
        if name == "market_search":
            return await search_markets(str(args.get("query", "")))
        key = "url" if name == "web_fetch" else "query"
        return await asyncio.to_thread(_TOOL_FUNCS[name], str(args.get(key, "")))
    except Exception as e:  # tools must never break research
        logger.info(f"tool {name} failed: {type(e).__name__}: {e}")
        return f"{name} failed ({type(e).__name__})."


def _assistant_message(msg) -> dict:
    d = msg.model_dump(exclude_none=True)
    keep = {"role": "assistant", "content": d.get("content") or ""}
    if d.get("tool_calls"):
        keep["tool_calls"] = d["tool_calls"]
    if d.get("reasoning_content"):
        keep["reasoning_content"] = d["reasoning_content"]
    return keep


async def run_free_research(prompt: str) -> str:
    MonetaryCostManager.raise_error_if_limit_would_be_reached()
    price_of(RESEARCH_MODEL)
    client = get_client()
    messages: list = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    used: dict[str, int] = {}
    calls = 0
    total = 0.0
    text = ""
    empty_replies = 0
    told_to_finish = False
    for turn in range(MAX_TOOL_CALLS + 5):
        final_turn = calls >= MAX_TOOL_CALLS or turn >= MAX_TOOL_CALLS + 4
        if final_turn and not told_to_finish:
            told_to_finish = True
            messages.append(
                {"role": "user", "content": "Tool budget used up. Write the final report now from what you have."}
            )
        try:
            # No automatic retry of a timed-out call: the same request would
            # most likely think just as long again. Nudge instead (below).
            response = await client.with_options(
                timeout=CALL_TIMEOUT, max_retries=0
            ).chat.completions.create(
                model=RESEARCH_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="none" if final_turn else "auto",
                max_tokens=MAX_TOKENS,
            )
        except openai.APITimeoutError:
            cost = worst_case_cost(RESEARCH_MODEL, MAX_TOKENS, sum(len(str(m)) for m in messages))
            total += cost
            add_cost(cost)
            empty_replies += 1
            logger.warning(f"Research call timed out after {CALL_TIMEOUT:.0f} s (empty reply {empty_replies})")
            if empty_replies > 2:
                break
            messages.append(
                {"role": "user", "content": "Your last reply took too long. Think briefly. "
                 + ("Write the final report now." if final_turn else
                    "Call the tools you need now, or write the final report if you have enough.")}
            )
            continue
        except (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError) as e:
            empty_replies += 1
            logger.warning(f"Research call failed ({type(e).__name__}, attempt {empty_replies}); retrying in 10 s")
            if empty_replies > 2:
                break
            await asyncio.sleep(10)
            continue
        cost = cost_of(RESEARCH_MODEL, response.usage)
        total += cost
        add_cost(cost)
        msg = response.choices[0].message
        if msg.tool_calls and not final_turn:
            messages.append(_assistant_message(msg))
            async def one(tc):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    return tc.id, "Invalid tool arguments (not JSON)."
                return tc.id, await _run_tool(tc.function.name, args, used)
            results = await asyncio.gather(*(one(tc) for tc in msg.tool_calls))
            calls += len(msg.tool_calls)
            for tc_id, content in results:
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": content})
            continue
        text = (msg.content or "").strip()
        if text:
            break
        # Seen live: the model spent its whole token budget thinking and
        # returned nothing. Ask again with a nudge rather than hand the
        # forecasters an empty report.
        empty_replies += 1
        logger.warning(
            f"Research model returned no text (finish={response.choices[0].finish_reason}, "
            f"empty reply {empty_replies})"
        )
        if empty_replies > 2:
            break
        messages.append(
            {
                "role": "user",
                "content": "Your last reply was empty: it ran out of tokens while thinking. "
                "Think briefly. "
                + ("Write the final report now." if final_turn else
                   "Call the tools you need now, or write the final report if you have enough."),
            }
        )
    logger.info(
        f"Research cost ${total:.4f} (model={RESEARCH_MODEL}, tool calls {used}, "
        f"empty replies {empty_replies})"
    )
    if not text:
        return (
            "RESEARCH FAILED: the research model returned no report for this question. "
            "Forecast from the question text and your own knowledge, and say so."
        )
    return text
