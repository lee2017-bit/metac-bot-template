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
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import requests
from forecasting_tools import MonetaryCostManager

from market_lookup import search_markets
from openai_compat import add_cost, cost_of, get_client, price_of

logger = logging.getLogger(__name__)

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL_OC", "zai-org/GLM-5.3")
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


def news_search_sync(query: str) -> str:
    query = " ".join(str(query).split())[:200]
    lines: list[str] = []
    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "ArtList",
                "maxrecords": 15,
                "format": "json",
                "timespan": "1m",
                "sort": "DateDesc",
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        data = r.json() if r.ok and r.text.strip().startswith("{") else {}
        for a in data.get("articles") or []:
            url = a.get("url") or ""
            if _blocked(url) or "metaculus" in (a.get("title") or "").lower():
                continue
            seen = a.get("seendate") or ""
            lines.append(
                f"- {seen[:8]} {a.get('domain')}: {a.get('title')} ({url})"
            )
    except Exception as e:
        logger.info(f"GDELT failed for {query!r}: {type(e).__name__}")
    source = "GDELT"
    if not lines:
        source = "Google News"
        try:
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
                items.append((when, f"- {when:%Y-%m-%d} {src}: {title}"))
            items.sort(key=lambda t: t[0], reverse=True)  # the feed is not in date order
            lines = [line for _, line in items[:15]]
        except Exception as e:
            logger.info(f"Google News RSS failed for {query!r}: {type(e).__name__}")
    if not lines:
        return f"No news found for '{query}'."
    how = (
        "open a URL with web_fetch to read an article."
        if source == "GDELT"
        else "no article links; fetch the publisher's or an official page with web_fetch if you know its URL."
    )
    return (
        f"News search ({source}) for '{query}' at {_now()}, newest first. Headlines only: "
        f"{how}\n" + "\n".join(lines)
    )


def wiki_search_sync(query: str) -> str:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 6,
                "format": "json",
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        hits = r.json().get("query", {}).get("search", [])
    except Exception as e:
        return f"Wikipedia search failed ({type(e).__name__})."
    if not hits:
        return f"No Wikipedia article found for '{query}'."
    out = []
    for h in hits:
        title = h.get("title", "")
        snippet = unescape(re.sub(r"<[^>]+>", "", h.get("snippet", "")))
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
            "description": "Search recent news (last month), newest first. Returns dated headlines with URLs. Use 2-6 keywords; you may search in the local language of the event.",
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
    "message is the report and must not call a tool."
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
    for turn in range(MAX_TOOL_CALLS + 3):
        final_turn = calls >= MAX_TOOL_CALLS or turn >= MAX_TOOL_CALLS + 2
        if final_turn:
            messages.append(
                {"role": "user", "content": "Tool budget used up. Write the final report now from what you have."}
            )
        response = await client.chat.completions.create(
            model=RESEARCH_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="none" if final_turn else "auto",
            max_tokens=8000,
        )
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
        break
    logger.info(f"Research cost ${total:.4f} (model={RESEARCH_MODEL}, tool calls {used})")
    return text
