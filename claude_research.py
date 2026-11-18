"""
Research step using Claude with Anthropic's server-side web search and web
fetch tools plus a client-side prediction-market lookup.

Why: the Metaculus LLM proxy closed on 2026-10-27, so with only an Anthropic key
the template's default researcher ("metaculus/gpt-5-search-api") fails. This
module replaces it. metaculus.com is blocked in search and fetch so the bot
cannot read (and copy) the community prediction.

Cost of each call is added to forecasting-tools' MonetaryCostManager, so it
shows up in the per-question cost and counts toward the spend cap.

The module does not depend on Metaculus question types: it takes a prompt and
returns a research write-up, so it can be reused by other forecasting code.
"""
from __future__ import annotations

import logging
import os

import anthropic
from forecasting_tools import MonetaryCostManager

from market_lookup import search_markets

logger = logging.getLogger(__name__)

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-5-5")
MAX_SEARCHES = int(os.getenv("RESEARCH_MAX_SEARCHES", "4"))
RESEARCH_EFFORT = os.getenv("RESEARCH_EFFORT", "low")
# The basic search tool. The newer web_search_20260209 runs searches from
# inside code execution, and in testing the model spent its search budget
# re-running the same query while debugging the result format (live run on
# 16 Nov: "only one web search before hitting the usage limit").
SEARCH_TOOL = os.getenv("RESEARCH_SEARCH_TOOL", "web_search_20250305")
# Page reading (0 = off). The basic fetch tool, for the same reason as above.
# It can only open URLs that already appear in the prompt or in search results.
MAX_FETCHES = int(os.getenv("RESEARCH_MAX_FETCHES", "0"))
FETCH_TOOL = os.getenv("RESEARCH_FETCH_TOOL", "web_fetch_20250910")
FETCH_MAX_TOKENS = int(os.getenv("RESEARCH_FETCH_MAX_TOKENS", "8000"))
# Prediction-market lookup calls allowed per question (0 = off).
MAX_MARKET_LOOKUPS = int(os.getenv("RESEARCH_MAX_MARKET_LOOKUPS", "0"))

# USD per token (input, output). Web search: $10 per 1000 searches. Web fetch
# has no charge beyond the tokens of the fetched page.
_PRICES = {
    "claude-fable-5-2": (8e-6, 40e-6),
    "claude-opus-5-5": (4e-6, 20e-6),
    "claude-sonnet-5-5": (2e-6, 10e-6),
    "claude-haiku-5-5": (0.1e-6, 0.5e-6),
}
_SEARCH_PRICE = 0.01

_BLOCKED_DOMAINS = ["metaculus.com"]

_MARKET_TOOL = {
    "name": "market_search",
    "description": (
        "Search Polymarket, Kalshi and Manifold for open prediction markets and return their exact "
        "titles, current prices, close dates and shortened rules. Use short keyword queries (2-5 words: "
        "the key entity plus the event), not full sentences. Call it when a market on the same or a "
        "closely related event could exist (elections, politics, economics, central banks, sports, "
        "crypto, company events, geopolitics). Returns 'no open market found' when nothing matches."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "2-5 keywords, e.g. 'Fed decision December'"}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # base_url is explicit so a stray ANTHROPIC_BASE_URL can't redirect calls.
        _client = anthropic.AsyncAnthropic(
            base_url="https://api.anthropic.com", max_retries=3, timeout=300
        )
    return _client


def _cost_of(response: anthropic.types.Message) -> float:
    inp, out = _PRICES.get(RESEARCH_MODEL, _PRICES["claude-opus-5-5"])
    u = response.usage
    tokens_in = (
        u.input_tokens
        + (u.cache_creation_input_tokens or 0) * 1.25
        + (u.cache_read_input_tokens or 0) * 0.1
    )
    searches = u.server_tool_use.web_search_requests if u.server_tool_use else 0
    return tokens_in * inp + u.output_tokens * out + searches * _SEARCH_PRICE


def _build_tools() -> list[dict]:
    tools: list[dict] = [
        {
            "type": SEARCH_TOOL,
            "name": "web_search",
            "max_uses": MAX_SEARCHES,
            "blocked_domains": _BLOCKED_DOMAINS,
        }
    ]
    if MAX_FETCHES > 0:
        tools.append(
            {
                "type": FETCH_TOOL,
                "name": "web_fetch",
                "max_uses": MAX_FETCHES,
                "max_content_tokens": FETCH_MAX_TOKENS,
                "blocked_domains": _BLOCKED_DOMAINS,
            }
        )
    if MAX_MARKET_LOOKUPS > 0:
        tools.append(_MARKET_TOOL)
    return tools


async def _run_market_tool(tool_input: object, used: int) -> tuple[str, bool]:
    """Returns (text, is_error)."""
    if used >= MAX_MARKET_LOOKUPS:
        return "Market lookup limit reached for this question; write up with what you have.", True
    query = tool_input.get("query") if isinstance(tool_input, dict) else None
    if not isinstance(query, str) or not query.strip():
        return "Invalid input: 'query' must be a non-empty string.", True
    try:
        return await search_markets(query), False
    except Exception as e:  # never let the lookup break research
        logger.warning(f"market_search failed: {type(e).__name__}: {e}")
        return "Market lookup failed; continue without it.", True


async def run_claude_web_research(prompt: str) -> str:
    MonetaryCostManager.raise_error_if_limit_would_be_reached()
    client = _get_client()
    messages: list = [{"role": "user", "content": prompt}]
    tools = _build_tools()
    total_cost = 0.0
    market_calls = searches = fetches = 0
    fetch_errors: list[str] = []
    response = None
    # Each pass is one request. pause_turn: the server-side loop stopped early,
    # send the turn back to resume. tool_use: the model asked for the
    # client-side market lookup, run it and send the result back.
    for _ in range(8):
        async with client.messages.stream(
            model=RESEARCH_MODEL,
            max_tokens=16000,
            output_config={"effort": RESEARCH_EFFORT},
            tools=tools,
            messages=messages,
        ) as stream:
            response = await stream.get_final_message()
        cost = _cost_of(response)
        total_cost += cost
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)
        used = response.usage.server_tool_use
        searches += getattr(used, "web_search_requests", 0) or 0
        fetches += getattr(used, "web_fetch_requests", 0) or 0
        for block in response.content:
            if block.type == "web_fetch_tool_result":
                code = getattr(block.content, "error_code", None)
                if code:
                    fetch_errors.append(code)
        if response.stop_reason == "pause_turn":
            messages = messages + [{"role": "assistant", "content": response.content}]
            continue
        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "market_search":
                    text, is_error = await _run_market_tool(block.input, market_calls)
                    market_calls += 1
                else:
                    text, is_error = f"Unknown tool {block.name}.", True
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": results},
            ]
            continue
        break
    if response.stop_reason == "refusal":
        logger.warning("Research call was refused; continuing without research.")
        return ""
    # Keep only the final write-up: text after the last tool block (earlier
    # text is the model's working notes between searches). Thinking blocks are
    # not tool blocks and carry no text of their own here.
    blocks = list(response.content)
    last_tool = max(
        (i for i, b in enumerate(blocks) if b.type not in ("text", "thinking", "redacted_thinking")),
        default=-1,
    )
    texts = [b.text for b in blocks[last_tool + 1 :] if b.type == "text"]
    if not texts:
        texts = [b.text for b in blocks if b.type == "text"]
    logger.info(
        f"Research cost ${total_cost:.4f} (stop={response.stop_reason}, searches={searches}, "
        f"fetches={fetches}, fetch_errors={fetch_errors}, market_lookups={market_calls})"
    )
    return "".join(texts).strip()
