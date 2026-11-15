"""
Research step using Claude with Anthropic's server-side web search tool.

Why: the Metaculus LLM proxy closed on 2026-10-27, so with only an Anthropic key
the template's default researcher ("metaculus/gpt-5-search-api") fails. This
module replaces it. metaculus.com is blocked in search so the bot cannot read
(and copy) the community prediction.

Cost of each call is added to forecasting-tools' MonetaryCostManager, so it
shows up in the per-question cost and counts toward the spend cap.
"""
from __future__ import annotations

import logging
import os

import anthropic
from forecasting_tools import MonetaryCostManager

logger = logging.getLogger(__name__)

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-5-5")
MAX_SEARCHES = int(os.getenv("RESEARCH_MAX_SEARCHES", "4"))
RESEARCH_EFFORT = os.getenv("RESEARCH_EFFORT", "low")

# USD per token (input, output). Web search: $10 per 1000 searches.
_PRICES = {
    "claude-fable-5-2": (8e-6, 40e-6),
    "claude-opus-5-5": (4e-6, 20e-6),
    "claude-sonnet-5-5": (2e-6, 10e-6),
    "claude-haiku-5-5": (0.1e-6, 0.5e-6),
}
_SEARCH_PRICE = 0.01

_BLOCKED_DOMAINS = ["metaculus.com"]

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


async def run_claude_web_research(prompt: str) -> str:
    MonetaryCostManager.raise_error_if_limit_would_be_reached()
    client = _get_client()
    messages: list = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": MAX_SEARCHES,
            "blocked_domains": _BLOCKED_DOMAINS,
        }
    ]
    total_cost = 0.0
    # pause_turn: the server-side loop stopped early; send the turn back to resume.
    for _ in range(3):
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
        if response.stop_reason != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": response.content}]
    if response.stop_reason == "refusal":
        logger.warning("Research call was refused; continuing without research.")
        return ""
    # Keep only the final write-up: text after the last tool block (earlier
    # text is the model's working notes between searches).
    blocks = list(response.content)
    last_tool = max(
        (i for i, b in enumerate(blocks) if b.type != "text"), default=-1
    )
    texts = [b.text for b in blocks[last_tool + 1 :] if b.type == "text"]
    if not texts:
        texts = [b.text for b in blocks if b.type == "text"]
    logger.info(f"Research cost ${total_cost:.4f}")
    return "".join(texts).strip()
