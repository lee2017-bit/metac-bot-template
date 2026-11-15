"""
Forecasting LLM that calls Claude through the Anthropic SDK directly, so that
reasoning effort can be set (high effort beat standard effort in all 8 paired
Metaculus baseline bots in Spring 2026) and several Claude models can be mixed
in one ensemble.

The model for each call comes from a context variable that the bot sets per
prediction (see FallTemplateBot2026._research_and_make_predictions), so one
question gets e.g. 2 Fable + 3 Opus forecasts.
"""
from __future__ import annotations

import contextvars
import logging
import os

import anthropic
from forecasting_tools import GeneralLlm, MonetaryCostManager

logger = logging.getLogger(__name__)

# USD per token (input, output).
PRICES = {
    "claude-fable-5-2": (8e-6, 40e-6),
    "claude-opus-5-5": (4e-6, 20e-6),
    "claude-sonnet-5-5": (2e-6, 10e-6),
    "claude-haiku-5-5": (0.1e-6, 0.5e-6),
}

current_model: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_model", default=None
)

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            base_url="https://api.anthropic.com", max_retries=3, timeout=900
        )
    return _client


def cost_of(model: str, usage) -> float:
    inp, out = PRICES.get(model, PRICES["claude-fable-5-2"])
    tokens_in = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0) * 1.25
        + (usage.cache_read_input_tokens or 0) * 0.1
    )
    return tokens_in * inp + usage.output_tokens * out


def parse_ensemble(spec: str) -> list[str]:
    """'claude-fable-5-2:2,claude-opus-5-5:3' -> [fable, fable, opus, opus, opus]"""
    models: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, count = part.partition(":")
        models += [name.strip()] * int(count or 1)
    return models


class ClaudeForecastLlm(GeneralLlm):
    def __init__(self, default_model: str, effort: str = "high") -> None:
        super().__init__(model=f"anthropic/{default_model}", temperature=None)
        self.default_model = default_model
        self.effort = effort

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        model = current_model.get() or self.default_model
        kwargs = {}
        if system_prompt:
            kwargs["system"] = system_prompt
        async with get_client().beta.messages.stream(
            model=model,
            max_tokens=32000,
            output_config={"effort": self.effort},
            # Server-side fallback if a safety classifier declines the request.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            response = await stream.get_final_message()
        cost = cost_of(response.model or model, response.usage)
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{model} declined the forecast request")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise RuntimeError(f"{model} returned no text (stop={response.stop_reason})")
        logger.info(f"{model} forecast call cost ${cost:.4f}")
        return f"[model: {model}]\n{text}"
