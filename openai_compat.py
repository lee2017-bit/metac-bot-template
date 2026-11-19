"""
Forecasting LLM for any OpenAI-compatible API (Nebius Token Factory now,
OpenRouter or another provider later).

Configuration (environment):
  OC_BASE_URL     e.g. https://api.tokenfactory.nebius.com/v1/
  OC_API_KEY_ENV  name of the variable that holds the key (default NEBIUS_API_KEY)
  OC_PRICES       optional JSON {"model": [usd_per_input_token, usd_per_output_token]}
                  overriding the built-in table below.

Every call's cost is computed from the token counts the API returns and added to
forecasting-tools' MonetaryCostManager, so the per-run hard limit and the spend
ledger (spend_guard.py) cover it. A model with no known price is refused, so
nothing can run uncounted.
"""
from __future__ import annotations

import json
import logging
import os

import openai
from forecasting_tools import GeneralLlm, MonetaryCostManager

from claude_llm import current_model

logger = logging.getLogger(__name__)

# USD per token (input, output). Nebius Token Factory list prices as published
# for November 2026; re-check against the live catalog before changing models.
PRICES: dict[str, tuple[float, float]] = {
    "moonshotai/Kimi-K3": (3.00e-6, 15.00e-6),
    "moonshotai/Kimi-K3.1-Pro": (2.00e-6, 10.00e-6),
    "zai-org/GLM-5.3": (1.40e-6, 4.40e-6),
    "zai-org/GLM-5.3-Flash": (0.15e-6, 0.50e-6),
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (0.30e-6, 1.20e-6),
    "Qwen/Qwen3.8-27B": (0.45e-6, 3.00e-6),
}
if os.getenv("OC_PRICES"):
    PRICES.update({k: tuple(v) for k, v in json.loads(os.environ["OC_PRICES"]).items()})

BASE_URL = os.getenv("OC_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
KEY_ENV = os.getenv("OC_API_KEY_ENV", "NEBIUS_API_KEY")

_client: openai.AsyncOpenAI | None = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        key = os.getenv(KEY_ENV)
        if not key:
            raise RuntimeError(f"{KEY_ENV} is not set")
        _client = openai.AsyncOpenAI(
            base_url=BASE_URL, api_key=key, max_retries=3, timeout=600
        )
    return _client


def price_of(model: str) -> tuple[float, float]:
    if model not in PRICES:
        raise RuntimeError(f"No price known for {model}; add it to PRICES or OC_PRICES")
    return PRICES[model]


def cost_of(model: str, usage) -> float:
    if usage is None:
        raise RuntimeError(f"{model} returned no token usage; refusing to count it as free")
    inp, out = price_of(model)
    # Cached input is billed lower by some providers; count it at full price
    # so the cap errs on the safe side.
    return (usage.prompt_tokens or 0) * inp + (usage.completion_tokens or 0) * out


def add_cost(cost: float) -> None:
    MonetaryCostManager.increase_current_usage_in_parent_managers(cost)


def worst_case_cost(model: str, max_tokens: int, prompt_chars: int) -> float:
    """A call that timed out returns no token counts but may still be billed:
    count it as if it had produced max_tokens, so the cap stays on the safe side."""
    inp, out = price_of(model)
    return (prompt_chars / 3) * inp + max_tokens * out


# Nebius returned at most ~16k output tokens (reasoning included) for Kimi-K3
# whatever max_tokens asked for; long hidden reasoning then cut the final
# answer off. Reasoning models partly follow a request to keep it short.
BREVITY = (
    "Your whole reply, hidden reasoning included, is limited to about 16,000 tokens. "
    "Keep your reasoning focused (well under 8,000 tokens) and always finish with the "
    "final answer in exactly the format the prompt asks for."
)


def parser_llm(model: str) -> GeneralLlm:
    """A GeneralLlm (litellm) on the same OpenAI-compatible endpoint, for the
    library's parsing and summarising calls. The model's price is registered
    with litellm so these calls are counted like every other call."""
    import litellm

    inp, out = price_of(model)
    for key in (f"openai/{model}", model):
        litellm.register_model(
            {key: {"input_cost_per_token": inp, "output_cost_per_token": out,
                   "litellm_provider": "openai", "mode": "chat"}}
        )
    return GeneralLlm(
        model=f"openai/{model}",
        api_base=BASE_URL,
        api_key=os.getenv(KEY_ENV),
        temperature=None,
        timeout=180,
    )


class OpenAICompatForecastLlm(GeneralLlm):
    def __init__(
        self,
        default_model: str,
        max_tokens: int = int(os.getenv("FORECAST_MAX_TOKENS_OC", "16000")),
    ) -> None:
        super().__init__(model=f"openai/{default_model}", temperature=None)
        self.default_model = default_model
        self.max_tokens = max_tokens

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        model = current_model.get() or self.default_model
        price_of(model)  # fail before spending if the price is unknown
        system = f"{system_prompt}\n\n{BREVITY}" if system_prompt else BREVITY
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await get_client().with_options(max_retries=1).chat.completions.create(
                model=model, messages=messages, max_tokens=self.max_tokens
            )
        except openai.APITimeoutError:
            add_cost(worst_case_cost(model, self.max_tokens, len(prompt)))
            raise
        cost = cost_of(model, response.usage)
        add_cost(cost)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        tokens = getattr(response.usage, "completion_tokens", None)
        if choice.finish_reason == "length" or not text:
            # A cut-off reply usually lacks the final answer; drop this one
            # forecast rather than let the parser guess from a fragment.
            raise RuntimeError(
                f"{model} reply cut off or empty (finish={choice.finish_reason}, "
                f"{tokens} output tokens, {len(text)} chars of text)"
            )
        logger.info(f"{model} forecast call cost ${cost:.4f} ({tokens} output tokens)")
        return f"[model: {model}]\n{text}"
