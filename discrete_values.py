"""
Per-outcome probabilities for discrete questions with few possible outcomes.

Asking for percentiles on a 5-30 outcome grid and interpolating smears mass
between outcomes (in one live case it even moved the most likely outcome).
Here the forecaster gives a probability for every outcome and the cumulative
distribution is built from those directly.

Pure functions, no Metaculus or LLM dependencies, so they can be unit-tested
and reused.
"""
from __future__ import annotations

import re

MARKER = "FINAL PROBABILITIES"

_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+"
_LINE = re.compile(
    rf"^(?:[\s>*•]|-(?=\s))*(?P<tail>below|above|less than|more than|under|over|lower than|higher than|<|>)?\s*"
    rf"(?P<value>{_NUM})[^:=\n]*[:=]\s*\**\s*(?P<prob>{_NUM})\s*%",
    re.IGNORECASE | re.MULTILINE,
)
_LOW_WORDS = {"below", "less than", "under", "lower than", "<"}


def outcome_values(lower_bound: float, upper_bound: float, cdf_size: int) -> list[float]:
    """The possible outcomes: centres of the cdf_size-1 equal bins between the bounds."""
    n = cdf_size - 1
    step = (upper_bound - lower_bound) / n
    return [lower_bound + (k + 0.5) * step for k in range(n)]


def step_decimals(step: float) -> int:
    for d in range(0, 7):
        if abs(round(step, d) - step) <= 1e-9 * max(1.0, abs(step)):
            return d
    return 6


def format_value(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return "0" if decimals == 0 and text == "-0" else text


def parse_outcome_probabilities(
    text: str, values: list[float]
) -> tuple[list[float], float, float]:
    """
    Read "value: NN%" lines (after the last MARKER if present) and return
    (probability per outcome, probability below the first outcome, probability
    above the last outcome), all as fractions that sum to 1.
    Raises ValueError when the lines do not describe a usable distribution.
    """
    if len(values) < 2:
        raise ValueError("need at least two outcomes")
    idx = text.upper().rfind(MARKER)
    section = text[idx + len(MARKER) :] if idx >= 0 else text
    step = values[1] - values[0]
    probs: dict[int, float] = {}
    below = above = None
    for m in _LINE.finditer(section):
        value = float(m.group("value").replace(",", ""))
        prob = float(m.group("prob").replace(",", ""))
        if not 0 <= prob <= 100:
            continue
        tail = (m.group("tail") or "").lower()
        if tail:
            if tail in _LOW_WORDS:
                below = prob
            else:
                above = prob
            continue
        k = round((value - values[0]) / step)
        if 0 <= k < len(values) and abs(values[k] - value) <= 0.26 * abs(step):
            probs[k] = prob  # a later line for the same outcome overrides an earlier one
    if len(probs) < max(2, (len(values) + 1) // 2):
        raise ValueError(f"only {len(probs)} of {len(values)} outcomes had a probability")
    total = sum(probs.values()) + (below or 0) + (above or 0)
    if not 85 <= total <= 115:
        raise ValueError(f"probabilities sum to {total:.1f}%, not 100%")
    inbound = [probs.get(k, 0.0) / total for k in range(len(values))]
    return inbound, (below or 0) / total, (above or 0) / total


def build_cdf(
    inbound: list[float],
    below: float,
    above: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
    uniform_floor: float = 0.03,
    min_open_tail: float = 0.01,
) -> list[float]:
    """
    Cumulative probabilities at the len(inbound)+1 points Metaculus expects:
    cdf[0] = P(below the first outcome), cdf[k+1] - cdf[k] = P(outcome k),
    1 - cdf[-1] = P(above the last outcome).
    A small uniform share is mixed into the in-range outcomes so that no
    outcome is left at (almost) zero, and open ends keep a minimum mass.
    """
    below = max(below, min_open_tail) if open_lower_bound else 0.0
    above = max(above, min_open_tail) if open_upper_bound else 0.0
    n = len(inbound)
    in_total = sum(inbound)
    if in_total <= 0:
        inbound, in_total = [1.0 / n] * n, 1.0
    target = 1.0 - below - above
    if target <= 0:
        raise ValueError("no probability left inside the range")
    mixed = [
        target * ((1 - uniform_floor) * p / in_total + uniform_floor / n) for p in inbound
    ]
    cdf = [below]
    for p in mixed:
        cdf.append(cdf[-1] + p)
    cdf[-1] = 1.0 - above  # remove float drift
    return cdf
