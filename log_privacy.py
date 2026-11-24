"""
Keep research and forecast reasoning out of the run logs.

GitHub Actions logs of a public repository can be read by anyone, and the
library logs every research report and every forecaster's reasoning in full
(often while the question is still open). This filter shortens any log record
longer than LOG_TEXT_LIMIT characters to its first line, so costs, URLs,
errors and the run summary stay visible but the text itself does not.

Turned on with LOG_PRIVATE_TEXT=1 (see main.py). Reports saved to the
reports/ folder are unaffected; they are what the bot posts as its private
comment and what later reviews read.
"""
from __future__ import annotations

import logging
import os

LIMIT = int(os.getenv("LOG_TEXT_LIMIT", "400"))


class HideLongText(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if len(msg) <= LIMIT:
            return True
        first = msg.strip().splitlines()[0] if msg.strip() else ""
        record.msg = f"{first[:200]} [... {len(msg) - min(len(first), 200)} characters not logged]"
        record.args = ()
        return True


def install() -> None:
    """Attach the filter to every handler of the root logger (records from
    all loggers pass through these handlers)."""
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, HideLongText) for f in handler.filters):
            handler.addFilter(HideLongText())
