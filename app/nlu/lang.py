"""Lightweight language detection (Arabic vs. English)."""

from __future__ import annotations

import re

from app.schemas import Language

# Arabic Unicode block (incl. supplement and presentation forms).
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def detect_language(text: str) -> Language:
    """Return :class:`Language.AR` if the text is predominantly Arabic, else EN.

    Uses the ratio of Arabic-script characters to total letters so that mixed
    strings (e.g. an Arabic sentence with a Latin name) still route correctly.
    """

    arabic = len(_ARABIC_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic == 0:
        return Language.EN
    if latin == 0:
        return Language.AR
    return Language.AR if arabic >= latin else Language.EN
