"""Parse structured input (command arguments, onboarding answers) into numbers; not intent detection."""

from __future__ import annotations

import re

_NUM = r"\d[\d,]*(?:\.\d+)?"
_MONEY = re.compile(
    rf"(?<![\w.:/#-])(?:(?:₹|rs\.?|inr)\s*)?({_NUM})\s*(k|lakhs?|lacs?|crores?|cr|l)?"
    rf"(?:\s*(?:/-|rs\b\.?|inr\b|rupees?\b))?(?![\w:/%-]|\.\d)",
    re.I,
)
_MULT = {"k": 1_000, "l": 100_000, "lac": 100_000, "lacs": 100_000, "lakh": 100_000, "lakhs": 100_000,
         "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000}  # fmt: skip


def _plausible(m: re.Match) -> bool:
    """Reject phone numbers and ids: 8+ plain digits with no comma, decimal point or k/l/cr suffix."""
    digits = m.group(1)
    return not (len(digits) >= 8 and digits.isdigit() and not m.group(2))


def parse_amount(text: str) -> float | None:
    """ "6000", "6k", "2l", "1.5 lakh", "₹6,000" -> 6000.0 ... ; None if it is not just an amount."""
    m = _MONEY.fullmatch((text or "").strip())
    if not m or not _plausible(m):
        return None
    value = _to_amount(m.group(1), m.group(2))
    return value if value > 0 else None


def _to_amount(num: str, suffix: str | None) -> float:
    value = float(num.replace(",", ""))
    return round(value * _MULT.get((suffix or "").lower(), 1), 2)


_MONTHS_WORD = re.compile(r"(\d+(?:\.\d+)?)\s*(months?|mos?|years?|yrs?|weeks?)\b", re.I)


def parse_goal(text: str) -> tuple[float, int] | None:
    """ "50000 in 6 months", "50k 6", "2 lakh in 1 year" -> (amount, months). None if unclear."""
    low = (text or "").lower()
    span = _MONTHS_WORD.search(low)
    if span:
        n = float(span.group(1))
        unit = span.group(2)[0]
        months = n * 12 if unit == "y" else n / 4.345 if unit == "w" else n
        rest = low[: span.start()] + " " + low[span.end() :]
    else:
        rest, months = low, 0.0
    amounts = [
        (m, _to_amount(m.group(1), m.group(2))) for m in _MONEY.finditer(rest) if _plausible(m)
    ]
    if not amounts:
        return None
    if not span:  # "50000 6": the small trailing number is the months
        if len(amounts) < 2 or not 0 < amounts[-1][1] <= 120:
            return None
        months = amounts[-1][1]
        amounts = amounts[:-1]
    amount = max(a for _, a in amounts)
    months = round(months)
    if amount <= 0 or not 1 <= months <= 600:
        return None
    return amount, months
