"""Small text-formatting helpers shared by the bot messages (Indian number grouping, compact money)."""

from __future__ import annotations

from datetime import date, datetime


def inr(x: float, decimals: int = 0) -> str:
    """₹1,23,456 (Indian digit grouping). Negative numbers keep a leading minus; -0.4 shows as ₹0."""
    x = round(x, decimals)
    sign = "−" if x < 0 else ""  # the same minus sign the entry lines use
    x = abs(x)
    whole, _, frac = f"{x:.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if decimals else "")


def short(x: float) -> str:
    """Compact money for tight spaces: ₹850, ₹12.5k, ₹3.2L, ₹1.4Cr."""
    a = abs(x)
    sign = "−" if round(x) < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.1f}Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.1f}L"
    if a >= 1e3:
        return f"{sign}₹{a / 1e3:.1f}k".replace(".0k", "k")
    return f"{sign}₹{a:,.0f}"


def change(pct: float | None, up_is_bad: bool = True) -> str:
    """'▲27%' / '▼6%' or '' when there is nothing to compare or the change is tiny."""
    if pct is None or abs(pct) < 1:
        return ""
    return f"{'▲' if pct > 0 else '▼'}{abs(pct):.0f}%"


def day(value: str | date | datetime) -> str:
    """'05 Oct' from an ISO date string / date / datetime."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value[:10])
    return value.strftime("%d %b")


def bar(percent: float, width: int = 10) -> str:
    """Text progress bar: ▰▰▰▱▱▱▱▱▱▱ (capped at full)."""
    filled = max(0, min(width, round(percent / 100 * width)))
    return "▰" * filled + "▱" * (width - filled)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")
