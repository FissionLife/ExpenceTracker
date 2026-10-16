"""What the bot sends back: text, optionally with an image / file and tappable buttons.

`Reply` is a plain description (no Telegram types), so the same logic serves Telegram, the dev
endpoint and the tests. app/bot/telegram.py turns a Reply into real Telegram messages.

Callback data (what a button sends back, max 64 bytes) is `<action>:<args>`:
    go:<screen>            open a screen         (home, stats, balance, history, budgets, keys, help)
    st:<period>[:<ctx>]    stats for a period    (7d, 30d, 90d, 1y, all, ...)
    ch:<kind>:<period>[:<ctx>]   one chart
    hi:<page>              history page
    undo:<tx id prefix>    undo a transaction
    ask:yes | ask:no       answer a pending confirmation
    imp:yes | imp:no       confirm / skip a statement import
    key:<test|pause|resume|del>:<n>   manage AI key number n
    ex:<period>            export CSV
    pay:<key>             retry a blocked spend on another account
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Btn:
    text: str
    data: str | None = None  # callback data
    url: str | None = None  # opens a link
    web_app: str | None = None  # opens a Telegram Mini App (https only)


@dataclass
class Reply:
    text: str = ""
    photo: bytes | None = None
    document: tuple[bytes, str] | None = None  # (file bytes, file name)
    rows: list[list[Btn]] = field(default_factory=list)
    edit: bool = False  # for button taps: update the message in place instead of sending a new one
    notice: str = ""  # short pop-up shown when a button is tapped ("Undone")


PERIODS = [("7d", "7 days"), ("30d", "30 days"), ("90d", "90 days"), ("1y", "1 year")]
CHARTS = [
    ("categories", "🍕 Categories"),
    ("trend", "📈 Trend"),
    ("merchants", "🏪 Merchants"),
    ("pace", "⏱ Pace"),
    ("budgets", "🎯 Budgets"),
    ("weekdays", "📅 Weekdays"),
    ("networth", "💰 Total money"),
    ("heatmap", "🗓 Heatmap"),
]


def home_rows() -> list[list[Btn]]:
    return [
        [Btn("📊 Stats", "go:stats"), Btn("💰 Balance", "go:balance")],
        [Btn("🧾 History", "go:history"), Btn("🎯 Budgets", "go:budgets")],
        [Btn("🖥 Dashboard", "go:dashboard"), Btn("🔑 AI keys", "go:keys")],
        [Btn("❓ Help", "go:help")],
    ]


def period_row(current: str, ctx: str = "") -> list[Btn]:
    suffix = f":{ctx}" if ctx else ""
    return [Btn(("• " if p == current else "") + label, f"st:{p}{suffix}") for p, label in PERIODS]


def chart_rows(period: str, ctx: str = "", exclude: str = "") -> list[list[Btn]]:
    suffix = f":{ctx}" if ctx else ""
    buttons = [
        Btn(label, f"ch:{kind}:{period}{suffix}") for kind, label in CHARTS if kind != exclude
    ]
    return [buttons[i : i + 3] for i in range(0, len(buttons), 3)]


def stats_rows(period: str, ctx: str = "") -> list[list[Btn]]:
    return [
        period_row(period, ctx),
        *chart_rows(period, ctx)[:2],
        [Btn("📄 Export CSV", f"ex:{period}"), Btn("🖥 Dashboard", "go:dashboard")],
    ]


def chart_nav_rows(kind: str, period: str, ctx: str = "") -> list[list[Btn]]:
    suffix = f":{ctx}" if ctx else ""
    return [
        [Btn("◀ Overview", f"st:{period}{suffix}"), Btn("📄 Export CSV", f"ex:{period}")],
        *chart_rows(period, ctx, exclude=kind)[:2],
    ]


def confirm_rows(pending_id: str | None = None) -> list[list[Btn]]:
    """Yes / Cancel for one specific question: the id stops an old button answering a newer question."""
    tag = f":{pending_id}" if pending_id else ""
    return [[Btn("✅ Yes, do it", f"ask:yes{tag}"), Btn("✖ Cancel", f"ask:no{tag}")]]


def import_rows() -> list[list[Btn]]:
    """History only (balances stay as they are), history plus balance update, or skip."""
    return [
        [Btn("📚 Import as history", "imp:hist"), Btn("💰 Import + update balance", "imp:apply")],
        [Btn("✖ Skip", "imp:no")],
    ]


# Telegram limits a button's data to 64 bytes. Up to 6 ids fit in the button itself; a bigger batch
# (a statement import, a long message) is remembered here and the button carries a short key instead.
_UNDO_BATCHES: dict[str, list[str]] = {}
_UNDO_BATCH_LIMIT = 500


def undo_batch(key: str) -> list[str] | None:
    return _UNDO_BATCHES.get(key)


# A blocked spend can be re-tried on another account. The retry args (amount, category, the other
# account...) don't fit in a 64-byte button, so they are remembered here under a short token.
_PAY_RETRIES: dict[str, dict] = {}
_PAY_RETRY_LIMIT = 500


def remember_pay_retry(args: dict) -> str:
    import secrets

    key = secrets.token_hex(4)
    if len(_PAY_RETRIES) >= _PAY_RETRY_LIMIT:
        _PAY_RETRIES.pop(next(iter(_PAY_RETRIES)))
    _PAY_RETRIES[key] = dict(args)
    return key


def pay_retry(key: str) -> dict | None:
    return _PAY_RETRIES.pop(key, None)  # single use: once tapped, forget it


def pay_alt_row(account: str, balance: float, args: dict) -> list[Btn]:
    """Buttons shown when a spend was blocked for insufficient funds.

    Two buttons: switch to another pipe that can cover the amount, or dismiss. Nothing is
    kept pending; the Cancel button just closes the keyboard back to home.
    """
    from app.bot import fmt

    key = remember_pay_retry(args)
    return [
        Btn(f"🏦 Use {account} instead ({fmt.inr(balance)})", f"pay:{key}"),
        Btn("❌ Cancel", "go:home"),
    ]


def cancel_row() -> list[Btn]:
    """Just a Cancel button, for a blocked spend with no other pipe that could cover it."""
    return [Btn("❌ Cancel", "go:home")]


def undo_row(tx_ids: list[str]) -> list[Btn]:
    if len(tx_ids) == 1:
        return [Btn("↩️ Undo", f"undo:{tx_ids[0][:12]}")]
    label = f"↩️ Undo ({len(tx_ids)})"
    if len(tx_ids) <= 6:
        return [Btn(label, "undo:" + ",".join(t[:8] for t in tx_ids))]
    import secrets

    key = secrets.token_hex(4)
    if len(_UNDO_BATCHES) >= _UNDO_BATCH_LIMIT:
        _UNDO_BATCHES.pop(next(iter(_UNDO_BATCHES)))  # forget the oldest
    _UNDO_BATCHES[key] = list(tx_ids)
    return [Btn(label, f"undo:b{key}")]


def history_rows(page: int, pages: int, undo_id: str | None = None) -> list[list[Btn]]:
    nav = []
    if page > 0:
        nav.append(Btn("◀ Newer", f"hi:{page - 1}"))
    if page < pages - 1:
        nav.append(Btn("Older ▶", f"hi:{page + 1}"))
    rows = [nav] if nav else []
    # a specific id, so tapping twice (or on an old message) cannot undo the NEXT entry back
    undo = f"undo:{undo_id[:12]}" if undo_id else "undo:last"
    rows.append([Btn("↩️ Undo last", undo), Btn("📄 Export CSV", "ex:all")])
    return rows


def keys_rows(key_ids: list[str]) -> list[list[Btn]]:
    """Buttons for the first 6 keys. They carry the key's id ('k' + 8 characters), not its position,
    so deleting key 1 and tapping an old "delete 2" button cannot remove a different key."""
    rows = []
    for n, key_id in enumerate(key_ids[:6], start=1):
        ref = f"k{key_id[:8]}"
        rows.append(
            [
                Btn(f"🧪 {n}", f"key:test:{ref}"),
                Btn(f"⏸ {n}", f"key:pause:{ref}"),
                Btn(f"▶ {n}", f"key:resume:{ref}"),
                Btn(f"🗑 {n}", f"key:del:{ref}"),
            ]
        )
    rows.append([Btn("🧪 Test all", "key:test:all"), Btn("📊 Status", "go:status")])
    return rows


def back_home() -> list[list[Btn]]:
    return [[Btn("🏠 Menu", "go:home")]]
