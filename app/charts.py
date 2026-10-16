"""Chart images (PNG) drawn from an analytics report. Works without a display.

    png = render_chart(report, "trend")        # one chart
    png = render_dashboard(report)             # a phone-friendly overview of everything useful

Each chart copes with empty or tiny data (it draws a short message instead of a blank box) and with
long labels / many categories. Uses matplotlib's object API (no global state), so it is safe to run in
a worker thread.
"""

from __future__ import annotations

import io
from typing import Any

from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# Palette: Cloudflare orange as the accent on neutral greys. The categorical colours are the same
# validated set the dashboard uses (see the colour notes in site/styles.css), orange first.
ACCENT = "#F6821F"
ACCENT_LIGHT = "#FBC896"
HEAT_LOW = "#FDE6CF"
HEAT_HIGH = "#B34700"  # the deep end of the orange ramp; also used for orange text on white
INK = "#18181B"
MUTED = "#6B6B73"
GRID = "#E7E7EA"
BG = "#FFFFFF"
GOOD = "#1BAF7A"
BAD = "#E34948"
WARN = "#EDA100"
CATEGORY_COLORS = [
    "#F6821F",
    "#4A3AA7",
    "#1BAF7A",
    "#E87BA4",
    "#008300",
    "#EDA100",
    "#E34948",
    "#2A78D6",
    "#5F6B7A",
    "#A8A29E",
]

KINDS = (
    "trend",
    "categories",
    "merchants",
    "monthly",
    "pace",
    "weekdays",
    "networth",
    "budgets",
    "heatmap",
    "stack",
    "sizes",
    "hours",
    "accounts",
)

_FONT = "DejaVu Sans"
for _name in ("Segoe UI", "Inter", "Arial"):  # nicer if present; DejaVu has the rupee sign
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        _FONT = _name
        break


def inr(x: float, pos: Any = None) -> str:
    """Compact rupees for axes and labels: 1,200 -> ₹1.2k, 250,000 -> ₹2.5L."""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.1f}Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.1f}L"
    if a >= 1e3:
        return f"{sign}₹{a / 1e3:.1f}k".replace(".0k", "k")
    return f"{sign}₹{a:,.0f}"


def _fig(w: float = 9, h: float = 5) -> tuple[Figure, Any]:
    fig = Figure(figsize=(w, h), dpi=120, facecolor=BG)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _style(ax)
    return fig, ax


def _style(ax) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(inr))


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(
        title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22 if subtitle else 12
    )
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def _empty(ax, message: str) -> None:
    ax.clear()
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=11,
        transform=ax.transAxes,
    )


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    return buf.getvalue()


def _short(text: str, n: int = 16) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _every(n_points: int, target: int = 7) -> int:
    """Show about `target` x-axis labels: never so many that neighbours touch."""
    return max(1, -(-n_points // target))


def _xlabels(ax, labels: list[str]) -> None:
    """Evenly spaced x labels, rotated only when there are still many."""
    step = _every(len(labels))
    shown = list(range(0, len(labels), step))
    ax.set_xticks(shown)
    crowded = len(shown) > 7
    ax.set_xticklabels(
        [labels[i] for i in shown],
        rotation=35 if crowded else 0,
        ha="right" if crowded else "center",
    )


def _legend_above(ax, ncol: int = 3) -> None:
    """Legend on the title row, right-aligned, so it never covers the data."""
    ax.legend(frameon=False, fontsize=9, ncol=ncol, loc="lower right", bbox_to_anchor=(1.0, 1.0))


# ---------------------------------------------------------------------------- individual charts


def chart_trend(ax, rep: dict) -> None:
    pts = rep["series"]["points"]
    group = rep["series"]["group"]
    if not pts or not any(p["spent"] or p["earned"] for p in pts):
        return _empty(ax, "No transactions in this period")
    xs = list(range(len(pts)))
    ax.bar(xs, [p["spent"] for p in pts], color=ACCENT, width=0.72, label="Spent", zorder=3)
    if len(pts) > 3:
        ax.plot(xs, [p["avg"] for p in pts], color=INK, linewidth=2, label="Trend", zorder=5)
    _xlabels(ax, [p["label"] for p in pts])
    unit = {"day": "per day", "week": "per week", "month": "per month"}[group]
    _title(ax, "Spending over time", f"{rep['query']['describe']} · {unit}")
    ax.set_ylim(top=max([q["spent"] for q in pts] or [1]) * 1.1)
    _legend_above(ax, 2)


def chart_categories(ax, rep: dict) -> None:
    cats = rep["categories"]
    if not cats:
        return _empty(ax, "No spending in this period")
    names = [_short(c["category"], 20) for c in cats][::-1]
    vals = [c["amount"] for c in cats][::-1]
    bars = ax.barh(names, vals, color=ACCENT, height=0.62, zorder=3)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ""))  # keep category names
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK, fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(inr))
    top = max(vals) or 1
    for bar, c in zip(bars, cats[::-1], strict=True):
        ch = c.get("change_pct")
        tag = "" if ch is None else f"  {'▲' if ch > 0 else '▼'}{abs(ch):.0f}%"
        ax.text(
            bar.get_width() + top * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{inr(c['amount'])}  ·  {c['share_pct']:.0f}%{tag}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, top * 1.38)
    _title(ax, "Where the money went", f"{rep['query']['describe']} · ▲▼ vs previous period")


def chart_merchants(ax, rep: dict) -> None:
    ms = rep["merchants"]
    if not ms:
        return _empty(ax, "No merchants recorded in this period")
    names = [_short(m["merchant"], 20) for m in ms][::-1]
    vals = [m["amount"] for m in ms][::-1]
    bars = ax.barh(names, vals, color=ACCENT, height=0.6, zorder=3)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK, fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(inr))
    top = max(vals) or 1
    for bar, m in zip(bars, ms[::-1], strict=True):
        ax.text(
            bar.get_width() + top * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{inr(m['amount'])} · {m['count']}x",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, top * 1.3)
    _title(ax, "Top merchants", rep["query"]["describe"])


def chart_monthly(ax, rep: dict) -> None:
    """Income vs spending per period (uses the series, so it adapts to the chosen range)."""
    pts = rep["series"]["points"]
    if not pts or not any(p["spent"] or p["earned"] for p in pts):
        return _empty(ax, "Nothing to compare yet")
    xs = list(range(len(pts)))
    w = 0.4
    ax.bar(
        [x - w / 2 for x in xs],
        [p["earned"] for p in pts],
        width=w,
        color=GOOD,
        label="Earned",
        zorder=3,
    )
    ax.bar(
        [x + w / 2 for x in xs],
        [p["spent"] for p in pts],
        width=w,
        color=ACCENT,
        label="Spent",
        zorder=3,
    )
    _xlabels(ax, [p["label"] for p in pts])
    _title(ax, "Earned vs spent", rep["query"]["describe"])
    ax.set_ylim(top=max([q[k] for q in pts for k in ("earned", "spent")] or [1]) * 1.1)
    _legend_above(ax, 2)


def chart_pace(ax, rep: dict) -> None:
    p = rep["pace"]
    if not p["this_month"] and not p["last_month"]:
        return _empty(ax, "No spending this month or last month yet")
    ax.plot(
        range(1, len(p["last_month"]) + 1),
        p["last_month"],
        color=MUTED,
        linewidth=2,
        linestyle="--",
        label="Last month",
    )
    ax.plot(
        range(1, len(p["this_month"]) + 1),
        p["this_month"],
        color=ACCENT,
        linewidth=2.6,
        label="This month",
    )
    if p["this_month"] and p["day"] < p["days_in_month"]:
        ax.plot(
            [p["day"], p["days_in_month"]],
            [p["spent"], p["projected"]],
            color=ACCENT,
            linewidth=1.6,
            linestyle=":",
            label="Projected",
        )
        ax.annotate(
            inr(p["projected"]),
            (p["days_in_month"], p["projected"]),
            textcoords="offset points",
            xytext=(-6, 8),
            ha="right",
            fontsize=9,
            color=ACCENT,
        )
    ax.set_xlim(1, max(p["days_in_month"], len(p["last_month"])))
    _title(
        ax,
        "This month vs last month",
        f"Spent so far {inr(p['spent'])} · last month total {inr(p['last_month_total'])}",
    )
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))


def chart_weekdays(ax, rep: dict) -> None:
    wd = rep["weekdays"]
    if not any(w["avg"] for w in wd):
        return _empty(ax, "Not enough spending to show a weekly pattern")
    top = max(w["avg"] for w in wd)
    colors = [ACCENT if w["avg"] == top else ACCENT_LIGHT for w in wd]
    ax.bar([w["day"] for w in wd], [w["avg"] for w in wd], color=colors, width=0.6, zorder=3)
    for i, w in enumerate(wd):
        ax.text(i, w["avg"] + top * 0.02, inr(w["avg"]), ha="center", fontsize=9, color=INK)
    _title(ax, "Which days cost the most", "Average spend per weekday")


def chart_networth(ax, rep: dict) -> None:
    pts = rep["net_worth"]
    if len(pts) < 2 or not any(p["total"] for p in pts):
        return _empty(ax, "Not enough history yet")
    xs = list(range(len(pts)))
    vals = [p["total"] for p in pts]
    ax.plot(xs, vals, color=ACCENT, linewidth=2.4, zorder=3)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or max(abs(hi) * 0.05, 1)
    ax.set_ylim(lo - pad, hi + pad)
    step = _every(len(pts))
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(
        [pts[i]["label"] for i in xs[::step]],
        rotation=0 if len(pts) < 16 else 35,
        ha="center" if len(pts) < 16 else "right",
    )
    delta = vals[-1] - vals[0]
    _title(
        ax,
        "Total money over time",
        f"Now {inr(vals[-1])} · {'+' if delta >= 0 else '-'}{inr(abs(delta))} over the range",
    )


def chart_budgets(ax, rep: dict) -> None:
    bs = rep["budgets"]
    if not bs:
        return _empty(ax, "No budgets set. Tell the bot: set food budget to 6000")
    names = [_short(b["category"], 18) for b in bs][::-1]
    pct = [min(b["percent"], 140) for b in bs][::-1]
    colors = [BAD if b["level"] == "over" else WARN if b["level"] == "warn" else GOOD for b in bs][
        ::-1
    ]
    ax.barh(names, [100] * len(bs), color=GRID, height=0.5, zorder=2)
    ax.barh(names, pct, color=colors, height=0.5, zorder=3)
    ax.axvline(100, color=MUTED, linewidth=1, linestyle="--", zorder=4)
    ax.grid(visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ""))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK, fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
    for i, b in enumerate(bs[::-1]):
        ax.text(
            min(b["percent"], 140) + 2,
            i,
            f"{inr(b['spent'])} of {inr(b['budget'])}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, 175)
    _title(ax, "Monthly budgets", "This month")


def chart_heatmap(ax, rep: dict) -> None:
    hm = rep["heatmap"]
    if not hm["cells"] or not hm["max"]:
        return _empty(ax, "No spending in the last 12 weeks")
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_axis_off()
    top = hm["max"]
    for c in hm["cells"]:
        shade = 0 if c["spent"] == 0 else 0.18 + 0.82 * min(1.0, c["spent"] / top)
        color = "#F1F1F3" if shade == 0 else _blend(HEAT_LOW, HEAT_HIGH, shade)
        ax.add_patch(_rect(c["week"], 6 - c["dow"], color))
    for i, d in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        ax.text(-0.7, 6 - i + 0.45, d, ha="right", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(-1.6, hm["weeks"])
    ax.set_ylim(-0.2, 7.2)
    ax.set_aspect("equal")
    ax.set_title(
        "Daily spending, last 12 weeks",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=12,
    )
    ax.text(
        hm["weeks"],
        -0.9,
        f"darker = more spent (peak {inr(top)})",
        ha="right",
        fontsize=8,
        color=MUTED,
    )


def _rect(x: float, y: float, color: str):
    from matplotlib.patches import FancyBboxPatch

    return FancyBboxPatch(
        (x + 0.06, y + 0.06),
        0.88,
        0.88,
        boxstyle="round,pad=0,rounding_size=0.15",
        linewidth=0,
        facecolor=color,
    )


def _blend(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b, strict=True))


def chart_stack(ax, rep: dict) -> None:
    st = rep["stack"]
    if not st["periods"]:
        return _empty(ax, "No spending in this period")
    xs = list(range(len(st["periods"])))
    bottom = [0.0] * len(xs)
    for i, cat in enumerate(st["categories"]):
        vals = [p["values"].get(cat, 0.0) for p in st["periods"]]
        ax.bar(
            xs,
            vals,
            bottom=bottom,
            color=CATEGORY_COLORS[i % len(CATEGORY_COLORS)],
            label=_short(cat, 16),
            width=0.7,
            edgecolor=BG,
            linewidth=0.8,
            zorder=3,
        )
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    _xlabels(ax, [p["label"] for p in st["periods"]])
    _title(ax, "What you spent on, over time", rep["query"]["describe"])
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=min(4, len(st["categories"])),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )


def chart_sizes(ax, rep: dict) -> None:
    sz = rep["sizes"]
    if not any(s["count"] for s in sz):
        return _empty(ax, "No expenses in this period")
    ax.bar([s["band"] for s in sz], [s["count"] for s in sz], color=ACCENT, width=0.6, zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
    top = max(s["count"] for s in sz)
    for i, s in enumerate(sz):
        if s["count"]:
            ax.text(i, s["count"] + top * 0.02, f"{s['count']}", ha="center", fontsize=9, color=INK)
    _title(ax, "How big are your expenses?", "Number of purchases by size")


def chart_hours(ax, rep: dict) -> None:
    hrs = rep["hours"]
    if not hrs:
        return _empty(ax, "Time of day is only known for chat-logged expenses")
    ax.bar([h["hour"] for h in hrs], [h["total"] for h in hrs], color=ACCENT, width=0.8, zorder=3)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    _title(ax, "Time of day", "When you spend")


def chart_accounts(ax, rep: dict) -> None:
    accs = rep["accounts"]
    if not accs:
        return _empty(ax, "No account activity in this period")
    names = [_short(a["account"], 16) for a in accs]
    xs = list(range(len(accs)))
    w = 0.4
    ax.bar(
        [x - w / 2 for x in xs],
        [a["earned"] for a in accs],
        width=w,
        color=GOOD,
        label="In",
        zorder=3,
    )
    ax.bar(
        [x + w / 2 for x in xs],
        [a["spent"] for a in accs],
        width=w,
        color=ACCENT,
        label="Out",
        zorder=3,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    _title(ax, "Money in and out, by account", rep["query"]["describe"])
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")


_DRAW = {
    "trend": chart_trend,
    "categories": chart_categories,
    "merchants": chart_merchants,
    "monthly": chart_monthly,
    "pace": chart_pace,
    "weekdays": chart_weekdays,
    "networth": chart_networth,
    "budgets": chart_budgets,
    "heatmap": chart_heatmap,
    "stack": chart_stack,
    "sizes": chart_sizes,
    "hours": chart_hours,
    "accounts": chart_accounts,
}

_SIZE = {"categories": (9, 5.6), "merchants": (9, 4.8), "heatmap": (9, 3.8), "budgets": (9, 4.6)}


def render_chart(rep: dict, kind: str) -> bytes:
    """One chart as PNG bytes. Raises ValueError for an unknown kind."""
    draw = _DRAW.get(kind)
    if draw is None:
        raise ValueError(f"Unknown chart '{kind}'. Choose from: {', '.join(KINDS)}")
    fig, ax = _fig(*_SIZE.get(kind, (9, 5)))
    draw(ax, rep)
    return _png(fig)


# ---------------------------------------------------------------------------- overview image


def render_dashboard(rep: dict) -> bytes:
    """A tall, phone-friendly overview: headline numbers, then the charts that matter."""
    s = rep["summary"]
    panels = ["trend", "categories", "pace", "budgets", "networth"]
    if rep["series"]["group"] != "day" or rep["query"]["days"] > 45:
        panels[0] = "monthly"
    panels = [p for p in panels if not (p == "budgets" and not rep["budgets"])]

    heights = [1.8] + [3.3 if p != "categories" else 3.9 for p in panels]
    fig = Figure(figsize=(8.2, sum(heights) + 0.6), dpi=130, facecolor=BG)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(
        len(heights),
        1,
        height_ratios=heights,
        hspace=0.62,
        left=0.2,
        right=0.96,
        top=0.985,
        bottom=0.025,
    )

    head = fig.add_subplot(grid[0])
    head.set_axis_off()
    head.text(
        -0.12, 0.98, "WalletLedger", fontsize=11, color=HEAT_HIGH, fontweight="bold", va="top"
    )
    head.text(-0.12, 0.80, rep["query"]["describe"], fontsize=10, color=MUTED, va="top")
    kpis = [
        ("Spent", inr(s["spent"]), s.get("spent_change_pct"), True),
        ("Earned", inr(s["earned"]), s.get("earned_change_pct"), False),
        ("Saved", inr(s["net"]), None, False),
        ("Per day", inr(s["avg_per_day"]), None, False),
    ]
    for i, (label, value, change, bad_if_up) in enumerate(kpis):
        x = -0.12 + i * 0.29
        head.text(x, 0.50, label, fontsize=9, color=MUTED, va="top")
        head.text(x, 0.40, value, fontsize=17, color=INK, fontweight="bold", va="top")
        if change is not None and abs(change) >= 1:
            up = change > 0
            color = (BAD if up else GOOD) if bad_if_up else (GOOD if up else BAD)
            head.text(
                x,
                0.03,
                f"{'▲' if up else '▼'} {abs(change):.0f}% vs before",
                fontsize=8.5,
                color=color,
                va="bottom",
            )
    for i, kind in enumerate(panels, start=1):
        ax = fig.add_subplot(grid[i])
        _style(ax)
        _DRAW[kind](ax, rep)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, pad_inches=0.2)
    return buf.getvalue()
