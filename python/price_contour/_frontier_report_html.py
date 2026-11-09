"""Render a frontier report (version 1) as one self-contained HTML page.

:func:`frontier_report_html` is a pure view of the report dict: it computes
nothing that isn't in the record except layout, so a page can be rebuilt from
JSON. The page has no scripts and loads nothing: charts are inline SVG with
hover values from SVG ``<title>``, and the tabs, the page-wide point switch and
the factor rail are CSS-only radio buttons. It opens offline, in MLflow's
artifact viewer (a sandboxed iframe) and prints.

Colour roles: blue is the starting point (the existing rates), orange is the
selected point's new rates. Every label, level and metadata value is
HTML-escaped; element ids are built from indices, never from labels.

The page structure, components and styling are adapted from t-boost's model
report (``python/t_boost/_html_report.py``, Apache-2.0).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from html import escape
from string import Formatter
from typing import Any

REPORT_VERSION = 1

# Page geometry (CSS px), as t-boost: a chart is drawn at the inner width of its
# card, so one SVG unit is one CSS pixel on a desktop screen.
_CONTENT = 1200
_GAP = 16
_INNER = 2 * 16 + 2
_W_FULL = _CONTENT - _INNER
_W_HALF = (_CONTENT - _GAP) // 2 - _INNER
_RAIL = 240
_H_ROW = 220
_FONT = 12.0

# Page-wide budgets (DESIGN_DECISIONS §19.5).
_TABLE_CELLS = 120_000
_SEGMENT_CELLS = 60_000
_TRACE_ROWS = 2_000
_COMPARE_ROWS = 300

# Beeswarm geometry (DESIGN_DECISIONS §19.4).
_ROW_H = 28.0
_BEESWARM_DOTS = 500
_PAGE_DOTS = 18_000
_BEESWARM_LABEL_W = 170.0
_BEESWARM_RIGHT = 84.0
_DOT_MIN_D = 6.0
_DOT_D_RANGE = 28.0
_JITTER = 0.36
_GOLDEN = 0.618034
_ZERO_DOMAIN = 0.01
_COLOUR_LOW = (11, 138, 251)
_COLOUR_HIGH = (255, 11, 81)
_COLOUR_STEPS = 10
_COLOUR_TITLE = "Level value (numeric) or sample size (categorical)"

# Other drawn marks, per point.
_TRANSITION_DOTS = 300
_TRACE_POINTS = 500

_PERCENTILES = ("p1", "p5", "p25", "p50", "p75", "p95", "p99")
_MAX_SLICES = 8

_LIGHT = (
    "--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;"
    "--grid:#e7e6e0;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--wash:rgba(11,11,11,.045);"
    "--data:#2a78d6;--model:#eb6834;--model-soft:#f6b89e;--other:#c3c2b9;--neutral:#b4b3aa;"
    "--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--good-ink:#006300;--link:#1c5cab;"
    "--hi:#c6ecc3;--lo:#fbeba6"
)
_DARK = (
    "--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8d86;"
    "--grid:#2c2c2a;--axis:#3d3d3a;--border:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);"
    "--data:#3987e5;--model:#d95926;--model-soft:#7a3a20;--other:#55544f;--neutral:#6d6c66;"
    "--good-ink:#0ca30c;--link:#86b6ef;--hi:#2c4f30;--lo:#544a1b"
)

_CSS = """
:root{color-scheme:light;@LIGHT@}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;@DARK@}}
:root[data-theme="dark"]{color-scheme:dark;@DARK@}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.app{max-width:@CONTENT@px;margin:0 auto;padding:0 16px 16px;box-sizing:content-box}
.top{padding:10px 0 0}
.top h1{font-size:20px;line-height:26px;margin:0;font-weight:650}
.top .sub{color:var(--ink2);font-size:13px;line-height:18px;margin:2px 0 0}
.bar{position:sticky;top:0;z-index:20;background:var(--page);display:flex;flex-wrap:wrap;
align-items:center;justify-content:space-between;gap:8px 16px;padding:8px 0;margin-bottom:12px;
border-bottom:1px solid var(--grid)}
.tabs{display:flex;flex-wrap:wrap;gap:2px}
.tabs label{display:block;padding:5px 12px;border:1px solid transparent;border-radius:7px;
cursor:pointer;color:var(--ink2);font-size:14px;line-height:20px;user-select:none}
.tabs label:hover{color:var(--ink);background:var(--wash)}
.switch{display:inline-flex;flex-wrap:wrap;border:1px solid var(--border);border-radius:8px;
overflow:hidden;background:var(--surface);font-size:13px;line-height:18px}
.switch .cap{padding:5px 10px;color:var(--muted)}
.switch label{padding:5px 10px;cursor:pointer;color:var(--ink2);border-left:1px solid var(--border);
user-select:none;font-variant-numeric:tabular-nums}
.switch label:hover{color:var(--ink)}
.sr{position:absolute;opacity:0;width:1px;height:1px;margin:0;pointer-events:none}
.sr.stay{position:fixed;top:0;left:0}
section.tab{display:flex;flex-direction:column;gap:@GAP@px}
.row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:@GAP@px;align-items:stretch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:12px 16px;min-width:0;display:flex;flex-direction:column}
.card h3{font-size:15px;line-height:20px;margin:0;font-weight:650}
.card .what{font-size:12.5px;line-height:17px;color:var(--ink2);margin:2px 0 8px}
.card.solo>h3,.card.solo>.head{margin-bottom:8px}
.card .head{display:flex;align-items:center;justify-content:space-between;gap:12px}
.card h4{font-size:13px;line-height:18px;margin:14px 0 6px;color:var(--ink2);font-weight:600}
.card>details,.card>div>details{margin-top:8px}
svg.chart{display:block;max-width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:@FONT@px;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
svg .lbl{fill:var(--ink2)}
svg .strong{fill:var(--ink);font-weight:600}
svg .gl{stroke:var(--grid);stroke-width:1}
svg .ax{stroke:var(--axis);stroke-width:1}
svg .ref{stroke:var(--ink2);stroke-width:1;opacity:.55}
svg .mk{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}
svg .line{stroke:var(--model);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .mdot{fill:var(--model);stroke:var(--surface);stroke-width:1.5}
svg .hollow{fill:var(--surface);stroke:var(--model);stroke-width:2}
svg .ddot{fill:var(--data);stroke:var(--surface);stroke-width:1.5}
svg .ring{fill:none;stroke:var(--ink);stroke-width:2}
svg .hbar{fill:var(--model)}
svg .flat{fill:var(--other)}
svg .hit{fill:transparent}
svg .hit:hover{fill:var(--wash)}
svg .dbar{fill:var(--data)}
svg .ln-d{stroke:var(--data);stroke-width:2;fill:none;stroke-linejoin:round}
svg .ln-m{stroke:var(--model);stroke-width:2;fill:none;stroke-linejoin:round}
svg .mk-d{stroke:var(--data);stroke-width:1.5;stroke-dasharray:4 3}
svg .mk-m{stroke:var(--model);stroke-width:1.5;stroke-dasharray:4 3}
svg .mk-t{stroke:var(--ink);stroke-width:1.5}
svg circle[class^=sw]{stroke:var(--surface);stroke-width:.6;opacity:.9}
svg .tr{stroke:var(--surface);stroke-width:1;opacity:.9}
svg .zm3{fill:var(--data)}
svg .zm2{fill:color-mix(in srgb,var(--data) 66%,var(--neutral))}
svg .zm1{fill:color-mix(in srgb,var(--data) 33%,var(--neutral))}
svg .z0{fill:var(--neutral)}
svg .zp1{fill:color-mix(in srgb,var(--model) 33%,var(--neutral))}
svg .zp2{fill:color-mix(in srgb,var(--model) 66%,var(--neutral))}
svg .zp3{fill:var(--model)}
.ptsel{display:none}
.dc{display:contents}
table{border-collapse:collapse;width:100%;font-size:13px;line-height:18px}
th,td{padding:5px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:baseline}
th{color:var(--ink2);font-weight:600;white-space:nowrap}
th:first-child,td:first-child{padding-left:0}
th:last-child,td:last-child{padding-right:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.tw{overflow-x:auto}
.tw.scroll{max-height:440px;overflow-y:auto}
.tw.scroll thead th{position:sticky;top:0;background:var(--surface);z-index:1}
.kv td:first-child{color:var(--ink2);width:32%;padding-right:12px}
td.g1{background:color-mix(in srgb,var(--hi) 20%,transparent)}
td.g2{background:color-mix(in srgb,var(--hi) 40%,transparent)}
td.g3{background:color-mix(in srgb,var(--hi) 60%,transparent)}
td.g4{background:color-mix(in srgb,var(--hi) 80%,transparent)}
td.g5{background:var(--hi)}
td.y1{background:color-mix(in srgb,var(--lo) 20%,transparent)}
td.y2{background:color-mix(in srgb,var(--lo) 40%,transparent)}
td.y3{background:color-mix(in srgb,var(--lo) 60%,transparent)}
td.y4{background:color-mix(in srgb,var(--lo) 80%,transparent)}
td.y5{background:var(--lo)}
details>summary{cursor:pointer;color:var(--ink2);font-size:13px;line-height:20px}
details[open]>summary{margin-bottom:6px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;line-height:18px;font-weight:600;
padding:0 8px 0 3px;border-radius:10px;border:1px solid var(--border);white-space:nowrap;color:var(--ink2)}
.badge i{font-style:normal;display:inline-block;width:14px;height:14px;border-radius:7px;color:#fff;
font-size:10px;line-height:14px;text-align:center}
.b-critical i{background:var(--crit)}.b-warning i{background:var(--warn);color:#0b0b0b}
.b-pass i{background:var(--good)}.b-info i{background:var(--muted)}
.counts{display:flex;gap:6px;flex-wrap:wrap}
.issues{list-style:none;margin:0;padding:0}
.issues li{display:grid;grid-template-columns:86px minmax(0,1fr);gap:10px;align-items:baseline;
padding:5px 0;border-top:1px solid var(--grid);font-size:13px;line-height:18px}
.issues li:first-child{border-top:0;padding-top:0}
.issues .badge{justify-self:start}
.link{color:var(--link);cursor:pointer;font-size:13px;line-height:18px}
.link:hover{text-decoration:underline}
.more{margin-top:auto;padding-top:6px}
.ok{color:var(--good-ink);font-size:13px;margin:4px 0}
.note{color:var(--ink2);font-size:13px;margin:4px 0}
.muted{color:var(--muted)}
.two{display:block}
.two .sub{display:block;color:var(--ink2);font-size:12px;line-height:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:@GAP@px}
.kpi .v{font-size:22px;line-height:28px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .d{font-size:13px;line-height:18px;color:var(--ink2);font-variant-numeric:tabular-nums}
.kpi .lb{font-size:13px;line-height:18px;color:var(--ink2)}
.fxgrid{display:grid;grid-template-columns:@RAIL@px minmax(0,1fr);gap:@GAP@px;align-items:start}
.rail{position:sticky;top:64px;max-height:calc(100vh - 134px);overflow-y:auto;padding:6px}
.rail .rl{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 8px;padding:7px 10px 8px;
border-radius:8px;cursor:pointer;user-select:none}
.rail .rl:hover{background:var(--wash)}
.rail .nm{font-size:13.5px;line-height:18px;font-weight:600;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.rail .pc{font-size:12px;line-height:18px;color:var(--ink2);font-variant-numeric:tabular-nums}
.rail .bars{grid-column:1/-1;display:flex;height:5px;border-radius:3px;overflow:hidden;
background:var(--grid)}
.rail .bars i{display:block;height:5px;background:var(--model)}
.chg{display:inline-block;width:64px;height:8px;position:relative;background:var(--grid);
border-radius:2px;vertical-align:middle;margin-left:6px}
.chg i{position:absolute;top:0;height:8px;background:var(--model);border-radius:2px}
.chg b{position:absolute;top:-1px;left:50%;width:1px;height:10px;background:var(--ink2)}
.fxgrid th,#segments th,#impact th{white-space:normal}
@media (max-width:980px){.row,.fxgrid{grid-template-columns:minmax(0,1fr)}
.rail{position:static;max-height:none}}
@media print{:root{@LIGHT@}body{background:#fff}.bar,.rail{display:none!important}
section.tab{display:flex!important;break-before:page}
.row,.fxgrid{grid-template-columns:minmax(0,1fr)}
.card{break-inside:avoid}.tw.scroll{max-height:none;overflow:visible}}
"""

_Fmt = Callable[[float], str]


# ---------------------------------------------------------------------- formatting ----------


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    x = float(v)
    if not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    magnitude = abs(x)
    if magnitude >= 1e4 or (magnitude >= 1e3 and x == round(x)):
        return f"{x:,.0f}"
    if magnitude >= 1e3:
        return f"{x:,.1f}"
    if magnitude < 1e-4:
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def _fmt(column: dict[str, str], value: float | None) -> str:
    """``value`` in the column's own format; not escaped."""
    return "—" if value is None else column["format"].format(value)


def _signed(column: dict[str, str], value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return _fmt(column, 0.0)
    return ("+" if value > 0 else "−") + _fmt(column, abs(value))


def _fine(column: dict[str, str], value: float | None) -> str:
    """A signed per-quote amount. A column's format suits portfolio totals, so keep its
    prefix and suffix but give small values enough digits to read."""
    if value is None:
        return "—"
    sign = "" if value == 0 else ("+" if value > 0 else "−")
    fields = list(Formatter().parse(column["format"]))
    prefix, spec = fields[0][0], fields[0][2] or ""
    suffix = "".join(field[0] for field in fields[1:])
    if spec.endswith("%"):
        return f"{sign}{prefix}{abs(value) * 100:.3g}%{suffix}"
    return f"{sign}{prefix}{_num(abs(value), 3)}{suffix}"


def _minus(text: str) -> str:
    return text.replace("-", "−")


def _signed_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.{digits}f}%".replace("-", "−")


def _share(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _display(r: dict[str, Any]) -> dict[str, Any]:
    """How a step reads; without a display, its own scenario value."""
    display = r["run"]["scenario_display"]
    if display is not None:
        return display
    return {
        "label": "Scenario value",
        "values": r["run"]["scenario_values"],
        "format": "{:.4g}",
        "target": None,
    }


def _disp(display: dict[str, Any], value: float | None) -> str:
    """A display value; not escaped."""
    return "—" if value is None else _minus(display["format"].format(value))


def _disp_signed(display: dict[str, Any], value: float | None) -> str:
    """A signed change in a display value; not escaped."""
    if value is None:
        return "—"
    pattern = display["format"].replace("{:+", "{:")
    text = pattern.format(abs(value))
    # A change that rounds to nothing reads as nothing, not as "−0.0%".
    return text if text == pattern.format(0.0) else ("+" if value > 0 else "−") + text


def _short(label: str, n: int = 22) -> str:
    return label if len(label) <= n else label[: max(n - 1, 1)] + "…"


def _text_w(s: str, size: float = _FONT) -> float:
    return len(s) * size * 0.56


# -------------------------------------------------------------------------- scales ----------


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    out: list[float] = []
    t = math.ceil(lo / step - 1e-9) * step
    while t <= hi + 1e-9 * step and len(out) < 50:
        out.append(round(t, 12))
        t += step
    return out


class _Lin:
    """Linear scale ``[lo, hi] -> [a, b]`` padded by ``pad`` of the span."""

    def __init__(
        self,
        lo: float,
        hi: float,
        a: float,
        b: float,
        *,
        pad: float = 0.06,
        zero: bool = False,
        n: int = 4,
    ) -> None:
        if hi <= lo:
            lo, hi = lo - (abs(lo) * 0.1 or 0.5), hi + (abs(hi) * 0.1 or 0.5)
        span = hi - lo
        lo2, hi2 = lo - span * pad, hi + span * pad
        if zero and lo >= 0:
            lo2 = 0.0
        self.lo, self.hi, self.a, self.b = lo2, hi2, a, b
        self.ticks = _ticks(lo2, hi2, n)

    def __call__(self, v: float) -> float:
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)


class _Svg:
    """An SVG drawn in CSS pixels."""

    def __init__(self, w: float, h: float, label: str) -> None:
        self.w, self.h, self.label = w, h, label
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(
        self, x: float, y: float, s: str, *, anchor: str = "start", cls: str = ""
    ) -> None:
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{a}{c}>{_e(s)}</text>')

    def hline(self, x1: float, x2: float, y: float, cls: str) -> None:
        self.add(
            f'<line class="{cls}" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )

    def vline(self, x: float, y1: float, y2: float, cls: str) -> None:
        self.add(
            f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>'
        )

    def hit(self, x: float, y: float, w: float, h: float, title: str) -> None:
        self.add(
            f'<rect class="hit" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}">'
            f"<title>{_e(title)}</title></rect>"
        )

    def render(self) -> str:
        return (
            f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
            f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}" role="img" '
            f'aria-label="{_e(self.label)}">' + "".join(self.parts) + "</svg>"
        )


def _bar(
    cls: str, x: float, y_top: float, w: float, y_base: float, r: float = 2.0
) -> str:
    """A column from the baseline up to ``y_top``, rounded at the data end only."""
    h = y_base - y_top
    if h <= 0.2 or w <= 0:
        return ""
    r = min(r, w / 2, h)
    return (
        f'<path class="{cls}" d="M{x:.1f},{y_base:.1f}V{y_top + r:.1f}Q{x:.1f},{y_top:.1f} '
        f"{x + r:.1f},{y_top:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y_top:.1f} "
        f'{x + w:.1f},{y_top + r:.1f}V{y_base:.1f}Z"/>'
    )


# ---------------------------------------------------------------------- components ----------


def _table(
    cols: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric: Iterable[int] = (),
    row_class: Sequence[str] | None = None,
    scroll: bool = False,
    tones: Sequence[Sequence[str]] | None = None,
) -> str:
    """Rows of already-escaped cell HTML."""
    num = set(numeric)
    head = "".join(
        f'<th class="{"n" if i in num else ""}">{_e(c)}</th>'
        for i, c in enumerate(cols)
    )
    body = []
    for r_i, row in enumerate(rows):
        rc = f' class="{row_class[r_i]}"' if row_class and row_class[r_i] else ""
        tone = tones[r_i] if tones is not None else []
        cells = []
        for i, c in enumerate(row):
            classes = " ".join(
                x
                for x in ("n" if i in num else "", tone[i] if i < len(tone) else "")
                if x
            )
            cells.append(
                f'<td class="{classes}">{c}</td>' if classes else f"<td>{c}</td>"
            )
        body.append(f"<tr{rc}>" + "".join(cells) + "</tr>")
    box = " scroll" if scroll else ""
    return (
        f'<div class="tw{box}"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _badge(status: str, text: str | None = None) -> str:
    icon = {"critical": "!", "warning": "!", "pass": "✓", "info": "i"}.get(status, "?")
    word = text if text is not None else {"info": "note"}.get(status, status)
    return f'<span class="badge b-{_e(status)}"><i aria-hidden="true">{icon}</i>{_e(word)}</span>'


def _card(
    title: str,
    body: str,
    *,
    what: str = "",
    extra: str = "",
    cls: str = "",
    titles: Sequence[str] | None = None,
) -> str:
    """A card; ``titles`` gives one title per point, shown with it by the switch."""
    shown = (
        "".join(f'<span class="pt pt-{i}">{_e(t)}</span>' for i, t in enumerate(titles))
        if titles
        else _e(title)
    )
    head = f'<h3 title="{_e(title)}">{shown}</h3>'
    if extra:
        head = f'<div class="head">{head}{extra}</div>'
    caption = f'<p class="what">{_e(what)}</p>' if what else ""
    classes = "card" + ("" if what else " solo") + (f" {cls}" if cls else "")
    return f'<div class="{classes}">{head}{caption}{body}</div>'


def _row(*cards: str) -> str:
    return '<div class="row">' + "".join(cards) + "</div>"


def _two(main: str, sub: str) -> str:
    return f'<span class="two">{main}<span class="sub">{sub}</span></span>'


class _Radio:
    """A CSS-only radio group. Its inputs sit first in ``.app``; its labels may sit in
    ``.bar`` or anywhere in ``.body``; content tagged ``{name} {name}-{i}`` shows only
    for option ``i``.

    A clicked label focuses its input and the browser scrolls to a focused element, so
    a group's inputs at the top of the page take the reader there. That suits tabs.
    ``keep_scroll`` fixes the inputs in the viewport, so choosing an option leaves the
    page where it is."""

    def __init__(
        self, name: str, n: int, default: int = 0, *, keep_scroll: bool = False
    ) -> None:
        self.name, self.n, self.default = name, max(n, 1), default
        self.input_class = "sr stay" if keep_scroll else "sr"

    def inputs(self) -> str:
        return "".join(
            f'<input class="{self.input_class}" type="radio" name="{self.name}" '
            f'id="{self.name}-{i}"'
            f"{' checked' if i == self.default else ''}>"
            for i in range(self.n)
        )

    def label(self, i: int, html: str, cls: str = "", title: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        t = f' title="{_e(title)}"' if title else ""
        return f'<label for="{self.name}-{i}"{c}{t}>{html}</label>'

    def tag(self, i: int) -> str:
        return f"{self.name} {self.name}-{i}"

    def css(self, selected: str) -> str:
        out = []
        for i in range(self.n):
            on = f"#{self.name}-{i}:checked"
            lab = f"label[for={self.name}-{i}]"
            out.append(f"{on}~.body .{self.name}:not(.{self.name}-{i}){{display:none}}")
            if selected:
                out.append(f"{on}~.bar {lab},{on}~.body {lab}{{{selected}}}")
            out.append(
                f"#{self.name}-{i}:focus-visible~.bar {lab},#{self.name}-{i}:focus-visible~"
                f".body {lab}{{outline:2px solid var(--data);outline-offset:-2px}}"
            )
        return "".join(out)


def _swarm_css() -> str:
    """The beeswarm's colour scale, one class per step."""
    out = []
    for k in range(_COLOUR_STEPS + 1):
        t = k / _COLOUR_STEPS
        red, green, blue = (
            round(a + (b - a) * t) for a, b in zip(_COLOUR_LOW, _COLOUR_HIGH)
        )
        out.append(f"svg .sw{k}{{fill:rgb({red},{green},{blue})}}")
    return "".join(out)


def _point_css(n: int) -> str:
    """The selected point's table rows washed and its chart markers shown."""
    return "".join(
        f"#pt-{i}:checked~.body .ptr-{i} td{{background:var(--wash)}}"
        f"#pt-{i}:checked~.body .ptsel-{i}{{display:inline}}"
        for i in range(n)
    )


# -------------------------------------------------------------------- report helpers --------


def _column(r: dict[str, Any], column_id: str) -> dict[str, str]:
    return next(c for c in r["columns"] if c["id"] == column_id)


def _objective(r: dict[str, Any]) -> dict[str, str]:
    return r["columns"][0]


def _baseline_name(r: dict[str, Any]) -> str:
    run = r["run"]
    if run["has_existing_rates"] or run["base_multiplier"] is not None:
        return "existing rates"
    return "no adjustment"


def _takeaway(r: dict[str, Any], point: dict[str, Any]) -> str:
    """One line on what the point does, computed from the data."""
    objective = _objective(r)
    pieces = [_change_phrase(objective, point["totals"][objective["id"]])]
    for constraint in r["run"]["swept"]:
        pieces.append(
            _change_phrase(_column(r, constraint), point["totals"][constraint])
        )
    head = f"{point['name']}: {pieces[0]}"
    return head if len(pieces) == 1 else f"{head}, for {', '.join(pieces[1:])}"


def _change_phrase(column: dict[str, str], entry: dict[str, Any]) -> str:
    amount = (
        _signed_pct(entry["pct"], 2)
        if entry["pct"] is not None
        else _signed(column, entry["change"])
    )
    return f"{amount} {column['label']}"


def _level_score(levels: list[dict[str, Any]]) -> float:
    """Quote-weighted mean of ``abs(log_change)``."""
    total = sum(lv["quotes"] for lv in levels)
    if total == 0:
        return 0.0
    return sum(lv["quotes"] * abs(lv["log_change"]) for lv in levels) / total


def _ranked_factors(point: dict[str, Any], top_n: int) -> list[int]:
    """Factor positions by size of change, descending; spec order on a tie."""
    scored = [
        (-_level_score(card["levels"]), f) for f, card in enumerate(point["rate_card"])
    ]
    scored.sort()
    return [f for _, f in scored[:top_n]]


# ---------------------------------------------------------------------- budgets -------------


def _segment_cells(r: dict[str, Any], limit: int) -> int:
    # Both of a level's tables: its impact, then its movement.
    per_row = 3 + 3 * len(r["columns"]) + 10
    total = 0
    for point in r["points"]:
        for segment in point["segments"]:
            levels = len(segment["levels"])
            total += (min(levels, limit) + (1 if levels > limit else 0)) * per_row
        for band in point["bands"]:
            total += len(band["levels"]) * per_row
    return total


def _segment_row_limit(r: dict[str, Any]) -> int | None:
    """The largest per-segment row limit within the segment budget, the rest rolled into
    "Other", or ``None`` if not even one row per segment fits (§19.5)."""
    top_n = r["display"]["top_n_levels"]
    if _segment_cells(r, top_n) <= _SEGMENT_CELLS:
        return top_n
    if _segment_cells(r, 0) > _SEGMENT_CELLS:
        return None
    lo, hi = 0, top_n
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if _segment_cells(r, mid) <= _SEGMENT_CELLS else (lo, mid)
    return lo


def _rate_row_limit(r: dict[str, Any]) -> int:
    """The largest per-factor row limit that fits the page's cell budget (§19.5)."""
    per_row = 7 + len(r["columns"])
    top_n = r["display"]["top_n_levels"]
    top_factors = r["display"]["top_n_factors"]
    counts = [
        len(point["rate_card"][f]["levels"])
        for point in r["points"]
        for f in _ranked_factors(point, top_factors)
    ]
    segment_limit = _segment_row_limit(r)
    budget = _TABLE_CELLS - (
        0 if segment_limit is None else _segment_cells(r, segment_limit)
    )

    def fits(limit: int) -> bool:
        return per_row * sum(min(limit, c) for c in counts) <= budget

    if fits(top_n):
        return top_n
    lo, hi = 0, top_n
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


# ----------------------------------------------------------------------- charts -------------


def _frontier_chart(r: dict[str, Any], dimension: str, w: float, h: float) -> str:
    objective, constraint = _objective(r), _column(r, dimension)
    points = r["points"]
    xs = [p["totals"][dimension]["point"] for p in points]
    ys = [p["totals"][objective["id"]]["point"] for p in points]
    base_x = points[0]["totals"][dimension]["baseline"]
    base_y = points[0]["totals"][objective["id"]]["baseline"]
    x_lo, x_hi = min(*xs, base_x), max(*xs, base_x)
    y_lo, y_hi = min(*ys, base_y), max(*ys, base_y)

    def y_label(t: float) -> str:
        return _fmt(objective, t)

    probe = _Lin(y_lo, y_hi, 0, 1, pad=0.12)
    left = max((_text_w(y_label(t)) for t in probe.ticks), default=30.0) + 12
    top, bottom = 20.0, 42.0
    x = _Lin(x_lo, x_hi, left, w - 14, pad=0.08, n=5)
    y = _Lin(y_lo, y_hi, h - bottom, top, pad=0.12)
    svg = _Svg(
        w, h, f"{objective['label']} against {constraint['label']} for each point"
    )
    for t in y.ticks:
        svg.hline(left, w - 14, y(t), "gl")
        svg.text(left - 6, y(t) + 4, y_label(t), anchor="end")
    for t in x.ticks:
        svg.text(x(t), h - bottom + 16, _fmt(constraint, t), anchor="middle")
    svg.hline(left, w - 14, h - bottom, "ax")
    svg.text(w - 14, h - 6, constraint["label"], anchor="end", cls="lbl")
    svg.text(left, 12, objective["label"], cls="lbl")

    slices = [s for s in r["comparisons"]["slices"] if s["dimension"] == dimension]
    hidden = max(len(slices) - _MAX_SLICES, 0)
    shown = slices[:_MAX_SLICES]
    for position, piece in enumerate(shown):
        coords = [(x(xs[i]), y(ys[i])) for i in piece["points"]]
        if len(coords) > 1:
            d = "M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in coords)
            opacity = 1.0 - 0.6 * position / max(len(shown) - 1, 1)
            svg.add(f'<path class="line" style="opacity:{opacity:.2f}" d="{d}"/>')
    drawn = {i for piece in shown for i in piece["points"]}
    # Points that achieved the same totals share one label.
    labels: dict[tuple[int, int], list[int]] = {}
    for i in sorted(drawn):
        labels.setdefault((round(x(xs[i])), round(y(ys[i]))), []).append(i)
    for i in sorted(drawn):
        point = points[i]
        px, py = x(xs[i]), y(ys[i])
        title = (
            f"{point['name']}: {objective['label']} {_fmt(objective, ys[i])}, "
            f"{constraint['label']} {_fmt(constraint, xs[i])}"
            + (", dominated" if point["dominated"] else "")
        )
        cls = "hollow" if point["dominated"] else "mdot"
        svg.add(
            f'<g><title>{_e(title)}</title><circle class="{cls}" cx="{px:.1f}" '
            f'cy="{py:.1f}" r="5"/></g>'
        )
        if point["highlighted"]:
            svg.add(f'<circle class="ring" cx="{px:.1f}" cy="{py:.1f}" r="9"/>')
        svg.add(
            f'<g class="ptsel ptsel-{i}"><circle class="ring" cx="{px:.1f}" cy="{py:.1f}" r="7"/>'
            f'<text class="strong" x="{px:.1f}" y="{py + 22:.1f}" text-anchor="middle">'
            f"{_e(point['name'])}</text></g>"
        )
    if len(points) <= 15:
        for (lx, ly), members in labels.items():
            svg.text(lx, ly - 9, "/".join(f"P{i}" for i in members), anchor="middle")
    bx, by = x(base_x), y(base_y)
    name = _baseline_name(r).capitalize()
    title = f"{name}: {objective['label']} {_fmt(objective, base_y)}, {constraint['label']} {_fmt(constraint, base_x)}"
    svg.add(
        f'<g><title>{_e(title)}</title><path class="ddot" d="M{bx:.1f},{by - 7:.1f}'
        f'L{bx + 7:.1f},{by:.1f}L{bx:.1f},{by + 7:.1f}L{bx - 7:.1f},{by:.1f}Z"/></g>'
    )
    key_x = w - 14 - _text_w(name)
    svg.add(
        f'<path class="ddot" d="M{key_x - 12:.1f},5L{key_x - 6:.1f},11'
        f'L{key_x - 12:.1f},17L{key_x - 18:.1f},11Z"/>'
    )
    svg.text(key_x, 15, name, cls="lbl")
    legend = ""
    if len(shown) > 1:
        names = [
            ", ".join(
                f"{_column(r, c)['label']} {v:g}" for c, v in piece["fixed"].items()
            )
            for piece in shown
        ]
        fade = "; ".join(names)
        legend = f'<p class="note">Lines, darkest first: {_e(fade)}.'
        legend += f" {hidden} more not drawn.</p>" if hidden else "</p>"
    return svg.render() + legend


def _histogram(
    column: dict[str, str], delta: dict[str, Any], w: float, h: float
) -> str:
    edges = delta["histogram"]["edges"]
    counts = delta["histogram"]["counts"]
    top, bottom = 22.0, 26.0
    left, right = 44.0, 12.0
    x = _Lin(edges[0], edges[-1], left, w - right, pad=0.0, n=5)
    y = _Lin(0.0, float(max(counts)), h - bottom, top, pad=0.0, zero=True, n=3)
    svg = _Svg(w, h, f"Distribution of the per-quote change in {column['label']}")
    for t in y.ticks:
        svg.hline(left, w - right, y(t), "gl")
        svg.text(left - 6, y(t) + 4, _num(int(t)), anchor="end")
    for i, count in enumerate(counts):
        x0, x1 = x(edges[i]), x(edges[i + 1])
        if x1 - x0 < 1:
            x0, x1 = (x0 + x1) / 2 - 6, (x0 + x1) / 2 + 6
        svg.add(
            _bar(
                "hbar",
                x0 + 0.5,
                y(float(count)),
                max(x1 - x0 - 1, 1.0),
                h - bottom,
                r=1.5,
            )
        )
        svg.hit(
            x0,
            top,
            max(x1 - x0, 1.0),
            h - bottom - top,
            f"{_fine(column, edges[i])} to {_fine(column, edges[i + 1])}: {count:,} quotes",
        )
    svg.hline(left, w - right, h - bottom, "ax")
    for t in x.ticks:
        svg.text(x(t), h - 8, _fine(column, t), anchor="middle")
    percentiles = delta["percentiles"]
    for key in ("p5", "p50", "p95"):
        value = percentiles[key]
        if edges[0] <= value <= edges[-1]:
            svg.vline(x(value), top - 4, h - bottom, "mk")
            svg.text(x(value), top - 8, key, anchor="middle", cls="lbl")
    return svg.render()


def _step_histogram(movement: dict[str, Any], w: float, h: float) -> str:
    changes = movement["histogram"]["changes"]
    counts = movement["histogram"]["counts"]
    tally = dict(zip(changes, counts))
    lo, hi = min(changes), max(changes)
    top, bottom = 10.0, 36.0
    left, right = 48.0, 12.0
    slot = (w - left - right) / (hi - lo + 1)
    y = _Lin(0.0, float(max(counts)), h - bottom, top, pad=0.0, zero=True, n=3)
    svg = _Svg(w, h, "How far quotes moved, in steps")
    for t in y.ticks:
        svg.hline(left, w - right, y(t), "gl")
        svg.text(left - 6, y(t) + 4, _num(int(t)), anchor="end")
    every = max(1, math.ceil(28 / slot))
    for change in range(lo, hi + 1):
        x0 = left + (change - lo) * slot
        count = tally.get(change, 0)
        bar_w = min(max(slot - 2, 1.0), 24.0)
        cls = "flat" if change == 0 else "hbar"
        svg.add(_bar(cls, x0 + (slot - bar_w) / 2, y(float(count)), bar_w, h - bottom))
        svg.hit(
            x0,
            top,
            slot,
            h - bottom - top,
            _minus(f"{change:+d} steps: {count:,} quotes"),
        )
        if change % every == 0:
            svg.text(
                x0 + slot / 2, h - bottom + 14, _minus(f"{change:+d}"), anchor="middle"
            )
    svg.hline(left, w - right, h - bottom, "ax")
    svg.text(w - right, h - 4, "Steps from baseline", anchor="end", cls="lbl")
    return svg.render()


def _bucket(c: float) -> str:
    k = max(-3, min(3, round(3 * c)))
    return "z0" if k == 0 else (f"zp{k}" if k > 0 else f"zm{-k}")


def _colour_positions(levels: list[dict[str, Any]]) -> tuple[list[float], str]:
    """Each level's place on the colour scale, 0 to 1, and what the scale is."""
    labels = [lv["label"] for lv in levels]
    try:
        values = [float(label) for label in labels]
        numeric = all(math.isfinite(v) for v in values)
    except ValueError:
        values, numeric = [], False
    basis = "level value" if numeric else "sample size"
    n = len(levels)
    if n == 1:
        return [0.5], basis
    keys = (
        [(v, label) for v, label in zip(values, labels)]
        if numeric
        else [(lv["quotes"], lv["label"]) for lv in levels]
    )
    positions = [0.0] * n
    for rank, i in enumerate(sorted(range(n), key=lambda i: keys[i])):
        positions[i] = rank / (n - 1)
    return positions, basis


def _beeswarm_rows(point: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """One row per factor with quotes, by quote-weighted mean ``abs(x)``."""
    rows = []
    for f, card in enumerate(point["rate_card"]):
        levels = [lv for lv in card["levels"] if lv["quotes"] > 0]
        total = sum(lv["quotes"] for lv in levels)
        if total == 0:
            continue
        xs = [
            lv["log_change"] if kind == "changes" else math.log(lv["new"])
            for lv in levels
        ]
        colours, basis = _colour_positions(levels)
        rows.append(
            {
                "factor": card["factor"],
                "position": f,
                "levels": levels,
                "xs": xs,
                "colours": colours,
                "basis": basis,
                "score": sum(lv["quotes"] * abs(x) for lv, x in zip(levels, xs))
                / total,
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["position"]))
    return rows


def _beeswarm_domain(r: dict[str, Any], kind: str) -> float:
    largest = max(
        (
            abs(x)
            for point in r["points"]
            for row in _beeswarm_rows(point, kind)
            for x in row["xs"]
        ),
        default=0.0,
    )
    return largest * 1.05 if largest > 0 else _ZERO_DOMAIN


def _beeswarm_layout(
    r: dict[str, Any], point: dict[str, Any], kind: str
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]], int, int]:
    """The rows shown, the dots drawn as (quotes, row, level), biggest first so small
    dots stay visible, and how many dots and factors aren't shown."""
    top_n = r["display"]["top_n_factors"]
    every = _beeswarm_rows(point, kind)
    rows = every[:top_n]
    dots = sorted(
        (
            (lv["quotes"], i, m)
            for i, row in enumerate(rows)
            for m, lv in enumerate(row["levels"])
        ),
        key=lambda d: (-d[0], d[1], d[2]),
    )
    limit = min(_BEESWARM_DOTS, _PAGE_DOTS // (2 * len(r["points"])))
    return rows, dots[:limit], max(len(dots) - limit, 0), len(every) - len(rows)


def _dot_title(row: dict[str, Any], m: int) -> str:
    lv = row["levels"][m]
    rate = (
        _rate(lv["new"])
        if lv["existing"] is None
        else f"{_rate(lv['existing'])} → {_rate(lv['new'])} "
        f"({_signed_pct(math.exp(lv['log_change']) - 1)})"
    )
    return (
        f"{row['factor']} = {lv['label']}\n{rate}\n{lv['quotes']:,} quotes\n"
        f"Colour: {row['basis']}"
    )


def _beeswarm(
    r: dict[str, Any],
    point: dict[str, Any],
    kind: str,
    domain: float,
    w: float,
) -> str:
    rows, dots, _, _ = _beeswarm_layout(r, point, kind)
    if not rows:
        return '<p class="note">No factor has any quotes.</p>'
    left = _BEESWARM_LABEL_W
    plot_w = w - left - _BEESWARM_RIGHT
    top, tail, axis_h = 14.0, 12.0, 44.0
    plot_h = len(rows) * _ROW_H
    height = top + plot_h + tail + axis_h
    pad = (_DOT_MIN_D + _DOT_D_RANGE) / 2 + 1

    def px(v: float) -> float:
        return left + pad + (v + domain) / (2 * domain) * (plot_w - 2 * pad)

    svg = _Svg(w, height, "Beeswarm of rates by factor")
    bottom = top + plot_h + tail
    for t in _ticks(-domain, domain, 8):
        svg.vline(px(t), top, bottom, "gl")
        svg.text(px(t), bottom + 16, _minus(f"{t + 0.0:g}"), anchor="middle")
    svg.vline(px(0.0), top, bottom, "ref")
    rate = r["run"]["rate_label"]
    direction = f"← decreases {rate} | increases {rate} →"
    title = (
        f"Change in relativity, log(new / existing) ({direction})"
        if kind == "changes"
        else f"Relativity, log({rate}) ({direction})"
    )
    svg.text(left + plot_w / 2, bottom + 36, title, anchor="middle", cls="lbl")

    for i, row in enumerate(rows):
        centre = top + (i + 0.5) * _ROW_H
        svg.add(
            f'<text class="lbl" x="{left - 8:.1f}" y="{centre + 4:.1f}" text-anchor="end">'
            f"<title>{_e(row['factor'])}</title>{_e(_short(row['factor']))}</text>"
        )
    largest = dots[0][0]
    for quotes, i, m in dots:
        row = rows[i]
        centre = top + (i + 0.5) * _ROW_H
        offset = ((m * _GOLDEN) % 1.0 - 0.5) * _JITTER * _ROW_H
        diameter = _DOT_MIN_D + _DOT_D_RANGE * math.sqrt(quotes / largest)
        shade = round(row["colours"][m] * _COLOUR_STEPS)
        svg.add(
            f'<circle class="sw{shade}" cx="{px(row["xs"][m]):.1f}" '
            f'cy="{centre + offset:.1f}" r="{diameter / 2:.1f}">'
            f"<title>{_e(_dot_title(row, m))}</title></circle>"
        )

    # The colour bar, high at the top.
    bar_x = left + plot_w + 18
    bar_h = plot_h * 0.7
    bar_top = top + (plot_h - bar_h) / 2
    piece = bar_h / (_COLOUR_STEPS + 1)
    for k in range(_COLOUR_STEPS + 1):
        svg.add(
            f'<rect class="sw{_COLOUR_STEPS - k}" x="{bar_x:.1f}" y="{bar_top + k * piece:.1f}" '
            f'width="14" height="{piece + 0.5:.1f}"/>'
        )
    svg.text(bar_x + 7, bar_top - 4, "High", anchor="middle")
    svg.text(bar_x + 7, bar_top + bar_h + 13, "Low", anchor="middle")
    if _text_w(_COLOUR_TITLE) <= plot_h:
        tx, ty = bar_x + 30, top + plot_h / 2
        svg.add(
            f'<text class="lbl" x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
            f'transform="rotate(90 {tx:.1f} {ty:.1f})">{_e(_COLOUR_TITLE)}</text>'
        )
    return svg.render()


def _beeswarm_caption(r: dict[str, Any], kind: str, point: dict[str, Any]) -> str:
    _, _, hidden_dots, hidden_rows = _beeswarm_layout(r, point, kind)
    what = (
        "Each level's change in relativity"
        if kind == "changes"
        else "Each level's new relativity"
    )
    text = (
        f"{what}, one dot per level, on the log scale. Bigger dots hold more quotes; colour "
        f"runs blue to red by level value, or by quotes where levels are categories."
    )
    if hidden_dots:
        text += f" The {hidden_dots:,} smallest levels aren't drawn; they are in the tables."
    if hidden_rows:
        text += f" {hidden_rows} more factors are in the Rate card list and the record."
    return text


def _value_axis(
    display: dict[str, Any], scenario: dict[str, Any]
) -> tuple[list[float], list[int], list[int]]:
    """Display values ascending with equal values combined, and the quotes on each at
    baseline and at the point. A step's display value needn't rise with the step."""
    combined: dict[float, list[int]] = {}
    for value, before, after in zip(
        display["values"], scenario["baseline_counts"], scenario["point_counts"]
    ):
        cell = combined.setdefault(value, [0, 0])
        cell[0] += before
        cell[1] += after
    values = sorted(combined)
    return values, [combined[v][0] for v in values], [combined[v][1] for v in values]


def _value_labels(
    svg: _Svg,
    display: dict[str, Any],
    values: list[float],
    left: float,
    slot: float,
    y: float,
) -> None:
    """Label every value that fits under a chart with one slot per value."""
    widest = max(_text_w(_disp(display, v)) for v in values) + 8
    every = max(1, math.ceil(widest / slot))
    for k, value in enumerate(values):
        if k % every == 0:
            svg.text(left + (k + 0.5) * slot, y, _disp(display, value), anchor="middle")


def _value_slot(values: list[float], value: float) -> float:
    """Where a value falls among increasing values, for a marker."""
    if value <= values[0]:
        return 0.0
    if value >= values[-1]:
        return float(len(values) - 1)
    k = next(i for i, v in enumerate(values) if v > value) - 1
    return k + (value - values[k]) / (values[k + 1] - values[k])


def _series_key(svg: _Svg, x: float, y: float, entries: list[tuple[str, str]]) -> None:
    """A small key along the top of a chart: (css class of a swatch, label)."""
    for cls, label in entries:
        svg.add(
            f'<rect class="{cls}" x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2"/>'
        )
        svg.text(x + 14, y, label, cls="lbl")
        x += 14 + _text_w(label) + 14


def _distribution_chart(
    r: dict[str, Any], point: dict[str, Any], w: float, h: float
) -> str:
    display = _display(r)
    scenario = point["scenario"]
    values, before, after = _value_axis(display, scenario)
    n = len(values)
    top, bottom, left, right = 26.0, 28.0, 52.0, 12.0
    slot = (w - left - right) / n
    y = _Lin(
        0.0, float(max(*before, *after, 1)), h - bottom, top, pad=0.0, zero=True, n=3
    )
    svg = _Svg(w, h, f"Quotes on each {display['label']} at baseline and at the point")
    for t in y.ticks:
        svg.hline(left, w - right, y(t), "gl")
        svg.text(left - 6, y(t) + 4, _num(int(t)), anchor="end")
    bar_w = max((slot - 2) / 2, 1.0)
    for k in range(n):
        x0 = left + k * slot
        svg.add(_bar("dbar", x0 + 1, y(float(before[k])), bar_w, h - bottom, r=1.5))
        svg.add(
            _bar("hbar", x0 + 1 + bar_w, y(float(after[k])), bar_w, h - bottom, r=1.5)
        )
        svg.hit(
            x0,
            top,
            slot,
            h - bottom - top,
            f"{_disp(display, values[k])}: {before[k]:,} quotes at baseline, "
            f"{after[k]:,} at the point",
        )
    svg.hline(left, w - right, h - bottom, "ax")
    _value_labels(svg, display, values, left, slot, h - 10)
    marks = [("mk-d", scenario["baseline_mean"]), ("mk-m", scenario["point_mean"])]
    if _target_on_scale(display):
        marks.append(("mk-t", display["target"]))
    for cls, value in marks:
        at = _value_slot(values, value)
        svg.vline(left + (at + 0.5) * slot, top - 4, h - bottom, cls)
    _series_key(svg, left, 12, [("dbar", "Baseline"), ("hbar", "Point")])
    return svg.render()


def _target_on_scale(display: dict[str, Any]) -> bool:
    """Whether the target can be drawn; past either end it would read as being there."""
    target = display["target"]
    return target is not None and (
        min(display["values"]) <= target <= max(display["values"])
    )


def _distribution_caption(r: dict[str, Any], point: dict[str, Any]) -> str:
    display = _display(r)
    scenario = point["scenario"]
    text = (
        f"Quotes on each step. Dashed lines are the means: {_disp(display, scenario['baseline_mean'])} "
        f"at baseline, {_disp(display, scenario['point_mean'])} at the point"
    )
    if display["target"] is not None:
        shown = _disp(display, display["target"])
        text += (
            f"; the solid line is the target, {shown}"
            if _target_on_scale(display)
            else f"; the target, {shown}, is off this scale"
        )
    return text


def _ecdf_chart(r: dict[str, Any], point: dict[str, Any], w: float, h: float) -> str:
    display = _display(r)
    scenario = point["scenario"]
    n_quotes = sum(scenario["baseline_counts"])
    values, baseline_counts, point_counts = _value_axis(display, scenario)
    n = len(values)

    def cumulative(counts: list[int]) -> list[float]:
        out, seen = [], 0
        for count in counts:
            seen += count
            out.append(seen / n_quotes)
        return out

    before = cumulative(baseline_counts)
    after = cumulative(point_counts)
    top, bottom, left, right = 26.0, 28.0, 44.0, 12.0
    slot = (w - left - right) / n
    y = _Lin(0.0, 1.0, h - bottom, top, pad=0.0, n=4)
    svg = _Svg(w, h, f"Cumulative share of quotes by {display['label']}")
    for t in y.ticks:
        svg.hline(left, w - right, y(t), "gl")
        svg.text(left - 6, y(t) + 4, f"{t * 100:.0f}%", anchor="end")

    def path(shares: list[float]) -> str:
        d = f"M{left:.1f},{y(shares[0]):.1f}"
        for k, share in enumerate(shares):
            d += f"V{y(share):.1f}H{left + (k + 1) * slot:.1f}"
        return d

    svg.add(f'<path class="ln-d" d="{path(before)}"/>')
    svg.add(f'<path class="ln-m" d="{path(after)}"/>')
    for k in range(n):
        svg.hit(
            left + k * slot,
            top,
            slot,
            h - bottom - top,
            f"At or below {_disp(display, values[k])}: {before[k]:.1%} at "
            f"baseline, {after[k]:.1%} at the point",
        )
    svg.hline(left, w - right, h - bottom, "ax")
    _value_labels(svg, display, values, left, slot, h - 10)
    _series_key(svg, left, 12, [("dbar", "Baseline"), ("hbar", "Point")])
    return svg.render()


def _ecdf_caption(r: dict[str, Any], point: dict[str, Any]) -> str:
    scenario = point["scenario"]
    before, after = scenario["baseline_counts"], scenario["point_counts"]
    n = sum(before)
    return (
        f"Share of quotes at or below each value. On the first step: {before[0] / n:.1%} at "
        f"baseline, {after[0] / n:.1%} at the point. On the last: {before[-1] / n:.1%} and "
        f"{after[-1] / n:.1%}"
    )


def _transition_cells(point: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    cells = sorted(
        point["scenario"]["transitions"],
        key=lambda t: (-t["quotes"], t["from"], t["to"]),
    )
    return cells[:_TRANSITION_DOTS], max(len(cells) - _TRANSITION_DOTS, 0)


def _colour_column(r: dict[str, Any]) -> dict[str, str]:
    """The first constraint, which the transition dots are coloured by."""
    constraints = r["run"]["constraints"]
    return _column(r, constraints[0]["name"]) if constraints else _objective(r)


def _transition_chart(
    r: dict[str, Any], point: dict[str, Any], w: float, h: float
) -> str:
    display = _display(r)
    colour = _colour_column(r)
    cells, _ = _transition_cells(point)
    n = r["run"]["n_steps"]
    top, bottom, left, right = 44.0, 44.0, 58.0, 16.0
    slot_x, slot_y = (w - left - right) / n, (h - top - bottom) / n

    def px(step: int) -> float:
        return left + (step + 0.5) * slot_x

    def py(step: int) -> float:
        return h - bottom - (step + 0.5) * slot_y

    svg = _Svg(w, h, f"Quotes by baseline and point {display['label']}")
    values = display["values"]
    widest = max(_text_w(_disp(display, v)) for v in values) + 8
    every_x = max(1, math.ceil(widest / slot_x))
    every_y = max(1, math.ceil(18 / slot_y))
    for k in range(n):
        if k % every_x == 0:
            svg.vline(px(k), top, h - bottom, "gl")
            svg.text(px(k), h - bottom + 16, _disp(display, values[k]), anchor="middle")
        if k % every_y == 0:
            svg.hline(left, w - right, py(k), "gl")
            svg.text(left - 6, py(k) + 4, _disp(display, values[k]), anchor="end")
    svg.add(
        f'<line class="ref" x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(n - 1):.1f}" y2="{py(n - 1):.1f}"/>'
    )
    if cells:
        largest = cells[0]["quotes"]
        size = max(abs(t["changes"][colour["id"]]) for t in cells) or 1.0
        for cell in cells:
            change = cell["changes"][colour["id"]]
            radius = (4 + 20 * math.sqrt(cell["quotes"] / largest)) / 2
            details = "\n".join(
                f"{c['label']}: {_signed(c, cell['changes'][c['id']])}"
                for c in r["columns"]
            )
            title = (
                f"{_disp(display, values[cell['from']])} → {_disp(display, values[cell['to']])}\n"
                f"{cell['quotes']:,} quotes\n{details}"
            )
            svg.add(
                f'<g><title>{_e(title)}</title><circle class="tr {_bucket(change / size)}" '
                f'cx="{px(cell["from"]):.1f}" cy="{py(cell["to"]):.1f}" r="{radius:.1f}"/></g>'
            )
    svg.text(
        left + (w - left - right) / 2,
        h - 6,
        f"{display['label']} at baseline",
        anchor="middle",
        cls="lbl",
    )
    svg.text(left, 34, f"{display['label']} at the point", cls="lbl")
    # The colour key.
    x0 = w - right - 7 * 12 - _text_w(f"raises {colour['label']}") - 6
    svg.text(x0 - 6, 14, f"lowers {colour['label']}", anchor="end", cls="lbl")
    for j, name in enumerate(("zm3", "zm2", "zm1", "z0", "zp1", "zp2", "zp3")):
        svg.add(f'<circle class="{name}" cx="{x0 + 5 + j * 12:.1f}" cy="10" r="5"/>')
    svg.text(x0 + 7 * 12 + 4, 14, f"raises {colour['label']}", cls="lbl")
    return svg.render()


def _axis_labels(ticks: list[float], pattern: str) -> list[str]:
    """Tick labels in a column's format, with more decimals if it can't tell them apart."""
    labels = [_minus(pattern.format(t)) for t in ticks]
    if len(ticks) < 2 or len(set(labels)) == len(labels):
        return labels
    fields = list(Formatter().parse(pattern))
    prefix, spec = fields[0][0], fields[0][2] or ""
    suffix = "".join(field[0] for field in fields[1:])
    head, digits, kind = re.fullmatch(r"(.*?)(?:\.(\d+))?([a-zA-Z%]?)", spec).groups()  # type: ignore[union-attr]
    kind = kind if kind in ("e", "E", "f", "F", "g", "G", "%") else "f"
    start = int(digits) if digits else 0
    for extra in range(1, 13):
        wider = f"{head}.{start + extra}{kind}"
        labels = [_minus(f"{prefix}{format(t, wider)}{suffix}") for t in ticks]
        if len(set(labels)) == len(labels):
            break
    return labels


def _band_bars(
    column: dict[str, str], levels: list[dict[str, Any]], w: float, h: float
) -> str:
    """The % change in one column for each band."""
    cid = column["id"]
    values = [
        None
        if lv["baseline"][cid] == 0
        else (lv["point"][cid] - lv["baseline"][cid]) / abs(lv["baseline"][cid])
        for lv in levels
    ]
    known = [v for v in values if v is not None]
    largest = max((abs(v) for v in known), default=0.0) or 0.01
    top, bottom, left, right = 12.0, 40.0, 52.0, 12.0
    y = _Lin(-largest, largest, h - bottom, top, pad=0.06, n=4)
    slot = (w - left - right) / max(len(levels), 1)
    svg = _Svg(w, h, f"Percentage change in {column['label']} by band")
    step = y.ticks[1] - y.ticks[0] if len(y.ticks) > 1 else largest
    digits = max(0, math.ceil(-math.log10(step * 100)))
    for t in y.ticks:
        svg.hline(left, w - right, y(t), "gl")
        label = "0%" if t == 0 else _signed_pct(t, digits)
        svg.text(left - 6, y(t) + 4, label, anchor="end")
    svg.hline(left, w - right, y(0.0), "ref")
    for k, (lv, value) in enumerate(zip(levels, values)):
        x0 = left + k * slot
        if value is not None:
            y0, y1 = sorted((y(0.0), y(value)))
            svg.add(
                f'<rect class="hbar" x="{x0 + slot * 0.2:.1f}" y="{y0:.1f}" '
                f'width="{slot * 0.6:.1f}" height="{max(y1 - y0, 1.0):.1f}" rx="2"/>'
            )
        reading = "no baseline" if value is None else _signed_pct(value, 2)
        svg.hit(
            x0,
            top,
            slot,
            h - bottom - top,
            f"{lv['label']}, mean {_num(lv['mean_value'], 4)}: {reading}, {lv['quotes']:,} quotes",
        )
        svg.text(
            x0 + slot / 2, h - bottom + 14, lv["label"], anchor="middle", cls="lbl"
        )
        svg.text(
            x0 + slot / 2, h - bottom + 28, _num(lv["mean_value"], 3), anchor="middle"
        )
    return svg.render()


def _band_lines(
    label: str,
    labels: list[str],
    before: list[float],
    after: list[float],
    fmt: _Fmt,
    pattern: str,
    w: float,
    h: float,
) -> str:
    """Baseline and point means for each band."""
    top, bottom, left, right = 24.0, 24.0, 64.0, 12.0
    every = [*before, *after]
    y = _Lin(min(every), max(every), h - bottom, top, pad=0.1, n=4)
    slot = (w - left - right) / max(len(labels), 1)
    svg = _Svg(w, h, f"{label} by band at baseline and at the point")
    for t, text in zip(y.ticks, _axis_labels(y.ticks, pattern)):
        svg.hline(left, w - right, y(t), "gl")
        svg.text(left - 6, y(t) + 4, text, anchor="end")

    def path(values: list[float]) -> str:
        return "M" + "L".join(
            f"{left + (k + 0.5) * slot:.1f},{y(v):.1f}" for k, v in enumerate(values)
        )

    svg.add(f'<path class="ln-d" d="{path(before)}"/>')
    svg.add(f'<path class="ln-m" d="{path(after)}"/>')
    for k, name in enumerate(labels):
        x0 = left + k * slot
        svg.hit(
            x0,
            top,
            slot,
            h - bottom - top,
            f"{name}: {fmt(before[k])} at baseline, {fmt(after[k])} at the point",
        )
        svg.text(x0 + slot / 2, h - 8, name, anchor="middle", cls="lbl")
    _series_key(svg, left, 12, [("dbar", "Baseline"), ("hbar", "Point")])
    return svg.render()


def _trace_chart(
    label: str,
    history: list[dict[str, Any]],
    values: list[float],
    bound: float | None,
    fmt: _Fmt,
    pattern: str,
    w: float,
    h: float,
) -> str:
    """A running total over the solve, one vertical band per coordinate-descent pass."""
    top, bottom, left, right = 20.0, 26.0, 72.0, 12.0
    every = [*values, *([] if bound is None else [bound])]
    y = _Lin(min(every), max(every), h - bottom, top, pad=0.08, n=3)
    n = len(values)

    def px(i: int) -> float:
        return left + (i + 0.5) / n * (w - left - right)

    svg = _Svg(w, h, f"{label} over the solve")
    for t, text in zip(y.ticks, _axis_labels(y.ticks, pattern)):
        svg.hline(left, w - right, y(t), "gl")
        svg.text(left - 6, y(t) + 4, text, anchor="end")
    stride = max(1, math.ceil(n / _TRACE_POINTS))
    keep = sorted({*range(0, n, stride), n - 1})
    svg.add(
        '<path class="line" d="M'
        + "L".join(f"{px(i):.1f},{y(values[i]):.1f}" for i in keep)
        + '"/>'
    )
    if bound is not None:
        svg.hline(left, w - right, y(bound), "mk-t")
    passes: dict[int, list[int]] = {}
    for i, row in enumerate(history):
        passes.setdefault(row["cd_iteration"], []).append(i)
    span = (w - left - right) / n
    for number, members in passes.items():
        x0 = left + members[0] * span
        if members[0] > 0:
            svg.vline(x0, top, h - bottom, "gl")
        last = members[-1]
        svg.hit(
            x0,
            top,
            span * len(members),
            h - bottom - top,
            f"Pass {number}: {label} {fmt(values[last])}"
            + ("" if bound is None else f", bound {fmt(bound)}"),
        )
        svg.text(x0 + span * len(members) / 2, h - 8, f"Pass {number}", anchor="middle")
    svg.text(left, 12, label, cls="lbl")
    return svg.render()


def _segment_bars(
    column: dict[str, str], rows: list[tuple[str, float]], w: float
) -> str:
    if not rows:
        return ""
    row_h = 24.0
    left = 160.0
    axis_h = 24.0
    height = len(rows) * row_h + axis_h
    largest = max(abs(v) for _, v in rows) or 1.0
    x = _Lin(-largest, largest, left, w - 16, pad=0.06, n=4)
    svg = _Svg(w, height, f"Change in {column['label']} by segment")
    for t in x.ticks:
        svg.vline(x(t), 0, height - axis_h, "gl")
        svg.text(x(t), height - 8, _signed(column, t), anchor="middle")
    svg.vline(x(0.0), 0, height - axis_h, "ref")
    for i, (label, value) in enumerate(rows):
        y = i * row_h
        svg.text(left - 8, y + 16, _short(label), anchor="end", cls="lbl")
        x0, x1 = sorted((x(0.0), x(value)))
        svg.add(
            f'<rect class="hbar" x="{x0:.1f}" y="{y + 6:.1f}" width="{max(x1 - x0, 1.0):.1f}" '
            f'height="12" rx="2"><title>{_e(label)}: {_e(_signed(column, value))}</title></rect>'
        )
    return svg.render()


# ------------------------------------------------------------------------- pages ------------


def _open(index: int, slug: str) -> str:
    return f'<section class="tab tb tb-{index}" id="{slug}">'


def _issue_lines(flags: list[dict[str, Any]], limit: int) -> str:
    lines = [
        f'<li>{_badge(f["status"])}<span class="t">{_e(f["message"])}</span></li>'
        for f in flags[:limit]
    ]
    return f'<ul class="issues">{"".join(lines)}</ul>' if lines else ""


def _review_card(r: dict[str, Any], pt: _Radio, tabs: _Radio, checks_tab: int) -> str:
    general = [f for f in r["flags"] if f["point"] is None]
    panels = []
    for point in r["points"]:
        own = [f for f in r["flags"] if f["point"] == point["index"]]
        lines = _issue_lines(general + own, 5) or '<p class="ok">No open findings.</p>'
        panels.append(f'<div class="{pt.tag(point["index"])}">{lines}</div>')
    link = tabs.label(checks_tab, "All checks and definitions →", cls="link")
    critical = sum(f["status"] == "critical" for f in r["flags"])
    warnings = sum(f["status"] == "warning" for f in r["flags"])
    counts = '<div class="counts">'
    counts += _badge("critical", f"{critical} critical") if critical else ""
    counts += _badge("warning", f"{warnings} warnings") if warnings else ""
    counts += "</div>"
    return _card(
        "Review", "".join(panels) + f'<div class="more">{link}</div>', extra=counts
    )


def _points_table(r: dict[str, Any]) -> str:
    columns = r["columns"]
    cols = [
        "Point",
        *[c["label"] for c in columns],
        "Moved up / down",
        "p5 / p95 move",
        "Rate card",
    ]
    numeric = list(range(1, len(columns) + 3))
    rows = []
    classes = []
    for point in r["points"]:
        binding = {a["constraint"]: a["binding"] for a in point["audit"]}
        cells = [
            _two(
                _e(point["name"]),
                ("dominated" if point["dominated"] else "efficient")
                + (" · recommended" if point["highlighted"] else ""),
            )
        ]
        for column in columns:
            entry = point["totals"][column["id"]]
            sub = f"{_signed(column, entry['change'])} ({_signed_pct(entry['pct'])})"
            if binding.get(column["id"]):
                sub += " · binding"
            cells.append(_two(_e(_fmt(column, entry["point"])), _e(sub)))
        n = (
            point["movement"]["up"]
            + point["movement"]["down"]
            + point["movement"]["unchanged"]
        )
        cells.append(
            f"{_share(point['movement']['up'] / n)} / {_share(point['movement']['down'] / n)}"
        )
        steps = point["movement"]["percentiles_steps"]
        cells.append(_minus(f"{steps['p5']:+d} / {steps['p95']:+d}"))
        cells.append(
            _e(point["artifact"])
            if point["artifact"]
            else '<span class="muted">—</span>'
        )
        rows.append(cells)
        classes.append(f"ptr-{point['index']}")
    return _table(cols, rows, numeric=numeric, row_class=classes)


def _overview(r: dict[str, Any], pt: _Radio, tabs: _Radio, checks_tab: int) -> str:
    objective = _objective(r)
    titles = [_takeaway(r, p) for p in r["points"]]
    review = _review_card(r, pt, tabs, checks_tab)
    cards = []
    for dimension in r["run"]["swept"]:
        cards.append(
            _card(
                f"{objective['label']} against {_column(r, dimension)['label']}",
                _frontier_chart(r, dimension, _W_HALF, _H_ROW + 20),
                titles=titles,
                what=f"Each point against {_baseline_name(r)}; hollow points are dominated",
            )
        )
    cards.append(review)
    grouped = [_row(*cards[i : i + 2]) for i in range(0, len(cards), 2)]
    table = _card(
        "Every point against the baseline",
        _points_table(r),
        what=f"Compared with {_baseline_name(r)}; the selected point is shaded",
    )
    return _open(0, "overview") + "".join(grouped) + table + "</section>"


def _kpis(r: dict[str, Any], point: dict[str, Any]) -> str:
    tiles = []
    for column in r["columns"]:
        entry = point["totals"][column["id"]]
        tiles.append(
            f'<div class="card kpi"><div class="lb">{_e(column["label"])}</div>'
            f'<div class="v">{_e(_fmt(column, entry["point"]))}</div>'
            f'<div class="d">{_e(_signed(column, entry["change"]))} '
            f"({_e(_signed_pct(entry['pct']))}) · was {_e(_fmt(column, entry['baseline']))}</div></div>"
        )
    return '<div class="kpis">' + "".join(tiles) + "</div>"


def _contribution_table(r: dict[str, Any], point: dict[str, Any]) -> str:
    rows = []
    for column in r["columns"]:
        delta = point["deltas"][column["id"]]
        rows.append(
            [
                _e(column["label"]),
                _e(_signed(column, point["totals"][column["id"]]["change"])),
                _e(_signed(column, delta["moved_up"])),
                _e(_signed(column, delta["moved_down"])),
            ]
        )
    return _table(
        ["Column", "Total change", "From quotes moved up", "From quotes moved down"],
        rows,
        numeric=[1, 2, 3],
    )


def _movement_table(r: dict[str, Any], point: dict[str, Any]) -> str:
    movement = point["movement"]
    n = movement["up"] + movement["down"] + movement["unchanged"]
    rows = [
        [
            "Clamped low",
            f"{point['health']['clamped_low']:,}",
            _share(point["health"]["clamped_low"] / n),
        ],
        [
            "Clamped high",
            f"{point['health']['clamped_high']:,}",
            _share(point["health"]["clamped_high"] / n),
        ],
    ]
    shares = _table(["Quotes", "Count", "Share"], rows, numeric=[1, 2])
    display = r["run"]["scenario_display"]
    steps = movement["percentiles_steps"]
    header = ["", *_PERCENTILES]
    percentile_rows = [["Steps", *[_minus(f"{steps[k]:+d}") for k in _PERCENTILES]]]
    if display is not None and movement["percentiles_display"] is not None:
        pattern = display["format"]
        values = movement["percentiles_display"]
        percentile_rows.append(
            [
                _e(display["label"]),
                *[
                    _e(pattern.format(values[k]).replace("-", "−"))
                    for k in _PERCENTILES
                ],
            ]
        )
    return shares + _table(header, percentile_rows, numeric=range(1, 8))


def _summary_table(r: dict[str, Any], point: dict[str, Any]) -> str:
    """Every headline number for a point, with its bounds and target."""
    run = r["run"]
    names = run["sensitivity_names"]
    display = _display(r)
    scenario = point["scenario"]
    n_quotes = sum(scenario["baseline_counts"])
    blanks = ["—"] * len(names)
    rows: list[list[str]] = []
    for column in r["columns"]:
        cid = column["id"]
        entry = point["totals"][cid]
        rows.append(
            [
                _e(column["label"]),
                _e(_fmt(column, entry["baseline"])),
                _e(_fmt(column, entry["point"])),
                *[_e(_fmt(column, point["sensitivity"][name][cid])) for name in names],
                _e(_signed(column, entry["change"])),
                _e(_signed_pct(entry["pct"], 2)),
                _e(_fmt(column, point["bounds"].get(cid))),
            ]
        )
    for ratio in point["ratios"]:

        def shown(value: float | None) -> str:
            return "—" if value is None else _e(ratio["format"].format(value))

        change = ratio["change"]
        rows.append(
            [
                _e(ratio["label"]),
                shown(ratio["baseline"]),
                shown(ratio["point"]),
                *[shown(ratio["sensitivity"][name]) for name in names],
                _e(_disp_signed({"format": ratio["format"]}, change)),
                "—",
                "—",
            ]
        )
    label = display["label"]
    for name, before, after, target in (
        (
            f"Mean {label}",
            scenario["baseline_mean"],
            scenario["point_mean"],
            display["target"],
        ),
        (f"Largest {label}", scenario["baseline_max"], scenario["point_max"], None),
    ):
        rows.append(
            [
                _e(name),
                _e(_disp(display, before)),
                _e(_disp(display, after)),
                *blanks,
                _e(_disp_signed(display, after - before)),
                "—",
                _e(_disp(display, target)),
            ]
        )
    values = display["values"]
    for name, position in (
        (f"Quotes on the first step ({_disp(display, values[0])})", 0),
        (f"Quotes on the last step ({_disp(display, values[-1])})", -1),
    ):
        before = scenario["baseline_counts"][position] / n_quotes
        after = scenario["point_counts"][position] / n_quotes
        rows.append(
            [
                _e(name),
                _share(before),
                _share(after),
                *blanks,
                _points_change(after - before),
                "—",
                "—",
            ]
        )
    cols = ["Metric", "Baseline", "Point", *names, "Change", "Change %", "Target"]
    table = _table(cols, rows, numeric=range(1, len(cols)))
    note = run["sensitivity_note"]
    return table + (f'<p class="note">{_e(note)}</p>' if note else "")


def _points_change(share_change: float) -> str:
    """A change in a share, in percentage points."""
    points = round(share_change * 100, 1)
    return "0.0pp" if points == 0 else _minus(f"{points:+.1f}pp")


def _direction_table(r: dict[str, Any], point: dict[str, Any]) -> str:
    display = _display(r)
    directions = point["movement"]["directions"]
    n = sum(d["quotes"] for d in directions.values())
    rows = []
    for key, name in (
        ("down", "Moved down"),
        ("unchanged", "Unchanged"),
        ("up", "Moved up"),
    ):
        entry = directions[key]
        rows.append(
            [
                name,
                f"{entry['quotes']:,}",
                _share(entry["quotes"] / n),
                _e(_disp_signed(display, entry["mean_display_change"])),
                *[
                    _e(_fine(column, entry["mean_changes"][column["id"]]))
                    for column in r["columns"]
                ],
            ]
        )
    cols = [
        "Direction",
        "Quotes",
        "Share",
        f"Mean change in {display['label']}",
        *[f"Mean change in {c['label']}" for c in r["columns"]],
    ]
    return _table(cols, rows, numeric=range(1, len(cols)))


def _forward_table(r: dict[str, Any], point: dict[str, Any]) -> str:
    columns = r["columns"]
    rows = []
    for entry in point["projection"]:
        cells = [f"{entry['horizon']:g} days"]
        for column in columns:
            total = entry["totals"][column["id"]]
            cells.append(
                _two(
                    _e(_fmt(column, total["point"])),
                    f"{_e(_signed(column, total['change']))} from {_e(_fmt(column, total['baseline']))}",
                )
            )
        rows.append(cells)
    cols = ["Horizon", *[c["label"] for c in columns]]
    return _table(cols, rows, numeric=range(1, len(cols)))


def _impact(r: dict[str, Any], index: int) -> str:
    panels = []
    display = _display(r)
    projection = r["run"]["projection"]
    for point in r["points"]:
        i = point["index"]
        cards = []
        for column in r["columns"]:
            delta = point["deltas"][column["id"]]
            below, above = delta["histogram"]["below"], delta["histogram"]["above"]
            cards.append(
                _card(
                    f"Per-quote change in {column['label']}",
                    _histogram(column, delta, _W_HALF, 180),
                    what=f"p1 to p99; {below:,} quotes below and {above:,} above the range",
                )
            )
        grid = "".join(_row(*cards[j : j + 2]) for j in range(0, len(cards), 2))
        summary = _card(
            "Summary",
            _summary_table(r, point),
            what=f"Against {_baseline_name(r)}; a target is a constraint's bound",
        )
        directions = _card(
            "Movement by direction",
            _direction_table(r, point),
            what="Per-quote means for quotes moved each way",
        )
        where = _card(
            "Where the change comes from",
            _contribution_table(r, point),
            what="The change in each column from quotes moved up and moved down; they can cancel",
        )
        hidden = _transition_cells(point)[1]
        colour = _colour_column(r)
        transitions = _card(
            f"{display['label']}: baseline against point",
            _transition_chart(r, point, _W_HALF, 360),
            what=(
                f"Each dot is a pair of steps, sized by quotes and coloured by the change in "
                f"{colour['label']}; on the line, nothing moved"
                + (f". {hidden:,} smaller pairs aren't drawn" if hidden else "")
            ),
        )
        distribution = _card(
            f"Distribution of {display['label']}",
            _distribution_chart(r, point, _W_HALF, 220),
            what=_distribution_caption(r, point),
        )
        cumulative = _card(
            f"Cumulative distribution of {display['label']}",
            _ecdf_chart(r, point, _W_HALF, 220),
            what=_ecdf_caption(r, point),
        )
        movement = _card(
            "How far quotes moved",
            _step_histogram(point["movement"], _W_HALF, 170)
            + _movement_table(r, point),
            what="Steps from each quote's baseline step",
        )
        forward = ""
        if projection is not None:
            forward = _card(
                "Forward projection",
                _forward_table(r, point),
                what=(
                    f"Totals scaled from the {projection['window_days']:g} days of data to each "
                    f"horizon. This assumes the book's volume and mix stay as observed."
                ),
            )
        panels.append(
            f'<div class="dc pt pt-{i}">'
            + _kpis(r, point)
            + summary
            + directions
            + where
            + _row(distribution, cumulative)
            + _row(transitions, movement)
            + grid
            + forward
            + "</div>"
        )
    return _open(index, "impact") + "".join(panels) + "</section>"


def _change_bar(log_change: float, band: float) -> str:
    if band <= 0:
        return ""
    width = min(abs(log_change) / band, 1.0) * 50
    left = 50 if log_change >= 0 else 50 - width
    return f'<span class="chg"><i style="left:{left:.1f}%;width:{width:.1f}%"></i><b></b></span>'


def _level_table(
    r: dict[str, Any], levels: list[dict[str, Any]], limit: int, band: float
) -> str:
    columns = r["columns"]
    ordered = sorted(
        levels, key=lambda lv: (-abs(lv["log_change"]), -lv["quotes"], lv["label"])
    )
    shown = ordered[:limit]
    rows = []
    for lv in shown:
        change = math.exp(lv["log_change"]) - 1
        mean = lv["mean_step_change"]
        rows.append(
            [
                _e(lv["label"]),
                _rate(lv["existing"]),
                _rate(lv["new"]),
                _signed_pct(change) + _change_bar(lv["log_change"], band),
                f"{lv['quotes']:,}",
                "—" if mean is None else _minus(f"{mean:+.2f}"),
                *[_e(_signed(c, lv["impacts"][c["id"]])) for c in columns],
                {"min": "at minimum", "max": "at maximum"}.get(lv["band_edge"], ""),
            ]
        )
    cols = [
        "Level",
        "Existing",
        "New",
        "Change",
        "Quotes",
        "Mean move",
        *[c["label"] for c in columns],
        "Band",
    ]
    table = _table(cols, rows, numeric=list(range(1, 6 + len(columns))), scroll=True)
    rest = len(levels) - len(shown)
    return table + (
        f'<p class="note">+ {rest:,} more levels in the record.</p>' if rest else ""
    )


def _rate_card_tab(
    r: dict[str, Any], index: int, pt: _Radio, fx: _Radio, order: list[int]
) -> str:
    top_factors = r["display"]["top_n_factors"]
    limit = _rate_row_limit(r)
    candidates = r["run"]["candidates"]
    band = max(abs(math.log(candidates["min"])), abs(math.log(candidates["max"])))
    run = r["run"]
    show_changes = run["has_existing_rates"] or run["base_multiplier"] is not None
    show_full = run["has_existing_rates"] or run["base_multiplier"] is None
    domains = {
        kind: _beeswarm_domain(r, kind)
        for kind, on in (("changes", show_changes), ("full", show_full))
        if on
    }
    position = {f: k for k, f in enumerate(order)}
    panels = []
    for point in r["points"]:
        i = point["index"]
        cards = []
        if show_changes:
            cards.append(
                _card(
                    "Rate changes",
                    _beeswarm(r, point, "changes", domains["changes"], _W_FULL),
                    what=_beeswarm_caption(r, "changes", point),
                )
            )
        if show_full:
            cards.append(
                _card(
                    "The full set of new rates",
                    _beeswarm(r, point, "full", domains["full"], _W_FULL),
                    what=_beeswarm_caption(r, "full", point),
                )
            )
        ranked = _ranked_factors(point, top_factors)
        n_factors = len(point["rate_card"])
        largest = max(
            (_level_score(point["rate_card"][f]["levels"]) for f in ranked), default=0.0
        )
        rail = []
        for f in ranked:
            card = point["rate_card"][f]
            score = _level_score(card["levels"])
            width = 0 if largest == 0 else score / largest * 100
            rail.append(
                fx.label(
                    position[f],
                    f'<span class="nm" title="{_e(card["factor"])}">{_e(card["factor"])}</span>'
                    f'<span class="pc">{_signed_pct(math.exp(score) - 1)}</span>'
                    f'<span class="bars"><i style="width:{width:.0f}%"></i></span>',
                    cls="rl",
                )
            )
        hidden = n_factors - len(ranked)
        note = (
            f'<p class="note">{hidden} more factors are in the record.</p>'
            if hidden
            else ""
        )
        detail = []
        for f in order:
            tag = f"{fx.tag(position[f])}"
            card = point["rate_card"][f]
            if f not in ranked:
                body = (
                    '<p class="note">This factor is not among this point\'s largest changes. '
                    "Its levels are in the record.</p>"
                )
            elif limit == 0:
                body = (
                    '<p class="note">The per-point tables were omitted to keep the page within '
                    "its size budget. The Compare tab and the record still hold them.</p>"
                )
            else:
                body = _level_table(r, card["levels"], limit, band)
            detail.append(
                f'<div class="{tag}">' + _card(card["factor"], body) + "</div>"
            )
        trimmed = (
            f'<p class="note">Showing up to {limit} levels per factor to keep the page within '
            "its size budget.</p>"
            if 0 < limit < r["display"]["top_n_levels"]
            else ""
        )
        panels.append(
            f'<div class="dc {pt.tag(i)}">'
            + "".join(cards)
            + '<div class="fxgrid"><div class="card rail">'
            + "".join(rail)
            + note
            + "</div><div>"
            + trimmed
            + "".join(detail)
            + "</div></div></div>"
        )
    return _open(index, "rate-card") + "".join(panels) + "</section>"


def _other_level(
    rest: list[dict[str, Any]], columns: list[dict[str, str]]
) -> dict[str, Any]:
    """Levels past the row limit as one row: totals add, means weight by quotes."""
    quotes = sum(lv["quotes"] for lv in rest)

    def mean_of(key: str) -> float | None:
        known = [lv for lv in rest if lv[key] is not None]
        weight = sum(lv["quotes"] for lv in known)
        return sum(lv[key] * lv["quotes"] for lv in known) / weight if weight else None

    largest = [
        lv["max_abs_display_change"]
        for lv in rest
        if lv["max_abs_display_change"] is not None
    ]
    return {
        "label": f"Other ({len(rest)})",
        "quotes": quotes,
        "baseline": {
            c["id"]: sum(lv["baseline"][c["id"]] for lv in rest) for c in columns
        },
        "point": {c["id"]: sum(lv["point"][c["id"]] for lv in rest) for c in columns},
        "max_abs_display_change": max(largest) if largest else None,
        **{
            key: mean_of(key)
            for key in (
                "mean_step_change",
                "baseline_display_mean",
                "point_display_mean",
                "mean_abs_display_change",
                "share_beyond_band",
                "share_at_first",
                "share_at_last",
            )
        },
    }


def _impact_rows_table(r: dict[str, Any], levels: list[dict[str, Any]]) -> str:
    columns = r["columns"]
    rows = []
    for lv in levels:
        mean = lv["mean_step_change"]
        cells = [
            _e(lv["label"]),
            f"{lv['quotes']:,}",
            "—" if mean is None else _minus(f"{mean:+.2f}"),
        ]
        for column in columns:
            before, after = lv["baseline"][column["id"]], lv["point"][column["id"]]
            change = after - before
            pct = None if before == 0 else change / abs(before)
            cells += [
                _e(_fmt(column, before)),
                _e(_fmt(column, after)),
                _two(_e(_signed(column, change)), _e(_signed_pct(pct))),
            ]
        rows.append(cells)
    cols = ["Level", "Quotes", "Mean move"]
    for column in columns:
        cols += [
            f"{column['label']} baseline",
            f"{column['label']} point",
            f"{column['label']} change",
        ]
    return _table(cols, rows, numeric=list(range(1, 3 + 3 * len(columns))), scroll=True)


def _movement_rows_table(r: dict[str, Any], levels: list[dict[str, Any]]) -> str:
    display = _display(r)
    band = r["run"]["movement_band"]
    label = display["label"]

    def share(value: float | None) -> str:
        return "—" if value is None else _share(value)

    rows = []
    for lv in levels:
        before, after = lv["baseline_display_mean"], lv["point_display_mean"]
        change = None if before is None or after is None else after - before
        cells = [
            _e(lv["label"]),
            f"{lv['quotes']:,}",
            _e(_disp(display, before)),
            _e(_disp(display, after)),
            _e(_disp_signed(display, change)),
            _e(_disp(display, lv["mean_abs_display_change"])),
            _e(_disp(display, lv["max_abs_display_change"])),
        ]
        if band is not None:
            cells.append(share(lv["share_beyond_band"]))
        cells += [share(lv["share_at_first"]), share(lv["share_at_last"])]
        rows.append(cells)
    cols = [
        "Level",
        "Quotes",
        f"Mean {label} baseline",
        f"Mean {label} point",
        "Mean change",
        "Mean move",
        "Largest move",
    ]
    if band is not None:
        cols.append(f"Beyond {_disp(display, band)}")
    cols += ["On first step", "On last step"]
    return _table(cols, rows, numeric=range(1, len(cols)), scroll=True)


def _segment_card(
    r: dict[str, Any],
    segment: dict[str, Any],
    limit: int,
) -> str:
    columns = r["columns"]
    objective = _objective(r)
    levels = segment["levels"]
    shown = levels[:limit]
    rest = levels[limit:]
    rows = [*shown, *([_other_level(rest, columns)] if rest else [])]
    bars = [
        (lv["label"], lv["point"][objective["id"]] - lv["baseline"][objective["id"]])
        for lv in rows
    ]
    return _card(
        f"By {segment['column']}",
        _segment_bars(objective, bars, _W_FULL)
        + _impact_rows_table(r, rows)
        + "<h4>Movement</h4>"
        + _movement_rows_table(r, rows),
        what=f"Change in {objective['label']} by segment, then every column and how far quotes moved",
    )


def _band_card(r: dict[str, Any], band: dict[str, Any]) -> str:
    objective = _objective(r)
    display = _display(r)
    levels = band["levels"]
    labels = [lv["label"] for lv in levels]
    colour = _colour_column(r)
    objective_id = objective["id"]

    def per_quote(value: float) -> str:
        return _fine(objective, value).lstrip("+")

    rows = _row(
        "<div>"
        + f"<h4>% change in {_e(objective['label'])}</h4>"
        + _band_bars(objective, levels, _W_HALF, 200)
        + "</div>",
        "<div>"
        + f"<h4>% change in {_e(colour['label'])}</h4>"
        + _band_bars(colour, levels, _W_HALF, 200)
        + "</div>",
    ) + _row(
        "<div>"
        + f"<h4>Mean {_e(objective['label'])} per quote</h4>"
        + _band_lines(
            objective["label"],
            labels,
            [lv["baseline"][objective_id] / lv["quotes"] for lv in levels],
            [lv["point"][objective_id] / lv["quotes"] for lv in levels],
            per_quote,
            objective["format"],
            _W_HALF,
            200,
        )
        + "</div>",
        "<div>"
        + f"<h4>Mean {_e(display['label'])}</h4>"
        + _band_lines(
            display["label"],
            labels,
            [lv["baseline_display_mean"] for lv in levels],
            [lv["point_display_mean"] for lv in levels],
            lambda v: _disp(display, v),
            display["format"],
            _W_HALF,
            200,
        )
        + "</div>",
    )
    tables = (
        "<details><summary>Show every band</summary>"
        + _impact_rows_table(r, levels)
        + "<h4>Movement</h4>"
        + _movement_rows_table(r, levels)
        + "</details>"
    )
    return _card(
        f"By {band['column']} band",
        rows + tables,
        what="Ten equal-count bands, lowest first, with the mean of the column under each",
    )


def _segments_tab(r: dict[str, Any], index: int, pt: _Radio) -> str:
    top_n = r["display"]["top_n_levels"]
    limit = _segment_row_limit(r)
    if limit is None:
        omitted = _card(
            "Segments",
            '<p class="note">The segment tables were omitted to keep the page within its size '
            "budget. Every level is in the record.</p>",
        )
        return _open(index, "segments") + omitted + "</section>"
    trimmed = (
        f'<p class="note">Showing up to {limit} levels per segment, with the rest in "Other", '
        "to keep the page within its size budget.</p>"
        if limit < top_n
        else ""
    )
    panels = [trimmed]
    for point in r["points"]:
        cards = [_segment_card(r, segment, limit) for segment in point["segments"]]
        cards += [_band_card(r, band) for band in point["bands"] if band["levels"]]
        panels.append(
            f'<div class="dc {pt.tag(point["index"])}">' + "".join(cards) + "</div>"
        )
    return _open(index, "segments") + "".join(panels) + "</section>"


def _tone(value: float, largest: float) -> str:
    if value == 0 or largest <= 0:
        return ""
    step = max(1, min(5, math.ceil(math.sqrt(abs(value) / largest) * 5)))
    return f"g{step}" if value > 0 else f"y{step}"


def _compare(r: dict[str, Any], index: int) -> str:
    points = r["points"]
    objective = _objective(r)
    trade_rows = []
    for piece in r["comparisons"]["slices"]:
        fixed = ", ".join(
            f"{_column(r, c)['label']} {v:g}" for c, v in piece["fixed"].items()
        )
        for step in piece["marginal"]:
            value = "—" if step["value"] is None else _minus(_num(step["value"]))
            trade_rows.append(
                [
                    _e(_column(r, piece["dimension"])["label"]),
                    _e(fixed) or '<span class="muted">—</span>',
                    _e(points[step["from"]]["name"]),
                    _e(points[step["to"]]["name"]),
                    value
                    if step["value"] is not None
                    else f'— <span class="muted">{_e(step["value_reason"])}</span>',
                ]
            )
    trade = _card(
        "The trade-off along the frontier",
        _table(
            ["Dimension", "Held at", "From", "To", "Marginal rate"],
            trade_rows,
            numeric=[4],
            scroll=True,
        )
        if trade_rows
        else '<p class="note">One point per slice, so no marginal rates.</p>',
        what=f"Change in {objective['label']} per unit of change in the constraint, between neighbouring points",
    )

    existing_by_factor = {
        card["factor"]: {lv["label"]: lv["existing"] for lv in card["levels"]}
        for card in points[0]["rate_card"]
    }
    candidates = []
    for factor in r["comparisons"]["rate_changes"]:
        for level in factor["levels"]:
            spread = math.log(max(level["rates"])) - math.log(min(level["rates"]))
            candidates.append((-spread, factor["factor"], level))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]["label"]))
    chosen = candidates[:_COMPARE_ROWS]
    logs: list[list[float]] = []
    for _, factor, level in chosen:
        start = existing_by_factor[factor][level["label"]]
        logs.append(
            [math.log(v / (1.0 if start is None else start)) for v in level["rates"]]
        )
    largest = max((abs(v) for row in logs for v in row), default=0.0)
    rows = []
    tones = []
    for (_, factor, level), row_logs in zip(chosen, logs):
        rows.append(
            [_e(factor), _e(level["label"]), *[f"{v:.3f}" for v in level["rates"]]]
        )
        tones.append(["", "", *[_tone(v, largest) for v in row_logs]])
    cols = ["Factor", "Level", *[f"P{p['index']}" for p in points]]
    capped = len(candidates) - len(chosen)
    note = (
        f'<p class="note">{capped:,} more level rows are in the record.</p>'
        if capped
        else ""
    )
    rates = _card(
        "Levels that differ most across points",
        _table(cols, rows, numeric=range(2, len(cols)), scroll=True, tones=tones)
        + note,
        what="New rates, shaded green above and yellow below the starting rate",
    )
    return _open(index, "compare") + trade + rates + "</section>"


def _history_rows(r: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Within the trace budget, keeping every point's final pass (§19.5)."""
    histories = {p["index"]: p["history"] for p in r["points"] if p["history"]}
    total = sum(len(h) for h in histories.values())
    if total <= _TRACE_ROWS:
        return histories
    out: dict[int, list[dict[str, Any]]] = {}
    remaining = _TRACE_ROWS
    for index, history in histories.items():
        last_pass = history[-1]["cd_iteration"]
        tail = [row for row in history if row["cd_iteration"] == last_pass][:remaining]
        remaining -= len(tail)
        out[index] = tail
    return out


def _trace_charts(r: dict[str, Any], point: dict[str, Any]) -> str:
    """The objective and each constraint over the solve, from the full trace."""
    history = point["history"]
    if not history:
        return ""
    objective = _objective(r)

    def fmt_of(column: dict[str, str]) -> _Fmt:
        return lambda v: _fmt(column, v)

    width = (_W_FULL - _GAP) // 2
    charts = [
        _trace_chart(
            objective["label"],
            history,
            [row["total_objective"] for row in history],
            None,
            fmt_of(objective),
            objective["format"],
            width,
            170,
        )
    ]
    for constraint in r["run"]["constraints"]:
        name = constraint["name"]
        column = _column(r, name)
        charts.append(
            _trace_chart(
                column["label"],
                history,
                [row["totals"][name] for row in history],
                point["bounds"][name],
                fmt_of(column),
                column["format"],
                width,
                170,
            )
        )
    return "".join(_row(*charts[i : i + 2]) for i in range(0, len(charts), 2))


def _definitions(r: dict[str, Any]) -> str:
    items = [
        ("Baseline", f"each quote at its starting price: {_baseline_name(r)}."),
        (
            "Change",
            "the point less the baseline. A percentage is the change over the absolute baseline.",
        ),
        (
            "Binding",
            "the constraint's slack is within the binding tolerance of its bound.",
        ),
        (
            "Clamped",
            "the quote's price falls past an end of the scenario grid, so it takes that end's step.",
        ),
        (
            "Dominated",
            "another point is at least as good on the objective and every constraint, and better on one.",
        ),
        (
            "Band edge",
            "the rate sits at the smallest or largest change the search could make.",
        ),
        ("Percentiles", "nearest rank on the per-quote values, with no interpolation."),
        (
            "Band",
            "ten equal-count groups of a column, lowest first, with ties broken by quote ID.",
        ),
    ]
    display = r["run"]["scenario_display"]
    if display is not None:
        items.append(
            (
                display["label"],
                "shown in natural units; each step has one display value and a change is "
                "that value's difference between the point's step and the baseline step.",
            )
        )
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in items)
    return (
        '<p class="note">This report describes the data it was given. Each point\'s totals '
        "matched the frontier exactly, but that can't prove the data is the data that was swept.</p>"
        f'<details><summary>Definitions</summary><table class="kv">{rows}</table></details>'
    )


def _checks(r: dict[str, Any], index: int, pt: _Radio) -> str:
    run = r["run"]
    points = r["points"]
    flag_rows = []
    for flag in r["flags"]:
        where = "All points" if flag["point"] is None else points[flag["point"]]["name"]
        flag_rows.append(
            [
                _badge(flag["status"]),
                _e(where),
                _e(flag["message"]),
                _num(flag["value"]),
                _num(flag["threshold"]),
            ]
        )
    flags = _card(
        "Every review flag",
        _table(
            ["Status", "Where", "Finding", "Value", "Threshold"],
            flag_rows,
            numeric=[3, 4],
            scroll=True,
        )
        if flag_rows
        else '<p class="ok">No open findings.</p>',
        extra="",
    )
    audit_rows = []
    for point in points:
        for row in point["audit"]:
            column = _column(r, row["constraint"])
            audit_rows.append(
                [
                    _e(point["name"]),
                    _e(column["label"]),
                    _e(_fmt(column, row["bound"])),
                    _e(_fmt(column, row["achieved"])),
                    _e(_signed(column, row["slack"])),
                    "yes" if row["met"] else "no",
                    "yes" if row["binding"] else "no",
                ]
            )
    audit = _card(
        "Constraint audit",
        _table(
            ["Point", "Constraint", "Bound", "Achieved", "Slack", "Met", "Binding"],
            audit_rows,
            numeric=[2, 3, 4],
            scroll=True,
        ),
    )
    names = [c["name"] for c in run["constraints"]]
    health_rows = [
        [
            _e(p["name"]),
            "yes" if p["health"]["converged"] else "no",
            f"{p['health']['iterations']}",
            *[_num(p["health"]["lambdas"][c]) for c in names],
            _share(p["health"]["clamp_rate"], 2),
            f"{p['health']['clamped_low']:,}",
            f"{p['health']['clamped_high']:,}",
        ]
        for p in points
    ]
    health = _card(
        "Solve health",
        _table(
            [
                "Point",
                "Converged",
                "Iterations",
                *[f"λ {c}" for c in names],
                "Clamp rate",
                "Clamped low",
                "Clamped high",
            ],
            health_rows,
            numeric=range(2, 6 + len(names)),
            scroll=True,
        ),
        what="Convergence means the rates stopped moving; compare the audit for whether constraints were met",
    )
    histories = _history_rows(r)
    trace = ""
    if histories:
        total = sum(len(p["history"]) for p in points if p["history"])
        shown = sum(len(h) for h in histories.values())
        panels = []
        for point in points:
            rows = histories.get(point["index"], [])
            body = [
                [
                    str(row["cd_iteration"]),
                    _e(row["factor"]),
                    _num(row["total_objective"]),
                    str(row["inner_iterations"]),
                    "yes" if row["inner_converged"] else "no",
                ]
                for row in rows
            ]
            table = _table(
                ["Pass", "Factor", "Objective", "Inner iterations", "Inner converged"],
                body,
                numeric=[0, 2, 3],
                scroll=True,
            )
            charts = _trace_charts(r, point)
            panels.append(
                f'<div class="{pt.tag(point["index"])}">{charts}'
                f"<details><summary>Show the coordinate-descent trace</summary>{table}</details></div>"
            )
        note = (
            f'<p class="note">Showing each point\'s final pass: {shown:,} of {total:,} rows.</p>'
            if shown < total
            else ""
        )
        trace = _card(
            "Convergence",
            note + "".join(panels),
            what="The running totals after each factor; a solid line is a constraint's bound",
        )
    facts: list[tuple[str, str]] = [
        ("Quotes", f"{run['n_quotes']:,}"),
        ("Scenario steps", f"{run['n_steps']:,}"),
        ("Objective", _column(r, run["objective"])["label"]),
        (
            "Constraints",
            ", ".join(
                f"{_column(r, c['name'])['label']} ({c['kind']})"
                for c in run["constraints"]
            ),
        ),
        ("Factors", f"{len(points[0]['rate_card']):,}"),
        (
            "Candidate range",
            f"{run['candidates']['min']:.4g} to {run['candidates']['max']:.4g} in {run['candidates']['steps']} steps",
        ),
        ("Existing rates", "supplied" if run["has_existing_rates"] else "none"),
        ("Coordinate-descent passes, at most", str(run["solver"]["max_cd_iterations"])),
        ("Coordinate-descent tolerance", f"{run['solver']['cd_tolerance']:g}"),
        ("Inner iterations, at most", str(run["solver"]["max_iter"])),
        ("Inner tolerance", f"{run['solver']['tolerance']:g}"),
        ("Per-quote starting point", run["base_multiplier"] or "none"),
        ("price-contour", r["library"]["price_contour_version"]),
        ("Generated", r["generated_at"]),
        *[(str(k), str(v)) for k, v in run["metadata"]],
    ]
    fact_rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in facts)
    run_card = _card("This run", f'<table class="kv">{fact_rows}</table>')
    how = _card("How to read this report", _definitions(r))
    return (
        _open(index, "checks")
        + flags
        + audit
        + health
        + trace
        + _row(run_card, how)
        + "</section>"
    )


def _header(r: dict[str, Any]) -> str:
    run = r["run"]
    facts = [
        f"{run['n_quotes']:,} quotes",
        f"{len(r['points'])} points",
        f"compared with {_baseline_name(r)}",
        f"price-contour {r['library']['price_contour_version']}",
        r["generated_at"].replace("T", " ").replace("+00:00", " UTC"),
    ]
    subtitle = f'<p class="sub">{_e(r["subtitle"])}</p>' if r["subtitle"] else ""
    return (
        f'<header class="top"><h1>{_e(r["title"])}</h1>{subtitle}'
        f'<p class="sub">{" · ".join(_e(x) for x in facts)}</p></header>'
    )


def frontier_report_html(report: dict[str, Any]) -> str:
    """Render a frontier report (version 1) as one self-contained HTML page.

    Raises ``ValueError`` for a record of another version.
    """
    version = report.get("report_version") if isinstance(report, dict) else None
    if version != REPORT_VERSION:
        raise ValueError(
            f"frontier_report_html renders report_version {REPORT_VERSION}, got {version!r}"
        )
    r = report
    has_segments = any(p["segments"] or p["bands"] for p in r["points"])
    names = ["Overview", "Impact", "Rate card"]
    if has_segments:
        names.append("Segments")
    names += ["Compare", "Checks"]
    tabs = _Radio("tb", len(names))
    points = r["points"]
    pt = _Radio("pt", len(points), r["default_point"], keep_scroll=True)

    top_factors = r["display"]["top_n_factors"]
    ranked_by_point = [_ranked_factors(p, top_factors) for p in points]
    order = sorted({f for ranked in ranked_by_point for f in ranked})
    default_ranked = ranked_by_point[r["default_point"]]
    default_factor = order.index(default_ranked[0]) if default_ranked else 0
    fx = _Radio("fx", len(order), default_factor, keep_scroll=True)

    checks_tab = len(names) - 1
    sections = [
        _overview(r, pt, tabs, checks_tab),
        _impact(r, 1),
        _rate_card_tab(r, 2, pt, fx, order),
    ]
    if has_segments:
        sections.append(_segments_tab(r, 3, pt))
    sections.append(_compare(r, len(names) - 2))
    sections.append(_checks(r, checks_tab, pt))

    switch = (
        '<div class="switch" role="group" aria-label="point"><span class="cap">point</span>'
        + "".join(
            pt.label(p["index"], f"P{p['index']}", title=p["name"]) for p in points
        )
        + "</div>"
    )
    bar = (
        '<nav class="bar"><div class="tabs">'
        + "".join(tabs.label(i, _e(t)) for i, t in enumerate(names))
        + f"</div>{switch}</nav>"
    )
    on = "background:var(--wash);color:var(--ink);box-shadow:inset 0 -2px 0 var(--data)"
    base = _CSS
    for token, value in (
        ("@LIGHT@", _LIGHT),
        ("@DARK@", _DARK),
        ("@CONTENT@", str(_CONTENT)),
        ("@GAP@", str(_GAP)),
        ("@FONT@", str(int(_FONT))),
        ("@RAIL@", str(_RAIL)),
    ):
        base = base.replace(token, value)
    css = (
        base
        + tabs.css(
            "background:var(--surface);color:var(--ink);border-color:var(--border)"
        )
        + pt.css(on)
        + fx.css("background:var(--wash);box-shadow:inset 3px 0 0 var(--model)")
        + _point_css(len(points))
        + _swarm_css()
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(r['title'])}</title><style>{css}</style></head><body>"
        f'<div class="app">{tabs.inputs()}{pt.inputs()}{fx.inputs()}{_header(r)}{bar}'
        f'<main class="body">{"".join(sections)}</main></div></body></html>'
    )
