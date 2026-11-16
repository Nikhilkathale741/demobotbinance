"""
Binance DEMO futures bot (multi-pair) - SUPERTREND + SUPPORT/RESISTANCE + NEWS FILTER.
Fixed per-trade exits: +3% take profit / -2% stop loss. DEMO / TESTNET ONLY.

Strategy (all must be true to take a trade):
  1. Supertrend gives the direction. UP -> longs only. DOWN -> shorts only.
  2. Price pulls back to a support (long) / resistance (short) level and rejects it
     (candle touches the level, closes back on the right side, in the trend direction).
  3. There is room to the next opposite level for the fixed 3% take-profit target.
  4. News filter: no new trades around high-impact USD news, and not on a news-spike candle.
  5. Stop loss is 2% and take profit is 3% from the actual entry fill.

Setup:
    pip install ccxt pandas numpy requests
    set BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET_KEY env vars
    python binance_demo_bot.py

NOTE: No bot can guarantee "no loss". Safety limits below only cap the damage.
"""
import os
import csv
import json
import time
import logging
from xml.sax.saxutils import escape
from datetime import datetime, timezone, timedelta

import ccxt
import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # news feed is optional
    requests = None

# ============================ CONFIG ============================
API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")

# "demo"    -> Binance Demo Trading keys (demo.binance.com)
# "testnet" -> Futures testnet keys (testnet.binancefuture.com)
MODE = "demo"

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "1000SATS/USDT:USDT",
    "1000PEPE/USDT:USDT",
    "1000SHIB/USDT:USDT",
    "1000BONK/USDT:USDT",
    "1000FLOKI/USDT:USDT",
    "1000RATS/USDT:USDT",
    "MEME/USDT:USDT",
    "BOME/USDT:USDT",
    "TURBO/USDT:USDT",
    "NEIRO/USDT:USDT",
    "PENGU/USDT:USDT",
]
MAX_OPEN_TRADES = 1000     # maximum simultaneous open positions across all symbols
TIMEFRAME = "1m"           # 1m = many trades but noisy; "5m" or "15m" = fewer, cleaner

# --- Supertrend ---
ST_ATR_LEN = 10
ST_MULT = 3.0
USE_EMA_FILTER = False     # True = also require price above (long) / below (short) the 200 EMA
EMA_LEN = 200

# --- Support / Resistance (swing-pivot levels) ---
PIVOT_N = 5                # a pivot needs N candles each side that are lower (high) / higher (low)
LEVEL_LOOKBACK = 300       # candles used to build levels
CLUSTER_PCT = 0.20         # pivots within this % of each other merge into one level
MIN_TOUCHES = 1            # raise to 2 for stronger (rarer) levels
ZONE_ATR = 0.35            # candle must come within this many ATRs of the level
SL_ATR = 0.25              # stop sits this many ATRs beyond the level / candle extreme
MIN_ROOM = 1.0             # nearest opposite level must be >= MIN_ROOM x risk away

# --- Per-trade exits are ROI on allocated position margin, NOT account balance ---
TAKE_PROFIT_ROI_PCT = 3.0  # +3% gross ROI on initial position margin
STOP_LOSS_ROI_PCT = 2.0    # -2% gross ROI on initial position margin
# At 5x leverage: +3% ROI ~= +0.60% price move; -2% ROI ~= -0.40% price move.
RR = TAKE_PROFIT_ROI_PCT / STOP_LOSS_ROI_PCT  # 1.5 reward-to-risk before fees/funding/slippage

# --- Risk management ---
RISK_PCT = 1.0             # % of balance risked per trade
LEVERAGE = 5
MAX_MARGIN_PER_TRADE = 1 # a single trade may use at most 1% of balance as margin
MAX_DAILY_LOSS_PCT = 3.0   # stop opening trades for the day after this drawdown
MAX_CONSEC_LOSSES = 3      # stop bot after this many losses in a row (all pairs combined)

# --- News filter ---
USE_NEWS_FEED = True       # auto-load this week's high-impact USD events (needs internet)
NEWS_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_COUNTRIES = {"USD"}
NEWS_IMPACTS = {"High"}
NEWS_BEFORE_MIN = 30       # no new trades this many minutes BEFORE an event
NEWS_AFTER_MIN = 30        # ... and this many minutes AFTER
SPIKE_ATR = 3.0            # skip if the signal candle is bigger than this many ATRs (news-like spike)
NEWS_EVENTS_UTC = [        # manual events, UTC, format "YYYY-MM-DD HH:MM"
    # "2026-11-18 13:30",
]

# --- Reports ---
REPORT_EVERY_N_TRADES = 5  # write a PDF report each time this many trades have CLOSED (0 = off)
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_JSON = os.path.join(BASE_DIR, "trade_log.json")   # every closed trade, kept across restarts
TRADE_LOG_CSV = os.path.join(BASE_DIR, "trades.csv")        # same data, opens in Excel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("bot")


def short(sym):
    return sym.split("/")[0]


def make_exchange():
    if not API_KEY or not SECRET:
        raise SystemExit("Set BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET_KEY env vars.")
    ex = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    if MODE == "demo" and hasattr(ex, "enable_demo_trading"):
        ex.enable_demo_trading(True)
    else:
        ex.set_sandbox_mode(True)
    ex.load_markets()
    return ex


# ------------------------- indicators -------------------------
def supertrend(df, n=ST_ATR_LEN, mult=ST_MULT):
    """Standard Supertrend. Returns (line, direction[+1 up / -1 down], atr)."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()

    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if c[i] > fub[i - 1]:
            d[i] = 1
        elif c[i] < flb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    line = np.where(d == 1, flb, fub)
    return line, d, atr


def add_indicators(df):
    line, d, atr = supertrend(df)
    df["st"], df["st_dir"], df["atr"] = line, d, atr
    df["ema"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()
    return df


def fetch_candles(ex, symbol):
    raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=500)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming candle
    return add_indicators(df)


# ------------------------ support / resistance ------------------------
def find_levels(df, n=PIVOT_N, lookback=LEVEL_LOOKBACK, cluster_pct=CLUSTER_PCT):
    """Swing-pivot levels (highs and lows pooled, so broken levels flip role).
    Pivots are only confirmed with n candles to their right -> no look-ahead.
    Returns list of (price, touches) sorted by price."""
    d = df.iloc[-lookback:].reset_index(drop=True)
    highs, lows = d["high"].to_numpy(float), d["low"].to_numpy(float)
    pts = []
    for i in range(n, len(d) - n):
        if highs[i] >= highs[i - n:i + n + 1].max():
            pts.append(highs[i])
        if lows[i] <= lows[i - n:i + n + 1].min():
            pts.append(lows[i])
    pts.sort()
    levels, group = [], []
    for p in pts:
        if group and p > np.mean(group) * (1 + cluster_pct / 100):
            levels.append((float(np.mean(group)), len(group)))
            group = []
        group.append(p)
    if group:
        levels.append((float(np.mean(group)), len(group)))
    return levels


# --------------------------- news filter ---------------------------
_news = {"events": [], "next_fetch": 0.0}


def refresh_news():
    """Load this week's high-impact events (cached 6h; retried in 15 min on failure)."""
    if not USE_NEWS_FEED or requests is None or time.time() < _news["next_fetch"]:
        return
    try:
        r = requests.get(NEWS_FEED_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        events = []
        for e in r.json():
            if e.get("country") in NEWS_COUNTRIES and e.get("impact") in NEWS_IMPACTS:
                try:
                    t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
                    events.append((t, e.get("title", "news")))
                except Exception:
                    continue
        _news["events"] = events
        _news["next_fetch"] = time.time() + 6 * 3600
        log.info(f"News feed loaded: {len(events)} high-impact events this week")
    except Exception as e:
        _news["next_fetch"] = time.time() + 15 * 60
        log.info(f"News feed unavailable ({e}) - using manual NEWS_EVENTS_UTC only")


def news_blocked(now=None):
    """Return the title of the news event we are inside the window of, else None."""
    now = now or datetime.now(timezone.utc)
    events = list(_news["events"])
    for s in NEWS_EVENTS_UTC:
        events.append((datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc), "manual event"))
    for t, title in events:
        if t - timedelta(minutes=NEWS_BEFORE_MIN) <= now <= t + timedelta(minutes=NEWS_AFTER_MIN):
            return title
    return None


# ----------------------------- signal -----------------------------
def get_signal(df):
    """Return ((side, entry, sl) or None, reason). ALL criteria must pass."""
    if len(df) < max(60, PIVOT_N * 4 + 10):
        return None, "not enough candles"

    c = df.iloc[-1]                       # last CLOSED candle = signal candle
    atr = float(c["atr"])
    if atr <= 0:
        return None, "ATR is zero"
    entry = float(c["close"])
    tol = ZONE_ATR * atr
    levels = [(p, t) for p, t in find_levels(df) if t >= MIN_TOUCHES]
    trend = "UP" if c["st_dir"] == 1 else "DOWN"

    if USE_EMA_FILTER and ((trend == "UP") != (c["close"] > c["ema"])):
        return None, f"Supertrend {trend} but 200 EMA disagrees"

    if (c["high"] - c["low"]) > SPIKE_ATR * atr:
        return None, f"Supertrend {trend} | spike candle (>{SPIKE_ATR} ATR) - news-like move, skipping"

    if trend == "UP":
        near = [p for p, _ in levels if p < entry and c["low"] <= p + tol]
        if not near:
            sup = [p for p, _ in levels if p < entry]
            gap = (entry - max(sup)) / atr if sup else float("nan")
            return None, f"UP trend | no support touched (nearest support {gap:.1f} ATR below)"
        lvl = max(near)
        if not c["close"] > c["open"]:
            return None, f"UP trend | touched support {lvl:.4f} but candle not bullish yet"
        sl = min(float(c["low"]), lvl) - SL_ATR * atr
        risk = entry - sl
        res = [p for p, _ in levels if p > entry]
        if res and (min(res) - entry) < entry * (TAKE_PROFIT_ROI_PCT / LEVERAGE) / 100:
            return None, f"UP trend | support bounce but resistance {min(res):.4f} is inside the +{TAKE_PROFIT_ROI_PCT:.1f}% ROI target zone"
        room = (f"the next resistance at {min(res):.6g} is beyond the +{TAKE_PROFIT_ROI_PCT:.1f}% ROI target"
                if res else "there is no resistance overhead, so the fixed target has open space")
        why = (f"Supertrend is UP, so only longs are allowed. Price pulled back to support {lvl:.6g} "
               f"(candle low {float(c['low']):.6g}) and closed bullish, rejecting the level. "
               f"Stop placed below the level; {room}. No high-impact news within {NEWS_BEFORE_MIN} min.")
        return ("buy", entry), why

    near = [p for p, _ in levels if p > entry and c["high"] >= p - tol]
    if not near:
        resi = [p for p, _ in levels if p > entry]
        gap = (min(resi) - entry) / atr if resi else float("nan")
        return None, f"DOWN trend | no resistance touched (nearest resistance {gap:.1f} ATR above)"
    lvl = min(near)
    if not c["close"] < c["open"]:
        return None, f"DOWN trend | touched resistance {lvl:.4f} but candle not bearish yet"
    sl = max(float(c["high"]), lvl) + SL_ATR * atr
    risk = sl - entry
    sup = [p for p, _ in levels if p < entry]
    if sup and (entry - max(sup)) < entry * (TAKE_PROFIT_ROI_PCT / LEVERAGE) / 100:
        return None, f"DOWN trend | resistance rejection but support {max(sup):.4f} is inside the +{TAKE_PROFIT_ROI_PCT:.1f}% ROI target zone"
    room = (f"the next support at {max(sup):.6g} is beyond the +{TAKE_PROFIT_ROI_PCT:.1f}% ROI target"
            if sup else "there is no support below, so the fixed target has open space")
    why = (f"Supertrend is DOWN, so only shorts are allowed. Price rallied to resistance {lvl:.6g} "
           f"(candle high {float(c['high']):.6g}) and closed bearish, rejecting the level. "
           f"Stop placed above the level; {room}. No high-impact news within {NEWS_BEFORE_MIN} min.")
    return ("sell", entry), why


# --------------------------- trading ---------------------------
def get_balance(ex):
    bal = ex.fetch_balance()
    usdt = bal.get("USDT") or {}
    total = usdt.get("total")
    if total is None:
        held = {k: v for k, v in (bal.get("total") or {}).items() if v}
        raise SystemExit(
            "No USDT found in your futures wallet.\n"
            f"Assets in this account: {held if held else 'none (wallet is empty)'}\n"
            "Fix: add demo USDT to the FUTURES wallet, "
            "and check MODE matches where you made the keys."
        )
    return float(total)


def symbols_with_position(ex):
    out = set()
    for p in ex.fetch_positions(SYMBOLS):
        if abs(float(p.get("contracts") or 0)) > 0:
            out.add(p["symbol"])
    return out


def trade_summary(ex, symbol, info):
    """Net PnL (after fees), average exit price and close time of the trade that just closed."""
    exit_side = "sell" if info["side"] == "buy" else "buy"
    pnl, exit_amt, exit_val, closed_ms, ok = 0.0, 0.0, 0.0, None, True
    try:
        for t in ex.fetch_my_trades(symbol, since=info["t0"]):
            pnl += float(t["info"].get("realizedPnl", 0) or 0)
            fee = t.get("fee") or {}
            if fee.get("currency") == "USDT":
                pnl -= float(fee.get("cost") or 0)
            if t.get("side") == exit_side:
                a = float(t["amount"])
                exit_amt += a
                exit_val += a * float(t["price"])
                closed_ms = max(closed_ms or 0, int(t["timestamp"]))
    except Exception as e:
        ok = False
        log.info(f"[{short(symbol)}] could not read fills/PnL: {e}")
    return {
        "ok": ok,
        "pnl": pnl,
        "exit": (exit_val / exit_amt) if exit_amt else None,
        "closed_ms": closed_ms or int(time.time() * 1000),
    }


# --------------------------- trade log + PDF report ---------------------------
def fp(x):
    return "n/a" if x is None else f"{x:.8g}"


def fts(ms):
    return datetime.fromtimestamp(ms / 1000).strftime("%d %b %H:%M:%S") if ms else "n/a"


def fdur(a_ms, b_ms):
    s = max(0, int((b_ms - a_ms) / 1000))
    return f"{s // 3600}h {s % 3600 // 60}m {s % 60}s" if s >= 3600 else f"{s // 60}m {s % 60}s"


def load_trade_log():
    try:
        with open(TRADE_LOG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_trade_log(trades):
    try:
        with open(TRADE_LOG_JSON, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)
        cols = ["n", "symbol", "side", "opened", "closed", "entry", "exit", "sl", "tp",
                "qty", "pnl", "r", "result", "reason"]
        with open(TRADE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for t in trades:
                w.writerow([t["n"], t["symbol"], t["side"], fts(t["opened_ms"]), fts(t["closed_ms"]),
                            fp(t["entry"]), fp(t["exit"]), fp(t["sl"]), fp(t["tp"]), t["qty"],
                            f"{t['pnl']:.4f}", "" if t["r"] is None else f"{t['r']:.2f}",
                            t["result"], t["reason"]])
    except Exception as e:
        log.info(f"could not save trade log: {e}")


def trade_stats(trs):
    n = len(trs)
    wins = sum(1 for t in trs if t["pnl"] > 0)
    rs = [t["r"] for t in trs if t["r"] is not None]
    return {
        "n": n, "wins": wins, "losses": n - wins,
        "win_rate": (wins / n * 100) if n else 0.0,
        "pnl": sum(t["pnl"] for t in trs),
        "avg_r": (sum(rs) / len(rs)) if rs else None,
    }


def make_report(all_trades, batch, path, balance=None):
    """Write the PDF report for `batch` (list of trades), with overall stats from `all_trades`."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

    GREEN, RED, DARK, GREY = colors.HexColor("#1a7f37"), colors.HexColor("#c62828"), \
        colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=DARK, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9, textColor=GREY, alignment=1)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=DARK, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9, leading=12)
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.5, leading=10)
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8, textColor=GREY, leading=10)

    def money(x):
        return f"{x:+.2f}"

    first, last = batch[0]["n"], batch[-1]["n"]
    sb, sa = trade_stats(batch), trade_stats(all_trades)
    story = [
        Paragraph("Trading Bot Report", h1),
        Paragraph(f"Trades #{first} to #{last} &nbsp;|&nbsp; Binance DEMO futures &nbsp;|&nbsp; "
                  f"Supertrend + Support/Resistance + News filter, +3% ROI TP / -2% ROI SL &nbsp;|&nbsp; "
                  f"Generated {datetime.now().strftime('%d %b %Y %H:%M')}", sub),
        Spacer(1, 8),
    ]

    # ---- summary ----
    def stat_row(label, s):
        return [label, str(s["n"]), f"{s['wins']} / {s['losses']}", f"{s['win_rate']:.0f}%",
                money(s["pnl"]), "n/a" if s["avg_r"] is None else f"{s['avg_r']:+.2f}R"]

    summ = Table(
        [["", "Trades", "Wins / Losses", "Win rate", "Net PnL (USDT)", "Avg result"],
         stat_row("This report", sb), stat_row("All trades so far", sa)],
        colWidths=[45 * mm, 28 * mm, 38 * mm, 28 * mm, 42 * mm, 32 * mm])
    summ.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("TEXTCOLOR", (4, 1), (4, 1), GREEN if sb["pnl"] >= 0 else RED),
        ("TEXTCOLOR", (4, 2), (4, 2), GREEN if sa["pnl"] >= 0 else RED),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ]))
    story += [summ]
    if balance is not None:
        story += [Spacer(1, 4), Paragraph(f"Account balance at report time: {balance:,.2f} USDT", small)]

    # ---- trade table ----
    story.append(Paragraph("Trades at a glance", h2))
    rows = [["#", "Coin", "Side", "Opened", "Closed", "Entry", "Exit", "PnL (USDT)", "Result", "R"]]
    for t in batch:
        rows.append([str(t["n"]), t["symbol"], t["side"].upper(), fts(t["opened_ms"]), fts(t["closed_ms"]),
                     fp(t["entry"]), fp(t["exit"]), money(t["pnl"]), t["result"],
                     "n/a" if t["r"] is None else f"{t['r']:+.2f}"])
    tbl = Table(rows, repeatRows=1,
                colWidths=[10 * mm, 24 * mm, 16 * mm, 34 * mm, 34 * mm, 28 * mm, 28 * mm, 28 * mm, 30 * mm, 14 * mm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), DARK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
    ]
    for i, t in enumerate(batch, start=1):
        style += [("TEXTCOLOR", (7, i), (7, i), GREEN if t["pnl"] > 0 else RED),
                  ("FONTNAME", (7, i), (7, i), "Helvetica-Bold")]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)

    # ---- per-trade reasons ----
    story.append(Paragraph("Why the bot took each trade", h2))
    for t in batch:
        col = "#1a7f37" if t["pnl"] > 0 else "#c62828"
        risk_usd = abs(t["entry"] - t["sl"]) * t["qty"]
        block = [
            Paragraph(f"<b>#{t['n']} &nbsp;{escape(t['symbol'])} {t['side'].upper()}</b> &nbsp;-&nbsp; "
                      f"{escape(t['result'])} &nbsp;<font color='{col}'><b>{money(t['pnl'])} USDT</b></font>", body),
            Paragraph(f"<b>Reason:</b> {escape(t['reason'])}", cell),
            Paragraph(f"<b>Plan:</b> entry {fp(t['entry'])}, stop {fp(t['sl'])}, target {fp(t['tp'])}, "
                      f"size {t['qty']}, planned risk about {risk_usd:.2f} USDT. "
                      f"<b>Outcome:</b> exit {fp(t['exit'])} after {fdur(t['opened_ms'], t['closed_ms'])}; "
                      f"net {money(t['pnl'])} USDT after fees.", cell),
            Spacer(1, 7),
        ]
        story.append(KeepTogether(block))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Demo account results. Planned exits target +3% gross ROI and -2% gross ROI on initial position margin "
        "(ROI converted to price triggers using configured leverage; fees/funding/slippage excluded). "
        "Realized results can differ because of "
        "execution, fees and slippage. A handful of trades is too few to judge a strategy. "
        "This is not financial advice.", small))

    doc = SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Trading Bot Report")
    doc.build(story)


def write_report(ex, trades):
    """Create the PDF for the latest batch of closed trades. Never raises."""
    batch = trades[-REPORT_EVERY_N_TRADES:]
    path = os.path.join(BASE_DIR, f"report_trades_{batch[0]['n']}-{batch[-1]['n']}_"
                                  f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    try:
        bal = None
        try:
            bal = get_balance(ex)
        except Exception:
            pass
        make_report(trades, batch, path, bal)
        log.info(f"REPORT SAVED: {path}")
    except ImportError:
        log.info("PDF report needs reportlab - run: pip install reportlab "
                 f"(your trades are still saved in {TRADE_LOG_CSV})")
    except Exception as e:
        log.info(f"Could not create PDF report: {e} (trades are still saved in {TRADE_LOG_CSV})")


def emergency_close(ex, symbol, side, qty, why):
    exit_side = "sell" if side == "buy" else "buy"
    log.info(f"[{short(symbol)}] EMERGENCY CLOSE ({why})")
    try:
        ex.create_order(symbol, "market", exit_side, qty, None, {"reduceOnly": True})
    except Exception as e:
        log.info(f"[{short(symbol)}] !!! could not close position - CLOSE IT MANUALLY: {e}")
    try:
        ex.cancel_all_orders(symbol)
    except Exception:
        pass


def max_order_qty(ex, symbol):
    """Largest quantity Binance allows in one market order for this symbol (None if unknown)."""
    m = ex.market(symbol)
    caps = []
    amax = (m["limits"].get("amount") or {}).get("max")
    if amax:
        caps.append(float(amax))
    for f in (m.get("info") or {}).get("filters", []):
        if f.get("filterType") in ("MARKET_LOT_SIZE", "LOT_SIZE"):
            q = float(f.get("maxQty") or 0)
            if q > 0:
                caps.append(q)
    return min(caps) if caps else None


def place_trade(ex, symbol, side, entry, reason=""):
    """Returns a dict describing the trade only if a position is open and protected by SL + TP,
    otherwise None."""
    tag = short(symbol)
    # Size the position using the price distance corresponding to the -2% ROI stop.
    # RISK_PCT controls account-balance risk sizing; it does NOT define the ROI target.
    risk_dist = entry * (STOP_LOSS_ROI_PCT / LEVERAGE) / 100
    bal = get_balance(ex)
    qty = (bal * RISK_PCT / 100) / risk_dist
    qty = min(qty, bal * LEVERAGE * MAX_MARGIN_PER_TRADE / entry)   # margin cap
    cap = max_order_qty(ex, symbol)
    if cap and qty > cap:
        log.info(f"[{tag}] Size capped to exchange max order qty {cap} (wanted {qty:.2f})")
        qty = cap
    qty = float(ex.amount_to_precision(symbol, qty))
    limits = ex.market(symbol)["limits"]
    min_qty = (limits.get("amount") or {}).get("min") or 0
    min_cost = (limits.get("cost") or {}).get("min") or 0
    if qty <= 0 or qty < min_qty or qty * entry < min_cost:
        log.info(f"[{tag}] Skip: size too small (qty={qty}, min qty={min_qty}, min notional={min_cost})")
        return None

    t0 = int(time.time() * 1000) - 5000   # fills are looked up from here (includes the entry fee)
    order = None
    for _ in range(5):   # if Binance still says "quantity too large" (-4005), halve and retry
        try:
            order = ex.create_order(symbol, "market", side, qty)
            break
        except ccxt.BaseError as e:
            if "-4005" not in str(e):
                raise
            qty = float(ex.amount_to_precision(symbol, qty / 2))
            log.info(f"[{tag}] Quantity too large, retrying with {qty}")
            if qty <= 0 or qty < min_qty or qty * entry < min_cost:
                log.info(f"[{tag}] Skip: size too small after reducing")
                return None
    if order is None:
        return None
    fill = order.get("average") or order.get("price")
    if not fill:
        try:
            fill = ex.fetch_order(order["id"], symbol).get("average")
        except Exception:
            fill = None
    fill = float(fill or entry)

    # Convert ROI targets to price movement using configured leverage.
    # Gross ROI ~= price move % * leverage; fees/funding/slippage are excluded.
    sl_move_pct = STOP_LOSS_ROI_PCT / LEVERAGE
    tp_move_pct = TAKE_PROFIT_ROI_PCT / LEVERAGE
    if side == "buy":
        sl = fill * (1 - sl_move_pct / 100)
        tp = fill * (1 + tp_move_pct / 100)
    else:
        sl = fill * (1 + sl_move_pct / 100)
        tp = fill * (1 - tp_move_pct / 100)
    exit_side = "sell" if side == "buy" else "buy"

    try:
        ex.create_order(symbol, "STOP_MARKET", exit_side, qty, None,
                        {"stopPrice": float(ex.price_to_precision(symbol, sl)), "reduceOnly": True})
        ex.create_order(symbol, "TAKE_PROFIT_MARKET", exit_side, qty, None,
                        {"stopPrice": float(ex.price_to_precision(symbol, tp)), "reduceOnly": True})
    except Exception as e:
        emergency_close(ex, symbol, side, qty, f"could not place SL/TP: {e}")
        return None

    log.info(f"[{tag}] OPENED {side.upper()} qty={qty} fill={fill} "
             f"SL={sl:.8g} (-{STOP_LOSS_ROI_PCT:.1f}% ROI; {sl_move_pct:.3f}% price) "
             f"TP={tp:.8g} (+{TAKE_PROFIT_ROI_PCT:.1f}% ROI; {tp_move_pct:.3f}% price)")
    return {"t0": t0, "opened_ms": int(time.time() * 1000), "side": side, "fill": fill,
            "sl": float(sl), "tp": float(tp), "qty": qty, "reason": reason}


def main():
    ex = make_exchange()

    try:   # one-way position mode is required for reduceOnly orders
        if str(ex.fapiPrivateGetPositionSideDual().get("dualSidePosition")).lower() == "true":
            raise SystemExit("Your account is in Hedge Mode. Switch Futures to One-way Mode and restart.")
    except SystemExit:
        raise
    except Exception:
        pass

    for s in SYMBOLS:
        if s not in ex.markets:
            raise SystemExit(f"{s} not found on this exchange - check the symbol name.")
        try:
            ex.set_margin_mode("isolated", s)
        except Exception:
            pass
        try:
            ex.set_leverage(LEVERAGE, s)
        except Exception as e:
            raise SystemExit(
                f"[{short(s)}] Could not set leverage to {LEVERAGE}x. "
                f"Stopping to avoid incorrect ROI-based TP/SL prices: {e}"
            )

    # clean up leftover orders from earlier runs on pairs that have no open position
    held = symbols_with_position(ex)
    for s in SYMBOLS:
        if s not in held:
            try:
                ex.cancel_all_orders(s)
            except Exception:
                pass
    if held:
        log.info("Positions already open (not managed by this run): "
                 + ", ".join(short(s) for s in held))

    refresh_news()
    day = datetime.now(timezone.utc).date()
    day_start_bal = get_balance(ex)
    consec_losses = 0
    last_candle_ts = {s: None for s in SYMBOLS}
    active = {}   # symbol -> details of the trade this run opened
    trades = load_trade_log()   # every closed trade, kept across restarts
    if trades:
        log.info(f"Loaded {len(trades)} earlier trades from {TRADE_LOG_JSON}")

    log.info(f"Bot started | balance={day_start_bal:.2f} USDT | pairs: "
             f"{', '.join(short(s) for s in SYMBOLS)} | {TIMEFRAME} | max open trades={MAX_OPEN_TRADES}")

    while True:
        try:
            refresh_news()
            now_day = datetime.now(timezone.utc).date()
            if now_day != day:
                day, day_start_bal = now_day, get_balance(ex)
                log.info("New day - daily loss counter reset")

            open_syms = symbols_with_position(ex)

            # --- detect trades that just closed ---
            for s in list(active):
                if s not in open_syms:
                    ex.cancel_all_orders(s)
                    info = active.pop(s)
                    res = trade_summary(ex, s, info)
                    pnl = res["pnl"]
                    risk_usd = abs(info["fill"] - info["sl"]) * info["qty"]
                    if not res["ok"]:
                        result = "Closed (PnL unknown)"
                    else:
                        # Classify by realized net PnL, not proximity to trigger prices.
                        result = "Take profit" if pnl > 0 else "Stop loss"
                    trades.append({
                        "n": (trades[-1]["n"] + 1) if trades else 1,
                        "symbol": short(s), "side": info["side"],
                        "opened_ms": info["opened_ms"], "closed_ms": res["closed_ms"],
                        "entry": info["fill"], "exit": res["exit"], "sl": info["sl"], "tp": info["tp"],
                        "qty": info["qty"], "pnl": pnl,
                        "r": (pnl / risk_usd) if risk_usd > 0 else None,
                        "result": result, "reason": info["reason"],
                    })
                    save_trade_log(trades)
                    consec_losses = consec_losses + 1 if pnl < 0 else 0
                    log.info(f"[{short(s)}] CLOSED ({result}) | PnL={pnl:.2f} USDT | "
                             f"consecutive losses={consec_losses} | trades so far={len(trades)}")
                    if REPORT_EVERY_N_TRADES and len(trades) % REPORT_EVERY_N_TRADES == 0:
                        write_report(ex, trades)
                    if consec_losses >= MAX_CONSEC_LOSSES:
                        log.info("Max consecutive losses hit - stopping bot.")
                        return

            # --- scan every pair for new setups ---
            for s in SYMBOLS:
                if s in open_syms or s in active:
                    continue
                if len(open_syms) >= MAX_OPEN_TRADES:
                    break

                df = fetch_candles(ex, s)
                ts = int(df.iloc[-1]["ts"])
                if ts == last_candle_ts[s]:  # already evaluated this closed candle
                    continue
                last_candle_ts[s] = ts

                dd = (day_start_bal - get_balance(ex)) / day_start_bal * 100
                if dd >= MAX_DAILY_LOSS_PCT:
                    log.info(f"Daily loss limit hit ({dd:.2f}%) - no new trades today")
                    continue
                title = news_blocked()
                if title:
                    log.info(f"[{short(s)}] News window ({title}) - skipping")
                    continue

                sig, why = get_signal(df)
                if not sig:
                    log.info(f"[{short(s)}] No setup - {why}")
                    continue
                log.info(f"[{short(s)}] {why}")

                opened = place_trade(ex, s, *sig, reason=why)
                if opened:
                    active[s] = opened
                    open_syms.add(s)

            time.sleep(10)

        except ccxt.NetworkError as e:
            log.info(f"Network error, retrying: {e}")
            time.sleep(15)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except SystemExit:
            raise
        except Exception as e:
            log.info(f"Error: {e}")
            time.sleep(15)


if __name__ == "__main__":
    main()