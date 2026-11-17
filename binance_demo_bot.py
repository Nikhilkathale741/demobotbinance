"""
Binance DEMO USDT-M futures bot — Chan-inspired intraday mean reversion.
DEMO ONLY by default. Market entries and bot-managed market exits (no resting orders).

Strategy:
- On the last CLOSED 1-minute candle, fit a rolling linear trend to log price.
- Calculate the current log-price residual z-score versus the rolling residual distribution.
- Require the residual series to pass a rolling Augmented Dickey-Fuller stationarity check.
- Long when z-score crosses back above -ENTRY_Z after being <= -ENTRY_Z.
- Short when z-score crosses back below +ENTRY_Z after being >= +ENTRY_Z.
- Optional volatility-spike guard.
- Exit at equal-distance stop/target: 2% ROI on isolated margin at configured leverage.
- Position size targets 1% of current USDT futures equity at the stop, subject to exchange precision.
- Only BTC, ETH, SOL. One open position at a time to cap aggregate exposure.

Important:
This is an implementation inspired by quantitative mean-reversion concepts discussed by
Ernest P. Chan, not a reproduction of a proprietary strategy or a claim of profitability.
Bot-managed exits require this process and network connection to remain online.
"""

import os
import csv
import json
import time
import logging
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ============================ CONFIG ============================
API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")
MODE = "demo"  # "demo" or "testnet"; never change to live until separately reviewed.

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
]

TIMEFRAME = "1m"
CANDLE_LIMIT = 300
POLL_SECONDS = 2
MAX_OPEN_TRADES = 1  # scan all 3 symbols, but only hold one position at a time.

# Chan-inspired residual mean reversion
REGRESSION_WINDOW = 120
ZSCORE_WINDOW = 60
ENTRY_Z = 2.0
ADF_PVALUE_MAX = 0.05
ADF_CHECK_EVERY_CANDLES = 5
MAX_CANDLE_RANGE_ATR = 3.0
ATR_LEN = 14

# Risk / exits
RISK_PCT = 1.0                 # account equity risk target at stop
STOP_ROI_PCT = 2.0             # loss target as % of allocated initial margin
TAKE_PROFIT_ROI_PCT = 2.0      # equal to stop for 1:1 R:R
LEVERAGE = 5
MAX_DAILY_LOSS_PCT = 3.0
MAX_CONSEC_LOSSES = 3
MIN_BALANCE_USDT = 5.0

# Reports / files
REPORT_EVERY_N_TRADES = 5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_JSON = os.path.join(BASE_DIR, "trade_log.json")
TRADE_LOG_CSV = os.path.join(BASE_DIR, "trades.csv")
STATE_JSON = os.path.join(BASE_DIR, "active_trade_state.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("chan_mean_reversion_bot")


# --------------------------- exchange ---------------------------
def make_exchange():
    if not API_KEY or not SECRET:
        raise SystemExit(
            "Missing credentials. Set BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET_KEY."
        )
    ex = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    if MODE == "demo":
        if not hasattr(ex, "enable_demo_trading"):
            raise SystemExit("Installed ccxt does not support Binance Demo Trading. Upgrade ccxt.")
        ex.enable_demo_trading(True)
    elif MODE == "testnet":
        ex.set_sandbox_mode(True)
    else:
        raise SystemExit("MODE must be 'demo' or 'testnet'. This script intentionally has no live mode.")
    ex.load_markets()
    return ex


def get_balance(ex):
    bal = ex.fetch_balance()
    usdt = bal.get("USDT") or {}
    total = usdt.get("total")
    if total is None:
        raise RuntimeError("USDT futures wallet balance not found. Fund the Binance DEMO futures wallet.")
    return float(total)


def get_open_positions(ex):
    positions = []
    for p in ex.fetch_positions(SYMBOLS):
        contracts = float(p.get("contracts") or 0)
        if abs(contracts) > 0:
            positions.append({
                "symbol": p["symbol"],
                "contracts": abs(contracts),
                "side": (p.get("side") or "").lower(),
            })
    return positions


# --------------------------- indicators ---------------------------
def add_atr(df, length=ATR_LEN):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / length, adjust=False).mean()
    return df


def calculate_residual_zscore(close, regression_window=REGRESSION_WINDOW,
                              z_window=ZSCORE_WINDOW):
    """
    Rolling log-price linear regression residual.
    The z-score measures deviation from the local fitted trend, not raw price vs SMA.
    """
    log_price = np.log(np.asarray(close, dtype=float))
    if len(log_price) < regression_window + z_window + 2:
        raise ValueError("Not enough candles for regression and z-score windows.")

    residuals = np.full(len(log_price), np.nan)
    x = np.arange(regression_window, dtype=float)

    for end in range(regression_window, len(log_price) + 1):
        y = log_price[end - regression_window:end]
        slope, intercept = np.polyfit(x, y, 1)
        residuals[end - 1] = y[-1] - (intercept + slope * x[-1])

    residual_series = pd.Series(residuals)
    mean = residual_series.rolling(z_window, min_periods=z_window).mean()
    std = residual_series.rolling(z_window, min_periods=z_window).std(ddof=1)
    zscore = (residual_series - mean) / std.replace(0, np.nan)
    return residual_series, zscore


_adf_cache = {}


def stationarity_ok(residuals, symbol, candle_ts):
    """Cache the ADF result for each symbol/candle; reject non-stationary residuals."""
    cached = _adf_cache.get(symbol)
    if cached and candle_ts - cached["ts"] < ADF_CHECK_EVERY_CANDLES * 60_000:
        return cached["ok"], cached["pvalue"]

    values = pd.Series(residuals).dropna().to_numpy(dtype=float)
    if len(values) < 50 or np.std(values) == 0:
        ok, pvalue = False, 1.0
    else:
        try:
            pvalue = float(adfuller(values[-REGRESSION_WINDOW:], maxlag=1,
                                    regression="c", autolag=None)[1])
            ok = pvalue <= ADF_PVALUE_MAX
        except Exception as exc:
            log.warning("[%s] ADF test failed: %s", symbol, exc)
            ok, pvalue = False, 1.0

    _adf_cache[symbol] = {"ts": candle_ts, "ok": ok, "pvalue": pvalue}
    return ok, pvalue


def prepare_candles(raw):
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    # Ignore the currently forming candle; signals only use completed candles.
    df = df.iloc[:-1].copy().reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return add_atr(df)


def get_signal(df, symbol):
    required = REGRESSION_WINDOW + ZSCORE_WINDOW + ATR_LEN + 5
    if len(df) < required:
        return None, "not enough closed candles"

    residuals, zscores = calculate_residual_zscore(df["close"].to_numpy())
    z = zscores.iloc[-1]
    prev_z = zscores.iloc[-2]
    row = df.iloc[-1]

    if pd.isna(z) or pd.isna(prev_z):
        return None, "z-score unavailable"
    if row["atr"] <= 0:
        return None, "ATR unavailable"

    candle_range = float(row["high"] - row["low"])
    if candle_range > MAX_CANDLE_RANGE_ATR * float(row["atr"]):
        return None, "volatility spike candle; skipping"

    stationary, pvalue = stationarity_ok(residuals, symbol, int(row["ts"]))
    if not stationary:
        return None, f"residual series not stationary (ADF p={pvalue:.3f})"

    # Require a cross back inside the extreme band to avoid blindly catching a falling/rising knife.
    if prev_z <= -ENTRY_Z and z > -ENTRY_Z:
        return ("buy", float(row["close"]), float(z)), (
            f"Mean-reversion LONG: residual z-score crossed back above -{ENTRY_Z:.1f} "
            f"({prev_z:.2f} -> {z:.2f}); ADF p={pvalue:.3f}"
        )
    if prev_z >= ENTRY_Z and z < ENTRY_Z:
        return ("sell", float(row["close"]), float(z)), (
            f"Mean-reversion SHORT: residual z-score crossed back below +{ENTRY_Z:.1f} "
            f"({prev_z:.2f} -> {z:.2f}); ADF p={pvalue:.3f}"
        )

    return None, f"no z-score re-entry (z={z:.2f}, ADF p={pvalue:.3f})"


# --------------------------- trade persistence ---------------------------
def load_trade_log():
    try:
        with open(TRADE_LOG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_trade_log(trades):
    with open(TRADE_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)
    cols = ["n", "symbol", "side", "opened_utc", "closed_utc", "entry", "exit",
            "sl", "tp", "qty", "risk_usdt", "pnl_usdt", "r_multiple", "result", "reason"]
    with open(TRADE_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)


def save_active_state(active):
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(active, f, indent=2)


def load_active_state():
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fetch_fill_price(ex, symbol, order):
    average = order.get("average") or order.get("price")
    if average:
        return float(average)
    try:
        fetched = ex.fetch_order(order["id"], symbol)
        if fetched.get("average"):
            return float(fetched["average"])
    except Exception:
        pass
    # Do not silently pretend the candle close was the fill. Query the live ticker.
    return float(ex.fetch_ticker(symbol)["last"])


def get_trade_pnl(ex, symbol, opened_ms, entry_side):
    pnl = 0.0
    exit_qty = 0.0
    exit_value = 0.0
    closed_ms = int(time.time() * 1000)
    exit_side = "sell" if entry_side == "buy" else "buy"
    try:
        fills = ex.fetch_my_trades(symbol, since=opened_ms - 5000)
        for trade in fills:
            info = trade.get("info") or {}
            pnl += float(info.get("realizedPnl") or 0)
            fee = trade.get("fee") or {}
            if fee.get("currency") == "USDT":
                pnl -= float(fee.get("cost") or 0)
            if trade.get("side") == exit_side:
                amount = float(trade.get("amount") or 0)
                exit_qty += amount
                exit_value += amount * float(trade.get("price") or 0)
                closed_ms = max(closed_ms, int(trade.get("timestamp") or closed_ms))
    except Exception as exc:
        log.warning("[%s] Could not fully retrieve realized PnL: %s", symbol, exc)
        return None, None, closed_ms
    exit_price = exit_value / exit_qty if exit_qty else None
    return pnl, exit_price, closed_ms


# --------------------------- sizing and execution ---------------------------
def position_quantity(ex, symbol, reference_price, balance):
    """
    Risk budget = 1% equity.
    Stop price distance = reference price * (STOP_ROI / leverage).
    Quantity = risk budget / price distance.
    This makes quantity dynamic with both balance and stop distance.
    """
    stop_price_fraction = (STOP_ROI_PCT / 100.0) / LEVERAGE
    stop_distance = reference_price * stop_price_fraction
    risk_budget = balance * RISK_PCT / 100.0
    if stop_distance <= 0:
        return None, None

    qty = risk_budget / stop_distance
    qty = float(ex.amount_to_precision(symbol, qty))
    market = ex.market(symbol)
    limits = market.get("limits") or {}
    min_qty = float((limits.get("amount") or {}).get("min") or 0)
    min_cost = float((limits.get("cost") or {}).get("min") or 0)
    if qty <= 0 or qty < min_qty or qty * reference_price < min_cost:
        log.info("[%s] Skip: calculated quantity below exchange minimum (qty=%s)", symbol, qty)
        return None, None

    # Isolated margin estimate; don't open if required margin exceeds available equity.
    required_margin = qty * reference_price / LEVERAGE
    if required_margin > balance * 0.90:
        log.info("[%s] Skip: required isolated margin %.2f exceeds 90%% of wallet %.2f",
                 symbol, required_margin, balance)
        return None, None

    return qty, risk_budget


def place_market_entry(ex, symbol, side, reference_price, reason):
    balance = get_balance(ex)
    if balance < MIN_BALANCE_USDT:
        log.warning("Balance %.2f below minimum; no trade.", balance)
        return None

    qty, risk_budget = position_quantity(ex, symbol, reference_price, balance)
    if qty is None:
        return None

    log.info("[%s] Sending MARKET %s qty=%s (estimated risk %.4f USDT)",
             symbol, side.upper(), qty, risk_budget)
    order = ex.create_order(symbol, "market", side, qty)
    fill = fetch_fill_price(ex, symbol, order)

    stop_move = (STOP_ROI_PCT / 100.0) / LEVERAGE
    target_move = (TAKE_PROFIT_ROI_PCT / 100.0) / LEVERAGE
    if side == "buy":
        sl = fill * (1.0 - stop_move)
        tp = fill * (1.0 + target_move)
    else:
        sl = fill * (1.0 + stop_move)
        tp = fill * (1.0 - target_move)

    sl = float(ex.price_to_precision(symbol, sl))
    tp = float(ex.price_to_precision(symbol, tp))
    actual_risk = abs(fill - sl) * qty
    state = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry": fill,
        "sl": sl,
        "tp": tp,
        "risk_usdt": actual_risk,
        "opened_ms": int(time.time() * 1000),
        "opened_utc": utc_now(),
        "reason": reason,
        "entry_order_id": order.get("id"),
    }
    log.info("[%s] OPEN %s | fill=%s qty=%s SL=%s TP=%s | planned R:R=1:1",
             symbol, side.upper(), fill, qty, state["sl"], state["tp"])
    return state


def close_position_market(ex, state, exit_reason):
    symbol = state["symbol"]
    exit_side = "sell" if state["side"] == "buy" else "buy"
    qty = float(state["qty"])
    log.warning("[%s] MARKET EXIT %s: %s", symbol, exit_side.upper(), exit_reason)
    order = ex.create_order(symbol, "market", exit_side, qty, None, {"reduceOnly": True})
    exit_price = fetch_fill_price(ex, symbol, order)
    return exit_price, order


# --------------------------- reporting ---------------------------
def trade_stats(trades):
    count = len(trades)
    wins = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) > 0)
    gross_profit = sum(float(t.get("pnl_usdt") or 0) for t in trades
                       if float(t.get("pnl_usdt") or 0) > 0)
    gross_loss = abs(sum(float(t.get("pnl_usdt") or 0) for t in trades
                         if float(t.get("pnl_usdt") or 0) < 0))
    return {
        "trades": count,
        "wins": wins,
        "losses": count - wins,
        "win_rate_pct": (wins / count * 100) if count else 0.0,
        "net_pnl_usdt": sum(float(t.get("pnl_usdt") or 0) for t in trades),
        "profit_factor": (gross_profit / gross_loss) if gross_loss else None,
    }


def write_pdf_report(trades, balance):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        batch = trades[-REPORT_EVERY_N_TRADES:]
        path = os.path.join(BASE_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Chan-Inspired Crypto Mean-Reversion Bot — DEMO Report", styles["Title"]),
            Paragraph(f"Generated {utc_now()} | Balance {balance:.2f} USDT", styles["Normal"]),
            Spacer(1, 6),
        ]
        stats = trade_stats(trades)
        story.append(Paragraph(
            f"Trades: {stats['trades']} | Win rate: {stats['win_rate_pct']:.1f}% | "
            f"Net PnL: {stats['net_pnl_usdt']:.4f} USDT | "
            f"Profit factor: {stats['profit_factor'] if stats['profit_factor'] is not None else 'n/a'}",
            styles["Normal"]))
        rows = [["#", "Symbol", "Side", "Entry", "Exit", "SL", "TP", "Qty", "PnL USDT", "Result"]]
        for trade in batch:
            rows.append([
                str(trade.get("n")), trade.get("symbol"), trade.get("side"),
                str(trade.get("entry")), str(trade.get("exit")),
                str(trade.get("sl")), str(trade.get("tp")),
                str(trade.get("qty")), f"{float(trade.get('pnl_usdt') or 0):.4f}",
                trade.get("result"),
            ])
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(table)
        doc = SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=10*mm,
                                rightMargin=10*mm, topMargin=10*mm, bottomMargin=10*mm)
        doc.build(story)
        log.info("PDF report saved: %s", path)
    except ImportError:
        log.info("Optional PDF report skipped; install reportlab to enable it.")
    except Exception as exc:
        log.exception("PDF report failed: %s", exc)


# --------------------------- main loop ---------------------------
def main():
    ex = make_exchange()

    # Require one-way mode because reduceOnly market exits are used.
    try:
        dual = ex.fapiPrivateGetPositionSideDual()
        if str(dual.get("dualSidePosition")).lower() == "true":
            raise SystemExit("Switch Binance Futures to One-way Mode before running this bot.")
    except SystemExit:
        raise
    except Exception as exc:
        log.warning("Could not verify position mode; check that Binance is in One-way Mode: %s", exc)

    for symbol in SYMBOLS:
        if symbol not in ex.markets:
            raise SystemExit(f"Symbol unavailable in this Binance market list: {symbol}")
        try:
            ex.set_margin_mode("isolated", symbol)
        except Exception as exc:
            log.info("[%s] Isolated margin setting not changed: %s", symbol, exc)
        ex.set_leverage(LEVERAGE, symbol)

    # Never take over an existing position after restart: its SL/TP state may be unknown.
    existing = get_open_positions(ex)
    if existing:
        raise SystemExit(
            "Existing futures position(s) detected. Bot will not take over unmanaged positions: "
            + ", ".join(f"{p['symbol']} ({p['side']} {p['contracts']})" for p in existing)
            + ". Close/manage them manually before starting."
        )

    # This bot creates no resting orders. If old open orders exist, stop rather than cancel user orders.
    for symbol in SYMBOLS:
        open_orders = ex.fetch_open_orders(symbol)
        if open_orders:
            raise SystemExit(
                f"Open order(s) already exist for {symbol}. Cancel/review them manually before starting."
            )

    active = load_active_state()
    if active:
        raise SystemExit(
            f"Stale {STATE_JSON} found. Review it and the exchange account; archive/delete it only "
            "after confirming there is no open position."
        )

    trades = load_trade_log()
    day = datetime.now(timezone.utc).date()
    day_start_balance = get_balance(ex)
    consecutive_losses = 0
    last_candle_ts = {symbol: None for symbol in SYMBOLS}

    log.info(
        "STARTED DEMO | balance=%.2f USDT | symbols=%s | timeframe=%s | "
        "risk=%.2f%% | stop=%.2f%% ROI | target=%.2f%% ROI | leverage=%sx",
        day_start_balance, ", ".join(SYMBOLS), TIMEFRAME, RISK_PCT,
        STOP_ROI_PCT, TAKE_PROFIT_ROI_PCT, LEVERAGE
    )

    while True:
        try:
            current_day = datetime.now(timezone.utc).date()
            if current_day != day:
                day = current_day
                day_start_balance = get_balance(ex)
                consecutive_losses = 0
                log.info("UTC day reset; starting balance %.2f USDT", day_start_balance)

            # Manage active position first. No exchange-side stop/TP orders are placed.
            if active:
                symbol = active["symbol"]
                ticker = ex.fetch_ticker(symbol)
                last = float(ticker.get("last") or ticker.get("close"))
                state = active
                reason = None

                if state["side"] == "buy":
                    if last <= state["sl"]:
                        reason = "STOP_LOSS"
                    elif last >= state["tp"]:
                        reason = "TAKE_PROFIT"
                else:
                    if last >= state["sl"]:
                        reason = "STOP_LOSS"
                    elif last <= state["tp"]:
                        reason = "TAKE_PROFIT"

                positions = get_open_positions(ex)
                still_open = any(p["symbol"] == symbol for p in positions)

                if reason and still_open:
                    exit_price, exit_order = close_position_market(ex, state, reason)
                    pnl, avg_exit, closed_ms = get_trade_pnl(
                        ex, symbol, state["opened_ms"], state["side"]
                    )
                    if pnl is None:
                        direction = 1 if state["side"] == "buy" else -1
                        pnl = (exit_price - state["entry"]) * state["qty"] * direction
                        avg_exit = exit_price

                    trade = {
                        "n": (int(trades[-1].get("n", 0)) + 1) if trades else 1,
                        "symbol": symbol,
                        "side": state["side"],
                        "opened_utc": state["opened_utc"],
                        "closed_utc": utc_now(),
                        "entry": state["entry"],
                        "exit": avg_exit or exit_price,
                        "sl": state["sl"],
                        "tp": state["tp"],
                        "qty": state["qty"],
                        "risk_usdt": state["risk_usdt"],
                        "pnl_usdt": float(pnl),
                        "r_multiple": (float(pnl) / state["risk_usdt"]) if state["risk_usdt"] else None,
                        "result": reason,
                        "reason": state["reason"],
                    }
                    trades.append(trade)
                    save_trade_log(trades)
                    active = {}
                    save_active_state(active)
                    consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
                    stats = trade_stats(trades)
                    log.info(
                        "CLOSED %s | PnL=%.4f USDT | R=%.2f | win rate=%.1f%% | PF=%s",
                        symbol, pnl, trade["r_multiple"] or 0,
                        stats["win_rate_pct"],
                        f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "n/a",
                    )
                    if REPORT_EVERY_N_TRADES and len(trades) % REPORT_EVERY_N_TRADES == 0:
                        write_pdf_report(trades, get_balance(ex))
                    if consecutive_losses >= MAX_CONSEC_LOSSES:
                        raise SystemExit("Maximum consecutive losses reached; stopping bot.")
                elif not still_open:
                    log.warning(
                        "[%s] Position disappeared from exchange without bot exit. "
                        "Check Binance order history and reconcile PnL manually.",
                        symbol
                    )
                    active = {}
                    save_active_state(active)

            # If flat, evaluate each symbol once per newly closed candle.
            if not active:
                balance = get_balance(ex)
                daily_loss_pct = ((day_start_balance - balance) / day_start_balance * 100
                                  if day_start_balance > 0 else 0)
                if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                    log.warning("Daily loss limit reached (%.2f%%); no new entries.", daily_loss_pct)
                    time.sleep(POLL_SECONDS)
                    continue

                for symbol in SYMBOLS:
                    raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
                    df = prepare_candles(raw)
                    if df.empty:
                        continue
                    candle_ts = int(df.iloc[-1]["ts"])
                    if candle_ts == last_candle_ts[symbol]:
                        continue
                    last_candle_ts[symbol] = candle_ts

                    signal, explanation = get_signal(df, symbol)
                    if not signal:
                        log.info("[%s] NO TRADE | %s", symbol, explanation)
                        continue

                    side, reference_price, zscore = signal
                    log.info("[%s] SIGNAL | %s | reference=%.8g | z=%.2f",
                             symbol, side.upper(), reference_price, zscore)

                    if get_open_positions(ex):
                        log.warning("Exchange position appeared before entry; skipping.")
                        break

                    active = place_market_entry(ex, symbol, side, reference_price, explanation)
                    if active:
                        save_active_state(active)
                        break

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log.warning("Stopped by user. Any open position remains on exchange; manage it immediately.")
            break
        except ccxt.NetworkError as exc:
            log.error("Network error: %s. Bot-managed exits may be delayed; check exchange manually.", exc)
            time.sleep(5)
        except SystemExit:
            raise
        except Exception as exc:
            log.exception("Loop error: %s", exc)
            time.sleep(5)


if __name__ == "__main__":
    main()
