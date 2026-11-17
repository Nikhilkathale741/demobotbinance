"""
Binance DEMO USDT-M Futures bot — Chan-inspired intraday mean reversion.
DEMO ONLY. This script intentionally has no live-trading mode.

Risk controls added:
- IST trading day (Asia/Kolkata) reset.
- Maximum 3 entries per IST day.
- Stop new entries after 2 losing trades in the IST day (not necessarily consecutive).
- Daily drawdown stop at 2.5% from start-of-day futures equity.
- Overall drawdown halt at 8% from persisted equity high-water mark.
- Close an active bot-managed position with a reduce-only market order on either
  drawdown breach; overall breach permanently halts new trading until manual review.
- 15-minute per-symbol cooldown after a losing trade.
- Persisted risk state, daily counters, cooldowns, high-water mark and processed candles.
- Daily Telegram text + PDF report at 23:59 IST, with separate delivery flags.

Important safety limitations:
- Entries/exits are market orders.
- Stop-loss/take-profit and drawdown exits are bot-managed, not exchange-side
  protective orders. The process and network must remain available.
- If an exchange position cannot be reconciled with the saved active state, the
  bot halts rather than guessing or taking over an unknown position.
- DEMO mode only. Validate all behavior in Binance Demo before any other use.
"""

import os
import csv
import json
import time
import uuid
import logging
import mimetypes
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import ccxt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ============================ CONFIG ============================
API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")
MODE = "demo"  # Intentionally fixed to Binance Demo Trading; no live mode exists.

CANDIDATE_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "AVAX/USDT:USDT",
    "ADA/USDT:USDT",
    "ZEC/USDT:USDT",
    "HYPE/USDT:USDT",
    "NEAR/USDT:USDT",
]
SYMBOLS = []  # populated by validate_symbols()

TIMEFRAME = "1m"
CANDLE_LIMIT = 300
POLL_SECONDS = 2
MAX_OPEN_TRADES = 1

# Chan-inspired residual mean reversion (unchanged)
REGRESSION_WINDOW = 120
ZSCORE_WINDOW = 60
ENTRY_Z = 2.0
ADF_PVALUE_MAX = 0.05
ADF_CHECK_EVERY_CANDLES = 5
MAX_CANDLE_RANGE_ATR = 3.0
ATR_LEN = 14

# Position risk / exits (existing settings retained)
RISK_PCT = 1.0
STOP_ROI_PCT = 2.0
TAKE_PROFIT_ROI_PCT = 2.0
LEVERAGE = 5
MIN_BALANCE_USDT = 5.0

# Confirmed daily / account-level controls
try:
    IST = ZoneInfo("Asia/Kolkata")
except ZoneInfoNotFoundError:  # Windows without the optional tzdata package
    IST = timezone(timedelta(hours=5, minutes=30))  # India has no daylight-saving changes
MAX_TRADES_PER_DAY = 3
MAX_LOSING_TRADES_PER_DAY = 2
DAILY_DRAWDOWN_LIMIT_PCT = 2.5
OVERALL_DRAWDOWN_LIMIT_PCT = 8.0
LOSS_COOLDOWN_MINUTES = 15
DAILY_REPORT_HOUR = 23
DAILY_REPORT_MINUTE = 59

# Reports / persistence
REPORT_EVERY_N_TRADES = 5  # retain the existing periodic PDF report as well
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_JSON = os.path.join(BASE_DIR, "trade_log.json")
TRADE_LOG_CSV = os.path.join(BASE_DIR, "trades.csv")
ACTIVE_STATE_JSON = os.path.join(BASE_DIR, "active_trade_state.json")
RISK_STATE_JSON = os.path.join(BASE_DIR, "risk_state.json")
DAILY_REPORT_DIR = os.path.join(BASE_DIR, "daily_reports")

# Telegram credentials must be configured in local environment variables.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("chan_mean_reversion_bot")
_EQUITY_FALLBACK_WARNED = False


# --------------------------- time / persistence ---------------------------
def now_ist():
    return datetime.now(IST)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ist_now_iso():
    return now_ist().isoformat()


def ist_day_key(value=None):
    value = value or now_ist()
    if value.tzinfo is None:
        value = value.replace(tzinfo=IST)
    return value.astimezone(IST).date().isoformat()


def parse_iso(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=IST)
    except (TypeError, ValueError):
        return None


def atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def save_active_state(active):
    atomic_write_json(ACTIVE_STATE_JSON, active or {})


def load_active_state():
    data = load_json(ACTIVE_STATE_JSON, {})
    return data if isinstance(data, dict) else {}


def save_risk_state(risk_state):
    atomic_write_json(RISK_STATE_JSON, risk_state)


def load_risk_state():
    data = load_json(RISK_STATE_JSON, {})
    return data if isinstance(data, dict) else {}


# --------------------------- exchange ---------------------------
def make_exchange():
    if not API_KEY or not SECRET:
        raise SystemExit(
            "Missing Binance Demo credentials. Set BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET_KEY."
        )
    ex = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    if not hasattr(ex, "enable_demo_trading"):
        raise SystemExit("Installed ccxt does not support Binance Demo Trading. Upgrade ccxt.")
    ex.enable_demo_trading(True)
    ex.load_markets()
    return ex


def validate_symbols(ex):
    supported, skipped = [], []
    for symbol in CANDIDATE_SYMBOLS:
        market = ex.markets.get(symbol)
        if not market:
            skipped.append((symbol, "not listed by Binance Demo market list"))
            continue
        if market.get("active") is False:
            skipped.append((symbol, "market inactive"))
            continue
        if not (market.get("swap") and market.get("linear") and market.get("quote") == "USDT"):
            skipped.append((symbol, "not an active USDT-margined linear perpetual"))
            continue
        supported.append(symbol)
    for symbol, reason in skipped:
        log.warning("[%s] Not scanning: %s", symbol, reason)
    if not supported:
        raise SystemExit("None of the 12 configured symbols are available as active USDT perpetuals.")
    log.info("Validated %d/%d symbols: %s", len(supported), len(CANDIDATE_SYMBOLS), ", ".join(supported))
    return supported


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


def get_equity(ex):
    """Return futures account equity (wallet balance + unrealized PnL where available)."""
    global _EQUITY_FALLBACK_WARNED
    balance = ex.fetch_balance()
    info = balance.get("info") or {}
    # Binance USD-M account response exposes totalMarginBalance, which includes unrealized PnL.
    for key in ("totalMarginBalance", "totalWalletBalance"):
        value = info.get(key)
        if value not in (None, ""):
            equity = float(value)
            if key == "totalWalletBalance" and info.get("totalUnrealizedProfit") not in (None, ""):
                equity += float(info["totalUnrealizedProfit"])
            if np.isfinite(equity):
                return equity
    usdt = balance.get("USDT") or {}
    total = usdt.get("total")
    if total is None:
        raise RuntimeError("Could not read USDT futures equity from Binance balance response.")
    if not _EQUITY_FALLBACK_WARNED:
        log.warning("Binance raw totalMarginBalance unavailable; using CCXT USDT total as equity fallback.")
        _EQUITY_FALLBACK_WARNED = True
    equity = float(total)
    if not np.isfinite(equity):
        raise RuntimeError("Binance returned non-finite USDT equity.")
    return equity


# --------------------------- indicators ---------------------------
def add_atr(df, length=ATR_LEN):
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1 / length, adjust=False).mean()
    return df


def calculate_residual_zscore(close, regression_window=REGRESSION_WINDOW, z_window=ZSCORE_WINDOW):
    """Rolling log-price regression residual and its rolling z-score."""
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
    return residual_series, (residual_series - mean) / std.replace(0, np.nan)


_adf_cache = {}


def stationarity_ok(residuals, symbol, candle_ts):
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
    df = df.iloc[:-1].copy().reset_index(drop=True)  # exclude forming candle
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = df[column].astype(float)
    return add_atr(df)


def get_signal(df, symbol):
    required = REGRESSION_WINDOW + ZSCORE_WINDOW + ATR_LEN + 5
    if len(df) < required:
        return None, "not enough closed candles"
    residuals, zscores = calculate_residual_zscore(df["close"].to_numpy())
    z, previous_z = zscores.iloc[-1], zscores.iloc[-2]
    row = df.iloc[-1]
    if pd.isna(z) or pd.isna(previous_z):
        return None, "z-score unavailable"
    if row["atr"] <= 0:
        return None, "ATR unavailable"
    candle_range = float(row["high"] - row["low"])
    if candle_range > MAX_CANDLE_RANGE_ATR * float(row["atr"]):
        return None, "volatility spike candle; skipping"
    stationary, pvalue = stationarity_ok(residuals, symbol, int(row["ts"]))
    if not stationary:
        return None, f"residual series not stationary (ADF p={pvalue:.3f})"
    if previous_z <= -ENTRY_Z and z > -ENTRY_Z:
        return ("buy", float(row["close"]), float(z)), (
            f"Mean-reversion LONG: z crossed above -{ENTRY_Z:.1f} ({previous_z:.2f} -> {z:.2f}); ADF p={pvalue:.3f}"
        )
    if previous_z >= ENTRY_Z and z < ENTRY_Z:
        return ("sell", float(row["close"]), float(z)), (
            f"Mean-reversion SHORT: z crossed below +{ENTRY_Z:.1f} ({previous_z:.2f} -> {z:.2f}); ADF p={pvalue:.3f}"
        )
    return None, f"no z-score re-entry (z={z:.2f}, ADF p={pvalue:.3f})"


# --------------------------- trade log / fills ---------------------------
def load_trade_log():
    data = load_json(TRADE_LOG_JSON, [])
    return data if isinstance(data, list) else []


def save_trade_log(trades):
    atomic_write_json(TRADE_LOG_JSON, trades)
    columns = [
        "n", "symbol", "side", "opened_utc", "opened_ist", "closed_utc", "closed_ist",
        "entry", "exit", "sl", "tp", "qty", "risk_usdt", "pnl_usdt", "pnl_accuracy", "risk_loss",
        "entry_slippage_pct", "exit_slippage_pct",
        "r_multiple", "result", "reason", "entry_order_id", "exit_order_id",
    ]
    tmp = TRADE_LOG_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, TRADE_LOG_CSV)


def fetch_fill_price(ex, symbol, order):
    # For a market order, use actual average/cost/filled or matched user trades;
    # never substitute the requested price or the latest ticker as a fake fill.
    average = order.get("average")
    if average:
        return float(average)
    order_id = order.get("id")
    if order_id:
        try:
            fetched = ex.fetch_order(order_id, symbol)
            if fetched.get("average"):
                return float(fetched["average"])
            if fetched.get("filled") and fetched.get("cost"):
                return float(fetched["cost"]) / float(fetched["filled"])
        except Exception as exc:
            log.warning("[%s] Could not retrieve order average fill: %s", symbol, exc)
        try:
            since = int(order.get("timestamp") or (time.time() * 1000)) - 10_000
            fills = ex.fetch_my_trades(symbol, since=since)
            matching = [t for t in fills if str(t.get("order")) == str(order_id)]
            qty = sum(float(t.get("amount") or 0) for t in matching)
            value = sum(float(t.get("amount") or 0) * float(t.get("price") or 0) for t in matching)
            if qty > 0:
                return value / qty
        except Exception as exc:
            log.warning("[%s] Could not calculate average from fills: %s", symbol, exc)
    raise RuntimeError(f"Could not verify average fill price for {symbol}; reconcile order {order_id} manually.")


def get_trade_pnl(ex, symbol, opened_ms, entry_side):
    realized_pnl = 0.0
    exit_qty = 0.0
    exit_value = 0.0
    closed_ms = int(time.time() * 1000)
    exit_side = "sell" if entry_side == "buy" else "buy"
    fills = ex.fetch_my_trades(symbol, since=opened_ms - 5000)
    if not fills:
        raise RuntimeError("No user fills returned for completed trade.")
    for trade in fills:
        info = trade.get("info") or {}
        realized_pnl += float(info.get("realizedPnl") or 0)
        fee = trade.get("fee") or {}
        if fee.get("currency") == "USDT":
            realized_pnl -= float(fee.get("cost") or 0)
        if trade.get("side") == exit_side:
            amount = float(trade.get("amount") or 0)
            exit_qty += amount
            exit_value += amount * float(trade.get("price") or 0)
            closed_ms = max(closed_ms, int(trade.get("timestamp") or closed_ms))
    if exit_qty <= 0:
        raise RuntimeError("No exit fills returned for completed trade.")
    return realized_pnl, exit_value / exit_qty, closed_ms


# --------------------------- sizing / orders ---------------------------
def position_quantity(ex, symbol, reference_price, equity):
    stop_fraction = (STOP_ROI_PCT / 100.0) / LEVERAGE
    stop_distance = reference_price * stop_fraction
    risk_budget = equity * RISK_PCT / 100.0
    if stop_distance <= 0:
        return None, None
    quantity = float(ex.amount_to_precision(symbol, risk_budget / stop_distance))
    market = ex.market(symbol)
    limits = market.get("limits") or {}
    min_qty = float((limits.get("amount") or {}).get("min") or 0)
    min_cost = float((limits.get("cost") or {}).get("min") or 0)
    if quantity <= 0 or quantity < min_qty or quantity * reference_price < min_cost:
        log.info("[%s] Skip: calculated quantity below exchange minimum (qty=%s)", symbol, quantity)
        return None, None
    required_margin = quantity * reference_price / LEVERAGE
    if required_margin > equity * 0.90:
        log.info("[%s] Skip: required isolated margin %.2f exceeds 90%% of equity %.2f",
                 symbol, required_margin, equity)
        return None, None
    return quantity, risk_budget


def place_market_entry(ex, symbol, side, reference_price, reason):
    equity = get_equity(ex)
    if equity < MIN_BALANCE_USDT:
        log.warning("Equity %.2f below minimum; no trade.", equity)
        return None
    quantity, risk_budget = position_quantity(ex, symbol, reference_price, equity)
    if quantity is None:
        return None
    try:
        pre_order_ticker = ex.fetch_ticker(symbol)
        pre_order_price = float(pre_order_ticker.get("last") or pre_order_ticker.get("close"))
        if pre_order_price <= 0:
            pre_order_price = None
    except Exception as exc:
        pre_order_price = None
        log.warning("[%s] Could not capture pre-entry quote for slippage: %s", symbol, exc)
    log.info("[%s] Sending DEMO MARKET %s qty=%s (estimated risk %.4f USDT)",
             symbol, side.upper(), quantity, risk_budget)
    order = ex.create_order(symbol, "market", side, quantity)
    fill = fetch_fill_price(ex, symbol, order)
    stop_move = (STOP_ROI_PCT / 100.0) / LEVERAGE
    target_move = (TAKE_PROFIT_ROI_PCT / 100.0) / LEVERAGE
    if side == "buy":
        stop_loss, take_profit = fill * (1.0 - stop_move), fill * (1.0 + target_move)
    else:
        stop_loss, take_profit = fill * (1.0 + stop_move), fill * (1.0 - target_move)
    stop_loss = float(ex.price_to_precision(symbol, stop_loss))
    take_profit = float(ex.price_to_precision(symbol, take_profit))
    entry_slippage_pct = (((fill - pre_order_price) / pre_order_price) * 100.0
                          * (1.0 if side == "buy" else -1.0)) if pre_order_price else None
    state = {
        "symbol": symbol,
        "side": side,
        "qty": quantity,
        "entry": fill,
        "signal_reference_price": float(reference_price),
        "pre_order_price": pre_order_price,
        "entry_slippage_pct": entry_slippage_pct,
        "sl": stop_loss,
        "tp": take_profit,
        "risk_usdt": abs(fill - stop_loss) * quantity,
        "opened_ms": int(order.get("timestamp") or time.time() * 1000),
        "opened_utc": utc_now(),
        "opened_ist": ist_now_iso(),
        "reason": reason,
        "entry_order_id": order.get("id"),
    }
    log.info("[%s] OPEN %s | fill=%s qty=%s SL=%s TP=%s | planned R:R=1:1",
             symbol, side.upper(), fill, quantity, stop_loss, take_profit)
    return state


def close_position_market(ex, state, exit_reason):
    symbol = state["symbol"]
    exit_side = "sell" if state["side"] == "buy" else "buy"
    quantity = float(state["qty"])
    try:
        ticker = ex.fetch_ticker(symbol)
        reference_exit = float(ticker.get("last") or ticker.get("close"))
    except Exception as exc:
        reference_exit = None
        log.warning("[%s] Could not capture pre-exit reference price for slippage: %s", symbol, exc)
    log.warning("[%s] DEMO MARKET EXIT %s: %s", symbol, exit_side.upper(), exit_reason)
    order = ex.create_order(symbol, "market", exit_side, quantity, None, {"reduceOnly": True})
    exit_price = fetch_fill_price(ex, symbol, order)
    return exit_price, order, reference_exit


# --------------------------- risk state ---------------------------
def reconstruct_day_counters(trades, day_key):
    entries, losses = 0, 0
    for trade in trades:
        opened = parse_iso(trade.get("opened_ist") or trade.get("opened_utc"))
        closed = parse_iso(trade.get("closed_ist") or trade.get("closed_utc"))
        if opened and ist_day_key(opened) == day_key:
            entries += 1
        if closed and ist_day_key(closed) == day_key and (trade.get("risk_loss") is True or float(trade.get("pnl_usdt") or 0) < 0):
            losses += 1
    return entries, losses


def initialize_risk_state(equity, trades):
    today = ist_day_key()
    entries, losses = reconstruct_day_counters(trades, today)
    state = {
        "schema_version": 2,
        "day_ist": today,
        "day_start_equity": equity,
        "day_end_equity": equity,
        "daily_trades": entries,
        "daily_losses": losses,
        "daily_halted": entries >= MAX_TRADES_PER_DAY or losses >= MAX_LOSING_TRADES_PER_DAY,
        "daily_halt_reason": "RECONSTRUCTED_DAILY_LIMIT" if entries >= MAX_TRADES_PER_DAY or losses >= MAX_LOSING_TRADES_PER_DAY else "",
        "daily_drawdown_breached": False,
        "cooldowns": {},
        "overall_peak_equity": equity,
        "overall_halted": False,
        "overall_halt_reason": "",
        "last_equity": equity,
        "last_equity_ist": ist_now_iso(),
        "daily_drawdown_pct": 0.0,
        "max_daily_drawdown_pct": 0.0,
        "overall_drawdown_pct": 0.0,
        "max_overall_drawdown_pct": 0.0,
        "last_candle_ts": {},
        "reports": {},
    }
    log.warning("Created new risk_state.json. Overall high-water mark starts at current equity %.4f USDT; prior peak cannot be reconstructed.", equity)
    return state


def reset_daily_state(risk_state, equity, new_day):
    risk_state.update({
        "day_ist": new_day,
        "day_start_equity": equity,
        "day_end_equity": equity,
        "daily_trades": 0,
        "daily_losses": 0,
        "daily_halted": False,
        "daily_halt_reason": "",
        "daily_drawdown_breached": False,
        "daily_drawdown_pct": 0.0,
        "max_daily_drawdown_pct": 0.0,
    })
    # Cooldowns and the account high-water mark carry across IST midnight.
    return risk_state


def update_equity_metrics(risk_state, equity):
    day_start = float(risk_state.get("day_start_equity") or equity)
    peak = max(float(risk_state.get("overall_peak_equity") or equity), equity)
    risk_state["overall_peak_equity"] = peak
    risk_state["last_equity"] = equity
    risk_state["last_equity_ist"] = ist_now_iso()
    risk_state["day_end_equity"] = equity
    risk_state["daily_drawdown_pct"] = max(0.0, (day_start - equity) / day_start * 100.0) if day_start > 0 else 0.0
    risk_state["max_daily_drawdown_pct"] = max(
        float(risk_state.get("max_daily_drawdown_pct") or 0), risk_state["daily_drawdown_pct"]
    )
    risk_state["overall_drawdown_pct"] = max(0.0, (peak - equity) / peak * 100.0) if peak > 0 else 0.0
    risk_state["max_overall_drawdown_pct"] = max(
        float(risk_state.get("max_overall_drawdown_pct") or 0), risk_state["overall_drawdown_pct"]
    )
    if risk_state["daily_drawdown_pct"] >= DAILY_DRAWDOWN_LIMIT_PCT:
        if not risk_state.get("daily_drawdown_breached"):
            log.error("DAILY DRAWDOWN LIMIT reached: %.3f%% >= %.3f%%", risk_state["daily_drawdown_pct"], DAILY_DRAWDOWN_LIMIT_PCT)
        risk_state["daily_drawdown_breached"] = True
        risk_state["daily_halted"] = True
        # Give a drawdown breach priority over the ordinary trade-count halt reason.
        risk_state["daily_halt_reason"] = "DAILY_DRAWDOWN_LIMIT"
    if risk_state["overall_drawdown_pct"] >= OVERALL_DRAWDOWN_LIMIT_PCT and not risk_state.get("overall_halted"):
        risk_state["overall_halted"] = True
        risk_state["overall_halt_reason"] = "OVERALL_DRAWDOWN_LIMIT"
        log.critical("OVERALL DRAWDOWN LIMIT reached: %.3f%% >= %.3f%%. Permanent halt set.", risk_state["overall_drawdown_pct"], OVERALL_DRAWDOWN_LIMIT_PCT)
    save_risk_state(risk_state)
    return risk_state


def daily_entry_block_reason(risk_state, symbol=None):
    if risk_state.get("overall_halted"):
        return "OVERALL_DRAWDOWN_HALT"
    if risk_state.get("daily_halted"):
        return risk_state.get("daily_halt_reason") or "DAILY_HALT"
    if int(risk_state.get("daily_trades", 0)) >= MAX_TRADES_PER_DAY:
        risk_state["daily_halted"] = True
        risk_state["daily_halt_reason"] = "MAX_TRADES_REACHED"
        save_risk_state(risk_state)
        return "MAX_TRADES_REACHED"
    if int(risk_state.get("daily_losses", 0)) >= MAX_LOSING_TRADES_PER_DAY:
        risk_state["daily_halted"] = True
        risk_state["daily_halt_reason"] = "TWO_LOSSES_REACHED"
        save_risk_state(risk_state)
        return "TWO_LOSSES_REACHED"
    if symbol:
        expiry = parse_iso((risk_state.get("cooldowns") or {}).get(symbol))
        if expiry and now_ist() < expiry:
            return f"SYMBOL_COOLDOWN_UNTIL_{expiry.strftime('%H:%M:%S')}_IST"
    return ""


def record_trade_entry(risk_state, active):
    # Persist the entry counter before the active state. If the process crashes in
    # between, startup detects the exchange position as unmanaged and halts safely,
    # while the daily trade count is not silently lost.
    risk_state["daily_trades"] = int(risk_state.get("daily_trades", 0)) + 1
    active["entry_day_ist"] = ist_day_key()
    if risk_state["daily_trades"] >= MAX_TRADES_PER_DAY:
        risk_state["daily_halted"] = True
        risk_state["daily_halt_reason"] = "MAX_TRADES_REACHED"
    save_risk_state(risk_state)
    save_active_state(active)


def close_and_record_trade(ex, active, exit_reason, trades, risk_state):
    """Close one managed position, reconcile realized PnL, then persist its completed trade."""
    exit_price, exit_order, reference_exit = close_position_market(ex, active, exit_reason)
    pnl_accuracy = "exchange_realized_pnl_after_usdt_fees"
    try:
        pnl, average_exit, closed_ms = get_trade_pnl(
            ex, active["symbol"], int(active["opened_ms"]), active["side"]
        )
    except Exception as exc:
        direction = 1 if active["side"] == "buy" else -1
        pnl = (exit_price - float(active["entry"])) * float(active["qty"]) * direction
        average_exit = exit_price
        closed_ms = int(time.time() * 1000)
        pnl_accuracy = "estimated_gross_before_fees"
        log.error("[%s] Could not retrieve exchange realized PnL/fees; fallback PnL excludes fees: %s",
                  active["symbol"], exc)

    closed_dt = datetime.fromtimestamp(closed_ms / 1000, tz=timezone.utc).astimezone(IST)
    is_risk_loss = float(pnl) < 0 or pnl_accuracy != "exchange_realized_pnl_after_usdt_fees"
    exit_side = "sell" if active["side"] == "buy" else "buy"
    exit_slippage_pct = None
    if reference_exit and reference_exit > 0:
        exit_slippage_pct = (((float(average_exit or exit_price) - reference_exit) / reference_exit) * 100.0
                             * (-1.0 if exit_side == "sell" else 1.0))
    trade = {
        "n": (int(trades[-1].get("n", 0)) + 1) if trades else 1,
        "symbol": active["symbol"],
        "side": active["side"],
        "opened_utc": active.get("opened_utc"),
        "opened_ist": active.get("opened_ist"),
        "closed_utc": datetime.fromtimestamp(closed_ms / 1000, tz=timezone.utc).isoformat(),
        "closed_ist": closed_dt.isoformat(),
        "entry": float(active["entry"]),
        "exit": float(average_exit or exit_price),
        "entry_slippage_pct": active.get("entry_slippage_pct"),
        "exit_slippage_pct": exit_slippage_pct,
        "sl": float(active["sl"]),
        "tp": float(active["tp"]),
        "qty": float(active["qty"]),
        "risk_usdt": float(active.get("risk_usdt") or 0),
        "pnl_usdt": float(pnl),
        "pnl_accuracy": pnl_accuracy,
        "risk_loss": is_risk_loss,
        "r_multiple": (float(pnl) / float(active["risk_usdt"])) if active.get("risk_usdt") else None,
        "result": exit_reason,
        "reason": active.get("reason", ""),
        "entry_order_id": active.get("entry_order_id"),
        "exit_order_id": exit_order.get("id"),
    }
    trades.append(trade)
    save_trade_log(trades)

    # If exchange realized PnL/fees could not be retrieved, classify the result
    # conservatively as a loss for risk limits (fallback PnL is gross, before fees).
    if is_risk_loss:
        risk_state["daily_losses"] = int(risk_state.get("daily_losses", 0)) + 1
        expiry = now_ist() + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
        risk_state.setdefault("cooldowns", {})[active["symbol"]] = expiry.isoformat()
        log.warning("[%s] Losing trade. %d/%d daily losses. Symbol cooldown until %s IST.",
                    active["symbol"], risk_state["daily_losses"], MAX_LOSING_TRADES_PER_DAY,
                    expiry.strftime("%H:%M:%S"))
    if int(risk_state.get("daily_losses", 0)) >= MAX_LOSING_TRADES_PER_DAY:
        risk_state["daily_halted"] = True
        risk_state["daily_halt_reason"] = "TWO_LOSSES_REACHED"
    save_active_state({})
    save_risk_state(risk_state)
    send_trade_telegram_alert("EXIT", trade, risk_state)
    log.info("CLOSED %s | %s | PnL=%.5f USDT (%s) | daily entries=%d losses=%d",
             trade["symbol"], exit_reason, pnl, pnl_accuracy,
             risk_state.get("daily_trades", 0), risk_state.get("daily_losses", 0))
    if REPORT_EVERY_N_TRADES and len(trades) % REPORT_EVERY_N_TRADES == 0:
        write_periodic_pdf_report(trades, float(risk_state.get("last_equity") or 0))
    return trade


# --------------------------- periodic PDF ---------------------------
def trade_stats(trades):
    count = len(trades)
    values = [float(t.get("pnl_usdt") or 0) for t in trades]
    wins = sum(1 for trade, value in zip(trades, values) if value > 0 and not trade.get("risk_loss", False))
    losses_count = sum(1 for trade, value in zip(trades, values) if trade.get("risk_loss", False) or value < 0)
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    return {
        "trades": count,
        "wins": wins,
        "losses": losses_count,
        "breakeven": count - wins - losses_count,
        "win_rate_pct": wins / count * 100 if count else 0.0,
        "net_pnl_usdt": sum(values),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
    }


def write_periodic_pdf_report(trades, equity):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        batch = trades[-REPORT_EVERY_N_TRADES:]
        path = os.path.join(BASE_DIR, f"trade_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        styles = getSampleStyleSheet()
        stats = trade_stats(trades)
        story = [Paragraph("Chan-Inspired Mean-Reversion Bot — DEMO Trade Batch", styles["Title"]),
                 Paragraph(f"Generated {utc_now()} | Equity {equity:.4f} USDT", styles["Normal"]), Spacer(1, 6),
                 Paragraph(f"Total trades: {stats['trades']} | Win rate: {stats['win_rate_pct']:.1f}% | "
                           f"Net PnL: {stats['net_pnl_usdt']:.4f} USDT", styles["Normal"])]
        rows = [["#", "Symbol", "Side", "Entry", "Exit", "Qty", "PnL USDT", "Accuracy", "Result"]]
        for t in batch:
            rows.append([str(t.get("n")), t.get("symbol"), t.get("side"), str(t.get("entry")),
                         str(t.get("exit")), str(t.get("qty")), f"{float(t.get('pnl_usdt') or 0):.4f}",
                         t.get("pnl_accuracy", ""), t.get("result", "")])
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                   ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story.append(table)
        SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=8*mm, rightMargin=8*mm,
                          topMargin=8*mm, bottomMargin=8*mm).build(story)
        log.info("Periodic PDF report saved: %s", path)
    except ImportError:
        log.warning("Periodic PDF skipped: install reportlab.")
    except Exception:
        log.exception("Periodic PDF report failed.")


# --------------------------- daily report / Telegram ---------------------------
def trades_closed_on(trades, day_key):
    result = []
    for trade in trades:
        closed = parse_iso(trade.get("closed_ist") or trade.get("closed_utc"))
        if closed and ist_day_key(closed) == day_key:
            result.append(trade)
    return result


def daily_report_text(day_key, day_trades, risk_state, active=None):
    stats = trade_stats(day_trades)
    start_equity = float(risk_state.get("day_start_equity") or 0)
    end_equity = float(risk_state.get("day_end_equity") or risk_state.get("last_equity") or start_equity)
    daily_dd = float(risk_state.get("daily_drawdown_pct") or 0)
    overall_dd = float(risk_state.get("overall_drawdown_pct") or 0)
    reason = risk_state.get("daily_halt_reason") or ("OVERALL_DRAWDOWN_HALT" if risk_state.get("overall_halted") else "No daily halt")
    entry_slips = [float(t["entry_slippage_pct"]) for t in day_trades if t.get("entry_slippage_pct") is not None]
    exit_slips = [float(t["exit_slippage_pct"]) for t in day_trades if t.get("exit_slippage_pct") is not None]
    avg_entry_slip = sum(entry_slips) / len(entry_slips) if entry_slips else None
    avg_exit_slip = sum(exit_slips) / len(exit_slips) if exit_slips else None
    lines = [
        f"📊 DAILY DEMO TRADING REPORT — {day_key} (IST)",
        "Strategy: Chan-inspired mean reversion | Market: Binance USDT-M Demo",
        f"Entries opened today: {int(risk_state.get('daily_trades', 0))}/{MAX_TRADES_PER_DAY}",
        f"Trades closed today: {stats['trades']} | Wins: {stats['wins']} | Losses: {stats['losses']} | Breakeven: {stats['breakeven']}",
        f"Win rate: {stats['win_rate_pct']:.2f}%",
        f"Net realized PnL: {stats['net_pnl_usdt']:+.5f} USDT",
        f"Average adverse entry slippage: {avg_entry_slip:+.5f}%" if avg_entry_slip is not None else "Average adverse entry slippage: n/a",
        f"Average adverse exit slippage: {avg_exit_slip:+.5f}%" if avg_exit_slip is not None else "Average adverse exit slippage: n/a",
        f"Profit factor: {stats['profit_factor']:.3f}" if stats["profit_factor"] is not None else "Profit factor: n/a",
        f"Start-of-day equity: {start_equity:.5f} USDT",
        f"Last recorded day-end equity: {end_equity:.5f} USDT",
        f"Current / max daily drawdown: {daily_dd:.3f}% / {float(risk_state.get('max_daily_drawdown_pct') or daily_dd):.3f}% (limit {DAILY_DRAWDOWN_LIMIT_PCT:.1f}%)",
        f"Current / max overall drawdown: {overall_dd:.3f}% / {float(risk_state.get('max_overall_drawdown_pct') or overall_dd):.3f}% (limit {OVERALL_DRAWDOWN_LIMIT_PCT:.1f}%)",
        f"Losing trades today: {int(risk_state.get('daily_losses', 0))}/{MAX_LOSING_TRADES_PER_DAY}",
        f"Trading status: {'HALTED' if risk_state.get('daily_halted') or risk_state.get('overall_halted') else 'ELIGIBLE'}",
        f"Halt reason: {reason}",
    ]
    if day_trades:
        lines.append("\nCLOSED TRADES:")
        for trade in day_trades:
            lines.append(
                f"• {trade.get('symbol')} {str(trade.get('side', '')).upper()} | "
                f"PnL {float(trade.get('pnl_usdt') or 0):+.5f} USDT | "
                f"{trade.get('result')} | entry slip {trade.get('entry_slippage_pct')}% | "
                f"exit slip {trade.get('exit_slippage_pct')}% | {trade.get('pnl_accuracy', 'PnL accuracy unknown')}"
            )
    else:
        lines.append("\nNo completed trades were recorded for this IST date.")
    if active:
        lines.append(
            f"\nOPEN POSITION AT REPORT TIME: {active.get('symbol')} "
            f"{str(active.get('side', '')).upper()} | entry {active.get('entry')} | "
            f"SL {active.get('sl')} | TP {active.get('tp')} | qty {active.get('qty')}. "
            "Bot continues managing it after this report."
        )
    lines.append("\nRisk note: PnL fallback estimates, if any, may exclude fees. Check the attached PDF and exchange history.")
    return "\n".join(lines)


def write_daily_pdf_report(day_key, day_trades, risk_state, active=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    os.makedirs(DAILY_REPORT_DIR, exist_ok=True)
    path = os.path.join(DAILY_REPORT_DIR, f"daily_report_{day_key}.pdf")
    styles = getSampleStyleSheet()
    stats = trade_stats(day_trades)
    start_equity = float(risk_state.get("day_start_equity") or 0)
    end_equity = float(risk_state.get("day_end_equity") or risk_state.get("last_equity") or start_equity)
    story = [
        Paragraph("Daily Trading Report — Binance DEMO USDT-M Futures", styles["Title"]),
        Paragraph(f"Trading date: {day_key} (Asia/Kolkata) | Generated: {ist_now_iso()}", styles["Normal"]),
        Spacer(1, 8),
    ]
    entry_slips = [float(t["entry_slippage_pct"]) for t in day_trades if t.get("entry_slippage_pct") is not None]
    exit_slips = [float(t["exit_slippage_pct"]) for t in day_trades if t.get("exit_slippage_pct") is not None]
    avg_entry_slip = sum(entry_slips) / len(entry_slips) if entry_slips else None
    avg_exit_slip = sum(exit_slips) / len(exit_slips) if exit_slips else None
    summary_rows = [
        ["Metric", "Value"],
        ["Entries opened", f"{int(risk_state.get('daily_trades', 0))}/{MAX_TRADES_PER_DAY}"],
        ["Trades closed", str(stats["trades"])],
        ["Wins / losses", f"{stats['wins']} / {stats['losses']}"],
        ["Win rate", f"{stats['win_rate_pct']:.2f}%"],
        ["Net realized PnL", f"{stats['net_pnl_usdt']:+.5f} USDT"],
        ["Average entry slippage (adverse positive)", f"{avg_entry_slip:+.5f}%" if avg_entry_slip is not None else "n/a"],
        ["Average exit slippage (adverse positive)", f"{avg_exit_slip:+.5f}%" if avg_exit_slip is not None else "n/a"],
        ["Profit factor", f"{stats['profit_factor']:.3f}" if stats["profit_factor"] is not None else "n/a"],
        ["Start-of-day equity", f"{start_equity:.5f} USDT"],
        ["Last recorded day-end equity", f"{end_equity:.5f} USDT"],
        ["Current / max daily drawdown", f"{float(risk_state.get('daily_drawdown_pct') or 0):.3f}% / {float(risk_state.get('max_daily_drawdown_pct') or 0):.3f}% (limit {DAILY_DRAWDOWN_LIMIT_PCT:.1f}%)"],
        ["Current / max overall drawdown", f"{float(risk_state.get('overall_drawdown_pct') or 0):.3f}% / {float(risk_state.get('max_overall_drawdown_pct') or 0):.3f}% (limit {OVERALL_DRAWDOWN_LIMIT_PCT:.1f}%)"],
        ["Daily losing trades", f"{int(risk_state.get('daily_losses', 0))}/{MAX_LOSING_TRADES_PER_DAY}"],
        ["Daily halt reason", risk_state.get("daily_halt_reason") or "None"],
        ["Overall halt", "YES — manual review required" if risk_state.get("overall_halted") else "No"],
    ]
    summary = Table(summary_rows, colWidths=[65*mm, 100*mm], repeatRows=1)
    summary.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                 ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                                 ("FONTSIZE", (0, 0), (-1, -1), 8),
                                 ("PADDING", (0, 0), (-1, -1), 5)]))
    story.extend([Paragraph("Daily summary", styles["Heading2"]), summary, Spacer(1, 10),
                  Paragraph("Completed trades", styles["Heading2"])])
    rows = [["#", "Symbol", "Side", "Opened (IST)", "Closed (IST)", "Entry", "Exit", "Qty", "PnL USDT", "Entry slip %", "Exit slip %", "Exit reason", "PnL source"]]
    for trade in day_trades:
        rows.append([
            str(trade.get("n", "")), trade.get("symbol", ""), str(trade.get("side", "")).upper(),
            (trade.get("opened_ist") or "")[:19].replace("T", " "),
            (trade.get("closed_ist") or "")[:19].replace("T", " "),
            f"{float(trade.get('entry') or 0):.8g}", f"{float(trade.get('exit') or 0):.8g}",
            f"{float(trade.get('qty') or 0):.8g}", f"{float(trade.get('pnl_usdt') or 0):+.5f}",
            f"{float(trade.get('entry_slippage_pct')):+.5f}" if trade.get("entry_slippage_pct") is not None else "n/a",
            f"{float(trade.get('exit_slippage_pct')):+.5f}" if trade.get("exit_slippage_pct") is not None else "n/a",
            trade.get("result", ""), trade.get("pnl_accuracy", ""),
        ])
    if not day_trades:
        rows.append(["—", "No completed trades", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
    detail = Table(rows, colWidths=[8*mm, 23*mm, 12*mm, 24*mm, 24*mm, 18*mm, 18*mm,
                                    16*mm, 18*mm, 17*mm, 17*mm, 22*mm, 36*mm], repeatRows=1)
    detail.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                ("FONTSIZE", (0, 0), (-1, -1), 6),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.extend([detail, Spacer(1, 10)])
    if active:
        open_rows = [["Open position", "Side", "Entry", "Stop", "Target", "Quantity"],
                     [active.get("symbol", ""), str(active.get("side", "")).upper(),
                      str(active.get("entry", "")), str(active.get("sl", "")),
                      str(active.get("tp", "")), str(active.get("qty", ""))]]
        open_table = Table(open_rows, repeatRows=1)
        open_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                        ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story.extend([Paragraph("Position still open at report time", styles["Heading2"]), open_table, Spacer(1, 8)])
    story.append(Paragraph("Risk settings: 3 entries/day; stop after 2 losing trades; 2.5% daily drawdown; "
                            "8% overall drawdown from persisted equity high-water mark; 15-minute per-symbol "
                            "cooldown after a losing trade. Bot-managed exits require the bot and network to remain online.",
                            styles["Italic"]))
    SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=7*mm, rightMargin=7*mm,
                      topMargin=8*mm, bottomMargin=8*mm).build(story)
    return path


def telegram_post(method, fields=None, file_path=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    if file_path:
        boundary = "----ChatGPTBoundary" + uuid.uuid4().hex
        body = bytearray()
        for key, value in (fields or {}).items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            body.extend(str(value).encode())
            body.extend(b"\r\n")
        filename = os.path.basename(file_path)
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'.encode())
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode())
        with open(file_path, "rb") as handle:
            body.extend(handle.read())
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        request = urllib.request.Request(url, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        payload = dict(fields or {})
        payload["chat_id"] = TELEGRAM_CHAT_ID
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram HTTP {exc.code}: {detail[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Telegram request failed: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API rejected request: {result}")
    return result


def send_startup_telegram_test(ex, equity, symbols, risk_state, active=None):
    """Send one Telegram startup/connection test after startup checks pass."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Startup Telegram test not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return False

    status = "RUNNING — monitoring for signals"
    if risk_state.get("overall_halted"):
        status = "STARTED BUT HALTED — overall drawdown/manual review required"
    elif risk_state.get("daily_halted"):
        status = f"STARTED — entries blocked today ({risk_state.get('daily_halt_reason') or 'daily limit'})"
    open_position = "None"
    if active:
        open_position = (
            f"{active.get('symbol')} {str(active.get('side', '')).upper()} | "
            f"entry {float(active.get('entry') or 0):.8g} | "
            f"SL {float(active.get('sl') or 0):.8g} | TP {float(active.get('tp') or 0):.8g}"
        )
    message = (
        "✅ BINANCE DEMO BOT STARTUP TEST\n"
        "Telegram connection: SUCCESS\n"
        "Environment: BINANCE DEMO USDT-M FUTURES (no live trading)\n"
        f"Status: {status}\n"
        f"Started (IST): {ist_now_iso()}\n"
        f"Futures equity: {float(equity):.5f} USDT\n"
        f"Symbols validated: {len(symbols)}/{len(CANDIDATE_SYMBOLS)}\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Today's entries: {int(risk_state.get('daily_trades', 0))}/{MAX_TRADES_PER_DAY}\n"
        f"Today's losses: {int(risk_state.get('daily_losses', 0))}/{MAX_LOSING_TRADES_PER_DAY}\n"
        f"Daily drawdown limit: {DAILY_DRAWDOWN_LIMIT_PCT:.2f}% | "
        f"Overall drawdown limit: {OVERALL_DRAWDOWN_LIMIT_PCT:.2f}%\n"
        f"Open position: {open_position}\n"
        "This is an automatic test message sent once every time the bot starts successfully."
    )
    try:
        telegram_post("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True})
        log.info("Telegram startup test message sent successfully.")
        return True
    except Exception as exc:
        log.error("Telegram startup test failed: %s", exc)
        return False


def send_trade_telegram_alert(event, trade_data, risk_state):
    """Send a best-effort, restart-aware Telegram alert for a filled entry or exit."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Trade %s alert not sent: Telegram is not configured.", event)
        return False

    risk_state.setdefault("trade_alerts", {})
    order_id = trade_data.get("entry_order_id") if event == "ENTRY" else trade_data.get("exit_order_id")
    alert_key = f"{event}:{order_id or trade_data.get('symbol', 'unknown')}"
    alert_state = risk_state["trade_alerts"].setdefault(alert_key, {})
    if alert_state.get("sent"):
        return True

    symbol = trade_data.get("symbol", "UNKNOWN")
    side = str(trade_data.get("side", "")).upper()
    entry = float(trade_data.get("entry") or 0)
    stop = float(trade_data.get("sl") or 0)
    target = float(trade_data.get("tp") or 0)
    qty = float(trade_data.get("qty") or 0)
    potential_loss = abs(entry - stop) * qty
    potential_profit = abs(target - entry) * qty

    if event == "ENTRY":
        message = (
            "🟢 BINANCE DEMO — TRADE OPENED\n"
            f"Symbol: {symbol}\n"
            f"Position: {side}\n"
            f"Entry fill: {entry:.8g}\n"
            f"Quantity: {qty:.8g}\n"
            f"Stop-loss: {stop:.8g}\n"
            f"Take-profit: {target:.8g}\n"
            f"Potential loss at SL: -{potential_loss:.5f} USDT\n"
            f"Potential profit at TP: +{potential_profit:.5f} USDT\n"
            f"Entry slippage: {float(trade_data['entry_slippage_pct']):+.5f}%\n"
            "Estimates exclude fees, funding and any additional slippage.\n"
            f"Time (IST): {trade_data.get('opened_ist', ist_now_iso())}"
        ) if trade_data.get("entry_slippage_pct") is not None else (
            "🟢 BINANCE DEMO — TRADE OPENED\n"
            f"Symbol: {symbol}\nPosition: {side}\nEntry fill: {entry:.8g}\n"
            f"Quantity: {qty:.8g}\nStop-loss: {stop:.8g}\nTake-profit: {target:.8g}\n"
            f"Potential loss at SL: -{potential_loss:.5f} USDT\n"
            f"Potential profit at TP: +{potential_profit:.5f} USDT\n"
            "Estimates exclude fees, funding and any additional slippage.\n"
            f"Time (IST): {trade_data.get('opened_ist', ist_now_iso())}"
        )
    else:
        exit_price = float(trade_data.get("exit") or 0)
        pnl = float(trade_data.get("pnl_usdt") or 0)
        exit_side = "SELL" if side == "BUY" else "BUY"
        message = (
            "🔴 BINANCE DEMO — POSITION CLOSED\n"
            f"Symbol: {symbol}\n"
            f"Position: {side} | Exit order side: {exit_side}\n"
            f"Entry fill: {entry:.8g}\n"
            f"Exit fill: {exit_price:.8g}\n"
            f"Stop-loss: {stop:.8g}\n"
            f"Take-profit: {target:.8g}\n"
            f"Potential loss at SL: -{potential_loss:.5f} USDT\n"
            f"Potential profit at TP: +{potential_profit:.5f} USDT\n"
            f"Realised PnL: {pnl:+.5f} USDT\n"
            f"Exit reason: {trade_data.get('result', 'UNKNOWN')}\n"
            f"PnL source: {trade_data.get('pnl_accuracy', 'unknown')}\n"
            f"Time (IST): {trade_data.get('closed_ist', ist_now_iso())}"
        )

    # Save an attempt timestamp before sending; successful delivery is persisted afterward.
    alert_state["last_attempt_ist"] = ist_now_iso()
    save_risk_state(risk_state)
    try:
        telegram_post("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True})
        alert_state["sent"] = True
        alert_state["sent_at_ist"] = ist_now_iso()
        save_risk_state(risk_state)
        log.info("Telegram %s alert sent for %s (order=%s).", event, symbol, order_id)
        return True
    except Exception as exc:
        alert_state["last_error"] = str(exc)
        save_risk_state(risk_state)
        log.error("Telegram %s alert failed for %s: %s", event, symbol, exc)
        return False


def maybe_send_daily_report(day_key, day_trades, risk_state, active=None):
    report_state = risk_state.setdefault("reports", {}).setdefault(day_key, {})
    if report_state.get("text_sent") and report_state.get("pdf_sent"):
        return

    # Build and persist the PDF once. If reportlab is missing, text delivery can
    # still proceed and PDF generation will be retried later.
    pdf_path = report_state.get("pdf_path")
    if not pdf_path or not os.path.isfile(pdf_path):
        now_ts = time.time()
        if now_ts - float(report_state.get("last_pdf_generate_attempt_ts", 0)) >= 60:
            report_state["last_pdf_generate_attempt_ts"] = now_ts
            save_risk_state(risk_state)
            try:
                pdf_path = write_daily_pdf_report(day_key, day_trades, risk_state, active)
                report_state["pdf_path"] = pdf_path
                report_state["pdf_generated"] = True
                save_risk_state(risk_state)
            except Exception as exc:
                report_state["last_pdf_error"] = str(exc)
                save_risk_state(risk_state)
                log.exception("Could not generate daily PDF report for %s. Install reportlab if missing.", day_key)
                pdf_path = None
        else:
            pdf_path = None

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        if not report_state.get("missing_telegram_warned"):
            log.warning("Daily report is saved locally where possible, but Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            report_state["missing_telegram_warned"] = True
            save_risk_state(risk_state)
        return

    now_ts = time.time()
    # Separate delivery flags reduce duplicate sends if only one of the two Telegram calls succeeds.
    if not report_state.get("text_sent") and now_ts - float(report_state.get("last_text_attempt_ts", 0)) >= 60:
        report_state["last_text_attempt_ts"] = now_ts
        save_risk_state(risk_state)
        try:
            message = daily_report_text(day_key, day_trades, risk_state, active)
            telegram_post("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": message, "disable_web_page_preview": True})
            report_state["text_sent"] = True
            report_state["text_sent_at"] = ist_now_iso()
            save_risk_state(risk_state)
            log.info("Telegram daily text report sent for %s.", day_key)
        except Exception as exc:
            report_state["last_text_error"] = str(exc)
            save_risk_state(risk_state)
            log.error("Telegram text report failed for %s: %s", day_key, exc)

    if pdf_path and not report_state.get("pdf_sent") and now_ts - float(report_state.get("last_pdf_attempt_ts", 0)) >= 60:
        report_state["last_pdf_attempt_ts"] = now_ts
        save_risk_state(risk_state)
        try:
            telegram_post("sendDocument", {"chat_id": TELEGRAM_CHAT_ID, "caption": f"Daily DEMO report — {day_key} (IST)"}, pdf_path)
            report_state["pdf_sent"] = True
            report_state["pdf_sent_at"] = ist_now_iso()
            save_risk_state(risk_state)
            log.info("Telegram daily PDF report sent for %s.", day_key)
        except Exception as exc:
            report_state["last_pdf_error"] = str(exc)
            save_risk_state(risk_state)
            log.error("Telegram PDF report failed for %s: %s", day_key, exc)


def maybe_run_daily_report(trades, risk_state, active=None, now=None):
    now = now or now_ist()
    day_key = ist_day_key(now)
    if (now.hour, now.minute) >= (DAILY_REPORT_HOUR, DAILY_REPORT_MINUTE):
        maybe_send_daily_report(day_key, trades_closed_on(trades, day_key), risk_state, active)


def rollover_ist_day(ex, trades, risk_state, equity, today_key, active=None):
    previous_day = risk_state.get("day_ist")
    if previous_day and previous_day != today_key:
        # If the process was stopped at 23:59, send the previous day's report on restart.
        maybe_send_daily_report(previous_day, trades_closed_on(trades, previous_day), risk_state, active)
        reset_daily_state(risk_state, equity, today_key)
        save_risk_state(risk_state)
        log.info("IST day reset to %s. New start-of-day equity: %.5f USDT", today_key, equity)
    elif not previous_day:
        risk_state["day_ist"] = today_key
        risk_state["day_start_equity"] = equity
        risk_state["day_end_equity"] = equity
        save_risk_state(risk_state)
    return risk_state


# --------------------------- main loop ---------------------------
def main():
    global SYMBOLS
    ex = make_exchange()
    SYMBOLS = validate_symbols(ex)

    # This bot's reduceOnly exits require One-way Mode.
    try:
        dual = ex.fapiPrivateGetPositionSideDual()
        if str(dual.get("dualSidePosition")).lower() == "true":
            raise SystemExit("Switch Binance Futures to One-way Mode before running this bot.")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Could not verify Binance One-way Mode safely: {exc}")

    for symbol in SYMBOLS:
        try:
            ex.set_margin_mode("isolated", symbol)
        except Exception as exc:
            log.info("[%s] Isolated margin setting not changed: %s", symbol, exc)
        ex.set_leverage(LEVERAGE, symbol)

    trades = load_trade_log()
    equity = get_equity(ex)
    risk_state = load_risk_state()
    if not risk_state or int(risk_state.get("schema_version", 0)) < 2:
        risk_state = initialize_risk_state(equity, trades)
        save_risk_state(risk_state)
    else:
        risk_state.setdefault("reports", {})
        risk_state.setdefault("trade_alerts", {})
        risk_state.setdefault("cooldowns", {})
        risk_state.setdefault("last_candle_ts", {})
        risk_state.setdefault("overall_halted", False)
        risk_state.setdefault("overall_halt_reason", "")
        risk_state.setdefault("daily_halted", False)
        risk_state.setdefault("daily_halt_reason", "")
        risk_state.setdefault("daily_drawdown_breached", False)
        risk_state.setdefault("daily_trades", 0)
        risk_state.setdefault("daily_losses", 0)
        risk_state.setdefault("overall_peak_equity", equity)
        risk_state.setdefault("day_start_equity", equity)
        risk_state.setdefault("day_end_equity", equity)
        risk_state.setdefault("max_daily_drawdown_pct", float(risk_state.get("daily_drawdown_pct") or 0))
        risk_state.setdefault("max_overall_drawdown_pct", float(risk_state.get("overall_drawdown_pct") or 0))

    active = load_active_state()
    existing = get_open_positions(ex)
    if existing:
        if not active:
            raise SystemExit(
                "Unmanaged Binance position detected. Bot will not guess its stop/target state: "
                + ", ".join(f"{p['symbol']} ({p['side']} {p['contracts']})" for p in existing)
                + ". Manage/close it manually and inspect the exchange before restarting."
            )
        if len(existing) != 1 or existing[0]["symbol"] != active.get("symbol"):
            raise SystemExit("Exchange positions do not match saved active_trade_state.json. Manual reconciliation required.")
        if existing[0]["side"] and existing[0]["side"] != active.get("side"):
            raise SystemExit("Saved active position side does not match Binance. Manual reconciliation required.")
        saved_qty = float(active.get("qty") or 0)
        exchange_qty = float(existing[0].get("contracts") or 0)
        if saved_qty <= 0 or abs(exchange_qty - saved_qty) > max(1e-8, saved_qty * 0.01):
            raise SystemExit("Saved active position quantity does not match Binance. Manual reconciliation required.")
        log.warning("Resuming saved managed position %s after restart; verify Binance position and active state.", active["symbol"])
    elif active:
        raise SystemExit(
            "Saved active_trade_state.json exists but Binance shows no matching position. "
            "Trade may have closed while bot was offline; reconcile order history before clearing state."
        )

    # No resting orders are created by this bot. Existing orders may be user orders, so do not cancel them.
    for symbol in SYMBOLS:
        open_orders = ex.fetch_open_orders(symbol)
        if open_orders:
            raise SystemExit(f"Open order(s) detected for {symbol}. Review/cancel manually before starting.")

    today_key = ist_day_key()
    if risk_state.get("day_ist") != today_key:
        risk_state = rollover_ist_day(ex, trades, risk_state, equity, today_key, active)
    else:
        risk_state = update_equity_metrics(risk_state, equity)
    save_risk_state(risk_state)

    log.info(
        "STARTED BINANCE DEMO | equity=%.5f USDT | symbols=%s | daily entries=%d/%d | losses=%d/%d | "
        "daily DD limit=%.2f%% | overall DD limit=%.2f%% | IST report=%02d:%02d",
        equity, ", ".join(SYMBOLS), risk_state.get("daily_trades", 0), MAX_TRADES_PER_DAY,
        risk_state.get("daily_losses", 0), MAX_LOSING_TRADES_PER_DAY,
        DAILY_DRAWDOWN_LIMIT_PCT, OVERALL_DRAWDOWN_LIMIT_PCT,
        DAILY_REPORT_HOUR, DAILY_REPORT_MINUTE,
    )

    # Send one connection/health test after all exchange and state reconciliation checks pass.
    send_startup_telegram_test(ex, equity, SYMBOLS, risk_state, active)

    while True:
        try:
            now = now_ist()
            today_key = ist_day_key(now)
            equity = get_equity(ex)
            if risk_state.get("day_ist") != today_key:
                risk_state = rollover_ist_day(ex, trades, risk_state, equity, today_key, active)
            risk_state = update_equity_metrics(risk_state, equity)

            # Drawdown breaches close the bot-managed open position before blocking new entries.
            if active and (risk_state.get("overall_halted") or risk_state.get("daily_drawdown_breached")):
                reason = "OVERALL_DRAWDOWN_LIMIT" if risk_state.get("overall_halted") else "DAILY_DRAWDOWN_LIMIT"
                close_and_record_trade(ex, active, reason, trades, risk_state)
                active = {}
                equity = get_equity(ex)
                risk_state = update_equity_metrics(risk_state, equity)

            # Normal stop / target management. These are bot-managed, not exchange-side orders.
            if active:
                symbol = active["symbol"]
                ticker = ex.fetch_ticker(symbol)
                last = float(ticker.get("last") or ticker.get("close"))
                exit_reason = None
                if active["side"] == "buy":
                    if last <= float(active["sl"]):
                        exit_reason = "STOP_LOSS"
                    elif last >= float(active["tp"]):
                        exit_reason = "TAKE_PROFIT"
                else:
                    if last >= float(active["sl"]):
                        exit_reason = "STOP_LOSS"
                    elif last <= float(active["tp"]):
                        exit_reason = "TAKE_PROFIT"
                positions = get_open_positions(ex)
                still_open = any(p["symbol"] == symbol for p in positions)
                if exit_reason and still_open:
                    close_and_record_trade(ex, active, exit_reason, trades, risk_state)
                    active = {}
                    equity = get_equity(ex)
                    risk_state = update_equity_metrics(risk_state, equity)
                elif not still_open:
                    raise RuntimeError(
                        f"[{symbol}] Saved active position disappeared from Binance without a bot-recorded exit. "
                        "Stopping for manual trade-history reconciliation."
                    )

            # Daily reporting is independent of whether entries are blocked.
            maybe_run_daily_report(trades, risk_state, active, now_ist())

            # Scan only when flat and all entry rules permit a new trade.
            if not active:
                block = daily_entry_block_reason(risk_state)
                if block:
                    log.info("No new entries: %s", block)
                else:
                    for symbol in SYMBOLS:
                        block = daily_entry_block_reason(risk_state, symbol)
                        if block:
                            log.info("[%s] Skip entry: %s", symbol, block)
                            continue
                        raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
                        df = prepare_candles(raw)
                        if df.empty:
                            continue
                        candle_ts = int(df.iloc[-1]["ts"])
                        last_seen = int((risk_state.get("last_candle_ts") or {}).get(symbol, 0))
                        if candle_ts <= last_seen:
                            continue
                        risk_state.setdefault("last_candle_ts", {})[symbol] = candle_ts
                        save_risk_state(risk_state)

                        signal, explanation = get_signal(df, symbol)
                        if not signal:
                            log.info("[%s] NO TRADE | %s", symbol, explanation)
                            continue
                        side, reference_price, zscore = signal
                        log.info("[%s] SIGNAL | %s | reference=%.8g | z=%.2f",
                                 symbol, side.upper(), reference_price, zscore)
                        if get_open_positions(ex):
                            raise RuntimeError("Exchange position appeared before entry; stopping for reconciliation.")

                        active = place_market_entry(ex, symbol, side, reference_price, explanation)
                        if active:
                            # Persist active position first; then increment the durable daily entry counter.
                            save_active_state(active)
                            record_trade_entry(risk_state, active)
                            send_trade_telegram_alert("ENTRY", active, risk_state)
                            break

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log.warning("Stopped by user. Any open position remains on Binance Demo; manage it immediately.")
            break
        except ccxt.NetworkError as exc:
            log.error("Binance network error: %s. Bot-managed stops may be delayed; check Demo account immediately.", exc)
            time.sleep(5)
        except ccxt.ExchangeError as exc:
            log.exception("Binance exchange error: %s. No new order will be attempted until next loop.", exc)
            time.sleep(5)
        except SystemExit:
            raise
        except Exception as exc:
            log.exception("Loop error: %s", exc)
            time.sleep(5)


if __name__ == "__main__":
    main()
