from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import pandas_ta as ta
import sys, time
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] v6 started; python={sys.version}", flush=True)

from pathlib import Path
import re

# ============================================================
# v30 Trade Window Portfolio Simulator (EventStudy-driven)
# - Fetch OHLCV 1m from Binance for 7-day trade window (+ warmup)
# - Build entry signals on-the-fly (A1 / C0) with v30 cross logic
# - Exit model uses ATR-based k/t + optional barrier x_bars
# - Portfolio:
#     initial_capital = 10,000 USDT
#     trade_size      = 1,000 USDT
#     max positions   = capital-only (maxAvail = floor(capital / trade_size))
#     realized PnL on close only
# - Runs baseline and barrier as separate portfolios
# - AUTO_CYCLE:
#     * runs both scenarios (A1 + C0)
#     * best-of(baseline, barrier) per scenario
#     * select winner across scenarios
#     * PrePaper (next week) runs BARRIER ONLY for winner
#     * writes selection JSON so PrePaper can run without prompts
# - PREPAPER_FROM_JSON:
#     * runs PrePaper (barrier only) from selection JSON, no prompts
# - Console format:
#     SIGNAL (blue) -> OPEN (blue) -> STOP/WINDOW_END (green/red)
# ============================================================

BINANCE_BASE = "https://api.binance.com"
INTERVAL = "1m"
LIMIT = 1000

FEE_RATE = 0.001  # 0.1% buy + 0.1% sell
WARMUP_DAYS = 7
ATR_LEN = 14

# ----------------------------
# Default portfolio
# ----------------------------
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_TRADE_SIZE = 1_000.0

# ----------------------------
# REGIME GUARD: 15m ADX gate
# ----------------------------
ADX_GATE_ENABLE = False        # set False to disable without code changes
ADX_TF = "15T"                # 15-minute
ADX_LEN = 14
ADX_MIN = 26.0                # gate threshold

# Apply gate to pyramiding too?
ADX_GATE_APPLY_TO_PYRAMID = True
ADX_PYR_MIN = 30.0            # PYRAMID gate threshold (stricter)

# ----------------------------
# DIRECTION FILTER (15m +DI/-DI)
# ----------------------------
DI_FILTER_ENABLE = False
DI_GATE_APPLY_TO_PYRAMID = True

# ----------------------------
# PYRAMIDING CONFIG
# ----------------------------
PYRAMID_ENABLE = True

# Maximum pyramid adds per BASE position (base itself is not counted).
# Example: 5 => base + up to 5 pyramid legs.
PYR_MAX_ADDS_CAP = 5

# Pyramid vol threshold rule:
# - If scenario has a min vol rule (A1 => >=1.5), use it.
# - If scenario vol_rule == ALL (C0), pyramid vol threshold is >= 1.0
PYR_VOL_THRESHOLD_ALL = 1.0

# Pyramiding requires RSI_SMA incremental AND vol_ratio threshold.
# If either fails once for a base, pyramiding ceases permanently for that base.

OUT_DIR = "forwardtest"
os.makedirs(OUT_DIR, exist_ok=True)

# Console colors (match your old preference)
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

# Set to a path string to force events from CSV; set to None to disable override.
EVENTS_CSV_OVERRIDE: str | None = None

# Optional: enable override by setting an environment variable:
#   PowerShell: $env:EVENTS_CSV_OVERRIDE=".\some_events.csv"
_env = os.getenv("EVENTS_CSV_OVERRIDE", "").strip()
if _env:
    EVENTS_CSV_OVERRIDE = _env

# If False: do NOT force-close open positions at trade_end.
# Open positions are carried beyond the window, but their P/L is NOT counted in the window summary.
FORCE_CLOSE_AT_WINDOW_END = False

# ----------------------------
# Walk-forward robustness (TRAIN slicing)
# ----------------------------
ROBUST_ENABLE = True
ROBUST_TRAIN_SLICES = 4              # 4 weeks inside TRAIN
ROBUST_MIN_POS_WEEKS = 3             # R1: >=3/4 positive weeks
ROBUST_WORST_WEEK_NET_MIN = -200.0   # R2 gate: worst week net_profit must be > -X (tune)
ROBUST_RANK_PRIMARY = "median_profit_over_maxdd"  # for display; we will rank by this after gates
ROBUST_WEEK_NET_MIN = 500.0      # R1b: each slice/week must earn at least this net profit
ROBUST_REQUIRE_ALL_WEEKS = True  # enforce 4/4 passing R1_week

def load_event_times_from_csv(path: str) -> set[pd.Timestamp]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "event_time" in df.columns:
        col = "event_time"
    elif "time" in df.columns:
        col = "time"
    else:
        raise ValueError(
            f"events csv must contain 'event_time' or 'time'. columns={list(df.columns)}"
        )

    t = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    return set(t.tolist())

# ----------------------------
# Console logging
# ----------------------------
def log_line(ts, action: str, symbol: str, price: float, extra: str = "", color: str = ""):
    ts_str = str(pd.to_datetime(ts, utc=True))[:16]
    msg = f"{ts_str} | {action:<10} | {symbol:<9} | Price {price:<10.6f} {extra}".rstrip()
    if color:
        print(f"{color}{msg}{COLOR_RESET}")
    else:
        print(msg)


def format_trade_id(pid: str) -> str:
    parts = pid.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-3:])  # v30_6_PYR1
    if len(parts) >= 3:
        return "_".join(parts[-2:])  # v30_6
    return pid


# ----------------------------
# JSON helpers
# ----------------------------
def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Portfolio helpers
# ----------------------------
def max_avail_slots(current_capital: float, trade_size: float) -> int:
    """Capital-only max positions availability (your rule #2)."""
    if trade_size <= 0:
        return 0
    return int(max(0, np.floor(current_capital / trade_size)))


def can_open_position(current_capital: float, trade_size: float, open_positions: int) -> bool:
    """
    Capital-only capacity:
      free_capital = capital - (open_positions * trade_size)
      must be >= trade_size
    """
    capital_in_use = open_positions * trade_size
    free_capital = current_capital - capital_in_use
    return free_capital >= trade_size

def compute_max_drawdown(eq: pd.DataFrame, capital_col: str = "capital_usdt") -> tuple[float, float]:
    """
    Returns (max_dd_pct, max_dd_usdt)
      - max_dd_pct is negative (e.g., -0.12 for -12%)
      - max_dd_usdt is negative (e.g., -350.0)
    """
    if eq is None or len(eq) == 0 or capital_col not in eq.columns:
        return (0.0, 0.0)

    s = pd.to_numeric(eq[capital_col], errors="coerce").dropna()
    if len(s) == 0:
        return (0.0, 0.0)

    peak = s.cummax()
    dd_usdt = s - peak
    dd_pct = (s / peak) - 1.0

    max_dd_usdt = float(dd_usdt.min())  # most negative
    max_dd_pct = float(dd_pct.min())    # most negative
    return (max_dd_pct, max_dd_usdt)


def add_drawdown_to_summary(summary_df: pd.DataFrame, eq: pd.DataFrame, label: str) -> pd.DataFrame:
    max_dd_pct, max_dd_usdt = compute_max_drawdown(eq, "capital_usdt")
    summary_df = summary_df.copy()
    summary_df["max_dd_pct"] = max_dd_pct
    summary_df["max_dd_usdt"] = max_dd_usdt
    # Optional: simple risk-adjusted score
    denom = abs(max_dd_usdt) if abs(max_dd_usdt) > 1e-9 else 0.0
    summary_df["profit_over_maxdd"] = (summary_df["net_profit"] / denom) if denom else float("inf")
    return summary_df

# ----------------------------
# PrePaper Monday 0800 UTC Helper
# ----------------------------
def next_monday_0800_utc(after_ts: pd.Timestamp) -> pd.Timestamp:
    """
    Returns the next Monday 08:00 UTC strictly AFTER after_ts.
    Rule A: PrePaper must start Monday 08:00 UTC.
    """
    t = pd.to_datetime(after_ts, utc=True)

    # move to next day 00:00 to ensure "strictly after"
    t = (t + pd.Timedelta(minutes=1)).floor("min")

    # pandas: Monday=0 ... Sunday=6
    days_ahead = (0 - t.weekday()) % 7
    candidate = t.normalize() + pd.Timedelta(days=days_ahead) + pd.Timedelta(hours=8)

    # if candidate is not strictly after t, jump 7 days
    if candidate <= t:
        candidate = candidate + pd.Timedelta(days=7)

    return candidate

# ----------------------------
# Monday-aligned 7d slices within TRAIN Helpers
# ----------------------------

def iter_monday_week_slices(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return list of [week_start, week_end) slices aligned to Monday 00:00 UTC.
    Only returns full 7-day slices fully contained in [start_utc, end_utc).
    """
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    # find first Monday 00:00 >= start_utc
    t = start_utc.floor("D")
    while t.weekday() != 0:
        t += pd.Timedelta(days=1)
    t = t.replace(hour=0, minute=0, second=0, microsecond=0)

    slices: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while True:
        w0 = t
        w1 = w0 + pd.Timedelta(days=7)
        if w0 < start_utc:
            t += pd.Timedelta(days=7)
            continue
        if w1 > end_utc:
            break
        slices.append((w0, w1))
        t += pd.Timedelta(days=7)

    return slices

# ----------------------------
# build_events_all_for_robustness
# ----------------------------
def build_events_all_for_robustness(
    d_features: pd.DataFrame,
    train_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    scenario: str,
) -> pd.DataFrame:
    """
    Full-range events used for robustness slicing.
    If EVENTS_CSV_OVERRIDE is set, uses CSV times (unfiltered).
    Otherwise uses build_events over [train_start, trade_end).
    """
    if EVENTS_CSV_OVERRIDE:
        forced_times = load_event_times_from_csv(EVENTS_CSV_OVERRIDE)
        ev = pd.DataFrame({"event_time": sorted(forced_times)})
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        return ev.reset_index(drop=True)

    # Non-override: generate from features across the full range
    ev = build_events(d_features, train_start, trade_end, scenario)
    ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
    return ev.reset_index(drop=True)

# ----------------------------
# Indicators (Wilder smoothing)
# ----------------------------
def wilders_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilders_rma(delta.clip(lower=0), length)
    avg_loss = wilders_rma(-delta.clip(upper=0), length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


# ----------------------------
# Binance OHLCV fetch
# ----------------------------
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_ohlcv_binance(symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    start_ms = int(start_utc.value // 10**6)
    end_ms = int(end_utc.value // 10**6)

    rows = []
    cur = start_ms
    while cur < end_ms:
        data = fetch_klines(symbol, INTERVAL, cur, end_ms)
        if not data:
            break

        rows.extend(data)
        last_open = data[-1][0]
        cur = last_open + 60_000  # next minute

        time.sleep(0.25)
        if len(data) < LIMIT:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "qav", "num_trades", "tb", "tq", "ignore"
        ],
    )
    df["time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df

# ----------------------------
# Evaluate ONE candidate across TRAIN slices
# ----------------------------

def eval_candidate_robustness_over_train(
    *,
    pair: str,
    scenario: str,
    ohlcv: pd.DataFrame,
    d_features: pd.DataFrame,
    events_all: pd.DataFrame,
    slices: list[tuple[pd.Timestamp, pd.Timestamp]],
    k: float,
    t: float,
    x_bars: int,
    initial_capital: float,
    trade_size: float,
) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for i, (w0, w1) in enumerate(slices, start=1):
        # events within slice
        ev = events_all[
            (events_all["event_time"] >= w0) & (events_all["event_time"] < w1)
        ].reset_index(drop=True)

        trades_df, eq_df, sim_counts = run_portfolio_sim(
            mode="baseline",  # robustness uses one mode consistently
            pair=pair,
            scenario=scenario,
            ohlcv=ohlcv,
            d_features=d_features,
            events=ev,
            trade_start=w0,
            trade_end=w1,
            k=k,
            t=t,
            x_bars=x_bars,
            initial_capital=initial_capital,
            trade_size=trade_size,
        )

        s = summarize_trades(trades_df, f"{scenario}-baseline")

        # drawdown
        max_dd_pct, max_dd_usdt = compute_max_drawdown(eq_df, "capital_usdt")
        s["max_dd_pct"] = float(max_dd_pct)
        s["max_dd_usdt"] = float(max_dd_usdt)

        # guard: avoid div by 0
        s["profit_over_maxdd"] = (float(s["net_profit"]) / abs(float(max_dd_usdt))) if float(max_dd_usdt) != 0 else float("inf")

        # slice metadata
        s["slice_ix"] = int(i)
        s["slice_start"] = w0
        s["slice_end"] = w1
        s["events_in_slice"] = int(len(ev))
        s["trades_in_slice"] = int(len(trades_df))
        s["opens_in_slice"] = int(sim_counts["opens_count"])
        s["closes_in_slice"] = int(sim_counts["closes_count"])
        s["open_positions_end"] = int(sim_counts["open_positions_end"])

        rows.append(s)

    df = pd.DataFrame(rows)

    # Per-slice robustness flags + aggregates
    if not df.empty:
        df["r1_week_pass"] = df["net_profit"].astype(float) >= float(ROBUST_WEEK_NET_MIN)
        df["r2_week_pass"] = df["net_profit"].astype(float) > float(ROBUST_WORST_WEEK_NET_MIN)

        pos_weeks = int(df["r1_week_pass"].sum())
        worst_week_net = float(df["net_profit"].astype(float).min())
        all_weeks_pass = bool(df["r1_week_pass"].all())
        median_net_profit = float(df["net_profit"].astype(float).median())
        median_profit_over_maxdd = float(df["profit_over_maxdd"].astype(float).median())
    else:
        pos_weeks = 0
        worst_week_net = 0.0
        all_weeks_pass = False
        median_net_profit = 0.0
        median_profit_over_maxdd = 0.0

    out: Dict[str, Any] = {
        "pair": pair,
        "scenario": scenario,
        "k": float(k),
        "t": float(t),
        "x_bars": int(x_bars),

        "n_slices": int(len(df)),
        "pos_weeks": int(pos_weeks),
        "all_weeks_pass": bool(all_weeks_pass),

        "worst_week_net_profit": float(worst_week_net),
        "median_net_profit": float(median_net_profit),
        "median_profit_over_maxdd": float(median_profit_over_maxdd),

        # useful for debugging / printing
        "robust_week_net_min": float(ROBUST_WEEK_NET_MIN),
        "df_slices": df,
    }
    return out

# ----------------------------
# Entry features + events (A1 / C0)
# ----------------------------
def compute_entry_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    d = ohlcv.copy()

    d["rsi"] = rsi_wilder(d["close"], 14)
    d["rsi_sma"] = sma(d["rsi"], 14)

    # SMMA200 (Wilder RMA(200))
    d["smma_200"] = wilders_rma(d["close"], 200)
    d["close_gt_smma_200"] = d["close"] > d["smma_200"]

    # Volume SMA(20) + ratio
    d["vol_sma_20"] = sma(d["volume"], 20)
    d["vol_gt_vol_sma"] = d["volume"] > d["vol_sma_20"]
    d["vol_ratio"] = d["volume"] / d["vol_sma_20"]

    # keep only valid rows
    return d.dropna().reset_index(drop=True)

def compute_adx_15m_maps(ohlcv_1m: pd.DataFrame) -> Dict[str, Dict[pd.Timestamp, float]]:
    """
    Compute ADX(+DI/-DI) on 15m bars from 1m OHLCV.
    Returns dict of maps keyed by 1m timestamps:
      {
        "adx_15m": {ts: val},
        "dmp_15m": {ts: val},  # +DI
        "dmn_15m": {ts: val},  # -DI
      }
    """
    d = ohlcv_1m[["time", "open", "high", "low", "close", "volume"]].copy()
    d = d.sort_values("time").set_index("time")

    o15 = d.resample(ADX_TF).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if o15.empty:
        return {"adx_15m": {}, "dmp_15m": {}, "dmn_15m": {}}

    adx_df = ta.adx(o15["high"], o15["low"], o15["close"], length=ADX_LEN)
    adx_col = f"ADX_{ADX_LEN}"
    dmp_col = f"DMP_{ADX_LEN}"
    dmn_col = f"DMN_{ADX_LEN}"

    if (
        adx_df is None
        or adx_df.empty
        or adx_col not in adx_df.columns
        or dmp_col not in adx_df.columns
        or dmn_col not in adx_df.columns
    ):
        return {"adx_15m": {}, "dmp_15m": {}, "dmn_15m": {}}

    o15["adx_15m"] = adx_df[adx_col]
    o15["dmp_15m"] = adx_df[dmp_col]
    o15["dmn_15m"] = adx_df[dmn_col]
    o15 = o15.dropna(subset=["adx_15m", "dmp_15m", "dmn_15m"]).copy()

    adx_series = o15["adx_15m"].reindex(d.index, method="ffill").dropna()
    dmp_series = o15["dmp_15m"].reindex(d.index, method="ffill").dropna()
    dmn_series = o15["dmn_15m"].reindex(d.index, method="ffill").dropna()

    return {
        "adx_15m": adx_series.to_dict(),
        "dmp_15m": dmp_series.to_dict(),
        "dmn_15m": dmn_series.to_dict(),
    }

import re

def _parse_possibility(possibility: str) -> dict:
    """
    Parse v30 candidate IDs like:
      C_FALSE__V_ALL__R_LT_4
      C_ALL__V_ALL__R_LT_1.5
      C_ALL__V_FALSE__R_LT_2

    In your naming:
      R == ratio_vol (your column is vol_ratio)
    """
    m = re.fullmatch(
        r"C_(TRUE|FALSE|ALL)__V_(TRUE|FALSE|ALL)__R_(LT|GE)_([0-9]+(?:\.[0-9]+)?)",
        possibility.strip().upper()
    )
    if not m:
        raise ValueError(f"Invalid possibility format: {possibility}")

    close_s, vol_s, r_op, r_val = m.group(1), m.group(2), m.group(3), float(m.group(4))
    return {"close": close_s, "vol": vol_s, "r_op": r_op, "r_value": r_val}

def get_exit_params_from_finalist(f: Dict[str, Any]) -> tuple[float, float, int]:
    """
    candidate_for_TRADE.json finalist schema:
      finalist["exit_params"] = {"k": ..., "t": ..., "x_bars": ...}
    """
    exitp = f.get("exit_params", {}) or {}
    return float(exitp["k"]), float(exitp["t"]), int(exitp["x_bars"])

def build_events(d, trade_start, trade_end, scenario):
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")

    scenario = scenario.strip().upper()

    # Base event generator (same as before): RSI_SMA cross up 51
    prev = d["rsi_sma"].shift(1)
    curr = d["rsi_sma"]
    cross_up_51 = (prev < 51.0) & (curr >= 51.0)

    ev = d.loc[cross_up_51].copy()
    ev = ev[(ev["time"] >= trade_start) & (ev["time"] < trade_end)].copy()
    
    print(f"[DEBUG][events] base cross_up_51 in-window count = {len(ev)}")

    if scenario == "A1":
        ev = ev[
            (ev["close_gt_smma_200"] == True) &
            (ev["vol_gt_vol_sma"] == True) &
            (ev["vol_ratio"] >= 1.5)
        ].copy()

    elif scenario == "C0":
        ev = ev[
            (ev["close_gt_smma_200"] == False) &
            (ev["vol_gt_vol_sma"] == True)
        ].copy()

    elif scenario.startswith("C_"):
        p = _parse_possibility(scenario)
        print(f"[DEBUG][events] parsed possibility = {p}")

        # close filter
        if p["close"] == "TRUE":
            before = len(ev)
            ev = ev[ev["close_gt_smma_200"] == True].copy()
            print(f"[DEBUG][events] close TRUE: {before} -> {len(ev)}")
        elif p["close"] == "FALSE":
            before = len(ev)
            ev = ev[ev["close_gt_smma_200"] == False].copy()
            print(f"[DEBUG][events] close FALSE: {before} -> {len(ev)}")
        else:
            print(f"[DEBUG][events] close ALL: {len(ev)} (no filter)")

        # vol filter
        if p["vol"] == "TRUE":
            before = len(ev)
            ev = ev[ev["vol_gt_vol_sma"] == True].copy()
            print(f"[DEBUG][events] vol TRUE: {before} -> {len(ev)}")
        elif p["vol"] == "FALSE":
            before = len(ev)
            ev = ev[ev["vol_gt_vol_sma"] == False].copy()
            print(f"[DEBUG][events] vol FALSE: {before} -> {len(ev)}")
        else:
            print(f"[DEBUG][events] vol ALL: {len(ev)} (no filter)")

        # R (ratio_vol) filter -> your column vol_ratio
        before = len(ev)
        if p["r_op"] == "LT":
            ev = ev[ev["vol_ratio"] < p["r_value"]].copy()
        else:
            ev = ev[ev["vol_ratio"] >= p["r_value"]].copy()
        print(f"[DEBUG][events] R {p['r_op']} {p['r_value']}: {before} -> {len(ev)}")

    else:
        raise ValueError(f"Unsupported scenario='{scenario}'. Expected 'A1', 'C0', or a 'C_*' possibility.")

    out = ev[[
        "time", "close", "rsi_sma", "smma_200",
        "close_gt_smma_200", "vol_gt_vol_sma", "vol_ratio"
    ]].rename(columns={"time": "event_time", "close": "entry_close"}).reset_index(drop=True)

    return out

# ----------------------------
# Positions
# ----------------------------
@dataclass
class Position:
    pid: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    atr_entry: float
    fixed_stop: float
    trail_dist: float
    peak_high: float
    bars_held: int = 0
    trailing_active: bool = False

    # --- pyramiding identity ---
    base_id: str = ""          # base leg id; base leg has base_id == pid
    is_pyramid: bool = False   # True for pyramid legs
    pyr_level: int = 0         # 0=base, 1..N pyramid level

    # --- pyramiding control (base leg only) ---
    pyr_ceased: bool = False   # once True: never pyramid again for this base
    pyr_adds_done: int = 0     # how many pyramid legs have been opened for this base


def open_position(
    pid: str,
    ts: pd.Timestamp,
    entry_price: float,
    atr_entry: float,
    k: float,
    t: float,
    trade_size: float,
    *,
    base_id: str,
    is_pyramid: bool,
    pyr_level: int
) -> Position:
    qty = trade_size / entry_price
    fixed_stop = entry_price - (k * atr_entry)
    trail_dist = t * atr_entry

    return Position(
        pid=pid,
        entry_time=ts,
        entry_price=entry_price,
        qty=qty,
        atr_entry=atr_entry,
        fixed_stop=fixed_stop,
        trail_dist=trail_dist,
        peak_high=entry_price,

        base_id=base_id,
        is_pyramid=is_pyramid,
        pyr_level=pyr_level,

        # base leg defaults:
        pyr_ceased=False if not is_pyramid else True,   # pyramids don't control pyramiding
        pyr_adds_done=0
    )


def close_position(pos: Position, ts: pd.Timestamp, exit_price: float, reason: str, trade_size: float) -> Dict[str, Any]:
    entry_val = trade_size
    exit_val = pos.qty * exit_price
    buy_fee = entry_val * FEE_RATE
    sell_fee = exit_val * FEE_RATE
    pnl = (exit_val - entry_val) - (buy_fee + sell_fee)

    return {
        "position_id": pos.pid,
        "entry_time": pos.entry_time,
        "exit_time": ts,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "qty": pos.qty,
        "atr_entry": pos.atr_entry,
        "fixed_stop": pos.fixed_stop,
        "trail_dist": pos.trail_dist,
        "bars_held": pos.bars_held,
        "reason": reason,
        "buy_fee_usdt": buy_fee,
        "sell_fee_usdt": sell_fee,
        "net_pnl_usdt": pnl,
    }


# ----------------------------
# Portfolio simulation (bar-by-bar)
# ----------------------------
def run_portfolio_sim(
    *,
    mode: str,
    pair: str,
    scenario: str,
    ohlcv: pd.DataFrame,
    d_features: pd.DataFrame,
    events: pd.DataFrame,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    k: float,
    t: float,
    x_bars: int,
    initial_capital: float,
    trade_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mode = mode.strip().lower()
    if mode not in {"baseline", "barrier"}:
        raise ValueError("mode must be baseline or barrier")

    # event times for O(1) check
    event_times = set(pd.to_datetime(events["event_time"], utc=True).tolist())

    # feature lookup by timestamp for SIGNAL printing
    feat = d_features.set_index("time")[[
        "rsi_sma", "smma_200", "close_gt_smma_200",
        "vol_gt_vol_sma", "vol_ratio"
    ]]
    feat_map = feat.to_dict("index")

    window = ohlcv[(ohlcv["time"] >= trade_start) & (ohlcv["time"] <= trade_end)].copy()
    if window.empty:
        return pd.DataFrame(), pd.DataFrame(), {"opens_count": 0, "closes_count": 0, "open_positions_end": 0}

    current_capital = float(initial_capital)
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    next_id = 1
    opens_count = 0
    closes_count = 0

    for _, bar in window.iterrows():
        ts = bar["time"]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        atr = float(bar.get("atr", np.nan))

        # 1) exits
        for pid, pos in list(positions.items()):
            pos.bars_held += 1
            pos.peak_high = max(pos.peak_high, h)

            if mode == "baseline":
                pos.trailing_active = True
            else:
                if pos.bars_held > x_bars:
                    pos.trailing_active = True

            trail_stop = -np.inf
            if pos.trailing_active:
                trail_stop = pos.peak_high - pos.trail_dist

            stop_level = max(pos.fixed_stop, trail_stop)

            if l <= stop_level:
                exit_price = max(o, stop_level)
                tr = close_position(pos, ts, exit_price, "STOP", trade_size)
                trades.append(tr)
                closes_count += 1

                current_capital += float(tr["net_pnl_usdt"])
                del positions[pid]

                pnl = float(tr["net_pnl_usdt"])
                color = COLOR_GREEN if pnl > 0 else COLOR_RED
                open_cnt = len(positions)
                avail = max_avail_slots(current_capital, trade_size)

                log_line(
                    ts, "STOP", pair, exit_price,
                    extra=f"| ID {format_trade_id(pos.pid):<10} | P/L ${pnl:>8.2f} | Cap ${current_capital:>10.2f} | Port {open_cnt:02d}/{avail:02d}",
                    color=color
                )

        # 2) entries
        if ts in event_times:
            f = feat_map.get(ts, {})
            rsi_sma = float(f.get("rsi_sma", np.nan))
            smma200 = float(f.get("smma_200", np.nan))
            c_gt = f.get("close_gt_smma_200", None)
            v_gt = f.get("vol_gt_vol_sma", None)
            vr = float(f.get("vol_ratio", np.nan))
                        # ADX/DI values (only used if gates are enabled)
            adx15 = np.nan
            dmp15 = np.nan
            dmn15 = np.nan
            if ADX_GATE_ENABLE or DI_FILTER_ENABLE:
                adx15 = float(f.get("adx_15m", np.nan))
                dmp15 = float(f.get("dmp_15m", np.nan))  # +DI
                dmn15 = float(f.get("dmn_15m", np.nan))  # -DI
            # adx15 = float(f.get("adx_15m", np.nan))
            # dmp15 = float(f.get("dmp_15m", np.nan))  # +DI
            # dmn15 = float(f.get("dmn_15m", np.nan))  # -DI

            # SIGNAL row (blue)
            log_line(
                ts, "SIGNAL", pair, c,
                extra=f"| RSI_SMA {rsi_sma:5.2f} | SMMA {smma200:.5f} | "
                    f"C>SMMA {str(c_gt):<5} | V>VSMA {str(v_gt):<5} | "
                    f"VR {vr:>5.2f}",
                color=COLOR_BLUE
            )

            # 1) ADX strength gate (existing)
            if ADX_GATE_ENABLE:
                if not np.isfinite(adx15) or adx15 < ADX_MIN:
                    log_line(ts, "SKIP_ADX", pair, c, extra=f"| ADX15 {adx15:>5.2f} < {ADX_MIN:.2f}")
                    continue

            # 2) DI direction gate (NEW)
            if DI_FILTER_ENABLE:
                if (not np.isfinite(dmp15)) or (not np.isfinite(dmn15)) or (dmp15 <= dmn15):
                    log_line(ts, "SKIP_DI", pair, c, extra=f"| DMP {dmp15:>5.2f} <= DMN {dmn15:>5.2f}")
                    continue

            # ATR must exist
            if not np.isfinite(atr) or atr <= 0:
                continue

            if can_open_position(current_capital, trade_size, open_positions=len(positions)):
                posid = f"{pair}_v30_{next_id}"
                pos = open_position(
                    posid, ts, entry_price=c, atr_entry=atr, k=k, t=t, trade_size=trade_size,
                    base_id=posid, is_pyramid=False, pyr_level=0
                )
                positions[posid] = pos
                opens_count += 1

                open_cnt = len(positions)
                avail = max_avail_slots(current_capital, trade_size)

                # OPEN row (blue), compact (no repeated k/t/SL)
                log_line(
                    ts, "OPEN", pair, c,
                    extra=f"| PosID {posid:<18} | Port {open_cnt:02d}/{avail:02d}",
                    color=COLOR_BLUE
                )
                next_id += 1
            else:
                open_cnt = len(positions)
                avail = max_avail_slots(current_capital, trade_size)
                log_line(
                    ts, "SKIP", pair, c,
                    extra=f"| no capital | Port {open_cnt:02d}/{avail:02d}"
                )

        # ============================================================
        # PYRAMIDING: attempt every bar after base OPEN until failure.
        # Conditions to add (MUST satisfy BOTH):
        #   1) rsi_sma(now) > rsi_sma(prev)
        #   2) vol_ratio(now) >= pyr_vol_min
        # If either fails ONCE for a base => base.pyr_ceased=True forever.
        #
        # Vol threshold rule:
        #   - A1 => >=1.5
        #   - C0 => >=1.0
        # ============================================================
        if PYRAMID_ENABLE:
            scen = scenario.upper()
            pyr_vol_min = 1.5 if scen == "A1" else PYR_VOL_THRESHOLD_ALL  # C0 => 1.0

            f_now = feat_map.get(ts, {})
                        # ADX/DI values (only used if gates are enabled)
            adx_now = np.nan
            dmp_now = np.nan
            dmn_now = np.nan
            if ADX_GATE_ENABLE or DI_FILTER_ENABLE:
                adx_now = float(f_now.get("adx_15m", np.nan))
                dmp_now = float(f_now.get("dmp_15m", np.nan))
                dmn_now = float(f_now.get("dmn_15m", np.nan))

            # adx_now = float(f_now.get("adx_15m", np.nan))
            # dmp_now = float(f_now.get("dmp_15m", np.nan))
            # dmn_now = float(f_now.get("dmn_15m", np.nan))

            # ADX gate for pyramiding (strength)
            if ADX_GATE_ENABLE and ADX_GATE_APPLY_TO_PYRAMID:
                if not np.isfinite(adx_now) or adx_now < ADX_PYR_MIN:
                    continue

            # DI gate for pyramiding (direction)
            if DI_FILTER_ENABLE and DI_GATE_APPLY_TO_PYRAMID:
                if (not np.isfinite(dmp_now)) or (not np.isfinite(dmn_now)) or (dmp_now <= dmn_now):
                    continue
            
            f_prev = feat_map.get(ts - pd.Timedelta(minutes=1), {})

            vr_now = float(f_now.get("vol_ratio", np.nan))
            rsi_now = float(f_now.get("rsi_sma", np.nan))
            rsi_prev = float(f_prev.get("rsi_sma", np.nan))

            ok_vol = np.isfinite(vr_now) and (vr_now >= pyr_vol_min)
            ok_rsi = np.isfinite(rsi_now) and np.isfinite(rsi_prev) and (rsi_now > rsi_prev)

            # iterate base legs only (not pyramids)
            for base in [p for p in positions.values() if (not p.is_pyramid and p.base_id == p.pid)]:
                if base.pyr_ceased:
                    continue

                # only start trying after at least 1 bar passed since base entry
                if base.bars_held < 1:
                    continue

                # cap number of adds
                if base.pyr_adds_done >= PYR_MAX_ADDS_CAP:
                    base.pyr_ceased = True
                    continue

                # if either condition fails once => cease forever for this base
                if not (ok_vol and ok_rsi):
                    base.pyr_ceased = True
                    continue

                # must have ATR for the pyramid leg at this bar
                if not (np.isfinite(atr) and atr > 0):
                    continue

                # capital check
                if not can_open_position(current_capital, trade_size, open_positions=len(positions)):
                    break

                # open next pyramid leg
                level = base.pyr_adds_done + 1
                posid = f"{base.pid}_PYR{level}"

                pos = open_position(
                    posid, ts, entry_price=c, atr_entry=atr, k=k, t=t, trade_size=trade_size,
                    base_id=base.pid, is_pyramid=True, pyr_level=level
                )
                positions[posid] = pos
                opens_count += 1
                base.pyr_adds_done += 1

                open_cnt = len(positions)
                avail = max_avail_slots(current_capital, trade_size)

                log_line(
                    ts, f"PYR{level}", pair, c,
                    extra=f"| PosID {posid:<18} | RSI {rsi_now:5.2f}>{rsi_prev:5.2f} | "
                        f"VR {vr_now:>4.2f}>={pyr_vol_min:.2f} | Port {open_cnt:02d}/{avail:02d}",
                    color=COLOR_BLUE
                )

        # 3) equity snapshot (realized only)
        equity_rows.append({
            "time": ts,
            "capital_usdt": current_capital,
            "open_positions": len(positions),
        })

    # 4) end-of-window forced closes (optional)
    open_positions_end = len(positions)
    sim_counts = {
        "opens_count": opens_count,
        "closes_count": closes_count,
        "open_positions_end": open_positions_end,
    }
    return pd.DataFrame(trades), pd.DataFrame(equity_rows), sim_counts


def summarize_trades(trades_df: pd.DataFrame, label: str) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {"label": label, "trades": 0, "net_profit": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0}

    net = float(trades_df["net_pnl_usdt"].sum())
    wins = trades_df[trades_df["net_pnl_usdt"] > 0]["net_pnl_usdt"]
    losses = trades_df[trades_df["net_pnl_usdt"] <= 0]["net_pnl_usdt"]

    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    wr = float((trades_df["net_pnl_usdt"] > 0).mean() * 100.0)
    avg = float(trades_df["net_pnl_usdt"].mean())

    return {"label": label, "trades": int(len(trades_df)), "net_profit": net, "win_rate": wr, "profit_factor": pf, "avg_pnl": avg}


def pick_best_mode_for_scenario(
    scenario: str,
    summary_baseline: Dict[str, Any],
    summary_barrier: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Best-of(baseline, barrier) for a single scenario.
    Primary: higher net_profit
    Tie-breakers: higher profit_factor, higher win_rate, higher avg_pnl, more trades
    """
    a = dict(summary_baseline)
    b = dict(summary_barrier)

    a["scenario"] = scenario
    b["scenario"] = scenario

    for k in ["profit_over_maxdd", "net_profit", "profit_factor", "win_rate", "avg_pnl", "trades"]:
        a[k] = float(a.get(k, 0) or 0)
        b[k] = float(b.get(k, 0) or 0)

    def score(x: Dict[str, Any]) -> tuple:
        return (
            x["profit_over_maxdd"],
            x["net_profit"],
            x["profit_factor"],
            x["win_rate"],
            x["avg_pnl"],
            x["trades"],
        )

    return a if score(a) >= score(b) else b


def choose_winner_across_scenarios(best_a1: Dict[str, Any], best_c0: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare best-of per scenario and pick winner.
    Uses same scoring rules as pick_best_mode_for_scenario.
    """
    def score(x: Dict[str, Any]) -> tuple:
        return (
            float(x.get("profit_over_maxdd", 0) or 0),
            float(x.get("net_profit", 0) or 0),
            float(x.get("profit_factor", 0) or 0),
            float(x.get("win_rate", 0) or 0),
            float(x.get("avg_pnl", 0) or 0),
            float(x.get("trades", 0) or 0),
        )

    return best_a1 if score(best_a1) >= score(best_c0) else best_c0
          
def choose_winner_across_candidates(best_list: list[Dict[str, Any]], *, pair_label: str) -> Dict[str, Any]:
    """
    Robust winner selection (hard constraints + maximize profit).

    Rule:
      - If no candidate passes constraints, DO NOT pick a winner for this pair/week.
        (Caller should treat this as "no trade".)
    """
    if not best_list:
        raise ValueError("best_list is empty")

    # ----------------------------
    # Robustness constraints (tuneable)
    # ----------------------------
    WIN_MIN_TRADES = 80
    WIN_MIN_PROFIT_FACTOR = 1.25
    WIN_MAX_ABS_DD_USDT = 800.0   # must have max_dd_usdt >= -800
    WIN_REQUIRE_POSITIVE_NET = True

    def passes_constraints(x: Dict[str, Any]) -> bool:
        trades = float(x.get("trades", 0) or 0)
        net = float(x.get("net_profit", 0) or 0)
        pf = float(x.get("profit_factor", 0) or 0)
        dd = float(x.get("max_dd_usdt", 0) or 0)  # negative drawdown

        if trades < WIN_MIN_TRADES:
            return False
        if pf < WIN_MIN_PROFIT_FACTOR:
            return False
        if dd < -WIN_MAX_ABS_DD_USDT:
            return False
        if WIN_REQUIRE_POSITIVE_NET and net <= 0:
            return False
        return True

    eligible = [x for x in best_list if passes_constraints(x)]

    if not eligible:
        print("\n" + "=" * 100)
        print(f"[WINNER][GATE] {pair_label}: NO TRADE THIS WEEK — no candidate met winner constraints "
              f"(min_trades={WIN_MIN_TRADES}, min_pf={WIN_MIN_PROFIT_FACTOR}, "
              f"max_abs_dd_usdt={WIN_MAX_ABS_DD_USDT}, require_pos_net={WIN_REQUIRE_POSITIVE_NET}).")
        print("=" * 100)
        return None

    # If constraints pass: maximize net_profit (primary), then profit_over_maxdd, then PF, then win_rate.
    def score_profit_first(x: Dict[str, Any]) -> tuple:
        return (
            float(x.get("net_profit", 0) or 0),
            float(x.get("profit_over_maxdd", 0) or 0),
            float(x.get("profit_factor", 0) or 0),
            float(x.get("win_rate", 0) or 0),
            float(x.get("avg_pnl", 0) or 0),
            float(x.get("trades", 0) or 0),
        )

    winner = eligible[0]
    for b in eligible[1:]:
        if score_profit_first(b) >= score_profit_first(winner):
            winner = b
    return winner

def run_one_scenario_both_modes(
    *,
    pair: str,
    scenario: str,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    ohlcv: pd.DataFrame,
    d_features: pd.DataFrame,
    k: float,
    t: float,
    x_bars: int,
    initial_capital: float,
    trade_size: float,
) -> Dict[str, Any]:
    """
    Runs baseline + barrier for one scenario and returns:
      - events
      - trades/eq for both modes
      - summaries for both modes
      - best-of selection for that scenario
    """
    events = build_events(d_features, trade_start, trade_end, scenario)
    
    events_all = events  # default (non-override): full-range events

    # --- OVERRIDE: replace events with CSV times if configured ---
    if EVENTS_CSV_OVERRIDE:
        forced_times = load_event_times_from_csv(EVENTS_CSV_OVERRIDE)

        events_all = pd.DataFrame({"event_time": sorted(forced_times)})
        events_all["event_time"] = pd.to_datetime(events_all["event_time"], utc=True)

        events = events_all[(events_all["event_time"] >= trade_start) &
                            (events_all["event_time"] < trade_end)].reset_index(drop=True)

        print(f"[events override] Using events from CSV: {EVENTS_CSV_OVERRIDE} | "
              f"events_all={len(events_all)} | events_trade_window={len(events)}")
    else:
        events_all = events
    # --- END OVERRIDE ---
    
    # ADD THESE LINES HERE
    print(f"[DEBUG] EVENTS USED FOR ENTRY (show first 10):")
    print(events.head(10).to_string(index=False))
    print(f"[DEBUG] TOTAL EVENTS USED: {len(events)}")
    
    trades_base, eq_base, _ = run_portfolio_sim(
        mode="baseline",
        pair=pair,
        scenario=scenario,
        ohlcv=ohlcv,
        d_features=d_features,
        events=events,
        trade_start=trade_start,
        trade_end=trade_end,
        k=k,
        t=t,
        x_bars=x_bars,
        initial_capital=initial_capital,
        trade_size=trade_size,
    )

    trades_barr, eq_barr, _ = run_portfolio_sim(
        mode="barrier",
        pair=pair,
        scenario=scenario,
        ohlcv=ohlcv,
        d_features=d_features,
        events=events,
        trade_start=trade_start,
        trade_end=trade_end,
        k=k,
        t=t,
        x_bars=x_bars,
        initial_capital=initial_capital,
        trade_size=trade_size,
    )

    s_base = summarize_trades(trades_base, f"{scenario}-baseline")
    dd_pct_base, dd_usdt_base = compute_max_drawdown(eq_base, "capital_usdt")
    s_base["max_dd_pct"] = dd_pct_base
    s_base["max_dd_usdt"] = dd_usdt_base
    s_base["profit_over_maxdd"] = (s_base["net_profit"] / abs(dd_usdt_base)) if dd_usdt_base != 0 else float("inf")

    s_barr = summarize_trades(trades_barr, f"{scenario}-barrier")
    dd_pct_barr, dd_usdt_barr = compute_max_drawdown(eq_barr, "capital_usdt")
    s_barr["max_dd_pct"] = dd_pct_barr
    s_barr["max_dd_usdt"] = dd_usdt_barr
    s_barr["profit_over_maxdd"] = (s_barr["net_profit"] / abs(dd_usdt_barr)) if dd_usdt_barr != 0 else float("inf")
    best = pick_best_mode_for_scenario(scenario, s_base, s_barr)

    return dict(
        scenario=scenario,
        events=events,
        trades_baseline=trades_base,
        trades_barrier=trades_barr,
        equity_baseline=eq_base,
        equity_barrier=eq_barr,
        summary_baseline=s_base,
        summary_barrier=s_barr,
        best=best,
    )


def strip_tz(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
                df[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return df

def regime_score_block(
    *,
    label: str,
    d_features: pd.DataFrame,
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adx_col: str = "adx_15m",
    adx_min: float = ADX_MIN,
) -> pd.DataFrame:
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)

    w = d_features[(d_features["time"] >= start) & (d_features["time"] < end)].copy()
    if w.empty or adx_col not in w.columns:
        return pd.DataFrame([{
            "window": label,
            "scope": "all_minutes",
            "minutes": 0,
            "pct_adx_ge_min": 0.0,
            "adx_p10": np.nan,
            "adx_p25": np.nan,
            "adx_p50": np.nan,
            "adx_p75": np.nan,
            "adx_p90": np.nan,
        }])

    adx_all = pd.to_numeric(w[adx_col], errors="coerce").dropna()
    minutes_all = int(len(adx_all))
    pct_all = float((adx_all >= adx_min).mean() * 100.0) if minutes_all else 0.0

    def q(s: pd.Series, p: float) -> float:
        return float(s.quantile(p)) if len(s) else np.nan

    rows = [{
        "window": label,
        "scope": "all_minutes",
        "minutes": minutes_all,
        "pct_adx_ge_min": pct_all,
        "adx_p10": q(adx_all, 0.10),
        "adx_p25": q(adx_all, 0.25),
        "adx_p50": q(adx_all, 0.50),
        "adx_p75": q(adx_all, 0.75),
        "adx_p90": q(adx_all, 0.90),
    }]

    # SIGNAL-time stats
    if events is not None and (not events.empty) and ("event_time" in events.columns):
        ev = events.copy()
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
        ev = ev[(ev["event_time"] >= start) & (ev["event_time"] < end)].copy()

        adx_map = w.set_index("time")[adx_col].to_dict()
        ev["adx_at_signal"] = ev["event_time"].map(adx_map)
        adx_sig = pd.to_numeric(ev["adx_at_signal"], errors="coerce").dropna()

        minutes_sig = int(len(adx_sig))
        pct_sig = float((adx_sig >= adx_min).mean() * 100.0) if minutes_sig else 0.0

        rows.append({
            "window": label,
            "scope": "signal_minutes",
            "minutes": minutes_sig,
            "pct_adx_ge_min": pct_sig,
            "adx_p10": q(adx_sig, 0.10),
            "adx_p25": q(adx_sig, 0.25),
            "adx_p50": q(adx_sig, 0.50),
            "adx_p75": q(adx_sig, 0.75),
            "adx_p90": q(adx_sig, 0.90),
        })

    return pd.DataFrame(rows)

def print_regime_score(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if df is None or df.empty:
        print("(no regime score)")
    else:
        print(df.to_string(index=False))

def get_exit_params_from_finalist(f: Dict[str, Any]) -> tuple[float, float, int]:
    exitp = f.get("exit_params", {}) or {}
    return float(exitp["k"]), float(exitp["t"]), int(exitp["x_bars"])

def main():
    pair = input("Pair (e.g. ACTUSDT): ").strip().upper()
    pair_label = pair
    run_mode_raw = input("Mode (MANUAL / AUTO_CYCLE): ")
    run_mode = (run_mode_raw or "").strip().upper().replace("-", "_") or "MANUAL"
    print(f"[DEBUG] run_mode_raw={run_mode_raw!r}  run_mode={run_mode!r}", flush=True)

    # -----------------------------
    # Schedule anchor: PrePaper start (Rule A)
    # -----------------------------
    prepaper_start_str = input("PrePaper START Monday (UTC) [YYYY-MM-DD]: ").strip()
    pre_start = pd.to_datetime(prepaper_start_str + " 00:00", utc=True)

    if pre_start.weekday() != 0:
        raise ValueError(f"PrePaper START must be a Monday date. Got: {pre_start}")

    pre_end = pre_start + pd.Timedelta(days=7)
    trade_end = pre_start
    trade_start = trade_end - pd.Timedelta(days=7)

    train_end = trade_start
    train_start = train_end - pd.Timedelta(days=30)

    print("\n" + "=" * 100)
    print(f"[SCHEDULE] TRAIN (UTC):    {train_start} -> {train_end}   (30d)")
    print(f"[SCHEDULE] TRADE (UTC):    {trade_start} -> {trade_end}   (7d)")
    print(f"[SCHEDULE] PREPAPER (UTC): {pre_start} -> {pre_end}   (7d)")
    print("=" * 100)

    if run_mode == "AUTO_":
        print("\n[AUTO] Enter tuned params from Excel study for EACH scenario.")
        k_a1 = float(input("A1 k (ATR-mult for fixed stop): ").strip())
        t_a1 = float(input("A1 t (ATR-mult for trailing dist): ").strip())
        x_a1 = int(input("A1 x_bars (barrier activation bars): ").strip())

        k_c0 = float(input("C0 k (ATR-mult for fixed stop): ").strip())
        t_c0 = float(input("C0 t (ATR-mult for trailing dist): ").strip())
        x_c0 = int(input("C0 x_bars (barrier activation bars): ").strip())
    else:
        # v30-driven candidates (non-interactive):
        # - Loads finalists from candidate_for_TRADE.json
        # - Picks a candidate via env var CANDIDATE, otherwise defaults to rank #1 (finalists[0])
        # - Uses the k/t/x_bars determined by the quantile selection code (already stored in JSON)
        import json, os

        CANDIDATE_TRADE_JSON = os.getenv("CANDIDATE_TRADE_JSON", "candidate_for_TRADE.json")
        with open(CANDIDATE_TRADE_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)

        finalists = d.get("finalists", [])
        if not finalists:
            raise ValueError(f"No finalists found in {CANDIDATE_TRADE_JSON}")
        
        if run_mode != "AUTO_CYCLE":
            AUTO_TOP_N = int(os.getenv("AUTO_TOP_N", "3"))  # cycle top N finalists in AUTO_CYCLE

            allowed = [f["possibility"] for f in finalists]
            requested = os.getenv("CANDIDATE", "").strip()

            if requested:
                match = next((f for f in finalists if f["possibility"] == requested), None)
                if match is None:
                    raise ValueError(f"Invalid CANDIDATE='{requested}'. Allowed: {allowed}")
                chosen = match
            else:
                chosen = finalists[0]  # default: best-ranked finalist

            scenario = chosen["possibility"]  # keep variable name 'scenario' so the rest of v6 still works
            k = float(chosen["exit_params"]["k"])
            t = float(chosen["exit_params"]["t"])
            x_bars = int(chosen["exit_params"]["x_bars"])

            print(f"Scenario (candidate possibility): {scenario}")
            print(f"k (ATR-mult for fixed stop): {k}")
            print(f"t (ATR-mult for trailing dist): {t}")
            print(f"x_bars (barrier activation bars): {x_bars}")

    initial_capital = DEFAULT_INITIAL_CAPITAL
    trade_size = DEFAULT_TRADE_SIZE

    print("\n" + "=" * 100)
    if run_mode == "AUTO_CYCLE":
        print(f"AUTO_CYCLE TRADE WINDOW (UTC): {trade_start} -> {trade_end} | Pair={pair}")
    else:
        print(f"TRADE WINDOW (UTC): {trade_start} -> {trade_end} | Pair={pair} | Scenario={scenario}")
    print(f"[PORT] initial=${initial_capital:,.2f} | trade_size=${trade_size:,.2f} | maxAvail starts at {int(initial_capital // trade_size)}")
    print("=" * 100)

    warmup_start = train_start - pd.Timedelta(days=WARMUP_DAYS)
    fetch_end = trade_end + (pd.Timedelta(days=7) if run_mode == "AUTO_CYCLE" else pd.Timedelta(days=0))
    print(f"[FETCH] {pair} {INTERVAL} (TRAIN warmup included): {warmup_start} -> {fetch_end}")

    ohlcv = get_ohlcv_binance(pair, warmup_start, fetch_end)
    if ohlcv.empty:
        print("No OHLCV fetched. Check pair/date.")
        return

    ohlcv["atr"] = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=ATR_LEN)
    ohlcv = ohlcv.dropna(subset=["atr"]).reset_index(drop=True)
    
    # ADD THIS:
    d_features = compute_entry_features(ohlcv)

    d = compute_entry_features(ohlcv)
    
    # -----------------------------
    # 15m ADX regime map (for gating) - attach for ALL modes (manual/auto + prepaper)
    # -----------------------------
    maps = {"adx_15m": {}, "dmp_15m": {}, "dmn_15m": {}}

    if ADX_GATE_ENABLE:
        maps = compute_adx_15m_maps(ohlcv)
        if not maps or not maps.get("adx_15m"):
            print("[ADX] WARNING: ADX/DI maps are empty; gates may block entries if enabled.")
        else:
            print("[ADX] 15m ADX/DI maps ready.")

        d["adx_15m"] = d["time"].map(maps.get("adx_15m", {}))
        d["dmp_15m"] = d["time"].map(maps.get("dmp_15m", {}))  # +DI
        d["dmn_15m"] = d["time"].map(maps.get("dmn_15m", {}))  # -DI
    else:
        d["adx_15m"] = np.nan
        d["dmp_15m"] = np.nan
        d["dmn_15m"] = np.nan

    if run_mode == "AUTO_CYCLE":

        # v30 finalists (self-contained load for AUTO_CYCLE)
        CANDIDATE_TRADE_JSON = os.getenv("CANDIDATE_TRADE_JSON", "candidate_for_TRADE.json")
        with open(CANDIDATE_TRADE_JSON, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        finalists = cand_data.get("finalists", [])
        if not finalists:
            raise ValueError(f"No finalists found in {CANDIDATE_TRADE_JSON}")

        AUTO_TOP_N = int(os.getenv("AUTO_TOP_N", "3"))
        cycle = finalists[:AUTO_TOP_N]  # ranked order from json (top N)
        print(f"[AUTO_CYCLE] AUTO_TOP_N={AUTO_TOP_N} | finalists={len(finalists)} | cycle_scenarios={[f['possibility'] for f in cycle]}", flush=True)

        # -----------------------------
        # TRADE WINDOW: run v30 finalists (C_* possibilities)
        # -----------------------------
        all_results = []
        all_summary_rows = []
        best_per_candidate = []
        params_by_scen = {str(f["possibility"]).strip().upper(): f for f in cycle}

        for f in cycle:
            scen = str(f["possibility"]).strip().upper()
            fk, ft, fx = get_exit_params_from_finalist(f)

            print("\n" + "-" * 100)
            print(f"[AUTO_CYCLE] Running finalist: {scen} | k={fk} t={ft} x_bars={fx}")

            res = run_one_scenario_both_modes(
                pair=pair, scenario=scen,
                trade_start=trade_start, trade_end=trade_end,
                ohlcv=ohlcv, d_features=d,
                k=fk, t=ft, x_bars=fx,
                initial_capital=initial_capital, trade_size=trade_size
            )

            all_results.append(res)
            all_summary_rows.append(res["summary_baseline"])
            all_summary_rows.append(res["summary_barrier"])
            best_per_candidate.append(res["best"])

        summary_trade = pd.DataFrame(all_summary_rows)

        print("\n" + "=" * 100)
        print("ALL CANDIDATES SUMMARY (TRADE WINDOW)")
        print("=" * 100)
        print(summary_trade.to_string(index=False))
            
        # -----------------------------
        # ROBUSTNESS (TRAIN walk-forward on finalists)
        # Gate R1b: all weeks net_profit >= ROBUST_WEEK_NET_MIN
        # Gate R2: worst-week net_profit > ROBUST_WORST_WEEK_NET_MIN
        # Ranking: median profit_over_maxdd, tie-break median net_profit
        # -----------------------------
        print(f"[DEBUG] ROBUST_ENABLE={ROBUST_ENABLE!r}")
        if ROBUST_ENABLE:
            print("[DEBUG] entered ROBUST block")
            train_slices = iter_monday_week_slices(train_start, trade_start)
            if len(train_slices) > ROBUST_TRAIN_SLICES:
                train_slices = train_slices[-ROBUST_TRAIN_SLICES:]

            print("\n" + "=" * 100)
            print("ROBUSTNESS CHECK (TRAIN weekly slices on finalists)")
            print("=" * 100)
            print(f"R1b (weekly_net>={ROBUST_WEEK_NET_MIN}): need all {len(train_slices)} weeks | "
                  f"R2 (worst_week_net>{ROBUST_WORST_WEEK_NET_MIN}): applied after R1b")
            for (w0, w1) in train_slices:
                print(f"  - {w0} -> {w1}")

            robust_rows = []

            # Build full-range events per scenario ONCE (so events_all is in-scope here)
            events_all_by_scen: dict[str, pd.DataFrame] = {}
            for cand in best_per_candidate:
                scen = str(cand["scenario"]).strip().upper()
                if scen not in events_all_by_scen:
                    events_all_by_scen[scen] = build_events_all_for_robustness(
                        d_features=d,
                        train_start=train_start,
                        trade_end=trade_end,
                        scenario=scen,
                    )

            for cand in best_per_candidate:
                scen = str(cand["scenario"]).strip().upper()
                exitp = (params_by_scen[scen].get("exit_params", {}) or {})
                k0 = float(exitp["k"])
                t0 = float(exitp["t"])
                x0 = int(exitp["x_bars"])

                rob = eval_candidate_robustness_over_train(
                    pair=pair,
                    scenario=scen,
                    ohlcv=ohlcv,
                    d_features=d,
                    events_all=events_all_by_scen[scen],  # <<< key fix
                    slices=train_slices,
                    k=k0,
                    t=t0,
                    x_bars=x0,
                    initial_capital=initial_capital,
                    trade_size=trade_size,
                )
                robust_rows.append(rob)

                print("\n" + "-" * 100)
                print(f"[ROBUST] {scen} | k={k0} t={t0} x={x0}")
                print(f"  R1b (weekly_net>={ROBUST_WEEK_NET_MIN}): {rob['pos_weeks']}/{rob['n_slices']} weeks pass (need all {rob['n_slices']})")
                print(f"  R2 (worst_week_net>{ROBUST_WORST_WEEK_NET_MIN}): {rob['worst_week_net_profit']:.2f}")
                print(f"     median_profit_over_maxdd: {rob['median_profit_over_maxdd']:.4f}")
                print(f"     median_net_profit: {rob['median_net_profit']:.2f}")
                    
                dfw = rob["df_slices"]
                if dfw is not None and not dfw.empty:
                    for _, row in dfw.iterrows():
                        w0 = row["slice_start"]
                        w1 = row["slice_end"]
                        nev = int(row.get("events_in_slice", 0))
                        nop = int(row.get("opens_in_slice", 0))
                        ncl = int(row.get("closes_in_slice", 0))
                        nopen_end = int(row.get("open_positions_end", 0))
                        net = float(row.get("net_profit", 0.0))
                        r1p = "PASS" if bool(row.get("r1_week_pass", False)) else "FAIL"
                        r2p = "PASS" if bool(row.get("r2_week_pass", False)) else "FAIL"
                        print(f"   - W{int(row['slice_ix'])} {w0} -> {w1} | events={nev:3d} | opens={nop:3d} | closes={ncl:3d} | open_end={nopen_end:2d} | net={net:+10.2f} | R1b_week={r1p} | R2_week={r2p}")

            # Apply gates (ROBUST)                
            gated = [
                r for r in robust_rows
                if bool(r.get("all_weeks_pass", False))  # R1b: 4/4 weeks net >= ROBUST_WEEK_NET_MIN
                and (float(r["worst_week_net_profit"]) > float(ROBUST_WORST_WEEK_NET_MIN))  # R2
            ]

            if not gated:
                print("\n" + "=" * 100)
                print(f"[ROBUST][GATE] {pair_label}: NO TRADE THIS WEEK — no finalist passed TRAIN robustness gates (R1b 4/4 + R2).")
                print(f"R1b (weekly_net>={ROBUST_WEEK_NET_MIN}): need all {len(train_slices)} weeks to pass")
                print(f"R2  (worst_week_net>{ROBUST_WORST_WEEK_NET_MIN}): worst week net must exceed threshold")
                print("=" * 100)
                return

            gated_sorted = sorted(
                gated,
                key=lambda r: (float(r["median_profit_over_maxdd"]), float(r["median_net_profit"])),
                reverse=True,
            )
            robust_pick = gated_sorted[0]

            print("\n" + "=" * 100)
            print("ROBUST WINNER (TRAIN walk-forward)")
            print("=" * 100)
            print(f"scenario={robust_pick['scenario']}")
            print(f"median_profit_over_maxdd={robust_pick['median_profit_over_maxdd']:.4f}")
            print(f"median_net_profit={robust_pick['median_net_profit']:.2f}")
            print(f"pos_weeks={robust_pick['pos_weeks']}/{robust_pick['n_slices']}")
            print(f"worst_week_net_profit={robust_pick['worst_week_net_profit']:.2f}")

            # Restrict finalists to robust scenario only before Trade-window final selection
            best_per_candidate = [
                x for x in best_per_candidate
                if str(x["scenario"]).strip().upper() == str(robust_pick["scenario"]).strip().upper()
            ]

        # Winner across candidates (same scoring as your choose_winner_across_scenarios)
        winner = choose_winner_across_candidates(best_per_candidate, pair_label=pair_label)
        if not winner:
            return
        win_scenario = str(winner["scenario"]).strip().upper()
        win_params = params_by_scen[win_scenario]
        exitp = win_params.get("exit_params", {}) or {}
        win_k = float(exitp["k"])
        win_t = float(exitp["t"])
        win_x = int(exitp["x_bars"])

        print("\n" + "=" * 100)
        print("WINNER (BEST-OF PER CANDIDATE)")
        print("=" * 100)

        print(f"scenario           : {winner.get('scenario')}")
        print(f"label              : {winner.get('label')}")
        print(f"trades             : {winner.get('trades')}")
        print(f"net_profit         : {winner.get('net_profit')}")
        print(f"win_rate           : {winner.get('win_rate')}")
        print(f"profit_factor      : {winner.get('profit_factor')}")
        print(f"avg_pnl            : {winner.get('avg_pnl')}")
        print(f"max_dd_pct         : {winner.get('max_dd_pct')}")
        print(f"max_dd_usdt        : {winner.get('max_dd_usdt')}")
        print(f"profit_over_maxdd  : {winner.get('profit_over_maxdd')}")
            
        # --- save trade window workbook ---
        ident = f"{trade_start.strftime('%Y-%m-%d_%H%M')}_to_{trade_end.strftime('%Y-%m-%d_%H%M')}"
        out_trade = os.path.join(OUT_DIR, f"forwardtest_TRADEWINDOW_7d_ALLCANDS_{ident}_{pair}.xlsx")
        with pd.ExcelWriter(out_trade, engine="openpyxl") as w:
            summary_trade.to_excel(w, sheet_name="summary_all_candidates", index=False)

        print(f"\nSaved: {out_trade}")
            
        # -----------------------------
        # PREPAPER (winner only): user-provided Monday 08:00 UTC for 7 days
        # -----------------------------
            
        print("\n" + "=" * 100)
                        
        # -----------------------------
        # PREPAPER (winner only): reuse same fetched ohlcv + features
        # -----------------------------
        print("\n" + "=" * 100)
        print(f"PREPAPER WINDOW (UTC): {pre_start} -> {pre_end} | Pair={pair} | Scenario={win_scenario}")
        print("=" * 100)

        res_pre = run_one_scenario_both_modes(
            pair=pair,
            scenario=win_scenario,
            trade_start=pre_start,
            trade_end=pre_end,
            ohlcv=ohlcv,
            d_features=d,
            k=win_k,
            t=win_t,
            x_bars=win_x,
            initial_capital=initial_capital,
            trade_size=trade_size,
        )

        summary_pre = pd.DataFrame([res_pre["summary_baseline"], res_pre["summary_barrier"]])

        print("\n" + "=" * 100)
        print("PREPAPER SUMMARY (WINNER ONLY)")
        print("=" * 100)
        print(summary_pre.to_string(index=False))

        best_pre = res_pre["best"]
        print("\n" + "=" * 100)
        print("PREPAPER WINNER MODE (baseline vs barrier)")
        print("=" * 100)
        print(f"scenario           : {best_pre.get('scenario')}")
        print(f"label              : {best_pre.get('label')}")
        print(f"trades             : {best_pre.get('trades')}")
        print(f"net_profit         : {best_pre.get('net_profit')}")
        print(f"win_rate           : {best_pre.get('win_rate')}")
        print(f"profit_factor      : {best_pre.get('profit_factor')}")
        print(f"avg_pnl            : {best_pre.get('avg_pnl')}")
        print(f"max_dd_pct         : {best_pre.get('max_dd_pct')}")
        print(f"max_dd_usdt        : {best_pre.get('max_dd_usdt')}")
        print(f"profit_over_maxdd  : {best_pre.get('profit_over_maxdd')}")

        ident_pre = f"{pre_start.strftime('%Y-%m-%d_%H%M')}_to_{pre_end.strftime('%Y-%m-%d_%H%M')}"
        out_pre = os.path.join(OUT_DIR, f"forwardtest_PREPAPER_7d_WINNER_{ident_pre}_{win_scenario}_{pair}.xlsx")

        with pd.ExcelWriter(out_pre, engine="openpyxl") as w:
            summary_pre.to_excel(w, sheet_name="summary", index=False)
            strip_tz(res_pre["events"].copy(), ["event_time"]).to_excel(w, sheet_name="events", index=False)
            strip_tz(res_pre["trades_baseline"].copy(), ["entry_time", "exit_time"]).to_excel(w, sheet_name="trades_baseline", index=False)
            strip_tz(res_pre["trades_barrier"].copy(), ["entry_time", "exit_time"]).to_excel(w, sheet_name="trades_barrier", index=False)
            strip_tz(res_pre["equity_baseline"].copy(), ["time"]).to_excel(w, sheet_name="equity_baseline", index=False)
            strip_tz(res_pre["equity_barrier"].copy(), ["time"]).to_excel(w, sheet_name="equity_barrier", index=False)

        print(f"\nSaved PrePaper workbook: {out_pre}")

        # IMPORTANT: stop here so we don't continue into MANUAL mode simulation below
        return

if __name__ == "__main__":
    main()