import pandas as pd
import pandas_ta as ta
import numpy as np

# -----------------------------
# V30 EVENT STUDY (ACTUSDT_1m.csv)
# -----------------------------
# Output format:
#   - each ROW = event_time (timestamp of RSI_SMA cross-up)
#   - each COLUMN = metric/field
#
# Event definition (cross-up):
#   prev rsi_sma <= 51 and current rsi_sma > 51
#
# For each event:
#   (2) record entry_close and entry_atr (ATR(14))
#   (3) scan forward until Close < entry_close (first time), then:
#       - highest High reached before that
#       - ATR multiple from entry_close to that high
#       - time from event_time to that max high
#
# NEW requested fields (at event candle):
#   - close_gt_smma_200: TRUE/FALSE  (close > SMMA_200)
#   - vol_gt_vol_sma: TRUE/FALSE    (volume > volume_SMA)
# -----------------------------

CSV_PATH = "ACTUSDT_1m.csv"

RSI_LEN = 14
RSI_SMA_LEN = 14
ATR_LEN = 14
RSI_SMA_LEVEL = 51.0

SMMA_LEN = 200           # as requested
VOL_SMA_LEN = 20         # default for "vol_sma" (change if you want)

STOP_ON_CLOSE_BELOW_ENTRY = True

TIME_COL_CANDIDATES = ["open_time", "open time", "time", "timestamp", "date", "datetime"]

TRADE_SIZE_USDT = 1000.0
FEE_RATE_PER_SIDE = 0.001  # 0.1% per side

def load_1m_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize headers: lower + strip
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify time column
    time_col = None
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(
            f"No time column found. Expected one of: {TIME_COL_CANDIDATES}. "
            f"Got columns: {list(df.columns)}"
        )

    df = df.rename(columns={time_col: "time"})

    # Parse time (your sample is ISO with +00:00, this handles it)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    # Clean numeric columns
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        raise ValueError("Missing required column: volume")

    # Your sample shows volume may be "-" -> coerce to NaN then fill with 0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    return df


def safe_round(value, decimals=8):
    """Round a value to specified decimals, returning np.nan for NaN values."""
    return round(float(value), decimals) if pd.notna(value) else np.nan


def wilders_rma(series: pd.Series, length: int) -> pd.Series:
    """
    Compute Wilder's RMA (also known as SMMA - Smoothed Moving Average).
    Compatible with TradingView's RMA behavior.
    
    Formula: RMA(length) = EWM with alpha=1/length, adjust=False
    
    This is equivalent to the recursive formula:
        RMA[i] = (RMA[i-1] * (length-1) + value[i]) / length
    
    Args:
        series: Input price series
        length: RMA period length
        
    Returns:
        Series with RMA values
    """
    return series.ewm(alpha=1/length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = ta.rsi(df["close"], length=RSI_LEN)
    df["rsi_sma"] = ta.sma(df["rsi"], length=RSI_SMA_LEN)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)

    # SMMA(200) - Wilder's RMA/Smoothed Moving Average
    # NOTE: Prior versions incorrectly used SMA (rolling mean) instead of SMMA.
    # This change aligns with Wilder's RMA/SMMA used in TradingView and elsewhere in this repo.
    df["smma_200"] = wilders_rma(df["close"], SMMA_LEN)

    # Volume SMA for vol > vol_sma
    df["vol_sma"] = df["volume"].rolling(window=VOL_SMA_LEN).mean()

    return df


def cross_up_mask(df: pd.DataFrame) -> pd.Series:
    prev = df["rsi_sma"].shift(1)
    curr = df["rsi_sma"]
    return (prev <= RSI_SMA_LEVEL) & (curr > RSI_SMA_LEVEL)


def analyze_events(df: pd.DataFrame) -> pd.DataFrame:
    mask = cross_up_mask(df)
    event_idxs = df.index[mask].tolist()

    rows = []
    for idx in event_idxs:
        entry_time = df.at[idx, "time"]
        entry_close = float(df.at[idx, "close"])
        entry_atr = df.at[idx, "atr"]
        entry_rsi_sma = df.at[idx, "rsi_sma"]

        smma_200 = df.at[idx, "smma_200"]
        vol_sma = df.at[idx, "vol_sma"]
        volume = float(df.at[idx, "volume"])

        close_gt_smma_200 = bool(pd.notna(smma_200) and (entry_close > float(smma_200)))
        vol_gt_vol_sma = bool(pd.notna(vol_sma) and (volume > float(vol_sma)))

        if pd.notna(vol_sma) and float(vol_sma) > 0:
            vol_ratio = volume / float(vol_sma)
        else:
            vol_ratio = np.nan

        vol_ratio_ge_15 = bool(pd.notna(vol_ratio) and vol_ratio >= 1.5)

        # Skip events before ATR/RSI is warmed up
        if pd.isna(entry_atr) or float(entry_atr) == 0 or pd.isna(entry_rsi_sma):
            continue

        entry_atr = float(entry_atr)
        entry_rsi_sma = float(entry_rsi_sma)

        max_high = float(df.at[idx, "high"])
        max_high_time = entry_time

        # NEW: track min low in the SAME window you already use
        min_low = float(df.at[idx, "low"])
        min_low_time = entry_time

        stop_idx = None

        # Scan forward until stop condition (close below entry)
        for j in range(idx + 1, len(df)):
            h = float(df.at[j, "high"])
            if h > max_high:
                max_high = h
                max_high_time = df.at[j, "time"]

            # NEW: update min low during the same scan
            l = float(df.at[j, "low"])
            if l < min_low:
                min_low = l
                min_low_time = df.at[j, "time"]

            if STOP_ON_CLOSE_BELOW_ENTRY:
                if float(df.at[j, "close"]) < entry_close:
                    stop_idx = j
                    break

        open_ended = stop_idx is None
        end_idx = (len(df) - 1) if open_ended else stop_idx
        stop_time = df.at[end_idx, "time"]

        time_to_max_high_min = (max_high_time - entry_time).total_seconds() / 60.0
        time_to_stop_min = (stop_time - entry_time).total_seconds() / 60.0

        atr_multiple_to_max = (max_high - entry_close) / entry_atr

        # NEW: adverse excursion (how far price went below entry) in ATR units
        # If min_low >= entry_close, this becomes <= 0; clamp to 0 for readability.
        atr_multiple_to_min = (entry_close - min_low) / entry_atr
        atr_multiple_to_min = max(0.0, atr_multiple_to_min)

        raw_move = max_high - entry_close

        # --- PnL model (fixed trade size, fees on both buy & sell) ---
        buy_px = entry_close
        sell_px = max_high

        qty = TRADE_SIZE_USDT / buy_px
        gross_pnl_usdt = (sell_px - buy_px) * qty

        buy_fee_usdt = (TRADE_SIZE_USDT) * FEE_RATE_PER_SIDE
        sell_notional_usdt = sell_px * qty
        sell_fee_usdt = sell_notional_usdt * FEE_RATE_PER_SIDE

        net_pnl_usdt = gross_pnl_usdt - buy_fee_usdt - sell_fee_usdt

        # Compute audit columns for debugging/validation
        close_minus_smma_200 = (entry_close - float(smma_200)) if pd.notna(smma_200) else np.nan
        vol_minus_vol_sma = (volume - float(vol_sma)) if pd.notna(vol_sma) else np.nan

        rows.append(
            {
                "event_time": entry_time,

                "entry_rsi_sma": entry_rsi_sma,

                "close_gt_smma_200": close_gt_smma_200,
                "vol_gt_vol_sma": vol_gt_vol_sma,
                "vol_ratio": safe_round(vol_ratio, 4),
                "vol_ratio_ge_15": vol_ratio_ge_15,

                "entry_close": entry_close,
                "entry_atr": entry_atr,

                # Audit columns for verifying calculations
                "smma_200": safe_round(smma_200, 8),
                "vol_sma": safe_round(vol_sma, 8),
                "close_minus_smma_200": safe_round(close_minus_smma_200, 8),
                "vol_minus_vol_sma": safe_round(vol_minus_vol_sma, 8),

                "max_high_before_stop": max_high,
                "max_high_time": max_high_time,
                "time_to_max_high_min": round(time_to_max_high_min, 2),

                # NEW fields
                "min_low_before_stop": min_low,
                "min_low_time": min_low_time,
                "atr_multiple_to_min": round(atr_multiple_to_min, 4),

                "stop_time": stop_time,
                "time_to_stop_min": round(time_to_stop_min, 2),
                "open_ended": bool(open_ended),

                "atr_multiple_to_max": round(atr_multiple_to_max, 4),
                "raw_move": raw_move,

                "qty": qty,
                "buy_fee_usdt": round(buy_fee_usdt, 6),
                "sell_fee_usdt": round(sell_fee_usdt, 6),
                "net_pnl_usdt": round(net_pnl_usdt, 6),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("event_time").set_index("event_time")
    return out


def summarize(out: pd.DataFrame) -> None:
    print("\n=== V30 Funnel Event Study: RSI_SMA cross-up > 51 ===")
    print(f"Events (after indicator warmup): {len(out)}")

    if out.empty:
        return

    # Self-check: Report NaN counts for audit columns
    print("\n--- Indicator Warmup Check ---")
    if "smma_200" in out.columns:
        nan_count = out["smma_200"].isna().sum()
        print(f"smma_200 NaN count: {nan_count} / {len(out)} ({100*nan_count/len(out):.1f}%)")
    if "vol_sma" in out.columns:
        nan_count = out["vol_sma"].isna().sum()
        print(f"vol_sma NaN count: {nan_count} / {len(out)} ({100*nan_count/len(out):.1f}%)")

    print("\n--- Filter counts ---")
    print(f"close_gt_smma_200 TRUE: {int(out['close_gt_smma_200'].sum())} / {len(out)}")
    print(f"vol_gt_vol_sma   TRUE: {int(out['vol_gt_vol_sma'].sum())} / {len(out)}")
    print(f"both TRUE: {int((out['close_gt_smma_200'] & out['vol_gt_vol_sma']).sum())} / {len(out)}")
    print(f"both TRUE + vol_ratio_ge_15 TRUE: {int((out['close_gt_smma_200'] & out['vol_gt_vol_sma'] & out['vol_ratio_ge_15']).sum())} / {len(out)}")

    print("\n--- ATR multiple to max (summary) ---")
    print(out["atr_multiple_to_max"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    
    # --- Net PnL (USDT, trade size $1000, fees 0.1% per side) ---
    print("\n--- Net PnL (USDT, trade size $1000, fees 0.1% per side) ---")

    def pnl_sum(df, mask, label):
        sub = df[mask]
        total = float(sub["net_pnl_usdt"].sum()) if len(sub) else 0.0
        avg = float(sub["net_pnl_usdt"].mean()) if len(sub) else 0.0
        print(f"{label}: trades={len(sub)} | total_net_pnl_usdt={total:.2f} | avg_net_pnl_usdt={avg:.2f}")

    BOTH_TRUE = out["close_gt_smma_200"] & out["vol_gt_vol_sma"]

    pnl_sum(out, pd.Series(True, index=out.index), "ALL events (RSI_SMA cross-up > 51)")
    pnl_sum(out, BOTH_TRUE, "both TRUE (close>smma_200 AND vol>vol_sma)")
    pnl_sum(out, BOTH_TRUE & out["vol_ratio_ge_15"], "both TRUE + vol_ratio_ge_15")

def main():
    df = load_1m_csv(CSV_PATH)
    df = compute_indicators(df)

    out = analyze_events(df)

    # overwrite previous file name (as you requested you renamed the first version)
    out_path = "v30_eventstudy_ACTUSDT_1m_rsi_sma_cross_gt51.csv"
    out.to_csv(out_path, index=True)

    summarize(out)
    print(f"\nSaved: {out_path}")
    if not out.empty:
        print("\nFirst 10 events (preview):")
        print(out.head(10).to_string())


if __name__ == "__main__":
    main()