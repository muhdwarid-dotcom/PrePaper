# PrePaper — End-to-End Trading Pipeline

## Overview

PrePaper is a quantitative pipeline that screens Binance perpetual pairs, runs an
event study to find high-probability setups, derives optimal exit parameters, and
feeds them into a live-forward trade window.

## Pipeline Stages

```
Stage 1A/1B  →  Stage 2  →  EventStudy (Steps 1-3)  →  Trade Window
```

| Stage | Script | Output |
|---|---|---|
| Stage 1A + 1B | `Optimizer_Stage1A_1B_v29R_CLEAN.py` | `results_v29R_30d/stage1B_behavior.csv` |
| Stage 2 (dual-TF optimizer) | `Optimizer_Stage2_v29R_DualTF_CLEAN.py` | `results_v29R_30d/stage2_intraday_dual_tf_improved.csv` |
| Step 1 — EventStudy fetch | `Funnel_Data_Test_V30_EventStudy.py` | `forwardtest/v30_eventstudy_{PAIR}_{INTERVAL}_*.csv` |
| Step 2 — EventStudy analysis | `eventstudy_analysis.py` | `forwardtest/eventstudy_list_summary_{PAIR}_{INTERVAL}_*.csv` |
| Step 3 — Exit param derivation | `Derive_k_t_from_PQ_windows.py` | `candidate_exit_params_{PAIR}_prepaper_{DATE}.json` |
| Trade window | `7_day_trade_window_forward_livefetch_v6+PrePaper.py` | Live trades |

## Quick Start — Single Command After Stage 2

After running Stage 1A/1B and Stage 2, use the glue script to run the full
eventstudy pipeline for the top-N symbols, automatically using each symbol's
`best_tf` (1m or 3m) as selected by the Stage 2 optimizer:

```bash
python run_screened_pipeline.py --prepaper-start 2025-12-01
```

This will:
1. Read `results_v29R_30d/stage2_intraday_dual_tf_improved.csv`
2. Select the top 10 symbols by `score_final`
3. For each symbol, run Steps 1–3 of the eventstudy pipeline using `best_tf`

### Options

```
--stage2-csv PATH         Path to Stage2 CSV (default: results_v29R_30d/stage2_intraday_dual_tf_improved.csv)
--prepaper-start DATE     PrePaper window start, e.g. 2025-12-01  [required]
--top-n N                 Number of symbols to process (default: 10)
--out-dir DIR             Output directory for eventstudy CSVs (default: forwardtest)
--dry-run                 Print commands without executing them
```

Example — process top 5, custom Stage2 CSV:

```bash
python run_screened_pipeline.py \
    --stage2-csv results_v29R_30d/stage2_intraday_dual_tf_improved.csv \
    --prepaper-start 2025-12-01 \
    --top-n 5
```

## Running Each Step Manually

### Step 1 — Generate Events CSV

```bash
# 1-minute data (default)
python Funnel_Data_Test_V30_EventStudy.py --pair ACTUSDT --prepaper-start 2025-12-01

# 3-minute data
python Funnel_Data_Test_V30_EventStudy.py --pair SHIBUSDT --prepaper-start 2025-12-01 --interval 3m
```

Output: `forwardtest/v30_eventstudy_{PAIR}_{INTERVAL}_rsi_sma_cross_gt51_prepaper_{DATE}.csv`

### Step 2 — Analyse Events

```bash
# 1-minute
python eventstudy_analysis.py \
    forwardtest/v30_eventstudy_ACTUSDT_1m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv \
    --pair ACTUSDT --prepaper-start 2025-12-01 --interval 1m --grid

# 3-minute
python eventstudy_analysis.py \
    forwardtest/v30_eventstudy_SHIBUSDT_3m_rsi_sma_cross_gt51_prepaper_2025-12-01.csv \
    --pair SHIBUSDT --prepaper-start 2025-12-01 --interval 3m --grid
```

Outputs:
- `forwardtest/eventstudy_list_summary_{PAIR}_{INTERVAL}_prepaper_{DATE}.csv`
- `forwardtest/top20_view_{PAIR}_{INTERVAL}_prepaper_{DATE}.csv`

### Step 3 — Derive Exit Parameters

```bash
# 1-minute (timeframe-minutes defaults to 1 when --interval 1m)
python Derive_k_t_from_PQ_windows.py --pair ACTUSDT --prepaper-start 2025-12-01 --interval 1m

# 3-minute (timeframe-minutes defaults to 3 when --interval 3m)
python Derive_k_t_from_PQ_windows.py --pair SHIBUSDT --prepaper-start 2025-12-01 --interval 3m
```

Output: `candidate_exit_params_{PAIR}_prepaper_{DATE}.json`

### Step 4 — Trade Window

```bash
cp candidate_exit_params_ACTUSDT_prepaper_2025-12-01.json candidate_for_TRADE.json
python 7_day_trade_window_forward_livefetch_v6+PrePaper.py
```

## Interval Support

Only `1m` and `3m` intervals are currently supported by the eventstudy pipeline.
The Stage 2 optimizer selects the better interval per symbol (`best_tf` column).

| Interval | `--interval` | `--timeframe-minutes` |
|---|---|---|
| 1 minute | `1m` | `1` |
| 3 minutes | `3m` | `3` |

## Environment Setup

```bash
pip install python-binance python-dotenv tenacity pyyaml pandas pandas_ta numpy requests
```

Copy `.env.example` to `.env` and fill in your Binance API credentials:

```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

> **Note:** The eventstudy pipeline (Steps 1–3) fetches data via the public Binance
> REST API and does **not** require API credentials. Only Stage 1A/1B and Stage 2
> require credentials (they use the private Binance Python client).

## Output Files

Generated output files are **not committed** to the repository.
Add `forwardtest/`, `results_v29R_30d/`, and `candidate_exit_params_*.json` to
`.gitignore` to keep the repository clean.
