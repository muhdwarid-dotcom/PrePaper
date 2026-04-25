#!/usr/bin/env python3
"""
run_screened_pipeline.py — Glue script for the v30 eventstudy pipeline.

Reads the Stage2 output CSV, selects the top-N symbols by ``score_final``,
and for each symbol runs the three-step eventstudy pipeline using the
``best_tf`` chosen by the optimizer (1m or 3m):

  Step 1 — Funnel_Data_Test_V30_EventStudy.py  (fetch + generate events CSV)
  Step 2 — eventstudy_analysis.py              (compute grid metrics / candidates)
  Step 3 — Derive_k_t_from_PQ_windows.py       (derive k, t, x_bars exit params)

USAGE:
  python run_screened_pipeline.py --prepaper-start 2025-12-01
  python run_screened_pipeline.py --prepaper-start 2025-12-01 --top-n 5
  python run_screened_pipeline.py \\
      --stage2-csv results_v29R_30d/stage2_intraday_dual_tf_improved.csv \\
      --prepaper-start 2025-12-01 --top-n 10

The script prints each sub-command before running it so you can reproduce
individual steps manually if needed.

Supported intervals: 1m, 3m  (as written in the Stage2 ``best_tf`` column).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_STAGE2_CSV = "results_v29R_30d/stage2_intraday_dual_tf_improved.csv"
DEFAULT_TOP_N = 10
SUPPORTED_INTERVALS = {"1m", "3m"}


def _interval_to_minutes(interval: str) -> int:
    """Map interval string to its bar duration in minutes."""
    return {"1m": 1, "3m": 3}.get(interval, 1)


def _run(cmd: list[str]) -> int:
    """Print and execute *cmd*; return the subprocess exit code."""
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def load_stage2(csv_path: str) -> pd.DataFrame:
    """Load and validate the Stage2 CSV, raising SystemExit on errors."""
    path = Path(csv_path)
    if not path.exists():
        print(
            f"Error: Stage2 CSV not found: {path}\n"
            f"Run Optimizer_Stage2_v29R_DualTF_CLEAN.py first to generate it, "
            f"or supply a different path with --stage2-csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(path)

    required_cols = {"symbol", "best_tf", "score_final"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"Error: Stage2 CSV is missing required columns: {missing}\n"
            f"Expected columns: {required_cols}",
            file=sys.stderr,
        )
        sys.exit(1)

    return df


def select_top_symbols(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return the top-N rows sorted by score_final descending."""
    df = df.dropna(subset=["score_final"]).sort_values("score_final", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def run_symbol(symbol: str, interval: str, prepaper_start: str, extra_funnel_args: list[str]) -> bool:
    """
    Run the 3-step pipeline for one symbol/interval combination.

    Returns True if all steps succeeded, False if any step failed.
    """
    tf_minutes = _interval_to_minutes(interval)

    print(f"\n{'='*70}")
    print(f"  Symbol : {symbol}")
    print(f"  Interval: {interval}  ({tf_minutes} min/bar)")
    print(f"  PrePaper: {prepaper_start}")
    print(f"{'='*70}")

    # Step 1 — generate events CSV
    print("\n[Step 1] Funnel_Data_Test_V30_EventStudy.py")
    rc = _run([
        sys.executable, "Funnel_Data_Test_V30_EventStudy.py",
        "--pair", symbol,
        "--prepaper-start", prepaper_start,
        "--interval", interval,
        *extra_funnel_args,
    ])
    if rc != 0:
        print(f"  ERROR: Step 1 failed for {symbol} ({interval}) — exit code {rc}", file=sys.stderr)
        return False

    # Step 2 — eventstudy analysis
    events_csv = (
        f"forwardtest/v30_eventstudy_{symbol}_{interval}"
        f"_rsi_sma_cross_gt51_prepaper_{prepaper_start}.csv"
    )
    print("\n[Step 2] eventstudy_analysis.py")
    rc = _run([
        sys.executable, "eventstudy_analysis.py",
        events_csv,
        "--pair", symbol,
        "--prepaper-start", prepaper_start,
        "--interval", interval,
        "--grid",
    ])
    if rc != 0:
        print(f"  ERROR: Step 2 failed for {symbol} ({interval}) — exit code {rc}", file=sys.stderr)
        return False

    # Step 3 — derive exit params
    print("\n[Step 3] Derive_k_t_from_PQ_windows.py")
    rc = _run([
        sys.executable, "Derive_k_t_from_PQ_windows.py",
        "--pair", symbol,
        "--prepaper-start", prepaper_start,
        "--interval", interval,
        "--timeframe-minutes", str(tf_minutes),
    ])
    if rc != 0:
        print(f"  ERROR: Step 3 failed for {symbol} ({interval}) — exit code {rc}", file=sys.stderr)
        return False

    print(f"\n  ✓ {symbol} ({interval}) — all steps completed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Glue script: reads Stage2 CSV, selects top-N symbols by score_final, "
            "and runs the v30 eventstudy pipeline (Steps 1-3) for each symbol "
            "using its best_tf (1m or 3m) from Stage2."
        )
    )
    parser.add_argument(
        "--stage2-csv",
        default=DEFAULT_STAGE2_CSV,
        help=(
            f"Path to Stage2 output CSV (default: {DEFAULT_STAGE2_CSV}). "
            "Must contain columns: symbol, best_tf, score_final."
        ),
    )
    parser.add_argument(
        "--prepaper-start",
        required=True,
        metavar="YYYY-MM-DD",
        help="PrePaper window start date (00:00 UTC), e.g. 2025-12-01.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top symbols to process by score_final (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--out-dir",
        default="forwardtest",
        help="Output directory for eventstudy CSVs (default: forwardtest).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    args = parser.parse_args()

    # Load Stage2 results
    stage2_df = load_stage2(args.stage2_csv)
    top_df = select_top_symbols(stage2_df, args.top_n)

    if top_df.empty:
        print("Error: No symbols found after filtering Stage2 CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Stage2 CSV   : {args.stage2_csv}")
    print(f"PrePaper start: {args.prepaper_start}")
    print(f"Top-N symbols : {len(top_df)}")
    print()
    print(top_df[["symbol", "best_tf", "score_final"]].to_string(index=False))

    extra_funnel = ["--out-dir", args.out_dir]

    # Run pipeline for each selected symbol
    failed: list[str] = []
    for _, row in top_df.iterrows():
        symbol = str(row["symbol"]).upper()
        interval = str(row["best_tf"]).strip().lower()

        if interval not in SUPPORTED_INTERVALS:
            print(
                f"Warning: Unsupported interval '{interval}' for {symbol} — skipping. "
                f"Supported: {sorted(SUPPORTED_INTERVALS)}",
                file=sys.stderr,
            )
            failed.append(f"{symbol} (unsupported interval '{interval}')")
            continue

        if args.dry_run:
            tf_minutes = _interval_to_minutes(interval)
            events_csv = (
                f"{args.out_dir}/v30_eventstudy_{symbol}_{interval}"
                f"_rsi_sma_cross_gt51_prepaper_{args.prepaper_start}.csv"
            )
            print(f"\n[DRY-RUN] {symbol} ({interval})")
            print(
                "  $",
                " ".join([
                    sys.executable, "Funnel_Data_Test_V30_EventStudy.py",
                    "--pair", symbol, "--prepaper-start", args.prepaper_start,
                    "--interval", interval, *extra_funnel,
                ]),
            )
            print(
                "  $",
                " ".join([
                    sys.executable, "eventstudy_analysis.py",
                    events_csv, "--pair", symbol,
                    "--prepaper-start", args.prepaper_start,
                    "--interval", interval, "--grid",
                ]),
            )
            print(
                "  $",
                " ".join([
                    sys.executable, "Derive_k_t_from_PQ_windows.py",
                    "--pair", symbol, "--prepaper-start", args.prepaper_start,
                    "--interval", interval, "--timeframe-minutes", str(tf_minutes),
                ]),
            )
            continue

        ok = run_symbol(symbol, interval, args.prepaper_start, extra_funnel)
        if not ok:
            failed.append(f"{symbol} ({interval})")

    # Summary
    print(f"\n{'='*70}")
    total = len(top_df)
    if args.dry_run:
        print(f"Dry-run complete — {total} symbol(s) would be processed.")
    elif not failed:
        print(f"All {total} symbol(s) processed successfully.")
        print("\nNext step: copy/rename each candidate_exit_params_*.json to")
        print("  candidate_for_TRADE.json  and run:")
        print("  python 7_day_trade_window_forward_livefetch_v6+PrePaper.py")
    else:
        succeeded = total - len(failed)
        print(f"{succeeded}/{total} symbol(s) succeeded.")
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
