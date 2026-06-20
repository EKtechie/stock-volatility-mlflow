"""
data_ingestion.py

Fetches historical OHLCV data for every ticker from Yahoo Finance.
No CSV, no manual downloads — data is pulled live via the yfinance API.

Input  : list of ticker symbols, start date, end date (from config.py)
Output : single long-format DataFrame with columns:
         Date · Open · High · Low · Close · Volume · Ticker

Design decisions:
    - Download one ticker at a time so a single failure doesn't kill the run
    - auto_adjust=True so stock splits don't look like price crashes
    - Flatten MultiIndex columns — newer yfinance versions return them
"""

import sys
import pandas as pd
import yfinance as yf

from src.config import TICKERS, START_DATE, END_DATE

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance >= 0.2.38 sometimes returns a MultiIndex column structure
    like (Close, RELIANCE.NS) instead of just Close.
    This flattens it back to single-level column names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_stock_data(
    tickers: list = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV for each ticker and combine into one DataFrame.

    Why long format (one row per ticker per date)?
    Easier to groupby Ticker in feature engineering —
    each ticker gets its own rolling windows without cross-contamination.

    Args:
        tickers : list of Yahoo Finance ticker symbols
        start   : start date string "YYYY-MM-DD"
        end     : end date string "YYYY-MM-DD"

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume, Ticker
    """
    print(f"\n{'='*60}")
    print("  STEP 1 — DATA INGESTION (Yahoo Finance)")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Period  : {start}  →  {end}\n")

    frames = []    # will hold one DataFrame per ticker
    failed = []    # track which tickers didn't load

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # adjusts for splits and dividends
                progress=False,     # suppress download bar in terminal
            )

            if raw.empty:
                # Yahoo returned nothing — bad symbol or no data in range
                print(f"  ⚠  {ticker:<18} no data returned, skipping.")
                failed.append(ticker)
                continue

            raw = _flatten_columns(raw)
            raw["Ticker"] = ticker

            # reset_index moves Date from index to a regular column
            raw.reset_index(inplace=True)

            # older yfinance names it "index" after reset — normalise it
            if "index" in raw.columns:
                raw.rename(columns={"index": "Date"}, inplace=True)

            raw["Date"] = pd.to_datetime(raw["Date"])

            # keep only the columns we need — drop anything extra yfinance adds
            frames.append(
                raw[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]
            )

            print(f"  ✓  {ticker:<18} {len(raw):>5,} rows")

        except Exception as exc:
            # network blip or bad ticker — log and continue
            print(f"  ✗  {ticker:<18} {exc}", file=sys.stderr)
            failed.append(ticker)

    if not frames:
        raise RuntimeError(
            "No data fetched. Check: (1) internet connection, "
            "(2) ticker symbols are valid, (3) date range contains trading days."
        )

    # combine all tickers into one DataFrame, sorted by ticker then date
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Fetched    : {df['Ticker'].nunique()} tickers")
    print(f"  Total rows : {len(df):,}")
    print(f"  Date range : {df['Date'].min().date()}  →  {df['Date'].max().date()}")

    if failed:
        print(f"  Skipped    : {failed}")

    return df

if __name__ == "__main__":
    # run this file directly to verify data fetching works
    # before touching feature engineering or training
    df = fetch_stock_data()
    print(f"\n  Sample:\n{df.head(3)}")
    print(f"\n  Tickers in data: {df['Ticker'].unique()}")