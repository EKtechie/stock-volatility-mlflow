"""
feature_engineering.py

Transforms raw OHLCV into a feature matrix for volatility forecasting.

Input  : long-format DataFrame from data_ingestion.py
Output : same DataFrame with 50+ feature columns + target_vol column

Target:
    5-day forward annualised realised volatility.
    At row T, target_vol = std(log returns T+1 to T+5) * sqrt(252)
    All features are computed from data up to T — zero look-ahead bias.

Feature groups:
    1. Technical indicators  — RSI, MACD, Bollinger Bands, ATR
    2. Price context         — SMA ratios, high/low spread, open/close gap
    3. Volume                — short and long run volume ratios
    4. Return stats          — rolling mean, std, skew, kurtosis (4 windows)
    5. Volatility history    — lag features + rolling min/mean/max
    6. Regime flags          — high vol regime, bull/bear trend
    7. Calendar              — day of week, month, quarter, week of year
    8. Ticker encoding       — ordinal so tree models can split on ticker
"""

import numpy as np
import pandas as pd

from src.config import (
    ANNUALISE_FACTOR,
    ATR_WINDOW,
    BB_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    ROLLING_WINDOWS,
    RSI_WINDOW,
    VOL_LAG_DAYS,
    VOLATILITY_WINDOW,
)

# ── Indicator helpers ─────────────────────────────────────────────────────────
# Each function takes a pandas Series and returns a pandas Series.
# They are private (_prefix) — only used inside this file.


def _rsi(series: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """
    Relative Strength Index — measures momentum.

    Formula: RSI = 100 - (100 / (1 + avg_gain / avg_loss))

    Why it matters for vol: overbought (RSI > 70) and oversold (RSI < 30)
    conditions often precede reversals. Reversals = volatility spikes.

    Why window=14: Wilder's original default. ~3 trading weeks.
    Short enough to react, long enough to filter daily noise.
    """
    delta = series.diff()

    # split moves into gains and losses — negative values become 0
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()

    rs = gain / (loss + 1e-9)    # 1e-9 prevents division by zero on flat days
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    """
    Moving Average Convergence Divergence — captures trend and momentum.

    Returns three series: macd_line, signal_line, histogram
        macd_line  = fast EMA - slow EMA
        signal     = EMA of macd_line (smoothed)
        histogram  = macd_line - signal (acceleration of trend)

    Why EMA not SMA: EMA weights recent prices more heavily.
    In fast markets, last week matters more than 3 weeks ago.

    Why these periods (12, 26, 9): industry standard since Gerald Appel's
    original 1970s paper. Still widely used because everyone else uses them
    — self-fulfilling signals in liquid markets.
    """
    ema_fast   = series.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow   = series.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram  = macd_line - signal
    return macd_line, signal, histogram


def _bollinger(series: pd.Series, window: int = BB_WINDOW):
    """
    Bollinger Bands — measures volatility relative to price level.

    Returns two series: bb_width, bb_position
        bb_width    = (upper - lower) / SMA  — how wide the bands are
        bb_position = (close - lower) / (upper - lower)  — where in the band

    Why bb_width matters: narrow bands = vol compression = breakout pending.
    After a squeeze, vol typically expands sharply. This is the 'Bollinger Squeeze'.

    Why bb_position matters: like RSI but in a different mathematical space.
    Near 1.0 = overbought, near 0.0 = oversold. Two features, different angles.

    Why window=20: one trading month. Standard across most trading systems.
    """
    sma      = series.rolling(window).mean()
    std      = series.rolling(window).std()
    upper    = sma + 2 * std
    lower    = sma - 2 * std
    width    = (upper - lower) / (sma + 1e-9)
    position = (series - lower) / (upper - lower + 1e-9)
    return width, position


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = ATR_WINDOW,
) -> pd.Series:
    """
    Average True Range — measures intraday volatility including gaps.

    True range = max of:
        1. High - Low              (intraday range)
        2. |High - prev Close|     (gap up scenario)
        3. |Low  - prev Close|     (gap down scenario)

    Why not just High - Low: a stock can gap up overnight and trade in a
    narrow range all day. High-Low misses the gap entirely. ATR captures it.

    Why window=14: Wilder's default, same reasoning as RSI.
    """
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window).mean()

# ── Per-ticker feature construction ───────────────────────────────────────────

def _build_features(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for a single ticker.
    Called via groupby — receives one ticker's rows, sorted by Date.

    Why a separate function per ticker:
    Rolling windows must not cross ticker boundaries.
    RELIANCE's 20-day SMA should only use RELIANCE's prices.
    groupby().apply() enforces this automatically.
    """
    g = grp.copy().sort_values("Date").reset_index(drop=True)

    close  = g["Close"]
    high   = g["High"]
    low    = g["Low"]
    volume = g["Volume"]

    # ── Log returns ───────────────────────────────────────────────────────────
    # Log returns are additive across time and approximately normally distributed.
    # Simple returns are multiplicative — harder to use in rolling statistics.
    log_ret      = np.log(close / close.shift(1))
    g["log_return"] = log_ret

    # ── Target — forward realised volatility ──────────────────────────────────
    # Step 1: backward-looking realised vol at time T
    rvol = log_ret.rolling(VOLATILITY_WINDOW).std() * np.sqrt(ANNUALISE_FACTOR)
    g["realized_vol"] = rvol

    # Step 2: shift backward to make it forward-looking
    # At row T, target_vol = vol of the NEXT 5 days — what we predict.
    # Negative shift = future value at current row. No leakage.
    g["target_vol"] = rvol.shift(-VOLATILITY_WINDOW)

    # ── Technical indicators ──────────────────────────────────────────────────
    g["rsi_14"]      = _rsi(close)

    macd, signal, hist = _macd(close)
    g["macd"]        = macd
    g["macd_signal"] = signal
    g["macd_hist"]   = hist    # histogram = momentum acceleration

    bb_width, bb_pos  = _bollinger(close)
    g["bb_width"]     = bb_width
    g["bb_position"]  = bb_pos

    atr              = _atr(high, low, close)
    g["atr_14"]      = atr
    g["atr_pct"]     = atr / (close + 1e-9)    # normalise by price level

    # ── Price context ─────────────────────────────────────────────────────────
    # How far is price from its own moving averages?
    # Positive = price above SMA (trend up), negative = below (trend down)
    g["price_vs_sma20"]  = close / (close.rolling(20).mean()  + 1e-9) - 1
    g["price_vs_sma50"]  = close / (close.rolling(50).mean()  + 1e-9) - 1
    g["price_vs_sma200"] = close / (close.rolling(200).mean() + 1e-9) - 1

    # intraday range as fraction of price — normalised ATR without smoothing
    g["hl_spread"] = (high - low) / (close + 1e-9)

    # did price close higher or lower than it opened?
    g["oc_gap"]    = (close - g["Open"]) / (g["Open"] + 1e-9)

    # ── Volume ────────────────────────────────────────────────────────────────
    # Volume spikes often precede or accompany vol spikes.
    # We measure volume relative to its own average — not absolute numbers.
    # A volume of 1M shares means nothing without knowing the normal level.
    g["vol_sma5_ratio"]  = volume / (volume.rolling(5).mean()  + 1e-9)
    g["vol_sma20_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

    # ── Rolling return statistics (4 time horizons) ───────────────────────────
    # Why 4 windows: short windows (5d) react fast but are noisy.
    # Long windows (63d) are stable but lag. The model learns which horizon
    # is most predictive — we don't need to decide upfront.
    for w in ROLLING_WINDOWS:
        roll = log_ret.rolling(w)
        g[f"ret_mean_{w}d"] = roll.mean()    # direction of returns
        g[f"ret_std_{w}d"]  = roll.std()     # magnitude of moves
        g[f"ret_skew_{w}d"] = roll.skew()    # asymmetry — more down days or up days?
        g[f"ret_kurt_{w}d"] = roll.kurt()    # fat tails — any extreme moves recently?

    # ── Volatility lag features ───────────────────────────────────────────────
    # Directly encodes volatility clustering: high vol today → high vol tomorrow.
    # Positive shift = past value at current row. These are features, not targets.
    for lag in VOL_LAG_DAYS:
        g[f"rvol_lag_{lag}d"] = rvol.shift(lag)

    # ── Rolling volatility statistics ─────────────────────────────────────────
    # shift(1) before rolling — use yesterday's vol to compute rolling stats.
    # Prevents the current row's vol from leaking into its own feature.
    lagged_rvol = rvol.shift(1)
    for w in ROLLING_WINDOWS:
        g[f"rvol_mean_{w}d"] = lagged_rvol.rolling(w).mean()
        g[f"rvol_max_{w}d"]  = lagged_rvol.rolling(w).max()
        g[f"rvol_min_{w}d"]  = lagged_rvol.rolling(w).min()

    # ── Regime flags ──────────────────────────────────────────────────────────
    # Binary features that tell the model what market state it is in.
    # Why 63 days: 1 quarter — long enough to define a regime, not a daily noise.
    # Tree models use these to partition feature space: "if high vol regime,
    # apply these rules; if low vol regime, apply different rules."
    g["high_vol_regime"]  = (rvol > rvol.rolling(63).mean()).astype(int)
    g["bull_trend"]       = (close > close.rolling(63).mean()).astype(int)

    # rate of change of vol — is vol accelerating or decelerating?
    g["vol_acceleration"] = rvol - rvol.shift(5)

    # ── Calendar features ─────────────────────────────────────────────────────
    # Vol has known seasonal patterns:
    # - Mondays often see higher vol (weekend news digested)
    # - January and October historically see elevated vol
    # - Q4 earnings season drives vol higher
    dt = pd.to_datetime(g["Date"])
    g["day_of_week"]  = dt.dt.dayofweek           # 0=Monday, 4=Friday
    g["month"]        = dt.dt.month
    g["quarter"]      = dt.dt.quarter
    g["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    return g

# ── Public API ────────────────────────────────────────────────────────────────

# Columns that are raw inputs or intermediate calculations — not model features.
# We exclude them when building the feature matrix for training.
_DROP_COLS = {
    "Date", "Open", "High", "Low", "Close", "Volume",
    "Ticker", "log_return", "realized_vol", "target_vol",
}


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply the full feature pipeline to raw OHLCV data.

    Why groupby apply:
    Each ticker must be processed independently. Rolling windows
    for TCS should not include rows from RELIANCE. groupby().apply()
    passes one ticker's data at a time to _build_features().

    Args:
        df : raw OHLCV DataFrame from fetch_stock_data()

    Returns:
        enriched_df  : DataFrame with all features + target_vol
        feature_cols : list of column names to use as model inputs
    """
    print(f"\n{'='*60}")
    print("  STEP 2 — FEATURE ENGINEERING")
    print(f"{'='*60}")

    enriched = (
        df.groupby("Ticker", group_keys=False)
        .apply(_build_features)
        .reset_index(drop=True)
    )

    # ordinal encode ticker — tree models can split on this
    # tells the model "AAPL behaves differently from ^NSEI"
    enriched["ticker_code"] = (
        enriched["Ticker"].astype("category").cat.codes
    )

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    # Rolling windows create NaN for the first N rows of each ticker.
    # A 200-day SMA needs 200 rows before it produces a value.
    # Rows without a valid target are also dropped (last 5 rows per ticker).
    n_before = len(enriched)
    enriched.dropna(subset=["target_vol"], inplace=True)
    enriched.dropna(inplace=True)
    n_after  = len(enriched)

    feature_cols = [c for c in enriched.columns if c not in _DROP_COLS]

    print(f"  Rows before cleaning : {n_before:,}")
    print(f"  Rows after  cleaning : {n_after:,}  (dropped {n_before - n_after:,})")
    print(f"  Features             : {len(feature_cols)}")
    print(f"  Target               : {VOLATILITY_WINDOW}-day forward annualised realised vol")

    return enriched, feature_cols

if __name__ == "__main__":
    # test feature engineering in isolation
    # run: python src/feature_engineering.py
    from src.data_ingestion import fetch_stock_data

    raw        = fetch_stock_data()
    df, feats  = engineer_features(raw)

    print(f"\n  Feature list ({len(feats)} total):")
    for i, f in enumerate(feats, 1):
        print(f"    {i:>2}. {f}")

    print(f"\n  Target stats:")
    print(df["target_vol"].describe().round(4))