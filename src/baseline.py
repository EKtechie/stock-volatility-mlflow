"""
baseline.py

Establishes the performance floor before any real model is trained.

Why baselines matter:
    A model that scores R²=0.75 sounds impressive until you find out
    that just copying yesterday's volatility scores R²=0.72.
    Baselines tell you how much your model actually learned vs
    how much was already in the data trivially.

Three baselines in order of increasing intelligence:
    1. Dummy (mean)    — predict the average every time, ignore all features
    2. Dummy (median)  — same but robust to vol spikes skewing the mean
    3. Persistence     — predict tomorrow's vol = today's vol (uses rvol_lag_1d)
    4. LinearRegression — simple ML baseline, uses all features linearly

Your XGBoost/LightGBM/CatBoost must beat ALL of these to add real value.
The persistence model is the hardest to beat — vol clustering is strong.
"""

import sys
import os

# make sure Python finds src/ when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data_ingestion import fetch_stock_data
from src.feature_engineering import engineer_features


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and print four regression metrics for one model.

    Why four metrics, not just one:
        RMSE  — penalises large errors heavily (good for risk — big mistakes matter)
        MAE   — average error in same units as vol (easy to explain to stakeholders)
        R²    — what fraction of variance does the model explain (0=nothing, 1=perfect)
        MAPE  — percentage error (useful when comparing across different vol levels)

    Args:
        name   : model name for display
        y_true : actual target values
        y_pred : model predictions

    Returns:
        dict of metric name → value
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # MAPE — Mean Absolute Percentage Error
    # 1e-9 prevents division by zero if target is exactly 0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    print(f"  {name:<30} "
          f"RMSE={rmse:.5f}  "
          f"MAE={mae:.5f}  "
          f"R²={r2:.4f}  "
          f"MAPE={mape:.2f}%")

    return {
        "model": name,
        "rmse" : rmse,
        "mae"  : mae,
        "r2"   : r2,
        "mape" : mape,
    }


# ── Train / test split ────────────────────────────────────────────────────────

def _split(df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.20):
    """
    Chronological 80/20 split — same split used in train.py.

    Why chronological and not random:
        Random split lets the model see 2023 data during training
        and then be 'tested' on 2021 data. That is not forecasting —
        it is interpolation. Chronological split tests true out-of-sample
        forecasting ability.
    """
    df_s  = df.sort_values("Date").reset_index(drop=True)
    split = int(len(df_s) * (1 - test_size))

    train = df_s.iloc[:split]
    test  = df_s.iloc[split:]

    X_tr = train[feature_cols].values
    y_tr = train["target_vol"].values
    X_te = test[feature_cols].values
    y_te = test["target_vol"].values

    print(f"  Train : {train['Date'].min().date()} → "
          f"{train['Date'].max().date()} ({len(train):,} rows)")
    print(f"  Test  : {test['Date'].min().date()} → "
          f"{test['Date'].max().date()} ({len(test):,} rows)\n")

    return X_tr, X_te, y_tr, y_te, test


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_baselines() -> list[dict]:
    """
    Run all baselines and print a comparison table.
    Call this before train.py — the numbers here become your benchmark.
    """
    raw_df      = fetch_stock_data()
    features_df, feature_cols = engineer_features(raw_df)

    print(f"\n  Target variable check:")
    print(features_df["target_vol"].describe().round(4))
    print(f"  Negative values : {(features_df['target_vol'] < 0).sum()}")
    print(f"  NaN count       : {features_df['target_vol'].isna().sum()}")
    print(f"  Values above 2.0: {(features_df['target_vol'] > 2.0).sum()}")

    print(f"\n{'='*60}")
    print("  BASELINES")
    print(f"{'='*60}\n")

    X_tr, X_te, y_tr, y_te, test_df = _split(features_df, feature_cols)

    results = []

    # ── 1. Dummy (mean) ───────────────────────────────────────────────────────
    # Predicts the training set mean for every single row.
    # Ignores all features completely.
    # If your real model barely beats this, your features are useless.
    # Expected R²: ~0 or slightly negative (by definition of R²)
    dummy_mean = DummyRegressor(strategy="mean")
    dummy_mean.fit(X_tr, y_tr)
    results.append(evaluate("Dummy (mean)", y_te, dummy_mean.predict(X_te)))

    # ── 2. Dummy (median) ─────────────────────────────────────────────────────
    # Predicts the training set median instead of mean.
    # Why median: vol distributions are right-skewed — crash spikes pull
    # the mean up. Median is more representative of the typical vol level.
    dummy_med = DummyRegressor(strategy="median")
    dummy_med.fit(X_tr, y_tr)
    results.append(evaluate("Dummy (median)", y_te, dummy_med.predict(X_te)))

    # ── 3. Persistence ────────────────────────────────────────────────────────
    # Predicts tomorrow's vol = yesterday's vol (rvol_lag_1d feature).
    # This is the hardest baseline to beat — volatility clusters strongly.
    # If XGBoost can't beat this, your 50 features added nothing beyond
    # what was already obvious in the raw vol series.
    # Expected R²: 0.60–0.75 (strong baseline due to clustering)
    if "rvol_lag_1d" in test_df.columns:
        persistence_pred = test_df["rvol_lag_1d"].values

        # some rows may be NaN at the boundary — mask them out
        mask = ~np.isnan(persistence_pred)
        results.append(
            evaluate("Persistence (rvol lag 1d)", y_te[mask], persistence_pred[mask])
        )
    else:
        print("  ⚠  rvol_lag_1d not found — check feature engineering output.")

    # ── 4. Linear Regression ──────────────────────────────────────────────────
    # First model that actually uses all features.
    # Establishes the ceiling of linear relationships in the data.
    # Gap between this and XGBoost = value of nonlinearity.
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    results.append(evaluate("LinearRegression", y_te, lr.predict(X_te)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Your MLflow models must beat Persistence R² to justify complexity.")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    results = run_baselines()

    # show sorted leaderboard
    df_res = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    df_res.index += 1
    print("\n  Baseline Leaderboard (sorted by RMSE):")
    print(df_res.to_string(index=True))

