"""
train.py

Trains 6 regression models, logs every experiment to MLflow, then
registers the best model in the MLflow Model Registry.

Why MLflow:
    Without it you end up with model_v3_final_FINAL.pkl and no record
    of what hyperparameters or features produced which results.
    MLflow tracks params, metrics, plots, and the model itself —
    every run is reproducible and auditable.

What gets logged per run:
    Params   : model hyperparameters + dataset metadata
    Tags     : data source, tickers, target definition
    Metrics  : train & test RMSE, MAE, R², MAPE
    Artifacts: feature importance plot, actual-vs-predicted plot
    Model    : serialised pipeline with input/output signature
"""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="catboost")

from src.config import (
    CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    TEST_SIZE,
    TICKERS,
    VOLATILITY_WINDOW,
)

# matplotlib needs this backend when running without a display (no GUI)
matplotlib.use("Agg")

# ── Model definitions ─────────────────────────────────────────────────────────
# Ordered from simplest to most complex. Each model answers a different
# question: does the data have linear structure? Does regularisation help?
# Does nonlinearity help? Does boosting beat bagging?

_MODELS = {
    "LinearRegression": LinearRegression(),

    "Ridge": Ridge(alpha=100.0),
    # alpha=1.0 — moderate L2 penalty. Shrinks coefficients on correlated
    # features (many of our rolling stats are correlated with each other).

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,        # prevents overfitting on 50+ features
        min_samples_leaf=10, # each leaf needs 10+ samples — smooths predictions
        n_jobs=-1,            # use all CPU cores
        random_state=42,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=5,          # shallow trees — boosting needs weak learners
        learning_rate=0.05,   # slow learning, more trees = better generalisation
        subsample=0.8,        # row sampling — reduces overfitting
        colsample_bytree=0.8, # column sampling — reduces overfitting
        random_state=42,
        verbosity=0,
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        num_leaves=63,        # LightGBM grows leaf-wise, not depth-wise
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    ),

    "CatBoost": CatBoostRegressor(
        iterations=100,
        depth=4,
        learning_rate=0.03,
        random_seed=42,
        verbose=0,
        l2_leaf_reg=5            
    ),
}


def _build_pipeline(name: str, model) -> Pipeline:
    """
    Linear models need scaled features — tree models don't.

    Why: LinearRegression and Ridge are sensitive to feature scale.
    A feature ranging 0-100 vs a feature ranging 0-0.001 distorts
    the optimisation and makes Ridge's penalty uneven across features.

    Tree models split on thresholds like 'is X > 0.5' — scale doesn't
    affect where that threshold lands, so scaling is unnecessary there.
    """
    if name in {"LinearRegression", "Ridge"}:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])

def _log_model(pipeline: Pipeline, model_name: str, signature, input_example):
    """
    Routes model logging to the correct MLflow flavor.

    Why not always use mlflow.sklearn.log_model:
    MLflow's sklearn flavor serialises models with skops by default —
    a safer alternative to pickle that only trusts pure scikit-learn
    object types. XGBoost, LightGBM, and CatBoost wrap their own native
    booster objects inside a sklearn-compatible API; skops correctly
    flags these as untrusted and refuses to serialise them.

    The fix is to use each library's own MLflow flavor. This also
    preserves full model fidelity — the model is saved in its native
    format and can be reloaded with that library's own inference
    engine, not just a generic sklearn predict() call.
    """
    inner_model = pipeline.named_steps["model"]

    if model_name == "XGBoost":
        mlflow.xgboost.log_model(
            inner_model, name="model",
            signature=signature, input_example=input_example,
        )
    elif model_name == "LightGBM":
        mlflow.lightgbm.log_model(
            inner_model, name="model",
            signature=signature, input_example=input_example,
        )
    elif model_name == "CatBoost":
        mlflow.catboost.log_model(
            inner_model, name="model",
            signature=signature, input_example=input_example,
        )
    else:
        # LinearRegression, Ridge, RandomForest — pure sklearn objects,
        # the default skops serialisation works without issue.
        # Log the full pipeline (includes the scaler for linear models).
        mlflow.sklearn.log_model(
            pipeline, name="model",
            signature=signature, input_example=input_example,
        )


# ── Train / test split ────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, feature_cols: list[str]):
    """
    Chronological 80/20 split. Same logic as baseline.py —
    train on the past, test on the future. No shuffling, ever.
    """
    df_s  = df.sort_values("Date").reset_index(drop=True)
    idx   = int(len(df_s) * (1 - TEST_SIZE))
    train = df_s.iloc[:idx]
    test  = df_s.iloc[idx:]

    X_train = train[feature_cols].to_numpy()
    y_train = train["target_vol"].to_numpy()
    X_test  = test[feature_cols].to_numpy()
    y_test  = test["target_vol"].to_numpy()

    print(f"\n  Train : {train['Date'].min().date()} → "
          f"{train['Date'].max().date()} ({len(train):,} rows)")
    print(f"  Test  : {test['Date'].min().date()} → "
          f"{test['Date'].max().date()} ({len(test):,} rows)")

    return X_train, X_test, y_train, y_test


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Same four metrics as baseline.py — keeps comparisons apples-to-apples."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "r2"  : float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100),
    }

# ── Plots — saved as MLflow artifacts ─────────────────────────────────────────

def _plot_feature_importance(pipeline: Pipeline, feature_cols: list[str],
                              model_name: str, top_n: int = 20):
    """
    Shows which features the model relied on most.
    Useful for explaining the model to a non-technical stakeholder —
    'the model weighs RSI and recent vol lags most heavily' is a sentence
    a risk manager can actually act on.
    """
    inner = pipeline.named_steps["model"]

    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_          # tree models
    elif hasattr(inner, "coef_"):
        imp = np.abs(inner.coef_)                  # linear models — use |coef|
    else:
        return None

    idx = np.argsort(imp)[-top_n:]                 # top N by importance
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(feature_cols)[idx], imp[idx], color="steelblue")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    path = os.path.join(tempfile.gettempdir(), f"{model_name}_importance.png")
    fig.savefig(path, dpi=120)
    plt.close()
    return path


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """
    Two diagnostic plots side by side:
        Left  : actual vs predicted — points should hug the diagonal line
        Right : residual distribution — should be centred on 0, roughly normal

    Why this matters: RMSE alone hides WHERE the model fails.
    A model can have good average RMSE but consistently underpredict
    during high-vol periods — exactly when accuracy matters most.
    This plot makes that visible at a glance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    sample = min(len(y_true), 800)   # cap points for readability
    axes[0].scatter(y_true[:sample], y_pred[:sample], alpha=0.35, s=8)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Volatility")
    axes[0].set_ylabel("Predicted Volatility")
    axes[0].set_title(f"{model_name} — Actual vs Predicted")
    axes[0].legend()

    # Residual distribution
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60, color="salmon", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1, linestyle="--")
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_title(f"{model_name} — Residual Distribution")

    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), f"{model_name}_predictions.png")
    fig.savefig(path, dpi=120)
    plt.close()
    return path

# ── Main training loop ────────────────────────────────────────────────────────

def run_experiments(df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """
    Trains every model in _MODELS, logs each as a separate MLflow run.

    Why one run per model:
    MLflow's comparison UI lets you select multiple runs and see metrics
    side by side. Separate runs = clean comparison table.
    """
    print(f"\n{'='*60}")
    print("  STEP 3 — MODEL TRAINING & MLFLOW LOGGING")
    print(f"{'='*60}")

    # tracking_uri tells MLflow where to store run data — a local folder here
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = temporal_split(df, feature_cols)

    results = []

    for model_name, base_model in _MODELS.items():
        print(f"\n  ▶  {model_name}")
        pipeline = _build_pipeline(model_name, base_model)

        # start_run() opens a new MLflow run — everything inside this
        # block gets attached to this specific run_id
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # ── Log params ────────────────────────────────────────────────
            # Why: reproducibility. 6 months from now you can see exactly
            # what hyperparameters produced this result.
            params = base_model.get_params()
            params.update(
                model_type=model_name,
                n_features=len(feature_cols),
                train_rows=len(X_train),
                test_rows=len(X_test),
                volatility_window=VOLATILITY_WINDOW,
            )
            mlflow.log_params(params)

            # ── Log tags ──────────────────────────────────────────────────
            # Why: searchable metadata. Lets you filter runs in the UI
            # by data source or target definition later.
            mlflow.set_tags({
                "data_source": "Yahoo Finance (yfinance)",
                "tickers":     ",".join(TICKERS),
                "target":      f"{VOLATILITY_WINDOW}-day forward annualised realised vol",
                "split_type":  "chronological",
            })

            # ── Train ─────────────────────────────────────────────────────
            pipeline.fit(X_train, y_train)

            # ── Evaluate ──────────────────────────────────────────────────
            y_pred_tr = pipeline.predict(X_train)
            y_pred_te = pipeline.predict(X_test)

            tr_m = _metrics(y_train, y_pred_tr)
            te_m = _metrics(y_test,  y_pred_te)

            mlflow.log_metrics({f"train_{k}": v for k, v in tr_m.items()})
            mlflow.log_metrics({f"test_{k}":  v for k, v in te_m.items()})

            print(f"     RMSE={te_m['rmse']:.5f}  R²={te_m['r2']:.4f}")

            # ── Log artifacts (plots) ────────────────────────────────────
            fi_path = _plot_feature_importance(pipeline, feature_cols, model_name)
            if fi_path:
                mlflow.log_artifact(fi_path, artifact_path="plots")

            pred_path = _plot_predictions(y_test, y_pred_te, model_name)
            mlflow.log_artifact(pred_path, artifact_path="plots")

            # ── Log the model itself ─────────────────────────────────────
            # signature records input/output schema — loading this model
            # later will validate incoming data matches what it expects
            signature = infer_signature(X_train, y_pred_tr)
            _log_model(pipeline, model_name, signature, X_train[:5])

            results.append({
                "run_id":     run_id,
                "model_name": model_name,
                **{f"test_{k}":  v for k, v in te_m.items()},
                **{f"train_{k}": v for k, v in tr_m.items()},
            })

    return results

# ── Model registry ─────────────────────────────────────────────────────────────

def register_best_model(results: list[dict]):
    """
    Ranks all runs by test RMSE, registers the winner in MLflow's
    Model Registry, and tags it with the 'champion' alias.

    Why a registry, not just picking the best run manually:
    The registry gives version control for models. 'champion' always
    points to the current best — when you retrain later and get a
    better model, you just move the alias. Old versions stay traceable.
    """
    print(f"\n{'='*60}")
    print("  STEP 4 — MODEL REGISTRY")
    print(f"{'='*60}")

    lb = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)
    lb.index += 1

    print("\n  Leaderboard (sorted by test RMSE):\n")
    print(lb[["model_name", "test_rmse", "test_mae", "test_r2", "test_mape"]]
          .to_string(index=True))

    best = lb.iloc[0]
    print(f"\n  Best model : {best['model_name']}")
    print(f"  Test RMSE  : {best['test_rmse']:.5f}")
    print(f"  Test R²    : {best['test_r2']:.4f}")

    # register the winning run's model artifact
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    print(f"\n  Registered '{REGISTERED_MODEL_NAME}' version {mv.version}")

    # point the 'champion' alias at this version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version)
    print(f"  Alias '{CHAMPION_ALIAS}' → version {mv.version}")

    print(f"\n  View results:  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    return best.to_dict(), lb


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_ingestion import fetch_stock_data
    from src.feature_engineering import engineer_features

    raw_df = fetch_stock_data()
    features_df, feature_cols = engineer_features(raw_df)

    results = run_experiments(features_df, feature_cols)
    register_best_model(results)

