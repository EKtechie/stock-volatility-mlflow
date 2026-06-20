"""
tune.py

Hyperparameter tuning using RandomizedSearchCV with TimeSeriesSplit.

Why TimeSeriesSplit instead of default KFold:
    KFold shuffles data randomly into folds — a fold can contain future
    data in training and past data in validation. This leaks information
    and makes CV scores look better than real-world performance will be.
    TimeSeriesSplit keeps validation folds strictly after training folds.

Why RandomizedSearchCV instead of GridSearchCV:
    Exhaustive grid search over 5 hyperparameters x 4 values each x 5 CV
    folds = thousands of model fits. RandomizedSearchCV samples a fixed
    number of combinations — n_iter=30 typically finds near-optimal
    params in a fraction of the compute time.

Each search logs its own MLflow run with the best params and CV score,
so the tuning process itself is auditable, not just the final model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import mlflow

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, TEST_SIZE
from src.data_ingestion import fetch_stock_data
from src.feature_engineering import engineer_features
from src.train import temporal_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ── Hyperparameter search spaces ──────────────────────────────────────────────
# Ranges chosen to bracket the defaults used in train.py — wide enough to
# find improvement, narrow enough to avoid wasting trials on absurd values.

_SEARCH_SPACES = {
    "RandomForest": {
        "model": RandomForestRegressor(n_jobs=-1, random_state=42),
        "params": {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [5, 8, 10, 15, None],
            "min_samples_leaf": [5, 10, 20, 30],
            "max_features":     ["sqrt", "log2", 0.5, 0.8],
        },
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=0),
        "params": {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [3, 4, 5, 6, 8],
            "learning_rate":    [0.01, 0.03, 0.05, 0.1],
            "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
        },
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, verbose=-1),
        "params": {
            "n_estimators":     [100, 200, 300, 500],
            "num_leaves":       [15, 31, 63, 127],
            "learning_rate":    [0.01, 0.03, 0.05, 0.1],
            "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        },
    },
    "CatBoost": {
        "model": CatBoostRegressor(random_seed=42, verbose=0),
        "params": {
            "iterations":    [100, 200, 300, 500],
            "depth":         [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg":   [1, 3, 5, 7, 9],
        },
    },
}

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SEARCH_SPACES["Ridge"] = {
    "model": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]),
    "params": {
        "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    },
}

_SEARCH_SPACES["Lasso"] = {
    "model": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=5000)),
    ]),
    "params": {
        "model__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
    },
}

_SEARCH_SPACES["ElasticNet"] = {
    "model": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=5000)),
    ]),
    "params": {
        "model__alpha":    [0.001, 0.01, 0.1, 0.5, 1.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
}


# ── Single model tuning ───────────────────────────────────────────────────────

def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 30,
    n_splits: int = 5,
) -> dict:
    """
    Run RandomizedSearchCV with TimeSeriesSplit for one model.

    Args:
        model_name : key into _SEARCH_SPACES
        X_train    : training features (only the training portion —
                     never touch the test set during tuning)
        y_train    : training targets
        n_iter     : number of random param combinations to try
        n_splits   : number of TimeSeriesSplit folds

    Returns:
        dict with best_params, best_cv_rmse, and the fitted best estimator

    Why we only use X_train here, never X_test:
        The test set must stay completely unseen until final evaluation.
        Tuning on test data (even indirectly through repeated peeking)
        is a subtle form of overfitting called "test set leakage."
    """
    spec  = _SEARCH_SPACES[model_name]
    model = spec["model"]
    param_dist = spec["params"]

    # TimeSeriesSplit: each fold's validation set comes strictly after
    # its training set. No shuffling — order is preserved from the input.
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # RMSE scorer — sklearn maximises by default, so we negate
    # (lower RMSE = better, but RandomizedSearchCV looks for higher score)
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer,
        cv=tscv,
        n_jobs=-1,           # parallelise across CV folds
        random_state=42,
        verbose=1,
    )

    print(f"\n  Tuning {model_name}  ({n_iter} trials x {n_splits} folds = "
          f"{n_iter * n_splits} fits)")

    search.fit(X_train, y_train)

    best_rmse = -search.best_score_   # un-negate back to positive RMSE

    print(f"  Best CV RMSE : {best_rmse:.5f}")
    print(f"  Best params  : {search.best_params_}")

    return {
        "model_name":    model_name,
        "best_params":   search.best_params_,
        "best_cv_rmse":  best_rmse,
        "best_estimator": search.best_estimator_,
    }

# ── Tune all candidate models ─────────────────────────────────────────────────

def run_tuning(df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """
    Tunes every model in _SEARCH_SPACES and logs each search as an
    MLflow run, so the tuning process itself is tracked — not just
    the final chosen model.

    Why log the tuning runs separately from train.py's runs:
        Keeps your experiment history clean. You can later look back
        and see "this is what I searched, this is what won" without
        mixing it into the main leaderboard.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = temporal_split(df, feature_cols)

    results = []

    for model_name in _SEARCH_SPACES:
        with mlflow.start_run(run_name=f"{model_name}_tuning"):
            mlflow.set_tag("stage", "hyperparameter_tuning")

            result = tune_model(model_name, X_train, y_train)

            # log the search space itself for reproducibility
            mlflow.log_param("n_iter", 30)
            mlflow.log_param("cv_splits", 5)
            mlflow.log_param("cv_strategy", "TimeSeriesSplit")

            # log the best params found — prefixed so they don't collide
            # with each other if you compare runs side by side later
            mlflow.log_params(
                {f"best_{k}": v for k, v in result["best_params"].items()}
            )
            mlflow.log_metric("best_cv_rmse", result["best_cv_rmse"])

            # evaluate the tuned model on the held-out test set
            # this is the FIRST and ONLY time test data is touched
            y_pred_test = result["best_estimator"].predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mlflow.log_metric("test_rmse", test_rmse)

            print(f"  Test RMSE (tuned) : {test_rmse:.5f}\n")

            result["test_rmse"] = test_rmse
            results.append(result)

    return results

if __name__ == "__main__":
    raw_df = fetch_stock_data()
    features_df, feature_cols = engineer_features(raw_df)

    tuning_results = run_tuning(features_df, feature_cols)

    # leaderboard sorted by test RMSE
    lb = pd.DataFrame([
        {"model": r["model_name"], "test_rmse": r["test_rmse"],
         "cv_rmse": r["best_cv_rmse"]}
        for r in tuning_results
    ]).sort_values("test_rmse").reset_index(drop=True)
    lb.index += 1

    print(f"\n{'='*60}")
    print("  TUNING LEADERBOARD")
    print(f"{'='*60}\n")
    print(lb.to_string(index=True))

    best = tuning_results[
        np.argmin([r["test_rmse"] for r in tuning_results])
    ]
    print(f"\n  Best tuned model : {best['model_name']}")
    print(f"  Best params      : {best['best_params']}")


