# Quick alpha extension check — run this once, don't need to save it
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.data_ingestion import fetch_stock_data
from src.feature_engineering import engineer_features
from src.train import temporal_split

raw_df = fetch_stock_data()
features_df, feature_cols = engineer_features(raw_df)
X_train, X_test, y_train, y_test = temporal_split(features_df, feature_cols)

tscv = TimeSeriesSplit(n_splits=5)

for alpha in [50, 100, 150, 200, 300, 500, 1000]:
    fold_rmses = []
    for tr_idx, val_idx in tscv.split(X_train):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
        pipe.fit(X_train[tr_idx], y_train[tr_idx])
        pred = pipe.predict(X_train[val_idx])
        fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))
    print(f"alpha={alpha:>6}   cv_rmse={np.mean(fold_rmses):.5f}")