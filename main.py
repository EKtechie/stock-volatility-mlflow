import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import fetch_stock_data
from src.feature_engineering import engineer_features
from src.train import run_experiments, register_best_model

def main():
    print("\n" + "="*60)
    print("  STOCK VOLATILITY FORECASTING PIPELINE")
    print("="*60)

    # Step 1 — pull live data from Yahoo Finance
    raw_df = fetch_stock_data()

    # Step 2 — build 50+ features
    features_df, feature_cols = engineer_features(raw_df)

    # Step 3 — train 6 models, log everything to MLflow
    results = run_experiments(features_df, feature_cols)

    # Step 4 — register the best model as champion
    register_best_model(results)

    print("\n✅  Pipeline complete.")
    print("    Run:  mlflow ui --backend-store-uri mlruns")
    print("    Open: http://127.0.0.1:5000\n")

if __name__ == "__main__":
    main()