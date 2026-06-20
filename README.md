# Stock Volatility Forecasting Pipeline

A regression pipeline that forecasts 5-day forward annualised stock volatility using live market data, 50+ engineered features, and a benchmarked comparison of 10 models — fully tracked with MLflow, including experiment logging, hyperparameter tuning, and model registry.

No static CSV files. All data is pulled live from Yahoo Finance at runtime.

---

## Why volatility, not price

Predicting future stock _price_ is close to impossible — markets are near-efficient and prices follow a roughly random walk. Volatility is different: it clusters (high volatility tends to follow high volatility) and mean-reverts, giving it genuine, learnable structure. This is also a real business problem — volatility forecasts feed directly into:

- **Risk management** — Value-at-Risk (VaR) calculations for capital allocation
- **Options pricing** — the only non-observable input in Black-Scholes-style pricing
- **Portfolio construction** — volatility-targeted position sizing
- **Algorithmic trading** — dynamic position sizing based on expected risk

---

## Pipeline overview

```
Yahoo Finance (yfinance)
        │
        ▼
Feature Engineering (50+ features)
        │
        ▼
Baselines (Dummy, Persistence, LinearRegression)
        │
        ▼
Model Training (10 models) ──► MLflow Experiment Tracking
        │
        ▼
Hyperparameter Tuning (RandomizedSearchCV + TimeSeriesSplit)
        │
        ▼
MLflow Model Registry ──► Champion model (alias: "champion")
```

---

## Project structure

```
stock-volatility-mlflow/
├── src/
│   ├── __init__.py
│   ├── config.py              # tickers, dates, windows, MLflow settings
│   ├── data_ingestion.py      # live OHLCV fetch via yfinance
│   ├── feature_engineering.py # 50+ feature pipeline + target definition
│   ├── baseline.py            # Dummy / Persistence / LinearRegression baselines
│   ├── train.py                # 6 models, MLflow logging, model registry
│   └── tune.py                  # RandomizedSearchCV + TimeSeriesSplit tuning
├── main.py                     # runs the full pipeline end to end
├── requirements.txt
└── README.md
```

---

## Data

Pulled live via `yfinance` — no CSV, no manual downloads. Mixed market and asset types to avoid overfitting to a single regime:

| Category          | Tickers                                                 |
| ----------------- | ------------------------------------------------------- |
| Indian blue chips | RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS |
| US tech           | AAPL, MSFT, GOOGL                                       |
| Indices           | ^NSEI (Nifty 50), ^GSPC (S&P 500)                       |

Date range and ticker list are configurable in `src/config.py`.

---

## Target variable

**5-day forward annualised realised volatility.**

```python
log_return   = log(close_t / close_t-1)
realised_vol = std(log_return, window=5) * sqrt(252)
target       = realised_vol.shift(-5)   # forward-looking, no leakage
```

All features are computed strictly from data available up to time T — the target is the only forward-looking value, and it's never used to compute a feature.

---

## Feature engineering (50+ features)

| Group                | Features                                                                |
| -------------------- | ----------------------------------------------------------------------- |
| Technical indicators | RSI, MACD (+ signal, histogram), Bollinger Band width/position, ATR     |
| Price context        | SMA ratios (20/50/200d), high-low spread, open-close gap                |
| Volume               | 5d / 20d volume ratios                                                  |
| Rolling return stats | mean, std, skew, kurtosis across 4 windows (5/10/21/63d)                |
| Volatility history   | 6 lag features (1/2/3/5/10/21d) + rolling min/mean/max across 4 windows |
| Regime flags         | high-volatility regime, bull/bear trend, volatility acceleration        |
| Calendar             | day of week, month, quarter, week of year                               |
| Ticker encoding      | ordinal code (captures baseline volatility differences across assets)   |

---

## Modeling approach

### Baseline hierarchy (established before any real model)

| Model                           | Purpose                                                        |
| ------------------------------- | -------------------------------------------------------------- |
| `DummyRegressor (mean/median)`  | Absolute floor — beats nothing                                 |
| Persistence (`vol_t = vol_t-1`) | Domain-specific floor — exploits volatility clustering         |
| `LinearRegression`              | First real model — establishes the linear-relationship ceiling |

A model that can't beat persistence hasn't learned anything beyond what's trivially already in the raw data.

### Models benchmarked

`LinearRegression` · `Ridge` · `Lasso` · `ElasticNet` · `RandomForest` · `XGBoost` · `LightGBM` · `CatBoost`

### Validation strategy

**Chronological train/test split** (80/20, sorted by date) — never randomly shuffled, since random splitting on time series data leaks future information into training and produces misleadingly optimistic results.

**Hyperparameter tuning** uses `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation — standard `KFold` would shuffle data into folds containing future information relative to their own validation set, which is a leakage bug, not a style choice.

---

## Results

### Baseline performance

| Model                   | Test RMSE | Test R² |
| ----------------------- | --------- | ------- |
| LinearRegression        | 0.1235    | 0.379   |
| Dummy (median)          | 0.1567    | -0.000  |
| Persistence (lag-1 vol) | 0.1610    | -0.057  |
| Dummy (mean)            | 0.1644    | -0.101  |

### Final leaderboard (after hyperparameter tuning)

| Rank | Model             | Test RMSE  | CV RMSE    |
| ---- | ----------------- | ---------- | ---------- |
| 1    | **Ridge (α=100)** | **0.1229** | 0.1464     |
| 2    | ElasticNet        | 0.1247     | **0.1397** |
| 3    | Lasso             | 0.1248     | 0.1394     |
| 4    | CatBoost          | 0.1257     | 0.1493     |
| 5    | XGBoost           | 0.1267     | 0.1469     |
| 6    | RandomForest      | 0.1267     | 0.1430     |
| 7    | LightGBM          | 0.1281     | 0.1477     |

**Champion: Ridge regression (α=100)**, registered in the MLflow Model Registry under the `champion` alias.

---

## Key finding

Every regularized linear model outperformed every tuned tree ensemble — including XGBoost, LightGBM, and CatBoost after a full `RandomizedSearchCV` + `TimeSeriesSplit` search. This isn't an undertuned-tree artifact; the trees were given a fair, properly time-series-aware search and still lost.

The most likely explanation: this dataset has a low signal-to-noise ratio (typical of financial forecasting — markets are close to efficient) and significant multicollinearity among the engineered features (several rolling-window features capture overlapping information). Tree ensembles have enough capacity to overfit that noise; regularized linear models, especially with the heavy shrinkage Ridge converged to (α=100 over the default α=1), generalize better by averaging across correlated, noisy signals instead of chasing them.

**A secondary nuance worth noting:** Ridge won on the single held-out test period, but Lasso and ElasticNet had _better average cross-validation RMSE_ across the historical `TimeSeriesSplit` folds — which span more volatile periods (2020 COVID crash, 2022 rate hikes) than the calmer final test window. This suggests Lasso/ElasticNet may generalize more robustly across different market regimes, even though Ridge edged ahead on this specific test slice. Ridge was selected as champion for simplicity and its test-set edge, with this trade-off documented rather than ignored.

> **Context on R² ≈ 0.38:** This is a strong result for financial volatility forecasting, not a weak one. Published quantitative finance research typically reports R² in the 0.2–0.5 range for this kind of target — markets are close to efficient, so a fully predictable volatility signal would already be arbitraged away. An R² above 0.3 here represents a real, exploitable signal.

---

## MLflow tracking

Every run logs:

- **Params** — full model hyperparameters + dataset metadata (feature count, row counts, split type)
- **Tags** — data source, ticker list, target definition
- **Metrics** — train/test RMSE, MAE, R², MAPE
- **Artifacts** — feature importance plot, actual-vs-predicted + residual distribution plots
- **Model** — serialised with an inferred input/output signature, using each library's native MLflow flavor (`mlflow.sklearn` / `mlflow.xgboost` / `mlflow.lightgbm` / `mlflow.catboost`) to preserve full model fidelity

Backend store: SQLite (`sqlite:///mlflow.db`) — MLflow's currently recommended local backend, replacing the deprecated filesystem-only store.

---

## How to run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Sanity check — verify data + features work end to end
python src/baseline.py

# 3. Run full training pipeline (6+ models, MLflow logging)
python src/train.py

# 4. Hyperparameter tuning (RandomizedSearchCV + TimeSeriesSplit)
python src/tune.py

# 5. View results in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
# open http://127.0.0.1:5000
```

`main.py` runs steps 1–3 as a single end-to-end script.

---

## Tech stack

`Python 3.12` · `yfinance` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `catboost` · `mlflow` (SQLite backend) · `matplotlib`

---

## Limitations

- **Live data dependency** — relies on Yahoo Finance's API at runtime; subject to occasional rate-limiting or schema changes
- **No regime-shift detection** — the model has no awareness of sudden structural breaks (surprise central bank decisions, geopolitical shocks, company-specific events). It forecasts based on historical patterns only
- **Not deployed publicly** — this is a local, fully reproducible MLflow pipeline. Not currently served as a live API or web app; run locally per the instructions above
- **Static training window** — the registered champion is trained on a fixed historical period. A production version would need a scheduled retraining cadence to stay current

---

## Possible extensions

- Export the champion model as a standalone artifact (decoupled from MLflow tracking infra) for lightweight serving
- Add a GARCH-family baseline for a domain-standard volatility model comparison
- Extend the feature set with options-market-implied volatility, if a data source becomes available
- Automated retraining pipeline with a scheduled MLflow run

---

## Author

**Eswarakumar J**
GitHub: [github.com/EKtechie](https://github.com/EKtechie)
LinkedIn: [linkedin.com/in/eswarakumar-j](https://linkedin.com/in/eswarakumar-j)
