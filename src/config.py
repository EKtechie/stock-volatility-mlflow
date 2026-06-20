# Configuration file for stock volatility analysis

# List of stock tickers to analyze
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "AAPL",
    "MSFT",
    "NVDA",
    "^NSEI",
    "^GSPC"
]

# Date range for historical data
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

# Volatility calculation parameters
VOLATILITY_WINDOW = 5
ANNUALISE_FACTOR  = 252
ROLLING_WINDOWS   = [5, 10, 21, 63]
VOL_LAG_DAYS      = [1, 2, 3, 5, 10, 21]

# Technical indicator parameters
RSI_WINDOW   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
BB_WINDOW    = 20
ATR_WINDOW   = 14

# Machine learning parameters
TEST_SIZE = 0.20

# ── MLflow ───────────────────────────────────────────────────────────────────
# Why sqlite instead of plain './mlruns' folder:
# As of MLflow 3.x, the filesystem-only backend is in maintenance mode.
# SQLite is a single local file, needs no server setup, and is the
# currently recommended local backend. Run metadata (params, metrics,
# tags) goes into mlflow.db — actual artifacts (plots, models) still
# get saved to the mlruns/ folder alongside it.
MLFLOW_TRACKING_URI    = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "stock-volatility-forecasting"
REGISTERED_MODEL_NAME  = "StockVolatilityForecaster"
CHAMPION_ALIAS         = "champion"

