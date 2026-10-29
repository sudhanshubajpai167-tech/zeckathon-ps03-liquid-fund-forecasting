"""
Configuration constants for the Zeckathon PS03 project.
Liquid Fund Cash Buffer Forecasting.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Data file names
# ──────────────────────────────────────────────
RAW_FLOWS_FILE = DATA_DIR / "amfi_monthly_flows.csv"
RAW_MACRO_FILE = DATA_DIR / "macro_features.csv"
FEATURE_FILE = DATA_DIR / "features_dataset.csv"

# ──────────────────────────────────────────────
# Date range for data
# ──────────────────────────────────────────────
DATA_START = "2018-04-01"      # Post SEBI recategorization
DATA_END = "2026-09-30"        # Latest available

# ──────────────────────────────────────────────
# Model parameters
# ──────────────────────────────────────────────
SEASONAL_PERIOD = 12            # Monthly seasonality
FORECAST_HORIZON = 3            # Predict 3 months ahead
BACKTEST_ORIGINS = 24           # 24 rolling-origin windows

# ──────────────────────────────────────────────
# Prediction interval levels
# ──────────────────────────────────────────────
PI_80_ALPHA = 0.20              # 80% prediction interval → alpha = 0.20
PI_95_ALPHA = 0.05              # 95% prediction interval → alpha = 0.05

QUANTILES = {
    "lower_95": 0.025,
    "lower_80": 0.10,
    "median":   0.50,
    "upper_80": 0.90,
    "upper_95": 0.975,
}

# ──────────────────────────────────────────────
# Feature columns
# ──────────────────────────────────────────────
CALENDAR_FEATURES = [
    "month",
    "is_advance_tax_month",
    "is_fiscal_yearend",
    "is_quarter_end",
    "fy_quarter",
]

MACRO_FEATURES = [
    "repo_rate",
    "liquidity_surplus_cr",
    "tbill_91d_yield",
    "nifty_monthly_return",
    "nifty_monthly_volatility",
]

LAG_FEATURES = [
    "flow_lag1",
    "flow_lag2",
    "flow_lag3",
    "flow_lag6",
    "flow_lag12",
]

ROLLING_FEATURES = [
    "rolling_mean_3m",
    "rolling_std_3m",
    "rolling_mean_6m",
    "rolling_std_6m",
]

DERIVED_FEATURES = [
    "flow_rate_pct",           # net_flow / lagged_AUM * 100
    "flow_momentum",           # flow_lag1 - flow_lag2
    "rate_spread",             # tbill_91d_yield - repo_rate
]

ALL_FEATURES = (
    CALENDAR_FEATURES + MACRO_FEATURES + LAG_FEATURES +
    ROLLING_FEATURES + DERIVED_FEATURES
)

EXOG_FEATURES_SARIMAX = [
    "is_advance_tax_month",
    "is_fiscal_yearend",
    "repo_rate",
    "liquidity_surplus_cr",
    "nifty_monthly_return",
]

# ──────────────────────────────────────────────
# Decision layer
# ──────────────────────────────────────────────
SHORTFALL_PENALTY_MULTIPLIER = 2.0   # 2× penalty on redemption shortfall
PORTFOLIO_YIELD_ASSUMPTION = 0.065   # ~6.5% p.a. for liquid fund portfolio
OVERNIGHT_YIELD_ASSUMPTION = 0.055   # ~5.5% p.a. for overnight instruments
SEBI_MIN_LIQUID_ASSETS_PCT = 20.0    # SEBI mandate: ≥20% in liquid assets

# ──────────────────────────────────────────────
# LightGBM hyperparameters
# ──────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 12,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIG_SIZE = (14, 6)
COLOR_HISTORICAL = "#2C3E50"
COLOR_FORECAST = "#E74C3C"
COLOR_PI_80 = "rgba(231, 76, 60, 0.25)"
COLOR_PI_95 = "rgba(231, 76, 60, 0.10)"
