"""
Feature engineering pipeline for Zeckathon PS03.

Loads raw flow + macro CSVs, creates calendar / lag / rolling / derived
features, and saves the final modelling-ready dataset.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    RAW_FLOWS_FILE,
    RAW_MACRO_FILE,
    FEATURE_FILE,
    DATA_DIR,
    CALENDAR_FEATURES,
    LAG_FEATURES,
    ROLLING_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURES,
)


# =====================================================================
# 1. LOAD RAW DATA
# =====================================================================

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two raw CSVs produced by data_generator.py.

    Returns
    -------
    flows_df : pd.DataFrame
        Monthly liquid-fund flows.
    macro_df : pd.DataFrame
        Monthly macro-economic indicators.

    Raises
    ------
    FileNotFoundError
        If either CSV is missing (run data_generator.py first).
    """
    if not RAW_FLOWS_FILE.exists():
        raise FileNotFoundError(
            f"Flows file not found: {RAW_FLOWS_FILE}\n"
            "  → Run `python data_generator.py` first."
        )
    if not RAW_MACRO_FILE.exists():
        raise FileNotFoundError(
            f"Macro file not found: {RAW_MACRO_FILE}\n"
            "  → Run `python data_generator.py` first."
        )

    flows_df = pd.read_csv(RAW_FLOWS_FILE, parse_dates=["date"])
    macro_df = pd.read_csv(RAW_MACRO_FILE, parse_dates=["date"])

    print(f"  ✓ Loaded flows  : {len(flows_df):>4} rows  ({RAW_FLOWS_FILE.name})")
    print(f"  ✓ Loaded macro  : {len(macro_df):>4} rows  ({RAW_MACRO_FILE.name})")
    return flows_df, macro_df


# =====================================================================
# 2. CREATE FEATURES
# =====================================================================

def create_features(
    flows_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw flows + macro data.

    Steps
    -----
    1. Merge on ``date``
    2. Calendar features
    3. Lag features on ``net_flow_cr``
    4. Rolling statistics on ``net_flow_cr``
    5. Derived / interaction features
    6. Drop initial NaN rows (first 12 months, from lag-12)
    7. Save to ``data/features_dataset.csv``

    Parameters
    ----------
    flows_df, macro_df : pd.DataFrame
        Raw DataFrames (must share a ``date`` column).

    Returns
    -------
    pd.DataFrame
        Feature-enriched dataset ready for modelling.
    """
    # ── 0. Merge ─────────────────────────────────────────────────────
    print("  [2a] Merging flows + macro on date …")
    df = pd.merge(flows_df, macro_df, on="date", how="inner")
    print(f"       → {len(df)} rows after merge")

    # Ensure sorted
    df = df.sort_values("date").reset_index(drop=True)

    # ── 1. Calendar features ─────────────────────────────────────────
    print("  [2b] Creating calendar features …")
    df["month"] = df["date"].dt.month

    df["is_advance_tax_month"] = df["month"].isin([3, 6, 9, 12]).astype(int)

    df["is_fiscal_yearend"] = (df["month"] == 3).astype(int)

    df["is_quarter_end"] = df["month"].isin([3, 6, 9, 12]).astype(int)

    # Fiscal-year quarter: Apr-Jun → Q1, Jul-Sep → Q2, Oct-Dec → Q3, Jan-Mar → Q4
    df["fy_quarter"] = df["month"].map(
        {4: 1, 5: 1, 6: 1,
         7: 2, 8: 2, 9: 2,
         10: 3, 11: 3, 12: 3,
         1: 4, 2: 4, 3: 4}
    )

    # ── 2. Lag features on net_flow_cr ───────────────────────────────
    print("  [2c] Creating lag features …")
    lag_map = {
        "flow_lag1": 1,
        "flow_lag2": 2,
        "flow_lag3": 3,
        "flow_lag6": 6,
        "flow_lag12": 12,
    }
    for col, lag in lag_map.items():
        df[col] = df["net_flow_cr"].shift(lag)

    # ── 3. Rolling features on net_flow_cr ───────────────────────────
    print("  [2d] Creating rolling features …")
    df["rolling_mean_3m"] = df["net_flow_cr"].shift(1).rolling(window=3).mean()
    df["rolling_std_3m"]  = df["net_flow_cr"].shift(1).rolling(window=3).std()
    df["rolling_mean_6m"] = df["net_flow_cr"].shift(1).rolling(window=6).mean()
    df["rolling_std_6m"]  = df["net_flow_cr"].shift(1).rolling(window=6).std()

    # ── 4. Derived features ──────────────────────────────────────────
    print("  [2e] Creating derived features …")

    # flow_rate_pct: net_flow as % of previous month's AUM
    df["flow_rate_pct"] = df["net_flow_cr"] / df["aum_cr"].shift(1) * 100

    # flow_momentum: acceleration in flows (lag1 − lag2)
    df["flow_momentum"] = df["flow_lag1"] - df["flow_lag2"]

    # rate_spread: T-bill vs repo (market-implied liquidity premium)
    df["rate_spread"] = df["tbill_91d_yield"] - df["repo_rate"]

    # ── 5. Drop NaN rows from lagging (first 12 rows) ───────────────
    n_before = len(df)
    df = df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).reset_index(drop=True)
    n_after = len(df)
    print(f"  [2f] Dropped {n_before - n_after} initial rows with NaN → {n_after} rows remain")

    # ── 6. Save ──────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_FILE, index=False)
    print(f"  ✓ Feature dataset saved → {FEATURE_FILE}  ({len(df)} rows)")

    return df


# =====================================================================
# 3. MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Zeckathon PS03 – Feature Engineering Pipeline")
    print("=" * 60)
    print()

    # ── Load ─────────────────────────────────────────────────────────
    print("[1/2] Loading raw data …")
    flows, macro = load_raw_data()

    # ── Engineer ─────────────────────────────────────────────────────
    print()
    print("[2/2] Engineering features …")
    features = create_features(flows, macro)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("  FEATURE MATRIX SUMMARY")
    print("-" * 60)
    print(f"  Shape          : {features.shape}")
    print(f"  Date range     : {features['date'].min().date()} → {features['date'].max().date()}")
    print(f"  NaN remaining  : {features[ALL_FEATURES].isna().sum().sum()}")
    print()

    print("  Calendar features:")
    for c in CALENDAR_FEATURES:
        print(f"    {c:30s}  unique={features[c].nunique()}")

    print()
    print("  Feature stats (selected):")
    stat_cols = LAG_FEATURES + ROLLING_FEATURES + DERIVED_FEATURES
    print(features[stat_cols].describe().round(2).to_string())

    print()
    print("✅ Feature engineering complete.")
