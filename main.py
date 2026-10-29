"""
Main Pipeline — Zeckathon PS03
Liquid Fund Cash Buffer Forecasting

Run this script to execute the full pipeline:
  1. Generate synthetic data (or load real data)
  2. Feature engineering
  3. Exploratory data analysis
  4. Train & benchmark models
  5. Rolling-origin backtest
  6. Decision layer (cash buffer optimization)
  7. Generate final forecasts & recommendations

Usage:
    python main.py                  # Run full pipeline
    python main.py --skip-eda       # Skip EDA plots
    python main.py --models-only    # Only train and evaluate models
"""

import sys
import argparse
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from config import *


def step1_data():
    """Step 1: Generate or load data."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)

    from data_generator import generate_synthetic_data, generate_macro_data, save_data

    if RAW_FLOWS_FILE.exists() and RAW_MACRO_FILE.exists():
        print("  ✅ Data files already exist. Loading...")
    else:
        print("  📊 Generating synthetic data (replace with real AMFI/RBI data)...")
        flows_df = generate_synthetic_data()
        macro_df = generate_macro_data()
        save_data(flows_df, macro_df)
        print("  ✅ Synthetic data generated and saved.")

    return True


def step2_features():
    """Step 2: Feature engineering."""
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    from feature_engineering import load_raw_data, create_features

    flows_df, macro_df = load_raw_data()
    df = create_features(flows_df, macro_df)
    print(f"  ✅ Feature matrix created: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def step3_eda(df):
    """Step 3: Exploratory data analysis."""
    print("\n" + "=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    from eda import run_full_eda
    run_full_eda(df)

    return True


def step4_models_and_backtest(df):
    """Step 4 & 5: Train models and run rolling-origin backtest."""
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING & BACKTESTING")
    print("=" * 70)

    from models import SeasonalNaive, SARIMAXForecaster, LightGBMForecaster, EnsembleForecaster
    from backtesting import rolling_origin_backtest, compute_summary_metrics, plot_backtest_results

    # Define feature sets
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    exog_cols = [c for c in EXOG_FEATURES_SARIMAX if c in df.columns]

    print(f"  Features for ML model:    {len(feature_cols)} columns")
    print(f"  Exog vars for SARIMAX:    {len(exog_cols)} columns")
    print(f"  Backtest origins:         {BACKTEST_ORIGINS}")
    print(f"  Forecast horizon:         {FORECAST_HORIZON} months")

    # Run backtest
    results_df = rolling_origin_backtest(
        df=df,
        feature_cols=feature_cols,
        exog_cols=exog_cols,
        n_origins=BACKTEST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )

    # Summary
    summary = compute_summary_metrics(results_df)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Backtest Results")
    print("=" * 70)
    print(summary.to_string())

    # Plots
    plot_backtest_results(results_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_summary.csv")
    print("\n  ✅ Backtest results saved.")

    return results_df, summary


def step5_final_forecast(df):
    """Step 5: Generate the final 3-month forecast."""
    print("\n" + "=" * 70)
    print("STEP 5: FINAL 3-MONTH FORECAST")
    print("=" * 70)

    from models import SARIMAXForecaster, LightGBMForecaster

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    exog_cols = [c for c in EXOG_FEATURES_SARIMAX if c in df.columns]

    # Train on ALL available data
    train_y = df["net_flow_cr"]
    train_X = df[feature_cols] if feature_cols else None
    train_exog = df[exog_cols] if exog_cols else None

    # ── SARIMAX ──
    print("\n  Training SARIMAX on full dataset...")
    sarimax = SARIMAXForecaster()
    sarimax.fit(train_y, train_exog)

    # For future exog, we need to create features for the next 3 months
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=3, freq="MS")
    future_exog = pd.DataFrame({"date": future_dates})
    future_exog["month"] = future_exog["date"].dt.month
    future_exog["is_advance_tax_month"] = future_exog["month"].isin([3, 6, 9, 12]).astype(int)
    future_exog["is_fiscal_yearend"] = (future_exog["month"] == 3).astype(int)

    # For macro features, use last known values (persistence assumption)
    for col in ["repo_rate", "liquidity_surplus_cr", "nifty_monthly_return"]:
        if col in exog_cols:
            future_exog[col] = df[col].iloc[-1]

    future_exog_vals = future_exog[exog_cols]
    sarimax_forecast = sarimax.predict(3, future_exog_vals)

    # ── LightGBM ──
    print("  Training LightGBM on full dataset...")
    lgbm = LightGBMForecaster()
    lgbm.fit(train_y, train_X)

    # For LightGBM future features, build them similarly
    future_features = pd.DataFrame({"date": future_dates})
    future_features["month"] = future_features["date"].dt.month
    future_features["is_advance_tax_month"] = future_features["month"].isin([3, 6, 9, 12]).astype(int)
    future_features["is_fiscal_yearend"] = (future_features["month"] == 3).astype(int)
    future_features["is_quarter_end"] = future_features["month"].isin([3, 6, 9, 12]).astype(int)
    future_features["fy_quarter"] = future_features["month"].map(
        {4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 1: 4, 2: 4, 3: 4}
    )

    # Lags and rolling features: use last known values
    for col in feature_cols:
        if col not in future_features.columns:
            if col in df.columns:
                future_features[col] = df[col].iloc[-1]
            else:
                future_features[col] = 0

    # Ensure correct column order
    future_X = future_features[feature_cols]
    lgbm_forecast = lgbm.predict(3, future_X)

    # ── Print Results ──
    print("\n" + "-" * 60)
    print("3-MONTH FORECAST — SARIMAX")
    print("-" * 60)
    for i, date in enumerate(future_dates):
        month_name = date.strftime("%B %Y")
        print(f"  {month_name}:")
        print(f"    Point forecast:  ₹{sarimax_forecast['point'][i]:>10,.0f} Cr")
        print(f"    80% PI:          [₹{sarimax_forecast['lower_80'][i]:>10,.0f}, ₹{sarimax_forecast['upper_80'][i]:>10,.0f}]")
        print(f"    95% PI:          [₹{sarimax_forecast['lower_95'][i]:>10,.0f}, ₹{sarimax_forecast['upper_95'][i]:>10,.0f}]")

    print("\n" + "-" * 60)
    print("3-MONTH FORECAST — LightGBM")
    print("-" * 60)
    for i, date in enumerate(future_dates):
        month_name = date.strftime("%B %Y")
        print(f"  {month_name}:")
        print(f"    Point forecast:  ₹{lgbm_forecast['point'][i]:>10,.0f} Cr")
        print(f"    80% PI:          [₹{lgbm_forecast['lower_80'][i]:>10,.0f}, ₹{lgbm_forecast['upper_80'][i]:>10,.0f}]")
        print(f"    95% PI:          [₹{lgbm_forecast['lower_95'][i]:>10,.0f}, ₹{lgbm_forecast['upper_95'][i]:>10,.0f}]")

    # Save forecasts
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "sarimax_point": sarimax_forecast["point"],
        "sarimax_lower_80": sarimax_forecast["lower_80"],
        "sarimax_upper_80": sarimax_forecast["upper_80"],
        "sarimax_lower_95": sarimax_forecast["lower_95"],
        "sarimax_upper_95": sarimax_forecast["upper_95"],
        "lgbm_point": lgbm_forecast["point"],
        "lgbm_lower_80": lgbm_forecast["lower_80"],
        "lgbm_upper_80": lgbm_forecast["upper_80"],
        "lgbm_lower_95": lgbm_forecast["lower_95"],
        "lgbm_upper_95": lgbm_forecast["upper_95"],
    })
    forecast_df.to_csv(OUTPUT_DIR / "final_forecast.csv", index=False)
    print("\n  ✅ Final forecast saved to:", OUTPUT_DIR / "final_forecast.csv")

    return forecast_df, lgbm


def step6_decision_layer(df, forecast_df=None):
    """Step 6: Cash buffer decision layer."""
    print("\n" + "=" * 70)
    print("STEP 6: DECISION LAYER — CASH BUFFER OPTIMIZATION")
    print("=" * 70)

    from decision_layer import (
        naive_buffer_rule,
        p95_buffer_rule,
        adaptive_buffer_rule,
        backtest_buffer_rules,
        summarize_buffer_performance,
        sensitivity_analysis,
        plot_decision_analysis,
    )

    # Load backtest predictions if available
    backtest_file = OUTPUT_DIR / "backtest_results.csv"
    if backtest_file.exists():
        backtest_preds = pd.read_csv(backtest_file, parse_dates=["date"] if "date" in pd.read_csv(backtest_file, nrows=0).columns else [])
    else:
        backtest_preds = None

    # Run buffer rule backtest
    buffer_results = backtest_buffer_rules(df, backtest_preds)

    # Summary
    summary = summarize_buffer_performance(buffer_results)
    print("\n" + "-" * 60)
    print("BUFFER RULE COMPARISON")
    print("-" * 60)
    print(summary.to_string())

    # Sensitivity analysis
    sens = sensitivity_analysis(df, None, df["aum_cr"])
    print("\n" + "-" * 60)
    print("SENSITIVITY ANALYSIS (Buffer Percentile vs Cost)")
    print("-" * 60)
    print(sens.to_string())

    # Plots
    plot_decision_analysis(buffer_results, sens)

    # Final recommendation
    last_aum = df["aum_cr"].iloc[-1]
    if forecast_df is not None:
        # Use the worst-case (P95) from LightGBM forecast
        worst_outflow = -forecast_df["lgbm_lower_95"].min()  # most negative = worst outflow
        recommended_buffer_cr = max(worst_outflow, 0.03 * last_aum)
        recommended_buffer_pct = recommended_buffer_cr / last_aum * 100
    else:
        recommended_buffer_pct = 8.0  # default
        recommended_buffer_cr = recommended_buffer_pct / 100 * last_aum

    print("\n" + "=" * 70)
    print("📋 FINAL RECOMMENDATION")
    print("=" * 70)
    print(f"  Current AUM:              ₹{last_aum:>12,.0f} Cr")
    print(f"  Recommended Buffer:       {recommended_buffer_pct:>8.1f}% of AUM")
    print(f"  Buffer Amount:            ₹{recommended_buffer_cr:>12,.0f} Cr")
    print(f"  SEBI Min (20% liquid):    ₹{0.20 * last_aum:>12,.0f} Cr")
    print(f"  Estimated Monthly Cost:   ~{recommended_buffer_pct * (PORTFOLIO_YIELD_ASSUMPTION - OVERNIGHT_YIELD_ASSUMPTION) / 12 * 100:.1f} bps")

    # Save
    buffer_results.to_csv(OUTPUT_DIR / "buffer_backtest.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "buffer_summary.csv")
    print("\n  ✅ Decision layer analysis saved.")

    return buffer_results


def main():
    parser = argparse.ArgumentParser(description="Zeckathon PS03 — Liquid Fund Cash Buffer Pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--models-only", action="store_true", help="Only run model training")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║  🏦 ZECKATHON PS03 — Liquid Fund Cash Buffer Forecasting          ║")
    print("║  How Much Cash Must a Liquid Fund Hold on 15 March?               ║")
    print("╚" + "═" * 68 + "╝")

    # Step 1: Data
    step1_data()

    # Step 2: Features
    df = step2_features()

    # Step 3: EDA
    if not args.skip_eda and not args.models_only:
        step3_eda(df)

    # Step 4: Models & Backtest
    backtest_results, model_summary = step4_models_and_backtest(df)

    if args.models_only:
        print("\n✅ Models-only mode complete.")
        return

    # Step 5: Final forecast
    forecast_df, best_model = step5_final_forecast(df)

    # Step 6: Decision layer
    step6_decision_layer(df, forecast_df)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  ✅ PIPELINE COMPLETE                                             ║")
    print("║                                                                    ║")
    print("║  Next steps:                                                       ║")
    print("║    1. streamlit run dashboard.py   (launch dashboard)              ║")
    print("║    2. Review outputs/ folder       (plots & results)               ║")
    print("║    3. Replace synthetic data with real AMFI/RBI data               ║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
