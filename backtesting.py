"""
Rolling-origin backtesting framework for Liquid Fund Cash Buffer forecasting.

Trains SeasonalNaive, SARIMAX, and LightGBM at each origin, collects
per-step metrics (MAPE, MAE, RMSE, and pinball / quantile loss), and
produces summary tables and diagnostic plots.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ALL_FEATURES,
    BACKTEST_ORIGINS,
    EXOG_FEATURES_SARIMAX,
    FEATURE_FILE,
    FORECAST_HORIZON,
    OUTPUT_DIR,
    PLOT_STYLE,
    SEASONAL_PERIOD,
)
from models import (
    BaseForecaster,
    EnsembleForecaster,
    LightGBMForecaster,
    SARIMAXForecaster,
    SeasonalNaive,
    train_all_models,
)

# Minimum number of training observations (3 full years of monthly data)
MIN_TRAIN_SIZE = 36


def pinball_loss(y: float, q_hat: float, alpha: float) -> float:
    """Compute the pinball (quantile) loss for a single observation.

    L_α(y, q̂) = max(α·(y − q̂), (1 − α)·(q̂ − y))

    This is the asymmetric loss function used for evaluating quantile
    forecasts per the competition scoring rubric.  It penalises
    under-prediction by factor α and over-prediction by factor (1 − α).

    Parameters
    ----------
    y : float
        Actual observed value.
    q_hat : float
        Quantile forecast.
    alpha : float
        Quantile level (e.g. 0.025, 0.10, 0.50, 0.90, 0.975).

    Returns
    -------
    float
        Non-negative pinball loss.
    """
    return max(alpha * (y - q_hat), (1 - alpha) * (q_hat - y))


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def rolling_origin_backtest(
    df: pd.DataFrame,
    feature_cols: List[str] = ALL_FEATURES,
    exog_cols: List[str] = EXOG_FEATURES_SARIMAX,
    n_origins: int = BACKTEST_ORIGINS,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Run rolling-origin cross-validation over *n_origins* windows.

    At each origin the models are retrained on all data up to that
    point, and *horizon*-step-ahead forecasts are compared with the
    actual values.

    Parameters
    ----------
    df : DataFrame
        Feature dataset with ``net_flow_cr`` and all required columns.
    feature_cols : list[str]
        Feature columns for LightGBM.
    exog_cols : list[str]
        Exogenous columns for SARIMAX.
    n_origins : int
        Number of rolling-origin windows.
    horizon : int
        Forecast horizon (number of steps ahead).

    Returns
    -------
    pd.DataFrame
        One row per (origin, model, horizon_step) with actuals,
        predictions, intervals, and error metrics.
    """
    n = len(df)

    # The last usable origin must leave room for *horizon* actuals after it.
    last_origin_idx = n - horizon  # exclusive train end

    # Walk backward to determine start origin
    first_origin_idx = last_origin_idx - n_origins + 1
    if first_origin_idx < MIN_TRAIN_SIZE:
        first_origin_idx = MIN_TRAIN_SIZE
        effective_origins = last_origin_idx - first_origin_idx + 1
        warnings.warn(
            f"Not enough data for {n_origins} origins; using "
            f"{effective_origins} instead."
        )
    else:
        effective_origins = n_origins

    records: List[dict] = []
    origin_indices = range(first_origin_idx, first_origin_idx + effective_origins)

    for i, train_end in enumerate(origin_indices):
        print(
            f"Backtest origin {i + 1}/{effective_origins}: "
            f"train rows 0‥{train_end - 1}  →  forecast rows "
            f"{train_end}‥{train_end + horizon - 1}"
        )

        # ── train all models ───────────────────────────────────────────
        fitted = train_all_models(df, train_end, feature_cols, exog_cols)

        # ── future feature rows for this origin ────────────────────────
        future_rows = df.iloc[train_end : train_end + horizon]
        actuals = future_rows["net_flow_cr"].values
        future_X = future_rows[feature_cols] if feature_cols else None
        future_exog = future_rows[exog_cols] if exog_cols else None

        # ── predict each model ─────────────────────────────────────────
        for model_name, model in fitted.items():
            try:
                # Determine the right future_X to pass
                if model_name == "LightGBM":
                    preds = model.predict(h=horizon, future_X=future_X)
                elif model_name == "SARIMAX":
                    preds = model.predict(h=horizon, future_X=future_exog)
                else:
                    preds = model.predict(h=horizon)
            except Exception as exc:
                warnings.warn(
                    f"  ⚠ {model_name} prediction failed at origin "
                    f"{i + 1}: {exc}"
                )
                continue

            for step in range(horizon):
                actual = actuals[step]
                pred = preds["point"][step]
                ae = abs(actual - pred)
                ape = (
                    abs((actual - pred) / actual) * 100
                    if actual != 0
                    else np.nan
                )

                # Pinball (quantile) losses for each PI bound
                pl_025 = pinball_loss(actual, preds["lower_95"][step], 0.025)
                pl_10  = pinball_loss(actual, preds["lower_80"][step], 0.10)
                pl_50  = pinball_loss(actual, pred, 0.50)
                pl_90  = pinball_loss(actual, preds["upper_80"][step], 0.90)
                pl_975 = pinball_loss(actual, preds["upper_95"][step], 0.975)
                pl_avg = (pl_025 + pl_10 + pl_50 + pl_90 + pl_975) / 5.0

                records.append(
                    {
                        "origin": i + 1,
                        "train_end_idx": train_end,
                        "origin_date": (
                            df["date"].iloc[train_end - 1]
                            if "date" in df.columns
                            else train_end
                        ),
                        "model_name": model_name,
                        "horizon_step": step + 1,
                        "actual": actual,
                        "predicted": pred,
                        "lower_80": preds["lower_80"][step],
                        "upper_80": preds["upper_80"][step],
                        "lower_95": preds["lower_95"][step],
                        "upper_95": preds["upper_95"][step],
                        "mae": ae,
                        "mape": ape,
                        "pinball_q025": pl_025,
                        "pinball_q10": pl_10,
                        "pinball_q50": pl_50,
                        "pinball_q90": pl_90,
                        "pinball_q975": pl_975,
                        "pinball_avg": pl_avg,
                    }
                )

    results = pd.DataFrame(records)
    print(f"\n✓ Backtest complete – {len(results)} prediction rows collected.")
    return results


# ---------------------------------------------------------------------------
# Interval calibration
# ---------------------------------------------------------------------------

def compute_interval_calibration(
    results_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, float]:
    """Compute empirical coverage of 80 % and 95 % prediction intervals.

    Parameters
    ----------
    results_df : DataFrame
        Output of :func:`rolling_origin_backtest`.
    model_name : str
        Name of the model to evaluate.

    Returns
    -------
    dict
        ``{"cal_80": <float>, "cal_95": <float>}``
    """
    sub = results_df[results_df["model_name"] == model_name].copy()
    if sub.empty:
        return {"cal_80": np.nan, "cal_95": np.nan}

    in_80 = ((sub["actual"] >= sub["lower_80"]) & (sub["actual"] <= sub["upper_80"]))
    in_95 = ((sub["actual"] >= sub["lower_95"]) & (sub["actual"] <= sub["upper_95"]))

    return {
        "cal_80": float(in_80.mean()),
        "cal_95": float(in_95.mean()),
    }


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def compute_summary_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model average MAPE, MAE, RMSE, pinball loss, and
    interval calibration.

    Parameters
    ----------
    results_df : DataFrame
        Output of :func:`rolling_origin_backtest`.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by model name.
    """
    rows = []
    for model_name in results_df["model_name"].unique():
        sub = results_df[results_df["model_name"] == model_name]
        cal = compute_interval_calibration(results_df, model_name)
        row = {
            "model": model_name,
            "avg_MAPE": sub["mape"].mean(),
            "avg_MAE": sub["mae"].mean(),
            "RMSE": np.sqrt((sub["mae"] ** 2).mean()),
            "cal_80": cal["cal_80"],
            "cal_95": cal["cal_95"],
            "n_predictions": len(sub),
        }
        # Add pinball loss metrics if available
        if "pinball_avg" in sub.columns:
            row["avg_pinball_loss"] = sub["pinball_avg"].mean()
            row["pinball_q025"] = sub["pinball_q025"].mean()
            row["pinball_q10"] = sub["pinball_q10"].mean()
            row["pinball_q50"] = sub["pinball_q50"].mean()
            row["pinball_q90"] = sub["pinball_q90"].mean()
            row["pinball_q975"] = sub["pinball_q975"].mean()
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_backtest_results(results_df: pd.DataFrame) -> None:
    """Create and save diagnostic plots to OUTPUT_DIR.

    Generates three figures:
    1. Average MAPE bar chart by model
    2. Interval calibration comparison (80 % and 95 %)
    3. Actual-vs-predicted overlay for each model

    Parameters
    ----------
    results_df : DataFrame
        Output of :func:`rolling_origin_backtest`.
    """
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        pass  # fall back to default style

    model_names = sorted(results_df["model_name"].unique())

    # ── 1. MAPE bar chart ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    mapes = [
        results_df[results_df["model_name"] == m]["mape"].mean()
        for m in model_names
    ]
    bars = ax1.bar(model_names, mapes, color=["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"][:len(model_names)])
    ax1.set_ylabel("Average MAPE (%)")
    ax1.set_title("Backtest – Average MAPE by Model")
    for bar, val in zip(bars, mapes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
        )
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "backtest_mape.png", dpi=150)
    print(f"  → saved {OUTPUT_DIR / 'backtest_mape.png'}")
    plt.close(fig1)

    # ── 2. Calibration comparison ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    cal_80 = []
    cal_95 = []
    for m in model_names:
        c = compute_interval_calibration(results_df, m)
        cal_80.append(c["cal_80"] * 100)
        cal_95.append(c["cal_95"] * 100)

    x = np.arange(len(model_names))
    w = 0.35
    ax2.bar(x - w / 2, cal_80, w, label="80% PI coverage", color="#3498DB")
    ax2.bar(x + w / 2, cal_95, w, label="95% PI coverage", color="#2ECC71")
    ax2.axhline(80, ls="--", color="#3498DB", alpha=0.5, label="Target 80%")
    ax2.axhline(95, ls="--", color="#2ECC71", alpha=0.5, label="Target 95%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Prediction Interval Calibration")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 110)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "backtest_calibration.png", dpi=150)
    print(f"  → saved {OUTPUT_DIR / 'backtest_calibration.png'}")
    plt.close(fig2)

    # ── 3. Actual vs Predicted overlay per model ───────────────────────
    fig3, axes = plt.subplots(
        len(model_names), 1,
        figsize=(14, 4.5 * len(model_names)),
        sharex=True,
    )
    if len(model_names) == 1:
        axes = [axes]

    for ax, m in zip(axes, model_names):
        sub = results_df[results_df["model_name"] == m].sort_values(
            ["origin", "horizon_step"]
        )
        idx = np.arange(len(sub))

        ax.fill_between(
            idx,
            sub["lower_95"].values,
            sub["upper_95"].values,
            alpha=0.15,
            color="#E74C3C",
            label="95% PI",
        )
        ax.fill_between(
            idx,
            sub["lower_80"].values,
            sub["upper_80"].values,
            alpha=0.30,
            color="#E74C3C",
            label="80% PI",
        )
        ax.plot(idx, sub["actual"].values, "o-", ms=4, color="#2C3E50", label="Actual")
        ax.plot(idx, sub["predicted"].values, "s--", ms=4, color="#E74C3C", label="Predicted")
        ax.set_title(f"{m} – Actual vs Predicted (all backtest origins)")
        ax.set_ylabel("Net Flow (₹ Cr)")
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Prediction index (origin × horizon step)")
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "backtest_actual_vs_pred.png", dpi=150)
    print(f"  → saved {OUTPUT_DIR / 'backtest_actual_vs_pred.png'}")
    plt.close(fig3)

    # ── 4. Pinball loss bar chart ──────────────────────────────────────
    if "pinball_avg" in results_df.columns:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        pinball_means = [
            results_df[results_df["model_name"] == m]["pinball_avg"].mean()
            for m in model_names
        ]
        colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"][:len(model_names)]
        bars = ax4.bar(model_names, pinball_means, color=colors)
        ax4.set_ylabel("Average Pinball Loss (₹ Cr)")
        ax4.set_title("Backtest — Average Pinball Loss by Model")
        for bar, val in zip(bars, pinball_means):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pinball_means) * 0.02,
                f"{val:.1f}",
                ha="center",
                fontsize=10,
            )
        fig4.tight_layout()
        fig4.savefig(OUTPUT_DIR / "backtest_pinball_loss.png", dpi=150)
        print(f"  → saved {OUTPUT_DIR / 'backtest_pinball_loss.png'}")
        plt.close(fig4)

        # ── 5. Detailed pinball loss by quantile (grouped bar) ─────────
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        quantile_labels = ["q0.025", "q0.10", "q0.50", "q0.90", "q0.975"]
        quantile_cols = [
            "pinball_q025", "pinball_q10", "pinball_q50",
            "pinball_q90", "pinball_q975",
        ]
        x = np.arange(len(quantile_labels))
        n_models = len(model_names)
        w = 0.8 / max(n_models, 1)

        for j, m in enumerate(model_names):
            sub = results_df[results_df["model_name"] == m]
            vals = [sub[col].mean() for col in quantile_cols]
            ax5.bar(
                x + j * w - (n_models - 1) * w / 2, vals, w,
                label=m, color=colors[j],
            )

        ax5.set_xticks(x)
        ax5.set_xticklabels(quantile_labels)
        ax5.set_ylabel("Average Pinball Loss (₹ Cr)")
        ax5.set_title("Pinball Loss by Quantile Level and Model")
        ax5.legend()
        fig5.tight_layout()
        fig5.savefig(OUTPUT_DIR / "backtest_pinball_by_quantile.png", dpi=150)
        print(f"  → saved {OUTPUT_DIR / 'backtest_pinball_by_quantile.png'}")
        plt.close(fig5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Liquid Fund Cash Buffer – Rolling-Origin Backtest")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\nLoading features from {FEATURE_FILE} ...")
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)[:8]} ...")

    # ── Drop rows with NaN in essential columns ────────────────────────
    essential = ["net_flow_cr"] + ALL_FEATURES
    before = len(df)
    df.dropna(subset=[c for c in essential if c in df.columns], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaNs → {len(df)} remaining.")

    # ── Run backtest ───────────────────────────────────────────────────
    print(f"\nRunning backtest: {BACKTEST_ORIGINS} origins, "
          f"{FORECAST_HORIZON}-step horizon ...\n")

    results = rolling_origin_backtest(
        df,
        feature_cols=ALL_FEATURES,
        exog_cols=EXOG_FEATURES_SARIMAX,
        n_origins=BACKTEST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )

    # ── Summary metrics ────────────────────────────────────────────────
    summary = compute_summary_metrics(results)
    print("\n" + "─" * 60)
    print("  BACKTEST SUMMARY")
    print("─" * 60)
    print(summary.to_string(float_format="{:.2f}".format))

    # ── Save artefacts ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    results.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "backtest_summary.csv")
    print(f"\n  → results saved to {OUTPUT_DIR / 'backtest_results.csv'}")
    print(f"  → summary saved to {OUTPUT_DIR / 'backtest_summary.csv'}")

    # ── Plots ──────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_backtest_results(results)

    print("\n✅ Done.")
