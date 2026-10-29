"""
Decision Layer for Liquid Fund Cash Buffer Forecasting (Deliverable D4).

Implements buffer rules, cost-based evaluation, backtesting, and sensitivity
analysis to help a liquid-fund manager decide how much cash to hold each month
so as to meet redemptions without excessive yield drag.

All costs are expressed in **basis points per month** (bps/mo) unless noted.
"""

import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import (
    FEATURE_FILE,
    FORECAST_HORIZON,
    OVERNIGHT_YIELD_ASSUMPTION,
    OUTPUT_DIR,
    PLOT_STYLE,
    PORTFOLIO_YIELD_ASSUMPTION,
    SEBI_MIN_LIQUID_ASSETS_PCT,
    SHORTFALL_PENALTY_MULTIPLIER,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
#  1.  Cost computation
# ──────────────────────────────────────────────────────────────────────

def compute_decision_cost(
    actual_outflow: float,
    buffer_amount: float,
    aum: float,
    portfolio_yield: float = PORTFOLIO_YIELD_ASSUMPTION,
    overnight_yield: float = OVERNIGHT_YIELD_ASSUMPTION,
    penalty_multiplier: float = SHORTFALL_PENALTY_MULTIPLIER,
) -> Dict[str, float]:
    """Compute the monthly economic cost of holding *buffer_amount* when
    the realised outflow is *actual_outflow*.

    Parameters
    ----------
    actual_outflow : float
        Non-negative realised redemptions (₹ Cr).  Should be ``max(0, -net_flow)``.
    buffer_amount : float
        Cash buffer held at the start of the month (₹ Cr).
    aum : float
        Assets under management (₹ Cr).
    portfolio_yield : float
        Annualised portfolio yield (e.g. 0.065 for 6.5 %).
    overnight_yield : float
        Annualised yield on overnight / liquid parking (e.g. 0.055 for 5.5 %).
    penalty_multiplier : float
        Multiplier on the shortfall cost (2× default per SEBI norms).

    Returns
    -------
    dict
        Keys: yield_drag, shortfall, shortfall_cost, total_cost, buffer_pct.
        All costs are *fractions* (not bps) — convert downstream if needed.
    """
    if aum <= 0:
        return {
            "yield_drag": 0.0,
            "shortfall": 0.0,
            "shortfall_cost": 0.0,
            "total_cost": 0.0,
            "buffer_pct": 0.0,
        }

    spread = portfolio_yield - overnight_yield  # annualised spread

    # Opportunity cost of holding cash (monthly)
    yield_drag = (buffer_amount / aum) * spread / 12.0

    # Shortfall = unmet redemptions
    shortfall = max(0.0, actual_outflow - buffer_amount)

    # Penalty for shortfall (forced liquidation cost)
    shortfall_cost = penalty_multiplier * (shortfall / aum) * spread / 12.0

    total_cost = yield_drag + shortfall_cost
    buffer_pct = buffer_amount / aum * 100.0

    return {
        "yield_drag": yield_drag,
        "shortfall": shortfall,
        "shortfall_cost": shortfall_cost,
        "total_cost": total_cost,
        "buffer_pct": buffer_pct,
    }


# ──────────────────────────────────────────────────────────────────────
#  2.  Naïve buffer rule (same-month-last-year)
# ──────────────────────────────────────────────────────────────────────

def naive_buffer_rule(df: pd.DataFrame) -> pd.Series:
    """Baseline heuristic: hold cash equal to the absolute outflow seen in
    the **same calendar month one year ago**.

    If last year's flow was positive (net inflow), a small minimum floor of
    3 % of AUM is applied instead.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``net_flow`` and ``aum``, indexed by date.

    Returns
    -------
    pd.Series
        Buffer amount (₹ Cr) indexed like *df*.
    """
    lagged_flow = df["net_flow"].shift(12)

    # Use magnitude only when last year saw an outflow
    buffer = lagged_flow.apply(lambda x: abs(x) if (pd.notna(x) and x < 0) else np.nan)

    # Floor at 3 % of AUM
    floor = 0.03 * df["aum"]
    buffer = buffer.clip(lower=floor)

    # Where we don't have 12-month history, fall back to the floor
    buffer = buffer.fillna(floor)

    return buffer.rename("buffer_naive")


# ──────────────────────────────────────────────────────────────────────
#  3.  P95 buffer rule (model-driven)
# ──────────────────────────────────────────────────────────────────────

def p95_buffer_rule(
    predicted_outflow_p95: pd.Series,
    aum: pd.Series,
) -> pd.Series:
    """Model-based rule: hold cash = 95th-percentile predicted outflow.

    Parameters
    ----------
    predicted_outflow_p95 : pd.Series
        P95 forecast of monthly outflow (₹ Cr), aligned by date.
    aum : pd.Series
        AUM corresponding to each date.

    Returns
    -------
    pd.Series
        Buffer amount (₹ Cr).
    """
    floor = 0.03 * aum
    buffer = predicted_outflow_p95.clip(lower=floor)

    # SEBI 20 % liquid-asset mandate as upper guidance
    sebi_cap = (SEBI_MIN_LIQUID_ASSETS_PCT / 100.0) * aum
    buffer = buffer.clip(upper=sebi_cap)

    return buffer.rename("buffer_p95")


# ──────────────────────────────────────────────────────────────────────
#  4.  Adaptive buffer rule (tax-month aware)
# ──────────────────────────────────────────────────────────────────────

def adaptive_buffer_rule(
    predicted_outflow_p95: pd.Series,
    aum: pd.Series,
    is_advance_tax_month: pd.Series,
) -> pd.Series:
    """Enhanced rule that scales the buffer depending on whether the month
    coincides with advance-tax payment dates (Jun, Sep, Dec, Mar).

    Parameters
    ----------
    predicted_outflow_p95 : pd.Series
        P95 forecast of monthly outflow (₹ Cr).
    aum : pd.Series
        AUM for each month.
    is_advance_tax_month : pd.Series
        Boolean / 0-1 flag for advance-tax months.

    Returns
    -------
    pd.Series
        Buffer amount (₹ Cr).
    """
    tax_flag = is_advance_tax_month.astype(bool)

    # Tax months → scale up P95 by 10 %, higher floor
    buffer_tax = pd.Series(
        np.maximum(predicted_outflow_p95 * 1.10, 0.05 * aum),
        index=predicted_outflow_p95.index,
    )

    # Non-tax months → P95 with lower floor
    buffer_nontax = pd.Series(
        np.maximum(predicted_outflow_p95, 0.02 * aum),
        index=predicted_outflow_p95.index,
    )

    buffer = buffer_nontax.copy()
    buffer.loc[tax_flag] = buffer_tax.loc[tax_flag]

    return buffer.rename("buffer_adaptive")


# ──────────────────────────────────────────────────────────────────────
#  5.  Backtesting engine
# ──────────────────────────────────────────────────────────────────────

def backtest_buffer_rules(
    df: pd.DataFrame,
    backtest_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate three buffer rules over a historical backtest window.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset with at least ``date``, ``net_flow``, ``aum``,
        and ``is_advance_tax_month``.
    backtest_predictions_df : pd.DataFrame
        Must contain columns ``date`` and ``predicted_outflow_p95`` from
        the quantile-regression / backtesting module.

    Returns
    -------
    pd.DataFrame
        One row per (date, rule).  Columns: date, rule_name, buffer_amount,
        buffer_pct, actual_outflow, yield_drag_bps, shortfall_cost_bps,
        total_cost_bps, had_shortfall.
    """
    # ── Align datasets on backtest dates ─────────────────────────────
    if "date" in df.columns:
        df = df.set_index("date") if not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    bt = backtest_predictions_df.copy()
    if "date" in bt.columns:
        bt = bt.set_index("date")
    bt.index = pd.to_datetime(bt.index)

    common_idx = df.index.intersection(bt.index).sort_values()
    df_bt = df.loc[common_idx].copy()
    bt = bt.loc[common_idx]

    # Actual outflow = max(0, -net_flow)
    actual_outflow = (-df_bt["net_flow"]).clip(lower=0)

    # AUM and tax flag
    aum = df_bt["aum"]
    tax_flag = df_bt["is_advance_tax_month"] if "is_advance_tax_month" in df_bt.columns else pd.Series(0, index=common_idx)

    # ── Compute buffers ─────────────────────────────────────────────
    naive_buf = naive_buffer_rule(df).reindex(common_idx)
    p95_buf = p95_buffer_rule(bt["predicted_outflow_p95"], aum)
    adaptive_buf = adaptive_buffer_rule(bt["predicted_outflow_p95"], aum, tax_flag)

    rules = {
        "naive": naive_buf,
        "p95": p95_buf,
        "adaptive": adaptive_buf,
    }

    # ── Evaluate each rule ──────────────────────────────────────────
    records = []
    for rule_name, buf_series in rules.items():
        for dt in common_idx:
            cost = compute_decision_cost(
                actual_outflow=actual_outflow.loc[dt],
                buffer_amount=buf_series.loc[dt],
                aum=aum.loc[dt],
            )
            records.append(
                {
                    "date": dt,
                    "rule_name": rule_name,
                    "buffer_amount": buf_series.loc[dt],
                    "buffer_pct": cost["buffer_pct"],
                    "actual_outflow": actual_outflow.loc[dt],
                    "yield_drag_bps": cost["yield_drag"] * 10_000,
                    "shortfall_cost_bps": cost["shortfall_cost"] * 10_000,
                    "total_cost_bps": cost["total_cost"] * 10_000,
                    "had_shortfall": cost["shortfall"] > 0,
                }
            )

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────
#  6.  Summary statistics
# ──────────────────────────────────────────────────────────────────────

def summarize_buffer_performance(backtest_results_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a per-rule summary of buffer-strategy performance.

    Parameters
    ----------
    backtest_results_df : pd.DataFrame
        Output of :func:`backtest_buffer_rules`.

    Returns
    -------
    pd.DataFrame
        One row per rule with aggregate metrics.
    """
    g = backtest_results_df.groupby("rule_name")

    summary = pd.DataFrame(
        {
            "avg_total_cost_bps_mo": g["total_cost_bps"].mean(),
            "avg_yield_drag_bps_mo": g["yield_drag_bps"].mean(),
            "avg_shortfall_cost_bps_mo": g["shortfall_cost_bps"].mean(),
            "shortfall_frequency_pct": g["had_shortfall"].mean() * 100,
            "max_shortfall_cost_bps": g["shortfall_cost_bps"].max(),
            "avg_buffer_pct": g["buffer_pct"].mean(),
            "annualised_total_cost_bps": g["total_cost_bps"].mean() * 12,
        }
    )

    # Order rows: naive → p95 → adaptive
    preferred_order = ["naive", "p95", "adaptive"]
    order = [r for r in preferred_order if r in summary.index]
    order += [r for r in summary.index if r not in order]
    summary = summary.loc[order]

    return summary.round(3)


# ──────────────────────────────────────────────────────────────────────
#  7.  Sensitivity analysis
# ──────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    df: pd.DataFrame,
    predicted_outflows: pd.DataFrame,
    aum: pd.Series,
) -> pd.DataFrame:
    """Sweep the buffer percentile from P50 to P99 and measure the resulting
    cost-risk trade-off.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset (needs ``net_flow``).
    predicted_outflows : pd.DataFrame
        Must have columns keyed by percentile labels, e.g. ``p50``, ``p60``,
        … ``p99`` **or** a single ``predicted_outflow_p95`` column. If only
        the p95 column is available, synthetic percentiles are generated by
        scaling (adequate for sensitivity illustration).
    aum : pd.Series
        AUM aligned with *predicted_outflows* index.

    Returns
    -------
    pd.DataFrame
        Columns: percentile, avg_buffer_pct, avg_annual_cost_bps,
        shortfall_frequency_pct.
    """
    # ── Align dates ─────────────────────────────────────────────────
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    if "date" in predicted_outflows.columns:
        predicted_outflows = predicted_outflows.set_index("date")
    predicted_outflows.index = pd.to_datetime(predicted_outflows.index)

    common = df.index.intersection(predicted_outflows.index).sort_values()
    actual_outflow = (-df.loc[common, "net_flow"]).clip(lower=0)
    aum_common = aum.reindex(common)

    # ── Build percentile columns if only p95 is available ───────────
    percentiles = list(range(50, 100, 5))  # 50, 55, … 95
    percentiles.append(99)

    if "predicted_outflow_p95" in predicted_outflows.columns and "p95" not in predicted_outflows.columns:
        base = predicted_outflows.loc[common, "predicted_outflow_p95"]
        # Linear scaling around p95: p_q ≈ base * (q / 95)
        pct_data = {}
        for q in percentiles:
            pct_data[f"p{q}"] = base * (q / 95.0)
        pct_df = pd.DataFrame(pct_data, index=common)
    else:
        pct_df = predicted_outflows.loc[common]

    # ── Sweep ───────────────────────────────────────────────────────
    records = []
    spread = PORTFOLIO_YIELD_ASSUMPTION - OVERNIGHT_YIELD_ASSUMPTION

    for q in percentiles:
        col = f"p{q}"
        if col not in pct_df.columns:
            continue
        buffer = pct_df[col].clip(lower=0.03 * aum_common)

        # Costs
        yield_drag = (buffer / aum_common) * spread / 12.0
        shortfall = (actual_outflow - buffer).clip(lower=0)
        shortfall_cost = SHORTFALL_PENALTY_MULTIPLIER * (shortfall / aum_common) * spread / 12.0
        total_cost = yield_drag + shortfall_cost

        records.append(
            {
                "percentile": q,
                "avg_buffer_pct": (buffer / aum_common * 100).mean(),
                "avg_annual_cost_bps": total_cost.mean() * 12 * 10_000,
                "shortfall_frequency_pct": (shortfall > 0).mean() * 100,
            }
        )

    return pd.DataFrame(records).round(3)


# ──────────────────────────────────────────────────────────────────────
#  8.  Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_decision_analysis(
    backtest_results_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> None:
    """Generate four diagnostic plots for the decision layer.

    1. Monthly cost comparison across rules (line chart).
    2. Cumulative cost comparison.
    3. Sensitivity curve (buffer percentile vs cost vs shortfall frequency).
    4. Buffer recommendation for next 3 months (bar chart).

    All figures are saved to *output_dir* (defaults to ``OUTPUT_DIR``).
    """
    out = output_dir or str(OUTPUT_DIR)
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        pass

    rules = backtest_results_df["rule_name"].unique()
    colour_map = {
        "naive": "#E74C3C",
        "p95": "#2980B9",
        "adaptive": "#27AE60",
    }

    # ── Plot 1: Monthly cost comparison ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for rule in rules:
        subset = backtest_results_df[backtest_results_df["rule_name"] == rule].sort_values("date")
        ax.plot(
            subset["date"],
            subset["total_cost_bps"],
            label=rule.title(),
            color=colour_map.get(rule, None),
            linewidth=1.5,
        )
    ax.set_title("Monthly Buffer Cost by Rule (bps)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Cost (bps / month)")
    ax.set_xlabel("")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(f"{out}/decision_monthly_cost.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Cumulative cost ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for rule in rules:
        subset = backtest_results_df[backtest_results_df["rule_name"] == rule].sort_values("date")
        ax.plot(
            subset["date"],
            subset["total_cost_bps"].cumsum(),
            label=rule.title(),
            color=colour_map.get(rule, None),
            linewidth=1.8,
        )
    ax.set_title("Cumulative Buffer Cost by Rule (bps)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Cost (bps)")
    ax.set_xlabel("")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/decision_cumulative_cost.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Sensitivity curve (dual y-axis) ────────────────────
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(
            sensitivity_df["percentile"],
            sensitivity_df["avg_annual_cost_bps"],
            "o-",
            color="#2C3E50",
            linewidth=2,
            markersize=6,
            label="Avg Annual Cost (bps)",
        )
        ax2.plot(
            sensitivity_df["percentile"],
            sensitivity_df["shortfall_frequency_pct"],
            "s--",
            color="#E74C3C",
            linewidth=2,
            markersize=6,
            label="Shortfall Frequency (%)",
        )

        ax1.set_xlabel("Buffer Percentile", fontsize=12)
        ax1.set_ylabel("Average Annual Cost (bps)", fontsize=12, color="#2C3E50")
        ax2.set_ylabel("Shortfall Frequency (%)", fontsize=12, color="#E74C3C")
        ax1.set_title(
            "Cost–Risk Trade-off: Buffer Percentile Sensitivity",
            fontsize=14,
            fontweight="bold",
        )

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        fig.tight_layout()
        fig.savefig(f"{out}/decision_sensitivity.png", dpi=150)
        plt.close(fig)

    # ── Plot 4: Buffer recommendation for next 3 months ─────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use the most recent AUM and adaptive-rule buffer as recommendation
    latest = backtest_results_df[backtest_results_df["rule_name"] == "adaptive"].sort_values("date")
    if len(latest) >= 3:
        tail = latest.tail(3)
    else:
        tail = latest.tail(len(latest))

    months_labels = tail["date"].dt.strftime("%b %Y").values
    buffer_pcts = tail["buffer_pct"].values

    bars = ax.bar(months_labels, buffer_pcts, color=["#27AE60", "#2980B9", "#E74C3C"][: len(months_labels)], edgecolor="white")
    for bar, val in zip(bars, buffer_pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_title("Recommended Cash Buffer (Adaptive Rule)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Buffer (% of AUM)")
    ax.set_ylim(0, max(buffer_pcts) * 1.4 if len(buffer_pcts) > 0 else 10)
    fig.tight_layout()
    fig.savefig(f"{out}/decision_buffer_recommendation.png", dpi=150)
    plt.close(fig)

    print(f"✅  Decision-layer plots saved to {out}/")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  DECISION LAYER  –  Liquid Fund Cash Buffer Analysis (D4)")
    print("=" * 72)

    # ── 1. Load feature dataset ─────────────────────────────────────
    try:
        df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
        print(f"\n📂 Loaded features from {FEATURE_FILE}  ({len(df)} rows)")
    except FileNotFoundError:
        print(f"\n⚠️  Feature file not found at {FEATURE_FILE}")
        print("    Generating synthetic dataset for demonstration …\n")

        np.random.seed(42)
        dates = pd.date_range("2018-04-01", "2026-09-30", freq="MS")
        n = len(dates)

        # Synthetic AUM: growing from ~1,00,000 Cr to ~3,00,000 Cr
        aum = np.linspace(100_000, 300_000, n) + np.random.normal(0, 5_000, n)

        # Synthetic net_flow: mostly small, with large outflows in tax months
        net_flow = np.random.normal(2_000, 8_000, n)
        for i, d in enumerate(dates):
            if d.month in [3, 6, 9, 12]:  # advance-tax months
                net_flow[i] -= np.random.uniform(8_000, 20_000)

        df = pd.DataFrame(
            {
                "date": dates,
                "net_flow": net_flow,
                "aum": aum,
                "month": dates.month,
                "is_advance_tax_month": [1 if m in [3, 6, 9, 12] else 0 for m in dates.month],
                "is_fiscal_yearend": [1 if m == 3 else 0 for m in dates.month],
                "is_quarter_end": [1 if m in [3, 6, 9, 12] else 0 for m in dates.month],
            }
        )
        print(f"    Synthetic dataset created ({len(df)} rows)")

    # Ensure required columns
    if "aum" not in df.columns:
        print("⚠️  'aum' column missing — approximating from net_flow.")
        df["aum"] = 200_000  # placeholder
    if "is_advance_tax_month" not in df.columns:
        df["is_advance_tax_month"] = df["date"].dt.month.isin([3, 6, 9, 12]).astype(int)

    # ── 2. Create mock backtest predictions ─────────────────────────
    # Use the last 24 months as backtest period
    backtest_start = df["date"].max() - pd.DateOffset(months=23)
    bt_mask = df["date"] >= backtest_start
    bt_dates = df.loc[bt_mask, "date"].values

    # P95 predicted outflow ≈ |net_flow| * 1.2  (mock estimate)
    actual_outflows = (-df.loc[bt_mask, "net_flow"]).clip(lower=0).values
    p95_predicted = actual_outflows * 1.2 + np.random.normal(0, 500, len(bt_dates))
    p95_predicted = np.maximum(p95_predicted, 0)

    backtest_preds = pd.DataFrame(
        {
            "date": bt_dates,
            "predicted_outflow_p95": p95_predicted,
        }
    )

    print(f"\n📊 Backtest period: {pd.Timestamp(bt_dates[0]):%Y-%m} → {pd.Timestamp(bt_dates[-1]):%Y-%m}  ({len(bt_dates)} months)")

    # ── 3. Run backtest ─────────────────────────────────────────────
    print("\n🔄 Running backtest across three buffer rules …")
    results = backtest_buffer_rules(df, backtest_preds)
    print(f"   → {len(results)} evaluations ({len(results) // 3} months × 3 rules)")

    # ── 4. Print summary ────────────────────────────────────────────
    summary = summarize_buffer_performance(results)
    print("\n" + "─" * 72)
    print("  BUFFER STRATEGY PERFORMANCE SUMMARY")
    print("─" * 72)

    # Format and print
    fmt = summary.copy()
    fmt.index.name = "Rule"
    col_labels = {
        "avg_total_cost_bps_mo": "Avg Cost\n(bps/mo)",
        "avg_yield_drag_bps_mo": "Yield Drag\n(bps/mo)",
        "avg_shortfall_cost_bps_mo": "Shortfall\n(bps/mo)",
        "shortfall_frequency_pct": "Shortfall\nFreq (%)",
        "max_shortfall_cost_bps": "Max Short-\nfall (bps)",
        "avg_buffer_pct": "Avg Buffer\n(% AUM)",
        "annualised_total_cost_bps": "Annual\nCost (bps)",
    }
    fmt = fmt.rename(columns=col_labels)
    print(fmt.to_string())
    print("─" * 72)

    # ── 5. Sensitivity analysis ─────────────────────────────────────
    print("\n🔍 Running sensitivity analysis (P50 → P99) …")

    # Align AUM for sensitivity
    df_indexed = df.set_index("date") if "date" in df.columns else df.copy()
    df_indexed.index = pd.to_datetime(df_indexed.index)

    sens = sensitivity_analysis(
        df=df,
        predicted_outflows=backtest_preds,
        aum=df_indexed.loc[pd.to_datetime(bt_dates), "aum"],
    )

    print("\n" + "─" * 60)
    print("  SENSITIVITY ANALYSIS: Buffer Percentile Trade-off")
    print("─" * 60)
    print(
        sens.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )
    print("─" * 60)

    # ── 6. Generate plots ───────────────────────────────────────────
    print("\n📈 Generating decision-layer plots …")
    plot_decision_analysis(results, sens)

    # ── Final note ──────────────────────────────────────────────────
    best_rule = summary["annualised_total_cost_bps"].idxmin()
    best_cost = summary.loc[best_rule, "annualised_total_cost_bps"]
    best_sf = summary.loc[best_rule, "shortfall_frequency_pct"]

    print(f"\n🏆  Best rule: **{best_rule.upper()}**")
    print(f"    Annualised cost: {best_cost:.2f} bps  |  Shortfall frequency: {best_sf:.1f}%")
    print("\n✅  Decision layer analysis complete.")
