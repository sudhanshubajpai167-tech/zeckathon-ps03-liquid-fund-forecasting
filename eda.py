"""
Exploratory Data Analysis for Liquid Fund Flow Forecasting.
Generates plots and summary statistics to understand the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from config import *


def load_data():
    """Load the feature-engineered dataset."""
    df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} months of data: {df['date'].min()} to {df['date'].max()}")
    return df


def summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS — Net Flow (₹ Crore)")
    print("=" * 70)

    flow = df["net_flow_cr"]
    print(f"  Count:       {len(flow)}")
    print(f"  Mean:        ₹{flow.mean():>12,.0f} Cr")
    print(f"  Std Dev:     ₹{flow.std():>12,.0f} Cr")
    print(f"  Min:         ₹{flow.min():>12,.0f} Cr")
    print(f"  25th %ile:   ₹{flow.quantile(0.25):>12,.0f} Cr")
    print(f"  Median:      ₹{flow.median():>12,.0f} Cr")
    print(f"  75th %ile:   ₹{flow.quantile(0.75):>12,.0f} Cr")
    print(f"  Max:         ₹{flow.max():>12,.0f} Cr")

    print(f"\n  Months with outflows (negative flow): {(flow < 0).sum()} / {len(flow)}")
    print(f"  Months with inflows  (positive flow): {(flow > 0).sum()} / {len(flow)}")

    print("\n  Monthly averages by month:")
    monthly_avg = df.groupby("month")["net_flow_cr"].mean()
    for m in range(1, 13):
        month_name = pd.Timestamp(2020, m, 1).strftime("%B")
        val = monthly_avg.get(m, 0)
        indicator = "📉" if val < 0 else "📈"
        print(f"    {month_name:>12s}: ₹{val:>10,.0f} Cr  {indicator}")

    print("\n  AUM range:")
    print(f"    Start: ₹{df['aum_cr'].iloc[0]:>12,.0f} Cr")
    print(f"    End:   ₹{df['aum_cr'].iloc[-1]:>12,.0f} Cr")
    print(f"    Growth: {(df['aum_cr'].iloc[-1] / df['aum_cr'].iloc[0] - 1) * 100:.1f}%")


def stationarity_tests(series, name="Net Flow"):
    """Run ADF and KPSS stationarity tests."""
    print(f"\n{'=' * 70}")
    print(f"STATIONARITY TESTS — {name}")
    print("=" * 70)

    # ADF test
    adf_result = adfuller(series.dropna(), autolag="AIC")
    print(f"\n  ADF Test:")
    print(f"    Test Statistic:  {adf_result[0]:.4f}")
    print(f"    p-value:         {adf_result[1]:.4f}")
    print(f"    Lags Used:       {adf_result[2]}")
    print(f"    Conclusion:      {'✅ Stationary' if adf_result[1] < 0.05 else '❌ Non-stationary'}")

    # KPSS test
    try:
        kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
        print(f"\n  KPSS Test:")
        print(f"    Test Statistic:  {kpss_result[0]:.4f}")
        print(f"    p-value:         {kpss_result[1]:.4f}")
        print(f"    Conclusion:      {'✅ Stationary' if kpss_result[1] > 0.05 else '❌ Non-stationary'}")
    except Exception as e:
        print(f"\n  KPSS Test: Could not run — {e}")

    return adf_result[1] < 0.05


def plot_time_series(df):
    """Plot the main time series of net flows and AUM."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Net Flows
    ax = axes[0]
    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in df["net_flow_cr"]]
    ax.bar(df["date"], df["net_flow_cr"], color=colors, alpha=0.7, width=20)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Net Flow (₹ Cr)")
    ax.set_title("Monthly Net Flows — Liquid + Overnight Funds", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))

    # Highlight advance-tax months
    for _, row in df[df["is_advance_tax_month"] == 1].iterrows():
        ax.axvline(x=row["date"], color="orange", alpha=0.2, linewidth=8)

    # Plot 2: AUM
    ax = axes[1]
    ax.plot(df["date"], df["aum_cr"], color=COLOR_HISTORICAL, linewidth=2)
    ax.fill_between(df["date"], df["aum_cr"], alpha=0.1, color=COLOR_HISTORICAL)
    ax.set_ylabel("AUM (₹ Cr)")
    ax.set_title("Assets Under Management", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))

    # Plot 3: Flow as % of AUM
    if "flow_rate_pct" in df.columns:
        ax = axes[2]
        colors2 = ["#E74C3C" if v < 0 else "#27AE60" for v in df["flow_rate_pct"]]
        ax.bar(df["date"], df["flow_rate_pct"], color=colors2, alpha=0.7, width=20)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Flow Rate (%)")
        ax.set_title("Net Flow as % of AUM (Normalized)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_time_series_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_time_series_overview.png")


def plot_seasonality(df):
    """Box plot showing monthly seasonality in flows."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of raw flows by month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax = axes[0]
    monthly_data = [df[df["month"] == m]["net_flow_cr"].values for m in range(1, 13)]
    bp = ax.boxplot(monthly_data, labels=month_names, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        color = "#E74C3C" if (i + 1) in [3, 6, 9, 12] else "#3498DB"
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Net Flow (₹ Cr)")
    ax.set_title("Monthly Flow Distribution\n(Red = Advance Tax Months)", fontweight="bold")

    # Average flow by month
    ax = axes[1]
    avg_by_month = df.groupby("month")["net_flow_cr"].mean()
    colors = ["#E74C3C" if m in [3, 6, 9, 12] else "#3498DB" for m in range(1, 13)]
    ax.bar(month_names, avg_by_month.values, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Avg Net Flow (₹ Cr)")
    ax.set_title("Average Monthly Flow Pattern", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_seasonality.png")


def plot_acf_pacf(df):
    """Plot ACF and PACF."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    plot_acf(df["net_flow_cr"].dropna(), ax=axes[0], lags=24, title="Autocorrelation (ACF)")
    plot_pacf(df["net_flow_cr"].dropna(), ax=axes[1], lags=24, title="Partial Autocorrelation (PACF)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_acf_pacf.png")


def plot_decomposition(df):
    """Seasonal decomposition."""
    try:
        result = seasonal_decompose(
            df.set_index("date")["net_flow_cr"],
            model="additive",
            period=12
        )
        fig = result.plot()
        fig.set_size_inches(14, 10)
        fig.suptitle("Seasonal Decomposition of Net Flows", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_decomposition.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: 04_decomposition.png")
    except Exception as e:
        print(f"  Decomposition failed: {e}")


def plot_driver_correlations(df):
    """Correlation analysis between flows and potential drivers."""
    corr_cols = ["net_flow_cr"]
    for col in MACRO_FEATURES + ["is_advance_tax_month", "is_fiscal_yearend"]:
        if col in df.columns:
            corr_cols.append(col)

    corr_matrix = df[corr_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0,
                fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix", fontweight="bold")

    # Bar chart of correlations with net_flow
    ax = axes[1]
    flow_corr = corr_matrix["net_flow_cr"].drop("net_flow_cr").sort_values()
    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in flow_corr.values]
    ax.barh(flow_corr.index, flow_corr.values, color=colors, alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with Net Flow")
    ax.set_title("Driver Correlations", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_correlations.png")


def statistical_tests(df):
    """Run statistical tests for driver significance."""
    print(f"\n{'=' * 70}")
    print("STATISTICAL TESTS — Driver Significance")
    print("=" * 70)

    # T-test: advance-tax months vs non-advance-tax months
    tax_flows = df[df["is_advance_tax_month"] == 1]["net_flow_cr"]
    non_tax_flows = df[df["is_advance_tax_month"] == 0]["net_flow_cr"]
    t_stat, p_val = stats.ttest_ind(tax_flows, non_tax_flows)
    print(f"\n  Advance-Tax Effect (t-test):")
    print(f"    Tax months avg:     ₹{tax_flows.mean():>10,.0f} Cr")
    print(f"    Non-tax months avg: ₹{non_tax_flows.mean():>10,.0f} Cr")
    print(f"    t-statistic: {t_stat:.3f},  p-value: {p_val:.4f}")
    print(f"    Significant: {'✅ YES' if p_val < 0.05 else '❌ NO'}")

    # March (fiscal year-end) vs other months
    mar_flows = df[df["month"] == 3]["net_flow_cr"]
    other_flows = df[df["month"] != 3]["net_flow_cr"]
    t_stat2, p_val2 = stats.ttest_ind(mar_flows, other_flows)
    print(f"\n  Fiscal Year-End Effect — March (t-test):")
    print(f"    March avg:       ₹{mar_flows.mean():>10,.0f} Cr")
    print(f"    Other months:    ₹{other_flows.mean():>10,.0f} Cr")
    print(f"    t-statistic: {t_stat2:.3f},  p-value: {p_val2:.4f}")
    print(f"    Significant: {'✅ YES' if p_val2 < 0.05 else '❌ NO'}")

    # Correlation tests for continuous drivers
    for col in MACRO_FEATURES:
        if col in df.columns:
            valid = df[["net_flow_cr", col]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid["net_flow_cr"], valid[col])
                sig = "✅" if p < 0.05 else "  "
                print(f"\n  {col}:")
                print(f"    Pearson r = {r:+.3f},  p-value = {p:.4f}  {sig}")


def plot_scatter_drivers(df):
    """Scatter plots of net flow vs each continuous driver."""
    continuous_drivers = [c for c in MACRO_FEATURES if c in df.columns]
    n = len(continuous_drivers)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(continuous_drivers):
        ax = axes[i]
        valid = df[["net_flow_cr", col]].dropna()
        ax.scatter(valid[col], valid["net_flow_cr"], alpha=0.5, s=30, color="#3498DB")

        # Add trend line
        z = np.polyfit(valid[col], valid["net_flow_cr"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7)

        ax.set_xlabel(col)
        ax.set_ylabel("Net Flow (₹ Cr)")
        ax.set_title(f"Flow vs {col}", fontsize=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Net Flow vs Macro Drivers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_scatter_drivers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_scatter_drivers.png")


def run_full_eda(df):
    """Run the complete EDA pipeline."""
    print("\n" + "=" * 70)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Summary stats
    summary_statistics(df)

    # Stationarity
    stationarity_tests(df["net_flow_cr"], "Raw Net Flow")
    if "flow_rate_pct" in df.columns:
        stationarity_tests(df["flow_rate_pct"].dropna(), "Flow Rate (% of AUM)")

    # Statistical driver tests
    statistical_tests(df)

    # Plots
    print("\n  Generating plots...")
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    plot_time_series(df)
    plot_seasonality(df)
    plot_acf_pacf(df)
    plot_decomposition(df)
    plot_driver_correlations(df)
    plot_scatter_drivers(df)

    print("\n✅ EDA complete! All plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    df = load_data()
    run_full_eda(df)
