"""
Synthetic data generator for Zeckathon PS03.

Generates realistic monthly liquid fund flow data mimicking AMFI patterns
and macro-economic indicators for India (Apr 2018 – Sep 2026).

When real data becomes available, replace the CSV files in data/ and the
rest of the pipeline works unchanged.
"""

import numpy as np
import pandas as pd
from config import (
    DATA_DIR,
    DATA_START,
    DATA_END,
    RAW_FLOWS_FILE,
    RAW_MACRO_FILE,
)

# ── reproducibility ──────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)


# =====================================================================
# 1. LIQUID FUND FLOWS
# =====================================================================

def generate_synthetic_data(
    start_date: str = DATA_START,
    end_date: str = DATA_END,
) -> pd.DataFrame:
    """
    Generate monthly liquid-fund flow data that mirrors AMFI patterns.

    Returns
    -------
    pd.DataFrame
        Columns: date, net_flow_cr, gross_inflow_cr, gross_outflow_cr,
                 aum_cr, category
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    n = len(dates)

    # ── 1a. AUM growth trend (₹ cr) ─────────────────────────────────
    # Linear-ish ramp from 5,00,000 to 14,00,000 cr with slight
    # curvature (compounding).
    t = np.arange(n)
    aum_base = 500_000 * np.exp(t * np.log(14_00_000 / 5_00_000) / n)

    # ── 1b. Seasonal net-flow pattern ────────────────────────────────
    # Index net-flow as % of current AUM so that flows scale with size.
    month_num = dates.month

    # Base seasonal % of AUM
    seasonal_pct = np.zeros(n)
    for i, m in enumerate(month_num):
        if m == 3:      # March: massive outflow (advance tax + year-end)
            seasonal_pct[i] = RNG.uniform(-0.10, -0.06)
        elif m in (6, 9, 12):  # Other advance-tax quarters
            seasonal_pct[i] = RNG.uniform(-0.04, -0.015)
        elif m == 4:    # April: post-tax cash parking
            seasonal_pct[i] = RNG.uniform(0.04, 0.08)
        else:           # Normal months: slight positive / mixed
            seasonal_pct[i] = RNG.uniform(-0.01, 0.025)

    net_flow_cr = seasonal_pct * aum_base

    # ── 1c. Realistic noise ──────────────────────────────────────────
    noise = RNG.normal(0, 5_000, size=n)
    net_flow_cr += noise

    # ── 1d. Structural events ────────────────────────────────────────
    for i, d in enumerate(dates):
        # IL&FS crisis: Sep–Nov 2018 – massive outflows
        if d.year == 2018 and d.month == 9:
            net_flow_cr[i] -= RNG.uniform(40_000, 55_000)
        elif d.year == 2018 and d.month == 10:
            net_flow_cr[i] -= RNG.uniform(50_000, 70_000)
        elif d.year == 2018 and d.month == 11:
            net_flow_cr[i] -= RNG.uniform(20_000, 35_000)

        # COVID crash: Mar 2020 – huge outflow on top of year-end
        if d.year == 2020 and d.month == 3:
            net_flow_cr[i] -= RNG.uniform(40_000, 60_000)
        # COVID recovery inflow: Apr–May 2020
        if d.year == 2020 and d.month == 4:
            net_flow_cr[i] += RNG.uniform(20_000, 35_000)
        if d.year == 2020 and d.month == 5:
            net_flow_cr[i] += RNG.uniform(10_000, 20_000)

        # Franklin Templeton wound-up (Apr 2020) – spill-over caution
        if d.year == 2020 and d.month == 6:
            net_flow_cr[i] -= RNG.uniform(10_000, 20_000)

    # ── 1e. Gross flows decomposition ────────────────────────────────
    # Gross outflow: roughly 60-85% of AUM monthly (very high turnover
    # in liquid funds).  We model gross outflow as a fraction of AUM,
    # boosted in advance-tax months.
    gross_outflow_pct = np.where(
        np.isin(month_num, [3, 6, 9, 12]),
        RNG.uniform(0.20, 0.30, size=n),   # higher churn in tax months
        RNG.uniform(0.12, 0.20, size=n),   # lower in other months
    )
    gross_outflow_cr = np.abs(gross_outflow_pct * aum_base)

    # Gross inflow = gross outflow + net flow
    gross_inflow_cr = gross_outflow_cr + net_flow_cr
    # Ensure gross inflow is positive
    gross_inflow_cr = np.maximum(gross_inflow_cr, gross_outflow_cr * 0.60)

    # ── 1f. Build AUM path incorporating flows + returns ─────────────
    monthly_return = RNG.uniform(0.003, 0.006, size=n)  # ~4-7% p.a.
    aum = np.empty(n)
    aum[0] = 500_000
    for i in range(1, n):
        aum[i] = aum[i - 1] * (1 + monthly_return[i]) + net_flow_cr[i]
        # Floor at 200k cr (liquid funds never went *that* low)
        aum[i] = max(aum[i], 200_000)

    # Slight rescale so that the last AUM is in the neighbourhood of
    # 14 lakh cr (±1 lakh cr).
    target_final = 14_00_000
    scale = target_final / aum[-1]
    aum *= scale
    # Re-floor first observation
    aum[0] = max(aum[0], 450_000)

    # Round for realism
    net_flow_cr = np.round(net_flow_cr, 1)
    gross_inflow_cr = np.round(gross_inflow_cr, 1)
    gross_outflow_cr = np.round(gross_outflow_cr, 1)
    aum = np.round(aum, 1)

    df = pd.DataFrame({
        "date": dates,
        "net_flow_cr": net_flow_cr,
        "gross_inflow_cr": gross_inflow_cr,
        "gross_outflow_cr": gross_outflow_cr,
        "aum_cr": aum,
        "category": "Liquid + Overnight",
    })
    return df


# =====================================================================
# 2. MACRO-ECONOMIC DATA
# =====================================================================

def _build_repo_rate_series(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Realistic RBI repo-rate path (Apr 2018 → Sep 2026).

    Key moves:
    - Apr 2018: 6.00% → Jun 2018: 6.25% → Aug 2018: 6.50%
    - Feb 2019: first cut → down to 4.00% by May 2020
    - Held at 4.00% until May 2022
    - Hikes from May 2022 → Feb 2023 to 6.50%
    - Held 6.50% → Feb 2025 cut to 6.25%, Apr 2025 cut to 6.00%
    """
    repo = np.empty(len(dates))
    rate_map = {
        # (year, month): rate
        (2018,  4): 6.00, (2018,  5): 6.00, (2018,  6): 6.25,
        (2018,  7): 6.25, (2018,  8): 6.50, (2018,  9): 6.50,
        (2018, 10): 6.50, (2018, 11): 6.50, (2018, 12): 6.50,
        (2019,  1): 6.50, (2019,  2): 6.25, (2019,  3): 6.25,
        (2019,  4): 6.00, (2019,  5): 6.00, (2019,  6): 5.75,
        (2019,  7): 5.75, (2019,  8): 5.40, (2019,  9): 5.40,
        (2019, 10): 5.15, (2019, 11): 5.15, (2019, 12): 5.15,
        (2020,  1): 5.15, (2020,  2): 5.15, (2020,  3): 4.40,
        (2020,  4): 4.40, (2020,  5): 4.00, (2020,  6): 4.00,
        (2020,  7): 4.00, (2020,  8): 4.00, (2020,  9): 4.00,
        (2020, 10): 4.00, (2020, 11): 4.00, (2020, 12): 4.00,
        (2021,  1): 4.00, (2021,  2): 4.00, (2021,  3): 4.00,
        (2021,  4): 4.00, (2021,  5): 4.00, (2021,  6): 4.00,
        (2021,  7): 4.00, (2021,  8): 4.00, (2021,  9): 4.00,
        (2021, 10): 4.00, (2021, 11): 4.00, (2021, 12): 4.00,
        (2022,  1): 4.00, (2022,  2): 4.00, (2022,  3): 4.00,
        (2022,  4): 4.00, (2022,  5): 4.40, (2022,  6): 4.90,
        (2022,  7): 4.90, (2022,  8): 5.40, (2022,  9): 5.40,
        (2022, 10): 5.90, (2022, 11): 5.90, (2022, 12): 6.25,
        (2023,  1): 6.25, (2023,  2): 6.50, (2023,  3): 6.50,
        (2023,  4): 6.50, (2023,  5): 6.50, (2023,  6): 6.50,
        (2023,  7): 6.50, (2023,  8): 6.50, (2023,  9): 6.50,
        (2023, 10): 6.50, (2023, 11): 6.50, (2023, 12): 6.50,
        (2024,  1): 6.50, (2024,  2): 6.50, (2024,  3): 6.50,
        (2024,  4): 6.50, (2024,  5): 6.50, (2024,  6): 6.50,
        (2024,  7): 6.50, (2024,  8): 6.50, (2024,  9): 6.50,
        (2024, 10): 6.50, (2024, 11): 6.50, (2024, 12): 6.50,
        (2025,  1): 6.50, (2025,  2): 6.25, (2025,  3): 6.25,
        (2025,  4): 6.00, (2025,  5): 6.00, (2025,  6): 6.00,
        (2025,  7): 6.00, (2025,  8): 6.00, (2025,  9): 6.00,
        (2025, 10): 6.00, (2025, 11): 6.00, (2025, 12): 6.00,
        (2026,  1): 6.00, (2026,  2): 6.00, (2026,  3): 6.00,
        (2026,  4): 6.00, (2026,  5): 6.00, (2026,  6): 6.00,
        (2026,  7): 6.00, (2026,  8): 6.00, (2026,  9): 6.00,
    }
    for i, d in enumerate(dates):
        key = (d.year, d.month)
        repo[i] = rate_map.get(key, 6.00)
    return repo


def _build_nifty_series(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Realistic Nifty-50 month-end close path (Apr 2018 → Sep 2026).
    """
    # Anchor points (approximate month-end close)
    anchors = {
        (2018,  4): 10_700, (2018,  9): 10_900, (2018, 12): 10_800,
        (2019,  6): 11_900, (2019,  9): 11_500, (2019, 12): 12_200,
        (2020,  1): 12_000, (2020,  3):  8_600, (2020,  4):  9_900,
        (2020,  9): 11_200, (2020, 12): 13_900,
        (2021,  6): 15_800, (2021, 10): 18_100, (2021, 12): 17_400,
        (2022,  3): 17_500, (2022,  6): 15_800, (2022, 12): 18_100,
        (2023,  6): 19_200, (2023, 12): 21_700,
        (2024,  6): 23_700, (2024,  9): 25_800, (2024, 12): 23_600,
        (2025,  6): 24_500, (2025,  9): 25_800,
        (2026,  3): 25_000, (2026,  9): 26_000,
    }

    # Build linearly-interpolated path between anchors, then add noise
    idx_to_val: dict[int, float] = {}
    for i, d in enumerate(dates):
        key = (d.year, d.month)
        if key in anchors:
            idx_to_val[i] = float(anchors[key])

    sorted_keys = sorted(idx_to_val.keys())
    nifty = np.empty(len(dates))
    for k in range(len(sorted_keys) - 1):
        i_start = sorted_keys[k]
        i_end = sorted_keys[k + 1]
        v_start = idx_to_val[i_start]
        v_end = idx_to_val[i_end]
        for j in range(i_start, i_end + 1):
            frac = (j - i_start) / (i_end - i_start)
            nifty[j] = v_start + frac * (v_end - v_start)

    # Fill edges if any remain (before first anchor / after last)
    first_anchor_idx = sorted_keys[0]
    last_anchor_idx = sorted_keys[-1]
    for j in range(0, first_anchor_idx):
        nifty[j] = idx_to_val[first_anchor_idx]
    for j in range(last_anchor_idx + 1, len(dates)):
        nifty[j] = idx_to_val[last_anchor_idx]

    # Add realistic noise (±2%)
    nifty *= (1 + RNG.normal(0, 0.012, size=len(dates)))
    return np.round(nifty, 1)


def generate_macro_data(
    start_date: str = DATA_START,
    end_date: str = DATA_END,
) -> pd.DataFrame:
    """
    Generate macro-economic feature data for India (monthly).

    Returns
    -------
    pd.DataFrame
        Columns: date, repo_rate, reverse_repo_rate, liquidity_surplus_cr,
                 tbill_91d_yield, call_rate, nifty_close,
                 nifty_monthly_return, nifty_monthly_volatility
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    n = len(dates)

    # ── Repo rate ────────────────────────────────────────────────────
    repo = _build_repo_rate_series(dates)

    # ── Reverse repo rate ────────────────────────────────────────────
    # Pre-Apr-2022: spread of 0.25%; post-Apr-2022 RBI switched to SDF,
    # effective spread widened.
    reverse_repo = np.where(
        dates < pd.Timestamp("2022-04-01"),
        repo - 0.25,
        repo - 0.40,                        # SDF era (wider corridor)
    )
    reverse_repo = np.round(reverse_repo, 2)

    # ── Liquidity surplus (₹ cr) ─────────────────────────────────────
    liq = np.empty(n)
    for i, d in enumerate(dates):
        if d < pd.Timestamp("2020-03-01"):
            # Pre-COVID: moderate surplus / sometimes deficit
            liq[i] = RNG.uniform(-100_000, 150_000)
        elif d < pd.Timestamp("2021-06-01"):
            # COVID era: massive RBI injection → huge surplus
            liq[i] = RNG.uniform(300_000, 800_000)
        elif d < pd.Timestamp("2022-06-01"):
            # Withdrawal begins
            liq[i] = RNG.uniform(200_000, 600_000)
        elif d < pd.Timestamp("2023-06-01"):
            # Tighter conditions
            liq[i] = RNG.uniform(-50_000, 250_000)
        else:
            # 2023-mid onwards: mixed
            liq[i] = RNG.uniform(-200_000, 200_000)
    liq = np.round(liq, 0)

    # ── 91-day T-bill yield ──────────────────────────────────────────
    tbill_spread = RNG.uniform(-0.30, 0.10, size=n)
    tbill = repo + tbill_spread
    tbill = np.round(tbill, 2)

    # ── Call / overnight rate ────────────────────────────────────────
    call_spread = RNG.uniform(-0.15, 0.15, size=n)
    call_rate = repo + call_spread
    call_rate = np.round(call_rate, 2)

    # ── Nifty ────────────────────────────────────────────────────────
    nifty = _build_nifty_series(dates)

    nifty_ret = np.empty(n)
    nifty_ret[0] = 0.0
    for i in range(1, n):
        nifty_ret[i] = (nifty[i] / nifty[i - 1]) - 1.0
    nifty_ret = np.round(nifty_ret, 5)

    # ── Monthly realized volatility (annualized, approximate) ────────
    nifty_vol = RNG.uniform(10.0, 25.0, size=n)
    # Spike during COVID
    for i, d in enumerate(dates):
        if d.year == 2020 and d.month in (3, 4, 5):
            nifty_vol[i] = RNG.uniform(40.0, 65.0)
        elif d.year == 2018 and d.month in (10, 11):
            nifty_vol[i] = RNG.uniform(20.0, 35.0)
    nifty_vol = np.round(nifty_vol, 2)

    df = pd.DataFrame({
        "date": dates,
        "repo_rate": repo,
        "reverse_repo_rate": reverse_repo,
        "liquidity_surplus_cr": liq,
        "tbill_91d_yield": tbill,
        "call_rate": call_rate,
        "nifty_close": nifty,
        "nifty_monthly_return": nifty_ret,
        "nifty_monthly_volatility": nifty_vol,
    })
    return df


# =====================================================================
# 3. SAVE HELPERS
# =====================================================================

def save_data(flows_df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
    """Save generated DataFrames to CSV in the data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    flows_df.to_csv(RAW_FLOWS_FILE, index=False)
    print(f"  ✓ Flows data saved → {RAW_FLOWS_FILE}  ({len(flows_df)} rows)")

    macro_df.to_csv(RAW_MACRO_FILE, index=False)
    print(f"  ✓ Macro data saved → {RAW_MACRO_FILE}  ({len(macro_df)} rows)")


# =====================================================================
# 4. MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Zeckathon PS03 – Synthetic Data Generator")
    print("=" * 60)
    print()

    print("[1/3] Generating liquid-fund flow data …")
    flows = generate_synthetic_data()

    print("[2/3] Generating macro-economic data …")
    macro = generate_macro_data()

    print("[3/3] Saving to disk …")
    save_data(flows, macro)

    # ── Summary stats ────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("  FLOW DATA SUMMARY")
    print("-" * 60)
    print(flows[["net_flow_cr", "gross_inflow_cr", "gross_outflow_cr", "aum_cr"]]
          .describe().round(1).to_string())

    print()
    print("  Net-flow by month (mean, ₹ cr):")
    month_means = (
        flows.assign(month=flows["date"].dt.month)
        .groupby("month")["net_flow_cr"]
        .mean()
        .round(0)
    )
    for m, v in month_means.items():
        flag = " ← tax" if m in (3, 6, 9, 12) else ""
        print(f"    Month {m:>2}: {v:>10,.0f}{flag}")

    print()
    print("-" * 60)
    print("  MACRO DATA SUMMARY")
    print("-" * 60)
    print(macro.describe().round(2).to_string())

    print()
    print("✅ Data generation complete.")
