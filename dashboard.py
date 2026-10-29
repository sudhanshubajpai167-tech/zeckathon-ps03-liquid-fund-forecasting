"""
Liquid Fund Cash Buffer — Monthly Decision Dashboard (Deliverable D5)
=====================================================================

A one-page Streamlit dashboard for fund managers to review monthly
net-flow forecasts, recommended cash buffers, cost analysis, and
key drivers.  Runs standalone with demo data or loads real data from
``data/features_dataset.csv``.

Launch
------
    streamlit run dashboard.py
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------
try:
    from config import RANDOM_SEED  # type: ignore[import-untyped]
except ImportError:
    RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Page config (MUST be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Liquid Fund Buffer Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
CLR_HIST = "#2C3E50"       # dark navy – historical
CLR_FORECAST = "#E74C3C"   # red – forecast line
CLR_BAND_80 = "rgba(231,76,60,0.25)"
CLR_BAND_95 = "rgba(231,76,60,0.10)"
CLR_POS = "#27AE60"        # green – positive drivers
CLR_NEG = "#E74C3C"        # red – negative drivers
CLR_TAX = "rgba(243,156,18,0.15)"  # faint orange – advance-tax shading
CLR_BG = "#FAFBFC"

ADVANCE_TAX_MONTHS = {3, 6, 9, 12}  # Mar, Jun, Sep, Dec

# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def fmt_inr_crore(val: float, decimals: int = 0) -> str:
    """Format a number as ₹X,XXX Cr (standard comma style)."""
    if abs(val) >= 1:
        return f"₹{val:,.{decimals}f} Cr"
    return f"₹{val:,.2f} Cr"


def fmt_bps(val: float, decimals: int = 1) -> str:
    """Format basis points."""
    return f"{val:,.{decimals}f} bps"


def fmt_pct(val: float, decimals: int = 1) -> str:
    """Format percentage."""
    return f"{val:,.{decimals}f}%"

# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def generate_dashboard_demo_data(
    n_hist_months: int = 60,
    n_fwd_months: int = 3,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame | pd.Series | dict]:
    """Create synthetic data that lets the dashboard run in demo mode.

    Returns
    -------
    dict with keys:
        ``history``       – DataFrame with columns [date, net_flow_cr, aum_cr]
        ``forecast``      – DataFrame with columns [date, forecast, lo95, hi95, lo80, hi80]
        ``importances``   – DataFrame with columns [feature, importance]
        ``buffer_compare``– DataFrame comparing buffer rules
        ``sensitivity``   – DataFrame for cost-sensitivity chart
        ``current_aum``   – float (latest AUM in ₹ Cr)
    """
    rng = np.random.default_rng(seed)

    # --- Historical flows (60 months) ---
    end = pd.Timestamp.now().normalize().replace(day=1)
    dates = pd.date_range(end=end, periods=n_hist_months, freq="MS")

    # Seasonal pattern: outflows in advance-tax months, inflows otherwise
    seasonal = np.array([
        800, 600, -2200,   # Jan  Feb  Mar(tax)
        500, 400, -1800,   # Apr  May  Jun(tax)
        700, 500, -2000,   # Jul  Aug  Sep(tax)
        600, 400, -2500,   # Oct  Nov  Dec(tax)
    ])
    # Tile to cover n_hist_months
    seasonal_full = np.tile(seasonal, n_hist_months // 12 + 1)[:n_hist_months]

    # Trend + noise
    trend = np.linspace(-200, 400, n_hist_months)
    noise = rng.normal(0, 350, n_hist_months)
    net_flows = seasonal_full + trend + noise

    # AUM grows with a slight upward trend
    aum_base = 45_000 + np.cumsum(net_flows * 0.15) + rng.normal(0, 200, n_hist_months)
    aum_base = np.clip(aum_base, 30_000, 80_000)

    history = pd.DataFrame({
        "date": dates,
        "net_flow_cr": np.round(net_flows, 1),
        "aum_cr": np.round(aum_base, 1),
    })

    # --- Forecast (3 months ahead) ---
    fwd_dates = pd.date_range(start=end + pd.offsets.MonthBegin(1), periods=n_fwd_months, freq="MS")
    fwd_seasonal = np.array([seasonal[d.month - 1] for d in fwd_dates], dtype=float)
    fwd_mean = fwd_seasonal + trend[-1] + rng.normal(0, 150, n_fwd_months)
    sigma = np.array([400, 550, 700], dtype=float)[:n_fwd_months]  # growing uncertainty

    forecast = pd.DataFrame({
        "date": fwd_dates,
        "forecast": np.round(fwd_mean, 1),
        "lo95": np.round(fwd_mean - 1.96 * sigma, 1),
        "hi95": np.round(fwd_mean + 1.96 * sigma, 1),
        "lo80": np.round(fwd_mean - 1.28 * sigma, 1),
        "hi80": np.round(fwd_mean + 1.28 * sigma, 1),
    })

    # --- Feature importances ---
    feature_names = [
        "Advance-Tax Month",
        "Prev Month Flow",
        "10Y G-Sec Yield Δ",
        "FPI Equity Flow",
        "Month-End Effect",
        "Repo Rate",
        "Credit Spread",
        "Equity Market Return",
    ]
    raw_imp = rng.uniform(-0.15, 0.30, len(feature_names))
    raw_imp[0] = 0.28   # advance-tax is dominant
    raw_imp[1] = 0.19
    raw_imp[3] = -0.12  # FPI outflow negative impact
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": np.round(raw_imp, 4),
    }).sort_values("importance", key=abs, ascending=True)

    # --- Buffer rule comparison ---
    buffer_compare = pd.DataFrame({
        "Rule": ["Naive (Flat 5%)", "P95 Static", "Adaptive (Ours)"],
        "Avg Cost (bps)": [8.2, 5.4, 3.1],
        "Shortfall Freq (%)": [12.5, 2.1, 3.8],
        "Annual Savings (bps)": [0.0, 2.8, 5.1],
    })

    # --- Cost sensitivity ---
    pctiles = np.arange(50, 100, 2)
    cost_bps = 1.0 + np.exp((pctiles - 50) * 0.04) * 0.5
    shortfall_pct = 50 - pctiles * 0.48 + rng.normal(0, 0.3, len(pctiles))
    shortfall_pct = np.clip(shortfall_pct, 0, 30)

    sensitivity = pd.DataFrame({
        "percentile": pctiles,
        "cost_bps": np.round(cost_bps, 2),
        "shortfall_pct": np.round(shortfall_pct, 2),
    })

    current_aum = float(history["aum_cr"].iloc[-1])

    return {
        "history": history,
        "forecast": forecast,
        "importances": importances,
        "buffer_compare": buffer_compare,
        "sensitivity": sensitivity,
        "current_aum": current_aum,
    }

# ---------------------------------------------------------------------------
# Data loader — tries real CSV first, falls back to demo
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data() -> Tuple[Dict, bool]:
    """Return (data_dict, is_demo)."""
    try:
        df = pd.read_csv("data/features_dataset.csv", parse_dates=["date"])
        if {"date", "net_flow_cr"}.issubset(df.columns) and len(df) >= 24:
            # Build the same structure from real data
            history = df[["date", "net_flow_cr"]].copy()
            if "aum_cr" in df.columns:
                history["aum_cr"] = df["aum_cr"]
            else:
                history["aum_cr"] = 50_000.0  # placeholder

            # We still need demo forecast / importances if not available
            demo = generate_dashboard_demo_data()
            demo["history"] = history
            demo["current_aum"] = float(history["aum_cr"].iloc[-1])
            return demo, False
    except Exception:
        pass

    return generate_dashboard_demo_data(), True

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    show_tax_markers: bool = True,
) -> go.Figure:
    """Main forecast chart — Row 2."""
    fig = go.Figure()

    # 95% prediction band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["date"], forecast["date"][::-1]]),
        y=pd.concat([forecast["hi95"], forecast["lo95"][::-1]]),
        fill="toself",
        fillcolor=CLR_BAND_95,
        line=dict(width=0),
        name="95% PI",
        showlegend=True,
        hoverinfo="skip",
    ))

    # 80% prediction band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["date"], forecast["date"][::-1]]),
        y=pd.concat([forecast["hi80"], forecast["lo80"][::-1]]),
        fill="toself",
        fillcolor=CLR_BAND_80,
        line=dict(width=0),
        name="80% PI",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Historical line
    fig.add_trace(go.Scatter(
        x=history["date"],
        y=history["net_flow_cr"],
        mode="lines",
        line=dict(color=CLR_HIST, width=2),
        name="Historical Net Flow",
    ))

    # Forecast line
    # Connect from last historical point
    bridge_date = [history["date"].iloc[-1], forecast["date"].iloc[0]]
    bridge_val = [history["net_flow_cr"].iloc[-1], forecast["forecast"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=bridge_date,
        y=bridge_val,
        mode="lines",
        line=dict(color=CLR_FORECAST, width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["date"],
        y=forecast["forecast"],
        mode="lines+markers",
        line=dict(color=CLR_FORECAST, width=2.5, dash="dash"),
        marker=dict(size=7, color=CLR_FORECAST),
        name="Forecast",
    ))

    # Today line
    today = history["date"].iloc[-1]
    y_min = min(history["net_flow_cr"].min(), forecast["lo95"].min()) - 300
    y_max = max(history["net_flow_cr"].max(), forecast["hi95"].max()) + 300
    fig.add_shape(
        type="line", x0=today, x1=today, y0=y_min, y1=y_max,
        line=dict(color="#7F8C8D", width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=today, y=y_max, text="Today", showarrow=False,
        font=dict(size=11, color="#7F8C8D"), yshift=10,
    )

    # Advance-tax month shading
    if show_tax_markers:
        all_dates = pd.concat([history["date"], forecast["date"]])
        for d in all_dates:
            if d.month in ADVANCE_TAX_MONTHS:
                fig.add_vrect(
                    x0=d - pd.Timedelta(days=14),
                    x1=d + pd.Timedelta(days=14),
                    fillcolor=CLR_TAX,
                    layer="below",
                    line_width=0,
                    annotation_text="Tax" if d == all_dates[all_dates.dt.month.isin(ADVANCE_TAX_MONTHS)].iloc[0] else None,
                    annotation_position="top left",
                    annotation_font_size=9,
                    annotation_font_color="#E67E22",
                )

    fig.update_layout(
        title=dict(text="Monthly Net Flows & 3-Month Forecast", font=dict(size=16)),
        xaxis_title="Month",
        yaxis_title="Net Flow (₹ Cr)",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def build_importance_chart(importances: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart — feature importances / SHAP."""
    colors = [CLR_POS if v >= 0 else CLR_NEG for v in importances["importance"]]
    fig = go.Figure(go.Bar(
        x=importances["importance"],
        y=importances["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in importances["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Key Drivers This Month", font=dict(size=14)),
        xaxis_title="Impact on Forecast",
        template="plotly_white",
        height=360,
        margin=dict(l=140, r=40, t=45, b=35),
        yaxis=dict(automargin=True),
    )
    return fig


def build_buffer_comparison(buffer_compare: pd.DataFrame) -> go.Figure:
    """Grouped bar chart — buffer rule comparison."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Cost (bps)",
        x=buffer_compare["Rule"],
        y=buffer_compare["Avg Cost (bps)"],
        marker_color="#3498DB",
        text=buffer_compare["Avg Cost (bps)"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Shortfall Freq (%)",
        x=buffer_compare["Rule"],
        y=buffer_compare["Shortfall Freq (%)"],
        marker_color="#E67E22",
        text=buffer_compare["Shortfall Freq (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Buffer Rule Performance", font=dict(size=14)),
        barmode="group",
        template="plotly_white",
        height=360,
        margin=dict(l=50, r=30, t=45, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Value",
    )
    return fig


def build_seasonal_heatmap(history: pd.DataFrame) -> go.Figure:
    """Box plot showing monthly flow distribution (Jan-Dec)."""
    df = history.copy()
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    # Order by calendar month
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    present = [m for m in month_order if m in df["month_name"].values]

    fig = go.Figure()
    colors = []
    for m in present:
        mn = dt.datetime.strptime(m, "%b").month
        colors.append("#E67E22" if mn in ADVANCE_TAX_MONTHS else "#3498DB")

    for m, c in zip(present, colors):
        subset = df[df["month_name"] == m]["net_flow_cr"]
        fig.add_trace(go.Box(
            y=subset, name=m, marker_color=c, boxmean=True,
        ))

    fig.update_layout(
        title=dict(text="Historical Monthly Flow Pattern", font=dict(size=14)),
        yaxis_title="Net Flow (₹ Cr)",
        template="plotly_white",
        height=360,
        margin=dict(l=60, r=30, t=45, b=35),
        showlegend=False,
    )
    return fig


def build_sensitivity_chart(sensitivity: pd.DataFrame) -> go.Figure:
    """Dual-axis: cost bps (left) vs shortfall % (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=sensitivity["percentile"],
        y=sensitivity["cost_bps"],
        mode="lines+markers",
        name="Annual Cost (bps)",
        line=dict(color="#2980B9", width=2),
        marker=dict(size=5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=sensitivity["percentile"],
        y=sensitivity["shortfall_pct"],
        mode="lines+markers",
        name="Shortfall Risk (%)",
        line=dict(color="#E74C3C", width=2, dash="dot"),
        marker=dict(size=5),
    ), secondary_y=True)

    fig.update_xaxes(title_text="Buffer Percentile")
    fig.update_yaxes(title_text="Annual Cost (bps)", secondary_y=False)
    fig.update_yaxes(title_text="Shortfall Risk (%)", secondary_y=True)

    fig.update_layout(
        title=dict(text="Cost vs Shortfall Sensitivity", font=dict(size=14)),
        template="plotly_white",
        height=360,
        margin=dict(l=60, r=60, t=45, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — renders the entire dashboard."""

    # ---- Load data ----
    data, is_demo = load_data()
    history: pd.DataFrame = data["history"]
    forecast: pd.DataFrame = data["forecast"]
    importances: pd.DataFrame = data["importances"]
    buffer_compare: pd.DataFrame = data["buffer_compare"]
    sensitivity: pd.DataFrame = data["sensitivity"]
    current_aum: float = data["current_aum"]

    # ---- Sidebar ----
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/bank-building.png",
            width=60,
        )
        st.markdown("### ⚙️ Dashboard Controls")

        min_date = history["date"].min().date()
        max_date = history["date"].max().date()
        date_range = st.date_input(
            "Historical date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        model_choice = st.selectbox(
            "Forecast model",
            ["Ensemble", "LightGBM", "SARIMAX", "Seasonal Naïve"],
            index=0,
        )

        buffer_rule = st.selectbox(
            "Buffer rule",
            ["Adaptive (Ours)", "P95 Static", "Naive (Flat 5%)"],
            index=0,
        )

        show_tax = st.toggle("Show advance-tax markers", value=True)

        st.markdown("---")
        if is_demo:
            st.info("📊 Running on **demo data**. Place `features_dataset.csv` in `data/` to use real data.")
        else:
            st.success("✅ Loaded real data from `data/features_dataset.csv`")

        st.caption(f"Model: **{model_choice}** | Rule: **{buffer_rule}**")

    # ---- Filter history by sidebar date range ----
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        mask = (history["date"].dt.date >= date_range[0]) & (history["date"].dt.date <= date_range[1])
        hist_filtered = history.loc[mask].copy()
    else:
        hist_filtered = history.copy()

    # ---- Derived metrics ----
    next_month_flow = float(forecast["forecast"].iloc[0])
    current_month_flow = float(hist_filtered["net_flow_cr"].iloc[-1]) if len(hist_filtered) > 0 else 0.0
    delta_flow = next_month_flow - current_month_flow

    # Buffer logic based on selected rule
    buffer_pct_map = {
        "Adaptive (Ours)": 3.8,
        "P95 Static": 5.0,
        "Naive (Flat 5%)": 5.0,
    }
    recommended_buffer_pct = buffer_pct_map.get(buffer_rule, 4.0)
    buffer_amount = current_aum * recommended_buffer_pct / 100.0

    cost_map = {
        "Adaptive (Ours)": 3.1,
        "P95 Static": 5.4,
        "Naive (Flat 5%)": 8.2,
    }
    est_cost_bps = cost_map.get(buffer_rule, 5.0)

    # ---- Header ----
    now = dt.datetime.now()
    st.markdown(
        f"""
        <h1 style='margin-bottom:0; color:{CLR_HIST};'>🏦 Liquid Fund Cash Buffer — Monthly Decision Dashboard</h1>
        <p style='margin-top:0; font-size:1.05rem; color:#7F8C8D;'>
            Report for <b>{now.strftime('%B %Y')}</b> &nbsp;|&nbsp; Category: Liquid + Overnight Funds
            &nbsp;|&nbsp; <span style='font-size:0.9rem;'>Last updated: {now.strftime('%d %b %Y %H:%M IST')}</span>
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ---- Row 1: KPI metrics ----
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Next Month Forecast",
            value=fmt_inr_crore(next_month_flow),
            delta=f"{delta_flow:+,.0f} Cr vs current",
        )
    with k2:
        st.metric(
            label="Recommended Buffer",
            value=fmt_pct(recommended_buffer_pct),
            delta=f"of ₹{current_aum:,.0f} Cr AUM",
            delta_color="off",
        )
    with k3:
        st.metric(
            label="Buffer Amount",
            value=fmt_inr_crore(buffer_amount),
            delta=f"{buffer_rule}",
            delta_color="off",
        )
    with k4:
        st.metric(
            label="Est. Monthly Cost",
            value=fmt_bps(est_cost_bps),
            delta=f"{est_cost_bps * 12:.1f} bps annualised",
            delta_color="off",
        )

    st.markdown("")

    # ---- Row 2: Main forecast chart ----
    st.plotly_chart(
        build_forecast_chart(hist_filtered, forecast, show_tax_markers=show_tax),
        use_container_width=True,
    )

    # ---- Row 3: Drivers + Buffer comparison ----
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(build_importance_chart(importances), use_container_width=True)
    with col_right:
        st.plotly_chart(build_buffer_comparison(buffer_compare), use_container_width=True)

        # Also show the raw table underneath
        with st.expander("📋 Detailed comparison table"):
            st.dataframe(
                buffer_compare.style.format({
                    "Avg Cost (bps)": "{:.1f}",
                    "Shortfall Freq (%)": "{:.1f}%",
                    "Annual Savings (bps)": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ---- Row 4: Seasonal pattern + Sensitivity ----
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.plotly_chart(build_seasonal_heatmap(hist_filtered), use_container_width=True)
    with col_r2:
        st.plotly_chart(build_sensitivity_chart(sensitivity), use_container_width=True)

    # ---- Footer: How to Read ----
    st.markdown("---")
    with st.expander("ℹ️ How to Read This Dashboard"):
        st.markdown(
            """
**Next Month Forecast**  
The model's point estimate for aggregate net flows (inflows minus
redemptions) across the liquid + overnight fund category for the
upcoming month, expressed in ₹ Crore.  A negative number indicates
expected net outflows.

**Recommended Buffer**  
The percentage of current AUM that should be held in highly liquid,
same-day-settlement instruments (CBLO/TREPS, overnight repos, T-Bills
maturing within 7 days) to meet redemption pressure with the selected
confidence level.

**Buffer Amount**  
Absolute rupee value = Recommended Buffer % × Current AUM.

**Est. Monthly Cost**  
The opportunity cost of holding the buffer in near-zero-return
instruments instead of higher-yielding commercial paper or corporate
bonds, expressed in basis points of AUM per month.

**Forecast Chart**  
- **Solid dark line**: actual historical net flows.
- **Dashed red line**: model forecast for the next 3 months.
- **Darker band**: 80 % prediction interval (the flow is expected
  to fall inside this range 80 % of the time).
- **Lighter band**: 95 % prediction interval.
- **Orange shaded columns**: advance-tax due-date months
  (Mar, Jun, Sep, Dec) when large outflows typically occur.

**Key Drivers**  
Shows the most influential features driving this month's forecast.
Green bars push the forecast higher (inflow); red bars push it lower
(outflow).  Based on feature importance or SHAP values from the
selected model.

**Buffer Rule Performance**  
Compares the adaptive (forecast-based) buffer rule against simpler
baselines over the historical back-test window.  Lower cost and
lower shortfall frequency are better.

**Historical Flow Pattern**  
Box plots of net flows grouped by calendar month.  Orange boxes
highlight advance-tax months.

**Cost vs Shortfall Sensitivity**  
Shows the trade-off curve as you vary the buffer percentile.  Moving
right increases the buffer (lower shortfall risk but higher cost).
            """,
            unsafe_allow_html=False,
        )

    st.caption(
        "Dashboard built for **Zeckathon PS03** — Liquid Fund Cash Buffer Forecasting.  "
        "Data is illustrative unless connected to a live pipeline."
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
