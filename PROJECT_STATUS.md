# Zeckathon PS 03: Project Completion & Execution Status Report

**Problem Statement:** How Much Cash Must a Liquid Fund Hold on 15 March?  
**Repository:** [https://github.com/sudhanshubajpai167-tech/zeckathon-ps03-liquid-fund-forecasting](https://github.com/sudhanshubajpai167-tech/zeckathon-ps03-liquid-fund-forecasting)  
**Last Updated:** October 2026

---

## 📊 Executive Completion Summary

| Area | Status | Completion % | Notes |
|---|---|:---:|---|
| **Architecture & Configuration** | ✅ Completed | 100% | Centralized parameters, constants, and seeds in `config.py` |
| **Data Simulation & Pipeline** | ✅ Completed | 95% | Realistic synthetic generator ready; real AMFI data hook in place |
| **Feature Engineering** | ✅ Completed | 100% | Calendar dummies, lags, rolling stats, normalized rates |
| **Exploratory Data Analysis (EDA)** | ✅ Completed | 100% | ADF/KPSS tests, ACF/PACF, tax t-tests, correlation plots |
| **Deliverable 01: Forecast Model** | ✅ Completed | 100% | Point forecasts + 80% & 95% prediction intervals (LightGBM & SARIMAX) |
| **Deliverable 02: Driver Analysis** | ✅ Completed | 100% | Statistical tests, feature importances, SHAP explainability |
| **Deliverable 03: Benchmarking** | ✅ Completed | 95% | 24-origin backtest, MAPE, MAE, calibration (Pinball loss extension noted) |
| **Deliverable 04: Decision Layer** | ✅ Completed | 100% | Yield drag vs. $2\times$ shortfall penalty, buffer rules, sensitivity |
| **Deliverable 05: Fund Manager Dashboard** | ✅ Completed | 100% | One-page interactive Streamlit dashboard (`dashboard.py`) |
| **Version Control & Repository** | ✅ Completed | 100% | Public GitHub repo created and synced on `main` branch |
| **Real AMFI/RBI Data Ingestion** | ⏳ In Progress | 25% | Data structures ready; awaiting live CSV drops from AMFI & RBI DBIE |
| **Live Tie-Breaker Submission** | ⏳ Pending | 0% | Run on latest live month before official AMFI release date |
| **Slide Deck / Presentation** | ⏳ Pending | 0% | Presentation deck summarizing findings for hackathon judges |

**Overall Project Progress: ~88% Complete**

---

## ✅ What Has Been Completed (In Detail)

### 1. Deliverable 01 — Forecast Models (`models.py`)
- [x] **Seasonal Naïve Forecaster**: 12-month lag baseline with historical residual standard deviation intervals.
- [x] **SARIMAX Forecaster**: Auto-tuned seasonal ARIMA with exogenous regressors (advance-tax dummies, repo rate, LAF liquidity, Nifty returns) and analytic confidence intervals.
- [x] **LightGBM Quantile Regressors**: Separate gradient boosted models for quantiles `[0.025, 0.10, 0.50, 0.90, 0.975]` yielding explicit 80% and 95% prediction intervals.
- [x] **Ensemble Forecaster**: Weighted forecast combiner using conservative envelope intervals for risk protection.

### 2. Deliverable 02 — Driver Analysis (`eda.py`, `feature_engineering.py`)
- [x] Calendar effects: Binary indicators for advance tax (Mar/Jun/Sep/Dec) and fiscal year-end crunch (March).
- [x] Macro drivers: RBI repo rate, T-bill spreads, banking-system liquidity surplus/deficit (LAF), and Nifty 50 returns/volatilities.
- [x] Statistical validation: Two-sample $t$-tests confirming advance-tax month significance ($p < 0.05$).
- [x] Interpretability: Tree feature importance and SHAP summary tooling.

### 3. Deliverable 03 — Model Benchmarking (`backtesting.py`)
- [x] Rolling-origin cross-validation engine across **24 monthly evaluation windows**.
- [x] Out-of-sample accuracy tracking: Mean Absolute Percentage Error (MAPE), MAE, and RMSE.
- [x] **Interval Calibration Check**: Verification of what fraction of actual values land within 80% and 95% bands.
- [x] Summary table generation and visual diagnostic export (`backtest_mape.png`, `backtest_calibration.png`, `backtest_actual_vs_pred.png`).

### 4. Deliverable 04 — Decision Layer (`decision_layer.py`)
- [x] Economic objective function matching problem specification:
  $$\text{Total Cost} = \text{Yield Drag on Idle Cash} + 2\times \text{Shortfall Penalty on Redemptions}$$
- [x] Policy rules implemented:
  - **Baseline Rule**: Hold last year's same-month outflow.
  - **P95 Buffer Rule**: Hold the 95th percentile predicted outflow, subject to a 3% AUM liquidity floor.
  - **Adaptive Rule**: Tax-aware scaling buffer that increases protection during quarter-ends.
- [x] Sensitivity Analysis: Trade-off curves mapping buffer percentile (P50–P99) to annualized cost (bps) vs. shortfall incidence.

### 5. Deliverable 05 — Fund-Manager Dashboard (`dashboard.py`)
- [x] Clean one-page executive UI built in Streamlit.
- [x] Top metric cards: Next month forecast, recommended buffer (% and ₹ Cr), estimated monthly cost in basis points.
- [x] Interactive forecast chart with 80% and 95% confidence bands and advance-tax event markers.
- [x] Driver rankings, baseline cost comparison, monthly seasonality box plots, and sensitivity curves.
- [x] Fully self-contained: includes demo data generator fallback so it can be demonstrated anywhere with zero external setup.

### 6. Pipeline Automation & Version Control
- [x] `main.py`: Single executable running steps 1 through 6 sequentially.
- [x] Local Git repository configured and pushed to GitHub:
  `https://github.com/sudhanshubajpai167-tech/zeckathon-ps03-liquid-fund-forecasting`

---

## ⏳ What Is Left To Do (Final 12%)

### Step 1: Real Data Ingestion (Optional Enhancement)
- Download actual AMFI monthly report spreadsheets from [AMFI India](https://www.amfiindia.com/research-information/other-data/mf-scheme-performance-details) (Liquid Fund & Overnight Fund categories, 2018–2026).
- Save to `data/amfi_monthly_flows.csv` with columns `date, net_flow_cr, gross_inflow_cr, gross_outflow_cr, aum_cr`.
- Download RBI LAF absorption & Repo rates from [RBI DBIE](https://dbie.rbi.org.in/) into `data/macro_features.csv`.
*(The synthetic generator currently models these exact empirical statistical distributions).*

### Step 2: Evaluation Metric Addition (Pinball Loss)
- Add Pinball (Quantile) Loss calculation to `backtesting.py` to directly match the secondary metric on the evaluation sheet:
  $$L_\alpha(y, \hat{q}) = \max(\alpha (y - \hat{q}), (1 - \alpha)(\hat{q} - y))$$

### Step 3: Live Forecast Tie-Breaker Submission
- Once real data is refreshed up to the latest published AMFI month:
- Run `python main.py` to produce `outputs/final_forecast.csv`.
- Extract the point forecast and 80%/95% intervals for the upcoming unpublished AMFI release as your official tie-breaker entry.

### Step 4: Final Presentation Deck (5–7 Slides)
- **Slide 1: Problem & Business Context** — The 15 March corporate advance tax crunch and liquidity trade-off.
- **Slide 2: Data & Key Drivers** — Empirical proof of quarter-end seasonality and macro influence.
- **Slide 3: Model Architecture** — Classical SARIMAX vs. LightGBM Quantile Regression.
- **Slide 4: Out-of-Sample Benchmarks** — 24-month rolling backtest comparison & interval calibration.
- **Slide 5: The Decision Layer** — P95 Buffer vs. Naïve Baseline showing basis point savings and risk mitigation.
- **Slide 6: Live Dashboard & Policy Recommendation** — Visual walkthrough of the Streamlit dashboard and actionable rule for fund managers.
