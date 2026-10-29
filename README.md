# Zeckathon PS 03 — Liquid Fund Cash Buffer Forecasting

> **Problem:** How Much Cash Must a Liquid Fund Hold on 15 March?  
> Forecasting net flows for Indian liquid and overnight mutual fund categories and turning them into an optimal cash-buffer policy balancing yield drag against redemption shortfall risk.

---

## 📌 Project Overview

Corporate treasuries park surplus liquidity in liquid mutual funds and redeem in large bursts around quarter-end advance-tax payment deadlines (15 June, 15 September, 15 December, and especially **15 March** fiscal year-end).

This repository contains an end-to-end quantitative solution:
1. **Data Layer (`data_generator.py`)**: Realistic Indian financial cycle simulation (AMFI category-level net flows, AUM growth, RBI repo rate cycles, LAF liquidity, Nifty returns, and structural shocks like IL&FS 2018 and COVID-19 2020).
2. **Feature Engineering (`feature_engineering.py`)**: Calendar dummies (advance tax, fiscal year-end), lag features (1, 2, 3, 6, 12 months), rolling statistics (3M/6M mean and volatility), normalized flow rates (% of AUM), and rate spreads.
3. **Exploratory Data Analysis (`eda.py`)**: Statistical driver significance tests, stationarity checks (ADF/KPSS), ACF/PACF autocorrelation analysis, seasonal decomposition, and correlation matrices.
4. **Forecasting Models (`models.py`)**: Unified architecture implementing:
   - **Seasonal Naïve** (benchmark)
   - **SARIMAX** with exogenous macro and calendar regressors
   - **LightGBM Quantile Regressors** (median, 80% and 95% prediction intervals)
   - **Ensemble Forecaster** (weighted combination with conservative uncertainty envelopes)
5. **Backtesting Framework (`backtesting.py`)**: 24-origin expanding-window rolling backtest, evaluating MAPE, MAE, RMSE, and empirical prediction interval calibration.
6. **Decision Layer (`decision_layer.py`)**: Converts forecasts and uncertainty intervals into actionable cash buffer policies (`P95 Outflow`, `Adaptive Tax-Aware`, vs. `Naïve Last Year Outflow`), factoring in yield drag vs. $2\times$ shortfall penalties and SEBI's 20% liquid asset guidelines.
7. **Interactive Dashboard (`dashboard.py`)**: One-page Streamlit application for fund managers to review forecasts, uncertainty intervals, top drivers, and buffer recommendations on the 1st of each month.

---

## 📂 Repository Structure

```
zeckathon-ps03/
├── config.py                 # Central configurations, hyperparameters, and filepaths
├── requirements.txt          # Python dependencies
├── data_generator.py         # Synthetic AMFI and macro data generator
├── feature_engineering.py    # Time-series feature pipeline
├── eda.py                    # Exploratory data analysis & statistical tests
├── models.py                 # Seasonal Naïve, SARIMAX, LightGBM, and Ensemble forecasters
├── backtesting.py            # Rolling-origin cross-validation engine
├── decision_layer.py         # Cash buffer policy optimization & cost backtesting
├── dashboard.py              # Interactive Streamlit executive dashboard
├── main.py                   # Master pipeline runner
├── data/                     # Data directory (CSV datasets)
└── outputs/                  # Backtest metrics, plots, and forecast artifacts
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/sudhanshubajpai167-tech/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

This runs:
- Synthetic data generation (or loading from `data/`)
- Feature engineering
- Exploratory data analysis & plots
- Model training & rolling-origin backtesting
- 3-month forecast generation with 80% & 95% prediction intervals
- Cash buffer decision policy evaluation

### 3. Launch the Fund Manager Dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Evaluation & Metrics

- **Forecast Accuracy**: Evaluated using MAPE and MAE over 24 rolling-origin windows.
- **Interval Calibration**: Verified empirical coverage for 80% and 95% prediction intervals.
- **Economic Decision Cost**:
  $$\text{Cost} = \text{Yield Drag} + 2 \times \text{Shortfall Penalty}$$
  Demonstrating lower annualized cost and reduced tail-risk relative to historical naive benchmarks.
