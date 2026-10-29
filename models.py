"""
Forecasting models for Liquid Fund Cash Buffer prediction.

All models inherit from BaseForecaster and expose a consistent
fit / predict interface.  Prediction intervals are returned at
80 % and 95 % coverage levels.

Models implemented
------------------
* SeasonalNaive          – repeat-last-season baseline
* SARIMAXForecaster      – auto-tuned SARIMAX via pmdarima
* LightGBMForecaster     – quantile-regression gradient boosting
* EnsembleForecaster     – weighted combination of fitted models
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from config import (
    ALL_FEATURES,
    EXOG_FEATURES_SARIMAX,
    FORECAST_HORIZON,
    LGBM_PARAMS,
    PI_80_ALPHA,
    PI_95_ALPHA,
    QUANTILES,
    SEASONAL_PERIOD,
)

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseForecaster(ABC):
    """Common interface for all forecasting models."""

    @abstractmethod
    def fit(self, train_y: np.ndarray, train_X: Optional[pd.DataFrame] = None):
        """Fit the model on training data.

        Parameters
        ----------
        train_y : array-like
            Target time series (net_flow_cr).
        train_X : DataFrame, optional
            Exogenous / feature matrix aligned with *train_y*.
        """

    @abstractmethod
    def predict(
        self,
        h: int = FORECAST_HORIZON,
        future_X: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Produce *h*-step-ahead forecasts.

        Returns
        -------
        dict with keys
            ``point``, ``lower_80``, ``upper_80``, ``lower_95``, ``upper_95``
            – each a 1-D numpy array of length *h*.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""


# ---------------------------------------------------------------------------
# 1. Seasonal-Naïve baseline
# ---------------------------------------------------------------------------

class SeasonalNaive(BaseForecaster):
    """Repeat values from one full season ago (12 months)."""

    def __init__(self, season: int = SEASONAL_PERIOD):
        self._season = season
        self._y: Optional[np.ndarray] = None
        self._residual_std: float = 1.0

    # ---- fit ---------------------------------------------------------------
    def fit(self, train_y: np.ndarray, train_X=None):
        self._y = np.asarray(train_y, dtype=float)

        # Compute seasonal in-sample errors for interval estimation
        errors = []
        for t in range(self._season, len(self._y)):
            errors.append(self._y[t] - self._y[t - self._season])
        if errors:
            self._residual_std = float(np.std(errors, ddof=1))
        else:
            self._residual_std = float(np.std(self._y, ddof=1))
        return self

    # ---- predict -----------------------------------------------------------
    def predict(self, h: int = FORECAST_HORIZON, future_X=None):
        if self._y is None:
            raise RuntimeError("Model has not been fitted yet.")

        point = np.array([
            self._y[-(self._season - (i % self._season))]
            for i in range(h)
        ])

        # Growing uncertainty with horizon: σ * sqrt(k) where k = ceil((i+1)/season)
        z_80 = stats.norm.ppf(1 - PI_80_ALPHA / 2)
        z_95 = stats.norm.ppf(1 - PI_95_ALPHA / 2)

        widths_80 = np.array([
            z_80 * self._residual_std * np.sqrt(max(1, (i // self._season) + 1))
            for i in range(h)
        ])
        widths_95 = np.array([
            z_95 * self._residual_std * np.sqrt(max(1, (i // self._season) + 1))
            for i in range(h)
        ])

        return {
            "point": point,
            "lower_80": point - widths_80,
            "upper_80": point + widths_80,
            "lower_95": point - widths_95,
            "upper_95": point + widths_95,
        }

    @property
    def name(self) -> str:
        return "SeasonalNaive"


# ---------------------------------------------------------------------------
# 2. SARIMAX via pmdarima
# ---------------------------------------------------------------------------

class SARIMAXForecaster(BaseForecaster):
    """Auto-tuned SARIMAX model with exogenous regressors."""

    def __init__(self, exog_cols: Optional[List[str]] = None):
        self._exog_cols = exog_cols or EXOG_FEATURES_SARIMAX
        self._model = None  # fitted pmdarima ARIMA result

    # ---- fit ---------------------------------------------------------------
    def fit(self, train_y: np.ndarray, train_X: Optional[pd.DataFrame] = None):
        import pmdarima as pm

        y = np.asarray(train_y, dtype=float)
        exog = None
        if train_X is not None and self._exog_cols:
            exog = train_X[self._exog_cols].values

        try:
            self._model = pm.auto_arima(
                y,
                exogenous=exog,
                seasonal=True,
                m=SEASONAL_PERIOD,
                max_p=3,
                max_q=3,
                max_P=2,
                max_Q=2,
                d=None,
                D=None,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
        except Exception as exc:
            warnings.warn(
                f"auto_arima failed ({exc}); falling back to "
                "SARIMAX(1,0,1)(1,1,0,12)."
            )
            self._model = pm.ARIMA(
                order=(1, 0, 1),
                seasonal_order=(1, 1, 0, SEASONAL_PERIOD),
                suppress_warnings=True,
            )
            self._model.fit(y, exogenous=exog)

        return self

    # ---- predict -----------------------------------------------------------
    def predict(self, h: int = FORECAST_HORIZON, future_X=None):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        exog = None
        if future_X is not None and self._exog_cols:
            exog = future_X[self._exog_cols].values[:h]

        point, conf_80 = self._model.predict(
            n_periods=h,
            exogenous=exog,
            return_conf_int=True,
            alpha=PI_80_ALPHA,
        )
        _, conf_95 = self._model.predict(
            n_periods=h,
            exogenous=exog,
            return_conf_int=True,
            alpha=PI_95_ALPHA,
        )

        return {
            "point": np.asarray(point),
            "lower_80": conf_80[:, 0],
            "upper_80": conf_80[:, 1],
            "lower_95": conf_95[:, 0],
            "upper_95": conf_95[:, 1],
        }

    @property
    def name(self) -> str:
        return "SARIMAX"

    @property
    def fitted_model(self):
        """Access the underlying pmdarima model for diagnostics."""
        return self._model


# ---------------------------------------------------------------------------
# 3. LightGBM quantile-regression forecaster
# ---------------------------------------------------------------------------

class LightGBMForecaster(BaseForecaster):
    """Gradient-boosted quantile regression (one model per quantile).

    For multi-step-ahead forecasting the caller must provide a
    ``future_X`` DataFrame whose rows are the future time steps.
    All feature columns (including lags) must be pre-computed.
    The model simply predicts each row independently (direct /
    as-provided strategy) which keeps the hackathon implementation
    simple and transparent.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self._feature_cols = feature_cols or ALL_FEATURES
        self._quantile_models: Dict[str, object] = {}
        self._mean_model: Optional[object] = None
        self._feature_importances: Optional[pd.Series] = None

    # ---- fit ---------------------------------------------------------------
    def fit(self, train_y: np.ndarray, train_X: Optional[pd.DataFrame] = None):
        import lightgbm as lgb

        if train_X is None:
            raise ValueError("LightGBMForecaster requires train_X.")

        X = train_X[self._feature_cols].values
        y = np.asarray(train_y, dtype=float)

        base = dict(LGBM_PARAMS)  # copy so we don't mutate global

        # --- mean (point forecast) model ---
        mean_params = {**base, "objective": "regression"}
        self._mean_model = lgb.LGBMRegressor(**mean_params)
        self._mean_model.fit(X, y)

        self._feature_importances = pd.Series(
            self._mean_model.feature_importances_,
            index=self._feature_cols,
        ).sort_values(ascending=False)

        # --- quantile models ---
        for q_name, q_val in QUANTILES.items():
            q_params = {
                **base,
                "objective": "quantile",
                "alpha": q_val,
            }
            model = lgb.LGBMRegressor(**q_params)
            model.fit(X, y)
            self._quantile_models[q_name] = model

        return self

    # ---- predict -----------------------------------------------------------
    def predict(self, h: int = FORECAST_HORIZON, future_X=None):
        if self._mean_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if future_X is None:
            raise ValueError(
                "LightGBMForecaster.predict requires future_X "
                "with pre-computed features for each forecast step."
            )

        X = future_X[self._feature_cols].values[:h]

        point = self._mean_model.predict(X)
        lower_80 = self._quantile_models["lower_80"].predict(X)
        upper_80 = self._quantile_models["upper_80"].predict(X)
        lower_95 = self._quantile_models["lower_95"].predict(X)
        upper_95 = self._quantile_models["upper_95"].predict(X)

        return {
            "point": np.asarray(point),
            "lower_80": np.asarray(lower_80),
            "upper_80": np.asarray(upper_80),
            "lower_95": np.asarray(lower_95),
            "upper_95": np.asarray(upper_95),
        }

    # ---- feature importance ------------------------------------------------
    def get_feature_importance(self) -> pd.Series:
        """Return feature importances from the mean-regression model."""
        if self._feature_importances is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._feature_importances.copy()

    # ---- SHAP values -------------------------------------------------------
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the mean model.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (must contain ``self._feature_cols``).

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        import shap

        if self._mean_model is None:
            raise RuntimeError("Model has not been fitted yet.")

        explainer = shap.TreeExplainer(self._mean_model)
        shap_values = explainer.shap_values(X[self._feature_cols])
        return np.asarray(shap_values)

    @property
    def name(self) -> str:
        return "LightGBM"


# ---------------------------------------------------------------------------
# 4. Ensemble forecaster
# ---------------------------------------------------------------------------

class EnsembleForecaster(BaseForecaster):
    """Weighted combination of already-fitted forecasters.

    Prediction intervals are taken as the *widest* envelope across
    constituent models at each horizon step, which is a conservative
    strategy appropriate for a cash-buffer use case.
    """

    def __init__(
        self,
        forecasters: List[BaseForecaster],
        weights: Optional[np.ndarray] = None,
    ):
        self._forecasters = forecasters
        n = len(forecasters)
        self._weights = (
            np.asarray(weights, dtype=float)
            if weights is not None
            else np.ones(n) / n
        )

    # ---- fit (no-op: sub-models already fitted) ----------------------------
    def fit(self, train_y=None, train_X=None):
        """No-op – constituent models must already be fitted."""
        return self

    # ---- predict -----------------------------------------------------------
    def predict(self, h: int = FORECAST_HORIZON, future_X=None):
        sub_preds = [
            m.predict(h=h, future_X=future_X) for m in self._forecasters
        ]

        # Weighted point forecast
        point = np.zeros(h)
        for w, sp in zip(self._weights, sub_preds):
            point += w * sp["point"]

        # Conservative (widest) intervals
        lower_80 = np.min([sp["lower_80"] for sp in sub_preds], axis=0)
        upper_80 = np.max([sp["upper_80"] for sp in sub_preds], axis=0)
        lower_95 = np.min([sp["lower_95"] for sp in sub_preds], axis=0)
        upper_95 = np.max([sp["upper_95"] for sp in sub_preds], axis=0)

        return {
            "point": point,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
        }

    # ---- weight optimisation -----------------------------------------------
    def optimize_weights(
        self,
        val_y: np.ndarray,
        val_predictions: List[np.ndarray],
    ):
        """Find weights that minimise MAPE on a validation set.

        Parameters
        ----------
        val_y : array-like, shape (n,)
            Actual values for the validation period.
        val_predictions : list of array-like, each shape (n,)
            Point forecasts from each constituent model.
        """
        val_y = np.asarray(val_y, dtype=float)
        preds = np.column_stack(val_predictions)  # (n, k)

        def _mape(w):
            combined = preds @ w
            return np.mean(np.abs((val_y - combined) / np.where(val_y == 0, 1, val_y))) * 100

        n = len(self._forecasters)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n

        result = optimize.minimize(
            _mape,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            self._weights = result.x
        else:
            warnings.warn(
                "Weight optimisation did not converge; keeping equal weights."
            )

        return self

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def name(self) -> str:
        return "Ensemble"


# ---------------------------------------------------------------------------
# Utility: train all models at once
# ---------------------------------------------------------------------------

def train_all_models(
    df: pd.DataFrame,
    train_end_idx: int,
    feature_cols: List[str],
    exog_cols: List[str],
) -> Dict[str, BaseForecaster]:
    """Train SeasonalNaive, SARIMAX and LightGBM on the training partition.

    Parameters
    ----------
    df : DataFrame
        Full features dataset (must include ``net_flow_cr`` and all
        feature / exog columns).
    train_end_idx : int
        Row index (exclusive upper bound) for the training window.
    feature_cols : list[str]
        Columns used by LightGBM.
    exog_cols : list[str]
        Exogenous columns used by SARIMAX.

    Returns
    -------
    dict mapping model name → fitted BaseForecaster
    """
    train = df.iloc[:train_end_idx]
    train_y = train["net_flow_cr"].values
    train_X = train[feature_cols] if feature_cols else None

    models: Dict[str, BaseForecaster] = {}

    # 1) Seasonal Naïve
    sn = SeasonalNaive()
    sn.fit(train_y)
    models[sn.name] = sn

    # 2) SARIMAX
    sx = SARIMAXForecaster(exog_cols=exog_cols)
    try:
        sx.fit(train_y, train_X)
        models[sx.name] = sx
    except Exception as exc:
        warnings.warn(f"SARIMAX training failed: {exc}")

    # 3) LightGBM
    lgbm = LightGBMForecaster(feature_cols=feature_cols)
    try:
        lgbm.fit(train_y, train_X)
        models[lgbm.name] = lgbm
    except Exception as exc:
        warnings.warn(f"LightGBM training failed: {exc}")

    return models
