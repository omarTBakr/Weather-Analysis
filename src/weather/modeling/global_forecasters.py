"""Multi-step forecasters for the global mean temperature series.

Every forecaster shares one interface: given the history (a daily frame with `temperature_celsius` and the
`hour_sin` / `hour_cos` regressors) and the dates to forecast, return a `Forecast` for all of them in one shot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from weather.enums import GlobalForecasting, GlobalModel, GlobalSeriesColumns
from weather.feature_engineering.global_series import project_observation_hours, seasonal_design_matrix


@dataclass
class Forecast:
    """Point forecast and, if the model provides one, a 95% prediction interval of shape (horizon, 2)."""

    mean: np.ndarray
    interval: np.ndarray | None = None


class GlobalForecaster(ABC):
    """Base class: subclasses set `name` and implement `forecast`."""

    name: str

    @abstractmethod
    def forecast(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> Forecast:
        """Forecast `GlobalForecasting.TARGET` for every date in `index` using only `history`."""


class NaiveForecaster(GlobalForecaster):
    """Repeat the last observed value."""

    name = GlobalModel.NAIVE

    def forecast(self, history, index):
        return Forecast(np.repeat(history[GlobalForecasting.TARGET].iloc[-1], len(index)))


class SeasonalNaiveForecaster(GlobalForecaster):
    """Same calendar day one year earlier (gaps interpolated / edge-filled)."""

    name = GlobalModel.SEASONAL_NAIVE

    def forecast(self, history, index):
        y = history[GlobalForecasting.TARGET]
        return Forecast(y.reindex(index - pd.Timedelta(days=365)).interpolate().bfill().ffill().to_numpy())


class HoltWintersForecaster(GlobalForecaster):
    """Additive damped trend + additive yearly seasonality (exponential smoothing)."""

    name = GlobalModel.HOLT_WINTERS

    def forecast(self, history, index):
        model = ExponentialSmoothing(
            history[GlobalForecasting.TARGET],
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=365,
            initialization_method="estimated",
        ).fit()
        return Forecast(model.forecast(len(index)).to_numpy())


class _FourierForecaster(GlobalForecaster):
    """Shared helper for models regressing on trend + annual Fourier terms + observation-hour harmonics."""

    def __init__(self, fourier_order: int):
        self.fourier_order = fourier_order

    def _designs(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Design matrices for the history and for the forecast dates (future hours held at a recent mean)."""
        origin = history.index[0]
        X_hist = seasonal_design_matrix(history.index, origin, self.fourier_order, history[GlobalSeriesColumns.HOUR])
        X_future = seasonal_design_matrix(index, origin, self.fourier_order, project_observation_hours(history, index))
        return X_hist, X_future


class HarmonicRegressionForecaster(_FourierForecaster):
    """OLS on the seasonal design matrix; intervals are OLS observation prediction intervals."""

    name = GlobalModel.HARMONIC_REGRESSION

    def forecast(self, history, index):
        X_hist, X_future = self._designs(history, index)
        model = sm.OLS(history[GlobalForecasting.TARGET], sm.add_constant(X_hist)).fit()
        frame = model.get_prediction(sm.add_constant(X_future, has_constant="add")).summary_frame(
            alpha=GlobalForecasting.PI_ALPHA
        )
        return Forecast(frame["mean"].to_numpy(), frame[["obs_ci_lower", "obs_ci_upper"]].to_numpy())


class SarimaxFourierForecaster(_FourierForecaster):
    """Regression on the seasonal design matrix with ARIMA(1,0,1) errors."""

    name = GlobalModel.SARIMAX_FOURIER

    def forecast(self, history, index):
        X_hist, X_future = self._designs(history, index)
        # the constant lives in the exogenous design, so the ARIMA part has no trend term
        model = SARIMAX(
            history[GlobalForecasting.TARGET], exog=X_hist.assign(const=1.0), order=(1, 0, 1), trend="n"
        ).fit(disp=False, maxiter=500)
        fc = model.get_forecast(len(index), exog=X_future.assign(const=1.0))
        return Forecast(fc.predicted_mean.to_numpy(), fc.conf_int(alpha=GlobalForecasting.PI_ALPHA).to_numpy())


def build_global_forecasters(fourier_order: int) -> dict[str, GlobalForecaster]:
    """All forecasters in evaluation order, keyed by name."""
    forecasters = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(),
        HoltWintersForecaster(),
        HarmonicRegressionForecaster(fourier_order),
        SarimaxFourierForecaster(fourier_order),
    ]
    return {f.name: f for f in forecasters}
