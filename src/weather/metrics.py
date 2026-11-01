"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(np.where(denom == 0, 0.0, 2 * np.abs(y_pred - y_true) / np.where(denom == 0, 1, denom))) * 100
    # MAPE is unstable for temperatures near 0 degC -> only on |y| >= 1
    mask = np.abs(y_true) >= 1.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape),
        "smape": float(smape),
        "r2": float(r2_score(y_true, y_pred)),
        "within_2c": float(np.mean(np.abs(y_true - y_pred) <= 2.0) * 100),
    }
