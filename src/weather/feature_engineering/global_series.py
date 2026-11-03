"""The global mean temperature series and the regression features used to forecast it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather.enums import Calendar, GlobalSeriesColumns
from weather.feature_engineering.encoding import cyclical_encode


def build_global_series(daily: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """Daily mean over a FIXED set of well-covered locations.

    Only locations observed on at least `min_coverage` of all days are used, and days where fewer than 90% of
    them report are dropped (then short gaps re-interpolated), so changes in which cities report cannot create
    a fake trend. The number of locations is stored in `attrs["n_stable_locations"]`.
    """
    n_days = daily["date"].nunique()
    observed = daily[daily["observed"]]
    coverage = observed.groupby("location_name")["date"].nunique() / n_days
    stable = coverage[coverage >= min_coverage].index
    hour_sin, hour_cos = cyclical_encode(daily["local_hour"], Calendar.HOURS_PER_DAY)
    stable_days = daily.assign(hour_sin=hour_sin, hour_cos=hour_cos)[daily["location_name"].isin(stable)]
    series = stable_days.groupby("date").agg(
        temperature_celsius=("temperature_celsius", "mean"),
        precip_mm=("precip_mm", "mean"),
        humidity=("humidity", "mean"),
        local_hour=("local_hour", "mean"),
        hour_sin=("hour_sin", "mean"),
        hour_cos=("hour_cos", "mean"),
        n_locations=("location_name", "nunique"),
    )
    series = series[series["n_locations"] >= 0.9 * len(stable)].asfreq("D").interpolate(limit=3)
    series.attrs["n_stable_locations"] = len(stable)
    return series.rename_axis("date").reset_index()


def seasonal_design_matrix(
    index: pd.DatetimeIndex, origin: pd.Timestamp, fourier_order: int, hours: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Regressors for the global series: linear trend + annual Fourier terms (+ observation-hour harmonics).

    Args:
        index: the dates to build rows for.
        origin: date where the trend term `t` (in years) is zero, i.e. the first training date.
        fourier_order: number of annual sine/cosine pairs (`s1, c1, ..., sK, cK`).
        hours: optional `hour_sin` / `hour_cos` columns aligned on `index`, joined as extra regressors.
    """
    X = pd.DataFrame(index=index)
    X["t"] = (index - origin).days / Calendar.DAYS_PER_YEAR
    day_of_year = index.day_of_year
    for k in range(1, fourier_order + 1):
        X[f"s{k}"] = np.sin(2 * np.pi * k * day_of_year / Calendar.DAYS_PER_YEAR)
        X[f"c{k}"] = np.cos(2 * np.pi * k * day_of_year / Calendar.DAYS_PER_YEAR)
    if hours is not None:
        X = X.join(hours)
    return X


def project_observation_hours(history: pd.DataFrame, index: pd.DatetimeIndex, window_days: int = 30) -> pd.DataFrame:
    """Future observation-hour regressors: hold the mean of the last `window_days` days.

    The hour at which snapshots are taken is not known in advance, but it drifts slowly, so a recent average
    is a reasonable stand-in.
    """
    last = history[GlobalSeriesColumns.HOUR].iloc[-window_days:].mean()
    return pd.DataFrame({c: last[c] for c in last.index}, index=index)
