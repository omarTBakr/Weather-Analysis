"""Supervised features for next-day temperature forecasting on the city panel.

For every (city, day) row the features describe what is known *on that day* (lags, rolling stats, weather,
observation hour, calendar of the target day, geography), and `target` is the temperature `horizon` days later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather.enums import Calendar, PanelColumns
from weather.feature_engineering.encoding import cyclical_encode, encode_continent
from weather.utils.config import FeatureSettings


def _location_features(location_days: pd.DataFrame, settings: FeatureSettings) -> pd.DataFrame:
    """Lag, rolling, difference, exogenous and diurnal features plus the shifted target for ONE location."""
    # re-grid so that shifts are calendar-exact even across long gaps
    days = location_days.set_index("date").asfreq("D")
    temp = days[settings.target]
    out = pd.DataFrame(index=days.index)
    out["temp_t"] = temp
    for lag in settings.lags:
        out[f"temp_lag{lag}"] = temp.shift(lag)
    for window in settings.rolling_windows:
        out[f"temp_rollmean{window}"] = temp.rolling(window, min_periods=window).mean()
        out[f"temp_rollstd{window}"] = temp.rolling(window, min_periods=window).std()
    out["temp_diff1"] = temp - temp.shift(1)
    for col in settings.exog:
        out[col] = days[col]
    out["humidity_diff1"] = days["humidity"] - days["humidity"].shift(1)
    out["pressure_diff1"] = days["pressure_mb"] - days["pressure_mb"].shift(1)
    # observation clock time (diurnal cycle): next snapshot is taken at ~the same local time
    out["hour_sin"], out["hour_cos"] = cyclical_encode(days["local_hour"], Calendar.HOURS_PER_DAY)
    out["target"] = temp.shift(-settings.horizon)
    out["target_date"] = out.index + pd.Timedelta(days=settings.horizon)
    for col in PanelColumns.LOCATION:
        out[col] = days[col].ffill().bfill()
    return out.rename_axis("date").reset_index()


def _add_calendar_and_geo_features(features: pd.DataFrame) -> pd.DataFrame:
    """Season of the *target* day (hemisphere-aware), absolute latitude and an integer continent code."""
    features["doy_sin"], features["doy_cos"] = cyclical_encode(
        features["target_date"].dt.day_of_year, Calendar.DAYS_PER_YEAR
    )
    # hemisphere-aware seasonal phase: flips the annual cycle for the southern hemisphere
    hemisphere = np.sign(features["latitude"]).replace(0, 1)
    features["season_sin"] = features["doy_sin"] * hemisphere
    features["season_cos"] = features["doy_cos"] * hemisphere
    features["abs_latitude"] = features["latitude"].abs()
    features["continent_code"] = encode_continent(features["continent"])
    return features


def build_panel_features(daily: pd.DataFrame, settings: FeatureSettings) -> tuple[pd.DataFrame, list[str]]:
    """Build the supervised feature table from the daily panel.

    Args:
        daily: output of `build_daily_panel`.
        settings: target column, forecast horizon, lags, rolling windows and exogenous columns.

    Returns:
        `(features, feature_columns)`: the table (rows with any missing feature or target dropped) and the
        ordered list of model input columns (everything except `PanelColumns.NON_FEATURE`).
    """
    frames = [_location_features(g, settings) for _, g in daily.groupby("location_name", sort=False)]
    features = _add_calendar_and_geo_features(pd.concat(frames, ignore_index=True))
    feature_columns = [c for c in features.columns if c not in PanelColumns.NON_FEATURE]
    features = features.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
    return features, feature_columns
