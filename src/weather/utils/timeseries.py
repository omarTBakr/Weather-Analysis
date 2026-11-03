"""Small time-series helpers shared by several stages."""

from __future__ import annotations

import pandas as pd

from weather.enums import RollingBaseline


def location_rolling_median(df: pd.DataFrame, column: str) -> pd.Series:
    """Centred `RollingBaseline.WINDOW_DAYS`-row rolling median of `column`, computed separately per location.

    `df` must already be sorted by `location_name` and `date`; the result is aligned on `df.index`.
    """
    return df.groupby("location_name")[column].transform(
        lambda s: s.rolling(RollingBaseline.WINDOW_DAYS, center=True, min_periods=RollingBaseline.MIN_PERIODS).median()
    )


def month_start(dates: pd.Series) -> pd.Series:
    """First day of each date's calendar month (for monthly aggregation)."""
    return dates.dt.to_period("M").dt.to_timestamp()
