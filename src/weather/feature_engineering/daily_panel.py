"""Turn the cleaned observations into a regular daily grid per city (the "daily panel")."""

from __future__ import annotations

import pandas as pd

from weather.enums import PanelColumns


def build_daily_panel(clean: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """Reindex every location onto a continuous daily calendar.

    Args:
        clean: cleaned observations, one row per (location_name, date).
        max_gap_days: interior gaps up to this many days are linearly interpolated; longer gaps stay missing.

    Returns:
        One row per (location_name, date) with the numeric measurements, the static location attributes,
        `condition_text`, and an `observed` flag that is False for days filled in by interpolation.
        Days whose temperature is still missing are dropped.
    """
    per_location = []
    for location, obs in clean.groupby("location_name", sort=True):
        obs = obs.set_index("date").sort_index()
        calendar = pd.date_range(obs.index.min(), obs.index.max(), freq="D")
        day = obs[PanelColumns.DAILY_NUMERIC].reindex(calendar).interpolate(limit=max_gap_days, limit_area="inside")
        day["observed"] = obs["temperature_celsius"].reindex(calendar).notna()
        for col in PanelColumns.STATIC:
            day[col] = obs[col].iloc[-1]
        day["condition_text"] = obs["condition_text"].reindex(calendar)
        day["location_name"] = location
        per_location.append(day.rename_axis("date").reset_index())
    daily = pd.concat(per_location, ignore_index=True)
    return daily.dropna(subset=["temperature_celsius"]).reset_index(drop=True)
