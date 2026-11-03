"""Feature engineering: daily panel, supervised panel features, time split and the global mean series.

The `features` DVC stage lives in `weather.feature_engineering.run`.
"""

from weather.feature_engineering.daily_panel import build_daily_panel
from weather.feature_engineering.encoding import cyclical_encode, encode_continent
from weather.feature_engineering.global_series import (
    build_global_series,
    project_observation_hours,
    seasonal_design_matrix,
)
from weather.feature_engineering.panel_features import build_panel_features
from weather.feature_engineering.splits import assign_time_split

__all__ = [
    "assign_time_split",
    "build_daily_panel",
    "build_global_series",
    "build_panel_features",
    "cyclical_encode",
    "encode_continent",
    "project_observation_hours",
    "seasonal_design_matrix",
]
