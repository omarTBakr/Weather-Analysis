"""Project-wide constants, grouped by topic.

- `Split`, `PanelModel`, `PanelEnsemble`, `GlobalModel`: closed sets of names (`StrEnum`, equal to their strings).
- `CleaningColumns`, `CleaningRules`, `PanelColumns`, `GlobalSeriesColumns`: named column groups.
- `PanelTraining`, `PanelEnsembling`, `GlobalForecasting`: per-stage modelling constants.
- `KaggleSource`, `RollingBaseline`, `Exploration`, `AnomalyDetecting`, `Insights`: constants of the
  download, EDA, anomaly and `analyses` stages.
- `Geography`, `Calendar`, `VariableLabels`, `Palette`: reference data, labels and colours.
"""

from weather.enums.analysis import AnomalyDetecting, Exploration, KaggleSource, RollingBaseline
from weather.enums.columns import CleaningColumns, CleaningRules, GlobalSeriesColumns, PanelColumns
from weather.enums.geography import Geography
from weather.enums.insights import (
    AirQualityInsights,
    ClimateInsights,
    FeatureImportanceInsights,
    GeographicInsights,
    Insights,
    SpatialInsights,
)
from weather.enums.labels import VariableLabels
from weather.enums.models import (
    GlobalForecasting,
    GlobalModel,
    PanelEnsemble,
    PanelEnsembling,
    PanelModel,
    PanelTraining,
)
from weather.enums.palette import Palette
from weather.enums.splits import Split
from weather.enums.time import Calendar

__all__ = [
    "AirQualityInsights",
    "ClimateInsights",
    "FeatureImportanceInsights",
    "GeographicInsights",
    "SpatialInsights",
    "AnomalyDetecting",
    "Calendar",
    "CleaningColumns",
    "CleaningRules",
    "Exploration",
    "Geography",
    "GlobalForecasting",
    "GlobalModel",
    "GlobalSeriesColumns",
    "Insights",
    "KaggleSource",
    "Palette",
    "PanelColumns",
    "PanelEnsemble",
    "PanelEnsembling",
    "PanelModel",
    "PanelTraining",
    "RollingBaseline",
    "Split",
    "VariableLabels",
]
