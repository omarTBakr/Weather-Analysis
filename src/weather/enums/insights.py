"""Constants of the `analyses` stage, one class per analysis (plus the limits they share)."""

from __future__ import annotations

from weather.enums.palette import Palette


class Insights:
    """Limits shared by several analyses."""

    TOP_RANKED = 15  # bars in the top-N rankings (most polluted capitals, top features)


class ClimateInsights:
    """Per-city trend fits and climatology figures."""

    MIN_TREND_DAYS = 500  # cities with fewer daily observations get no per-city trend fit


class AirQualityInsights:
    """Pollutant vs weather analysis and the PM2.5-driver model."""

    WHO_PM25_GUIDELINE = 15  # WHO 24-hour PM2.5 guideline (µg/m³)
    PM25_PERMUTATION_SAMPLE = 15000  # test rows used for the PM2.5-driver permutation importance
    # fixed LightGBM configuration of the PM2.5-driver model (diagnostic only, not tuned)
    PM25_MODEL_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63}
    # (x, y, x label, y label) of the binned response-curve panels
    RESPONSE_CURVES = [
        ("wind_kph", "aq_pm25", "Wind (kph)", "PM2.5 (µg/m³)"),
        ("humidity", "aq_pm25", "Humidity (%)", "PM2.5 (µg/m³)"),
        ("temperature_celsius", "aq_o3", "Temperature (°C)", "O3 (µg/m³)"),
    ]


class FeatureImportanceInsights:
    """Feature-importance comparison of the LightGBM panel model."""

    # importance techniques = columns of the comparison table, in display order
    TECHNIQUES = ["LightGBM gain", "Permutation (ΔRMSE)", "Mean |SHAP|", "Mutual info."]
    # persistence features (current temperature and its rolling means), left out of the "secondary drivers" figure
    PERSISTENCE_PREFIXES = ("temp_t", "temp_rollmean")
    # features drawn as SHAP dependence plots (when present in the feature table)
    SHAP_DEPENDENCE_FEATURES = ["hour_cos", "pressure_diff1"]


class GeographicInsights:
    """Continent summaries and hottest / coldest capitals."""

    TOP_EXTREMES = 10  # hottest / coldest capitals
    # (variable, axis label) of the continent box-plot panels
    BOXPLOT_VARIABLES = [
        ("temperature_celsius", "Temperature (°C)"),
        ("humidity", "Humidity (%)"),
        ("wind_kph", "Wind (kph)"),
        ("aq_pm25", "PM2.5 (µg/m³, log)"),
    ]
    # column -> LaTeX header of the continent summary table
    CONTINENT_LATEX_HEADERS = {
        "cities": "Cities",
        "temp_mean": "Temp (°C)",
        "temp_std": "Temp SD",
        "precip_mean": "Precip (mm)",
        "humidity": "Humidity (\\%)",
        "wind": "Wind (kph)",
        "pm25_median": "PM2.5 med.",
        "uv": "UV",
    }


class SpatialInsights:
    """Spatial autocorrelation and maps."""

    # per-city variables tested for spatial autocorrelation
    MORAN_VARIABLES = ["temperature", "precip", "humidity", "pm25"]
    # (variable, title, colour map, unit) of the static map panels
    STATIC_MAPS = [
        ("temperature", "Mean temperature", Palette.DIV_CMAP, "°C"),
        ("precip", "Mean precipitation", Palette.SEQ_CMAP, "mm/obs"),
        ("humidity", "Mean humidity", Palette.SEQ_CMAP, "%"),
        ("pm25", "Median PM2.5", Palette.SEQ_CMAP, "µg/m³"),
    ]
    # (variable, title, plotly colour scale) of the interactive plotly maps
    INTERACTIVE_MAPS = [
        ("temperature", "Mean temperature (°C)", "RdBu_r"),
        ("pm25", "Median PM2.5 (µg/m³)", "YlOrRd"),
        ("precip", "Mean precipitation (mm/obs)", "Blues"),
    ]
