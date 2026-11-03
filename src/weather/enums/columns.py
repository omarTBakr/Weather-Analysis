"""Named groups of dataset columns, one class per pipeline step that uses them."""

from __future__ import annotations


class CleaningColumns:
    """Column groups of the `clean` stage."""

    # imperial duplicates of metric columns carry no extra information
    IMPERIAL_DUPLICATES = [
        "temperature_fahrenheit",
        "wind_mph",
        "pressure_in",
        "precip_in",
        "feels_like_fahrenheit",
        "visibility_miles",
        "gust_mph",
    ]
    # short names for the air-quality columns
    AIR_QUALITY_RENAMES = {
        "air_quality_Carbon_Monoxide": "aq_co",
        "air_quality_Ozone": "aq_o3",
        "air_quality_Nitrogen_dioxide": "aq_no2",
        "air_quality_Sulphur_dioxide": "aq_so2",
        "air_quality_PM2.5": "aq_pm25",
        "air_quality_PM10": "aq_pm10",
        "air_quality_us-epa-index": "aq_us_epa_index",
        "air_quality_gb-defra-index": "aq_gb_defra_index",
    }
    # per-location IQR fence (roughly symmetric variables);
    # visibility is excluded: it is reported as exactly 10 km most of the time (IQR = 0)
    IQR_CAPPED = ["temperature_celsius", "feels_like_celsius", "wind_kph", "gust_kph", "pressure_mb"]
    # zero-inflated / log-normal variables: global upper-quantile cap
    QUANTILE_CAPPED = ["precip_mm", "aq_co", "aq_o3", "aq_no2", "aq_so2", "aq_pm25", "aq_pm10"]
    # weather measurements nulled for a whole day when that day's provider batch is corrupted
    WEATHER = [
        "temperature_celsius",
        "feels_like_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "gust_kph",
        "cloud",
        "uv_index",
        "precip_mm",
        "visibility_km",
    ]
    # categorical columns filled with the per-location mode
    IMPUTED_CATEGORICAL = ["condition_text", "wind_direction", "moon_phase"]


class CleaningRules:
    """Physically plausible ranges (surface observations); values outside -> NaN, later imputed."""

    VALID_RANGES = {
        "temperature_celsius": (-60, 57),
        "feels_like_celsius": (-75, 70),
        "wind_kph": (0, 250),
        "gust_kph": (0, 350),
        "pressure_mb": (870, 1085),
        "precip_mm": (0, 400),
        "humidity": (0, 100),
        "cloud": (0, 100),
        "visibility_km": (0, 50),
        "uv_index": (0, 20),
        "aq_co": (0, 50000),
        "aq_o3": (0, 1000),
        "aq_no2": (0, 1000),
        "aq_so2": (0, 1000),
        "aq_pm25": (0, 2000),
        "aq_pm10": (0, 6000),
    }
    # every range-checked numeric column is also imputed
    IMPUTED_NUMERIC = list(VALID_RANGES)


class PanelColumns:
    """Column groups of the daily panel, the feature table and the predictions table."""

    # per-location attributes that never change over time: copied onto every day of the grid
    STATIC = ["country", "continent", "latitude", "longitude", "timezone"]
    # numeric measurements reindexed onto the daily grid and gap-filled
    DAILY_NUMERIC = [
        "temperature_celsius",
        "feels_like_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "gust_kph",
        "cloud",
        "precip_mm",
        "uv_index",
        "visibility_km",
        "aq_co",
        "aq_o3",
        "aq_no2",
        "aq_so2",
        "aq_pm25",
        "aq_pm10",
        "aq_us_epa_index",
        "aq_gb_defra_index",
        "local_hour",
    ]
    # location descriptors carried onto every feature row (forward/back-filled across re-gridded gaps)
    LOCATION = ["location_name", "country", "continent", "latitude", "longitude"]
    # kept on the feature table for bookkeeping but never fed to a model
    NON_FEATURE = {"date", "target", "target_date", "location_name", "country", "continent"}
    # identifier / bookkeeping columns carried alongside every model prediction in `predictions.parquet`
    PREDICTION_ID = ["location_name", "country", "continent", "date", "target_date", "split", "target"]
    # each model's forecasts are stored in `predictions.parquet` as column `pred_<model name>`
    PREDICTION_PREFIX = "pred_"


class GlobalSeriesColumns:
    """Columns of the global mean series used as regressors."""

    HOUR = ["hour_sin", "hour_cos"]
