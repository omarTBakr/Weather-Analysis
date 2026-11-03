"""Constants of the descriptive stages: data download, EDA, anomaly detection and the `analyses` stage."""

from __future__ import annotations


class KaggleSource:
    """Download endpoint and sanity check of the `download` stage."""

    REST_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}/{filename}"
    MIN_CSV_BYTES = 1_000_000  # anything smaller is an error page / empty archive, not the dataset


class RollingBaseline:
    """Each city's centred rolling-median temperature, the reference for corrupted-day and temporal anomalies."""

    WINDOW_DAYS = 15
    MIN_PERIODS = 5


class Exploration:
    """Constants of the `eda` stage figures."""

    TREND_ROLLING_DAYS = 30  # smoothing window of the global temperature trend line
    TOP_WETTEST_CAPITALS = 15
    TOP_CONDITIONS = 12  # most frequent `condition_text` values shown


class AnomalyDetecting:
    """Constants of the `anomaly` stage."""

    ISOLATION_TREES = 300
    MAD_TO_SIGMA = 0.6745  # robust z = 0.6745 * deviation / MAD is ~N(0, 1) for Gaussian data
    TOP_TABLE_ROWS = 50  # rows of `top_anomalies.csv`
    TOP_LATEX_ROWS = 10  # rows of `top_anomalies.tex`
    # columns of `top_anomalies.csv`
    TOP_TABLE_COLUMNS = [
        "date",
        "location_name",
        "country",
        "continent",
        "temperature_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "precip_mm",
        "aq_pm25",
        "iso_score",
        "temporal_flag",
    ]
    # column -> LaTeX header of `top_anomalies.tex` (dict order = column order)
    TOP_LATEX_HEADERS = {
        "date": "Date",
        "location_name": "Location",
        "country": "Country",
        "temperature_celsius": "Temp (°C)",
        "wind_kph": "Wind (kph)",
        "precip_mm": "Precip (mm)",
        "aq_pm25": "PM2.5",
    }
