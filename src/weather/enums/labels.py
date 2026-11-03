"""Human-readable labels for variables on figures and tables (the dict order is also the display order)."""

from __future__ import annotations


class VariableLabels:
    """Column name -> display label, one group per figure family."""

    # EDA summary table + correlation matrix
    EDA = {
        "temperature_celsius": "Temperature",
        "feels_like_celsius": "Feels like",
        "humidity": "Humidity",
        "pressure_mb": "Pressure",
        "wind_kph": "Wind",
        "gust_kph": "Gust",
        "cloud": "Cloud",
        "precip_mm": "Precip.",
        "uv_index": "UV index",
        "visibility_km": "Visibility",
        "aq_pm25": "PM2.5",
        "aq_pm10": "PM10",
        "aq_o3": "O3",
        "aq_no2": "NO2",
        "aq_co": "CO",
        "aq_so2": "SO2",
    }
    # variables standardised for multivariate anomaly detection (keys are the model inputs, in order)
    ANOMALY = {
        "temperature_celsius": "Temperature",
        "humidity": "Humidity",
        "pressure_mb": "Pressure",
        "wind_kph": "Wind",
        "precip_mm": "Precip.",
        "cloud": "Cloud",
        "uv_index": "UV",
        "visibility_km": "Visibility",
        "aq_pm25": "PM2.5",
        "aq_o3": "O3",
    }
    # air-quality analysis: pollutants (rows) vs weather drivers (columns)
    POLLUTANTS = {"aq_pm25": "PM2.5", "aq_pm10": "PM10", "aq_o3": "O3", "aq_no2": "NO2", "aq_so2": "SO2", "aq_co": "CO"}
    WEATHER = {
        "temperature_celsius": "Temperature",
        "humidity": "Humidity",
        "wind_kph": "Wind",
        "pressure_mb": "Pressure",
        "precip_mm": "Precip.",
        "cloud": "Cloud",
        "uv_index": "UV index",
        "visibility_km": "Visibility",
    }
    # US-EPA air-quality index value -> category
    EPA_INDEX = {
        1: "Good",
        2: "Moderate",
        3: "Unhealthy (sensitive)",
        4: "Unhealthy",
        5: "Very unhealthy",
        6: "Hazardous",
    }
