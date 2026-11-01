"""Shared settings, output directory and label dictionaries for the `analyses` stage."""

from __future__ import annotations

from weather.utils.config import get_settings

S = get_settings()
P = S.paths

OUT = P.figures / "analyses"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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
