import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_panel() -> pd.DataFrame:
    """Two synthetic cities, 120 days, with a known seasonal temperature signal."""
    rng = np.random.default_rng(0)
    rows = []
    for loc, lat, cont in [("Alpha", 45.0, "Europe"), ("Beta", -30.0, "South America")]:
        dates = pd.date_range("2025-01-01", periods=120, freq="D")
        temp = 15 + 8 * np.sin(np.arange(120) / 20) + rng.normal(0, 0.5, 120)
        for d, t in zip(dates, temp, strict=True):
            rows.append(
                {
                    "date": d,
                    "location_name": loc,
                    "country": f"{loc}land",
                    "continent": cont,
                    "latitude": lat,
                    "longitude": 10.0,
                    "timezone": "UTC",
                    "temperature_celsius": t,
                    "feels_like_celsius": t - 1,
                    "humidity": 60.0,
                    "pressure_mb": 1012.0,
                    "wind_kph": 10.0,
                    "gust_kph": 15.0,
                    "cloud": 40.0,
                    "precip_mm": 0.0,
                    "uv_index": 4.0,
                    "visibility_km": 10.0,
                    "local_hour": 12.0,
                    "observed": True,
                }
            )
    return pd.DataFrame(rows)
