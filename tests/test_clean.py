import numpy as np
import pandas as pd

from weather.clean import apply_valid_ranges, canonical_country, country_to_continent, null_corrupted_days


def test_country_to_continent_handles_overrides_and_lookup():
    assert country_to_continent("France") == "Europe"
    assert country_to_continent("Kyrghyzstan") == "Asia"
    assert country_to_continent("Fiji Islands") == "Oceania"
    assert country_to_continent("Brazil") == "South America"


def test_canonical_country_uses_majority_label():
    df = pd.DataFrame({"location_name": ["Warsaw"] * 4, "country": ["Poland", "Poland", "Polônia", "Польша"]})
    assert set(canonical_country(df)) == {"Poland"}


def test_invalid_values_become_nan():
    df = pd.DataFrame(
        {
            c: [0.0, 0.0]
            for c in [
                "temperature_celsius",
                "feels_like_celsius",
                "wind_kph",
                "gust_kph",
                "pressure_mb",
                "precip_mm",
                "humidity",
                "cloud",
                "visibility_km",
                "uv_index",
                "aq_co",
                "aq_o3",
                "aq_no2",
                "aq_so2",
                "aq_pm25",
                "aq_pm10",
            ]
        }
    )
    df["pressure_mb"] = [1013.0, 3006.0]
    df["aq_co"] = [200.0, -9999.0]
    report = {}
    out = apply_valid_ranges(df, report)
    assert np.isnan(out.loc[1, "pressure_mb"]) and np.isnan(out.loc[1, "aq_co"])
    assert out.loc[0, "pressure_mb"] == 1013.0
    assert report["invalid_set_to_nan"]["pressure_mb"] == 1


def test_corrupted_batch_day_is_nulled(daily_panel, tmp_path):
    df = daily_panel.copy()
    bad_day = df["date"].iloc[60]
    df.loc[df["date"] == bad_day, "temperature_celsius"] -= 20  # every city shifts at once
    report = {}
    out = null_corrupted_days(df, threshold=4.0, report=report, deviation_csv=tmp_path / "dev.csv")
    assert list(report["corrupted_days"]) == [str(bad_day.date())]
    assert out.loc[out["date"] == bad_day, "temperature_celsius"].isna().all()
    assert out.loc[out["date"] != bad_day, "temperature_celsius"].notna().all()
    assert (tmp_path / "dev.csv").exists()
