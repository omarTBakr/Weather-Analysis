"""Stage `clean`: validity checks, de-duplication, outlier capping, imputation, continent mapping."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from weather.config import CLEAN_PATH, METRICS, ROOT, ensure_dirs, load_params, setup_mlflow

# Imperial duplicates of metric columns carry no extra information.
REDUNDANT = [
    "temperature_fahrenheit", "wind_mph", "pressure_in", "precip_in",
    "feels_like_fahrenheit", "visibility_miles", "gust_mph",
]

RENAME = {
    "air_quality_Carbon_Monoxide": "aq_co",
    "air_quality_Ozone": "aq_o3",
    "air_quality_Nitrogen_dioxide": "aq_no2",
    "air_quality_Sulphur_dioxide": "aq_so2",
    "air_quality_PM2.5": "aq_pm25",
    "air_quality_PM10": "aq_pm10",
    "air_quality_us-epa-index": "aq_us_epa_index",
    "air_quality_gb-defra-index": "aq_gb_defra_index",
}

# Physically plausible ranges (surface observations). Values outside -> NaN, later imputed.
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

# Per-location IQR fence (roughly symmetric variables).
# visibility is excluded: it is reported as exactly 10 km most of the time (IQR = 0)
IQR_COLS = ["temperature_celsius", "feels_like_celsius", "wind_kph", "gust_kph", "pressure_mb"]
# Zero-inflated / log-normal variables: global upper-quantile cap.
HEAVY_TAIL_COLS = ["precip_mm", "aq_co", "aq_o3", "aq_no2", "aq_so2", "aq_pm25", "aq_pm10"]

NUMERIC_IMPUTE = list(VALID_RANGES)

# Countries pycountry-convert cannot resolve (or resolves to a non-continent code).
CONTINENT_OVERRIDES = {
    "Timor-Leste": "Asia", "East Timor": "Asia", "Vatican City": "Europe", "Kosovo": "Europe",
    "Bolivia": "South America", "Venezuela": "South America", "Iran": "Asia", "Syria": "Asia",
    "Laos": "Asia", "Vietnam": "Asia", "Russia": "Europe", "South Korea": "Asia", "North Korea": "Asia",
    "Tanzania": "Africa", "Moldova": "Europe", "Macedonia": "Europe", "North Macedonia": "Europe",
    "Democratic Republic of Congo": "Africa", "Congo": "Africa", "Ivory Coast": "Africa",
    "Cote d'Ivoire": "Africa", "Swaziland": "Africa", "Eswatini": "Africa", "Cape Verde": "Africa",
    "Micronesia": "Oceania", "Palestine": "Asia", "Turkey": "Asia", "Türkiye": "Asia",
    "Bahamas": "North America", "Gambia": "Africa", "Saint Vincent and the Grenadines": "North America",
    "Saint Kitts and Nevis": "North America", "Saint Lucia": "North America", "Brunei Darussalam": "Asia",
    "Brunei": "Asia", "Myanmar": "Asia", "Burma": "Asia", "Guinea-Bissau": "Africa",
    "Marshall Islands": "Oceania", "Solomon Islands": "Oceania", "Western Sahara": "Africa",
    "Fiji Islands": "Oceania", "Kyrghyzstan": "Asia", "Seychelles Islands": "Africa",
    "Antarctica": "Antarctica",
}

# Non-English / variant spellings that appear as a location's dominant country.
COUNTRY_FIXES = {
    "USA United States of America": "United States of America",
    "United States": "United States of America",
    "Turkey": "Türkiye",
}


def country_to_continent(country: str) -> str:
    if country in CONTINENT_OVERRIDES:
        return CONTINENT_OVERRIDES[country]
    import pycountry_convert as pc

    code_map = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
                "SA": "South America", "OC": "Oceania", "AN": "Antarctica"}
    for name in (country, country.replace("&", "and")):
        try:
            a2 = pc.country_name_to_country_alpha2(name, cn_name_format="default")
            return code_map[pc.country_alpha2_to_continent_code(a2)]
        except (KeyError, ValueError):
            continue
    return "Unknown"


def canonical_country(df: pd.DataFrame) -> pd.Series:
    """Each location keeps the country string it is most often reported with (fixes translated names)."""
    mode = df.groupby("location_name")["country"].agg(lambda s: s.value_counts().index[0])
    mode = mode.replace(COUNTRY_FIXES)
    return df["location_name"].map(mode)


def apply_valid_ranges(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    invalid = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["invalid_set_to_nan"] = invalid
    return df


WEATHER_COLS = ["temperature_celsius", "feels_like_celsius", "humidity", "pressure_mb", "wind_kph", "gust_kph",
                "cloud", "uv_index", "precip_mm", "visibility_km"]


def null_corrupted_days(df: pd.DataFrame, threshold: float, report: dict) -> pd.DataFrame:
    """Detect whole-batch provider glitches (e.g. Riyadh at 15 degC in August) and null that day's weather values.

    Statistic: per date, the median over all cities of |temperature - city's centred 15-day rolling median|.
    A normal day sits around 1 degC; a corrupted snapshot shifts most cities at once.
    """
    df = df.sort_values(["location_name", "date"])
    med = df.groupby("location_name")["temperature_celsius"].transform(
        lambda s: s.rolling(15, center=True, min_periods=5).median())
    day_dev = (df["temperature_celsius"] - med).abs().groupby(df["date"]).median()
    day_dev.rename("median_abs_dev_c").to_csv(METRICS / "daily_batch_deviation.csv")
    bad = day_dev[day_dev > threshold]
    df.loc[df["date"].isin(bad.index), WEATHER_COLS] = np.nan
    report["corrupted_days"] = {str(d.date()): round(float(v), 2) for d, v in bad.items()}
    report["corrupted_rows_nulled"] = int(df["date"].isin(bad.index).sum())
    return df


def cap_outliers(df: pd.DataFrame, iqr_k: float, q: float, report: dict) -> pd.DataFrame:
    capped = {}
    g = df.groupby("location_name")
    for col in IQR_COLS:
        q1 = g[col].transform(lambda s: s.quantile(0.25))
        q3 = g[col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        flag = (df[col] < lo) | (df[col] > hi)
        df[f"outlier_{col}"] = flag
        df[col] = df[col].clip(lo, hi)
        capped[col] = int(flag.sum())
    for col in HEAVY_TAIL_COLS:
        hi = df[col].quantile(q)
        flag = df[col] > hi
        df[f"outlier_{col}"] = flag
        df[col] = df[col].clip(upper=hi)
        capped[col] = int(flag.sum())
    df["is_outlier_any"] = df.filter(like="outlier_").any(axis=1)
    report["outliers_capped"] = capped
    return df


def impute(df: pd.DataFrame, limit: int, report: dict) -> pd.DataFrame:
    before = int(df[NUMERIC_IMPUTE].isna().sum().sum())
    df = df.sort_values(["location_name", "last_updated"])
    df[NUMERIC_IMPUTE] = (
        df.groupby("location_name")[NUMERIC_IMPUTE]
        .transform(lambda s: s.interpolate(limit=limit, limit_direction="both"))
    )
    df[NUMERIC_IMPUTE] = df.groupby("location_name")[NUMERIC_IMPUTE].transform(lambda s: s.fillna(s.median()))
    df[NUMERIC_IMPUTE] = df[NUMERIC_IMPUTE].fillna(df[NUMERIC_IMPUTE].median())
    for col in ["condition_text", "wind_direction", "moon_phase"]:
        df[col] = df.groupby("location_name")[col].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
        )
    report["missing_before_impute"] = before
    report["missing_after_impute"] = int(df[NUMERIC_IMPUTE].isna().sum().sum())
    return df


def clean(raw: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    cp = params["clean"]
    report: dict = {"raw_rows": len(raw), "raw_locations": int(raw["location_name"].nunique()),
                    "raw_countries": int(raw["country"].nunique()),
                    "raw_missing_cells": int(raw.isna().sum().sum())}
    df = raw.drop(columns=[c for c in REDUNDANT if c in raw]).rename(columns=RENAME).copy()
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["date"] = df["last_updated"].dt.normalize()
    # local clock time of the snapshot; it drifts from afternoon (2024) to early morning (2026),
    # which biases naive temperature trends -> kept as a covariate
    df["local_hour"] = df["last_updated"].dt.hour + df["last_updated"].dt.minute / 60
    df["condition_text"] = df["condition_text"].str.strip().str.title()

    n = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = n - len(df)

    counts = df.groupby("location_name")["date"].nunique()
    keep = counts[counts >= cp["min_obs_per_location"]].index
    report["locations_dropped_low_coverage"] = int((counts < cp["min_obs_per_location"]).sum())
    df = df[df["location_name"].isin(keep)].copy()
    report["rows_dropped_low_coverage"] = int(n - report["exact_duplicates_removed"] - len(df))

    df["country"] = canonical_country(df)
    df["continent"] = df["country"].map({c: country_to_continent(c) for c in df["country"].unique()})
    report["unknown_continent_countries"] = sorted(df.loc[df["continent"] == "Unknown", "country"].unique())

    # one observation per location-day (keep latest update of the day)
    n = len(df)
    df = df.sort_values("last_updated").drop_duplicates(["location_name", "date"], keep="last")
    report["same_day_duplicates_removed"] = n - len(df)

    df = apply_valid_ranges(df, report)
    df = null_corrupted_days(df, cp["corrupt_day_threshold_c"], report)
    df = cap_outliers(df, cp["iqr_k"], cp["heavy_tail_quantile"], report)
    df = impute(df, cp["interpolate_limit"], report)

    report.update({"clean_rows": len(df), "clean_locations": int(df["location_name"].nunique()),
                   "clean_countries": int(df["country"].nunique()),
                   "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
                   "continent_counts": df.groupby("continent")["location_name"].nunique().to_dict()})
    return df.reset_index(drop=True), report


def main() -> None:
    ensure_dirs()
    params = load_params()
    raw = pd.read_csv(ROOT / params["data"]["raw_path"])
    df, report = clean(raw, params)
    df.to_parquet(CLEAN_PATH, index=False)
    (METRICS / "clean_report.json").write_text(json.dumps(report, indent=2, default=str))

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="clean"):
        mlflow.set_tag("stage", "clean")
        mlflow.log_params(params["clean"])
        mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(METRICS / "clean_report.json"))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
