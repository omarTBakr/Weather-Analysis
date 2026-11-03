"""Stage `clean`: validity checks, de-duplication, outlier capping, imputation, continent mapping.

Run with `python -m weather.data.clean` (the DVC `clean` stage). Reads the raw Kaggle CSV and writes
`data/interim/clean.parquet` (one row per location-day) plus `clean_report.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weather.enums import CleaningColumns, CleaningRules, Geography
from weather.utils.config import CleanSettings, Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.timeseries import location_rolling_median


def country_to_continent(country: str) -> str:
    """Continent name for a country: manual overrides first, then pycountry-convert, else "Unknown"."""
    if country in Geography.CONTINENT_OVERRIDES:
        return Geography.CONTINENT_OVERRIDES[country]
    import pycountry_convert as pc

    for name in (country, country.replace("&", "and")):
        try:
            a2 = pc.country_name_to_country_alpha2(name, cn_name_format="default")
            return Geography.ISO_CONTINENT_NAMES[pc.country_alpha2_to_continent_code(a2)]
        except (KeyError, ValueError):
            continue
    return "Unknown"


def canonical_country(df: pd.DataFrame) -> pd.Series:
    """Each location keeps the country string it is most often reported with (fixes translated names)."""
    mode = df.groupby("location_name")["country"].agg(lambda s: s.value_counts().index[0])
    mode = mode.replace(Geography.COUNTRY_FIXES)
    return df["location_name"].map(mode)


def apply_valid_ranges(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Set values outside `CleaningRules.VALID_RANGES` to NaN (imputed later); counts go to `report["invalid_set_to_nan"]`."""
    invalid = {}
    for col, (lo, hi) in CleaningRules.VALID_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["invalid_set_to_nan"] = invalid
    return df


def null_corrupted_days(
    df: pd.DataFrame, threshold: float, report: dict, deviation_csv: Path | None = None
) -> pd.DataFrame:
    """Detect whole-batch provider glitches (e.g. Riyadh at 15 degC in August) and null that day's weather values.

    Statistic: per date, the median over all cities of |temperature - city's centred 15-day rolling median|
    (`RollingBaseline`). A normal day sits around 1 degC; a corrupted snapshot shifts most cities at once.
    """
    df = df.sort_values(["location_name", "date"])
    med = location_rolling_median(df, "temperature_celsius")
    day_dev = (df["temperature_celsius"] - med).abs().groupby(df["date"]).median()
    if deviation_csv is not None:
        day_dev.rename("median_abs_dev_c").to_csv(deviation_csv)
    bad = day_dev[day_dev > threshold]
    df.loc[df["date"].isin(bad.index), CleaningColumns.WEATHER] = np.nan
    report["corrupted_days"] = {str(d.date()): round(float(v), 2) for d, v in bad.items()}
    report["corrupted_rows_nulled"] = int(df["date"].isin(bad.index).sum())
    return df


def cap_outliers(df: pd.DataFrame, iqr_k: float, q: float, report: dict) -> pd.DataFrame:
    """Winsorise outliers instead of dropping rows, flagging each capped value.

    - `CleaningColumns.IQR_CAPPED`: clipped to the per-location fence [Q1 - k*IQR, Q3 + k*IQR].
    - `CleaningColumns.QUANTILE_CAPPED`: clipped above at the global `q` quantile.

    Adds one boolean `outlier_<col>` column per capped column and an `is_outlier_any` summary flag.
    """
    capped = {}
    g = df.groupby("location_name")
    for col in CleaningColumns.IQR_CAPPED:
        q1 = g[col].transform(lambda s: s.quantile(0.25))
        q3 = g[col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        flag = (df[col] < lo) | (df[col] > hi)
        df[f"outlier_{col}"] = flag
        df[col] = df[col].clip(lo, hi)
        capped[col] = int(flag.sum())
    for col in CleaningColumns.QUANTILE_CAPPED:
        hi = df[col].quantile(q)
        flag = df[col] > hi
        df[f"outlier_{col}"] = flag
        df[col] = df[col].clip(upper=hi)
        capped[col] = int(flag.sum())
    df["is_outlier_any"] = df.filter(like="outlier_").any(axis=1)
    report["outliers_capped"] = capped
    return df


def impute_missing(df: pd.DataFrame, limit: int, report: dict) -> pd.DataFrame:
    """Fill missing values: numeric -> per-location interpolation (up to `limit` steps), then per-location median,
    then global median; categorical -> per-location mode (or "Unknown")."""
    before = int(df[CleaningRules.IMPUTED_NUMERIC].isna().sum().sum())
    df = df.sort_values(["location_name", "last_updated"])
    df[CleaningRules.IMPUTED_NUMERIC] = df.groupby("location_name")[CleaningRules.IMPUTED_NUMERIC].transform(
        lambda s: s.interpolate(limit=limit, limit_direction="both")
    )
    df[CleaningRules.IMPUTED_NUMERIC] = df.groupby("location_name")[CleaningRules.IMPUTED_NUMERIC].transform(
        lambda s: s.fillna(s.median())
    )
    df[CleaningRules.IMPUTED_NUMERIC] = df[CleaningRules.IMPUTED_NUMERIC].fillna(
        df[CleaningRules.IMPUTED_NUMERIC].median()
    )
    for col in CleaningColumns.IMPUTED_CATEGORICAL:
        df[col] = df.groupby("location_name")[col].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
        )
    report["missing_before_impute"] = before
    report["missing_after_impute"] = int(df[CleaningRules.IMPUTED_NUMERIC].isna().sum().sum())
    return df


def _standardise_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop imperial duplicates, shorten air-quality names, derive `date` / `local_hour`, tidy `condition_text`."""
    df = (
        raw.drop(columns=[c for c in CleaningColumns.IMPERIAL_DUPLICATES if c in raw])
        .rename(columns=CleaningColumns.AIR_QUALITY_RENAMES)
        .copy()
    )
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["date"] = df["last_updated"].dt.normalize()
    # local clock time of the snapshot; it drifts from afternoon (2024) to early morning (2026),
    # which biases naive temperature trends -> kept as a covariate
    df["local_hour"] = df["last_updated"].dt.hour + df["last_updated"].dt.minute / 60
    df["condition_text"] = df["condition_text"].str.strip().str.title()
    return df


def _drop_duplicates_and_sparse_locations(df: pd.DataFrame, min_obs: int, report: dict) -> pd.DataFrame:
    """Remove exact duplicate rows, then locations observed on fewer than `min_obs` distinct days."""
    n = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = n - len(df)

    counts = df.groupby("location_name")["date"].nunique()
    keep = counts[counts >= min_obs].index
    report["locations_dropped_low_coverage"] = int((counts < min_obs).sum())
    df = df[df["location_name"].isin(keep)].copy()
    report["rows_dropped_low_coverage"] = int(n - report["exact_duplicates_removed"] - len(df))
    return df


def _assign_countries_and_continents(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Canonicalise each location's country and map it onto a continent."""
    df["country"] = canonical_country(df)
    df["continent"] = df["country"].map({c: country_to_continent(c) for c in df["country"].unique()})
    report["unknown_continent_countries"] = sorted(df.loc[df["continent"] == "Unknown", "country"].unique())
    return df


def _keep_latest_per_day(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """One observation per location-day: the latest update of the day."""
    n = len(df)
    df = df.sort_values("last_updated").drop_duplicates(["location_name", "date"], keep="last")
    report["same_day_duplicates_removed"] = n - len(df)
    return df


def clean_dataset(raw: pd.DataFrame, cp: CleanSettings, deviation_csv: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline on the raw Kaggle frame.

    Drops imperial duplicates, derives `date` / `local_hour`, removes duplicates and low-coverage locations,
    canonicalises countries and maps continents, keeps one observation per location-day, then applies the
    validity ranges, corrupted-day nulling, outlier capping and imputation.

    Args:
        raw: the Kaggle CSV as read by pandas.
        cp: thresholds of the `clean` settings section.
        deviation_csv: if given, the per-day batch-deviation statistic of `null_corrupted_days` is written there.

    Returns:
        `(clean, report)` where `report` collects the row/location counts of every step.
    """
    report: dict = {
        "raw_rows": len(raw),
        "raw_locations": int(raw["location_name"].nunique()),
        "raw_countries": int(raw["country"].nunique()),
        "raw_missing_cells": int(raw.isna().sum().sum()),
    }
    df = _standardise_columns(raw)
    df = _drop_duplicates_and_sparse_locations(df, cp.min_obs_per_location, report)
    df = _assign_countries_and_continents(df, report)
    df = _keep_latest_per_day(df, report)

    df = apply_valid_ranges(df, report)
    df = null_corrupted_days(df, cp.corrupt_day_threshold_c, report, deviation_csv)
    df = cap_outliers(df, cp.iqr_k, cp.heavy_tail_quantile, report)
    df = impute_missing(df, cp.interpolate_limit, report)

    report.update(
        {
            "clean_rows": len(df),
            "clean_locations": int(df["location_name"].nunique()),
            "clean_countries": int(df["country"].nunique()),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "continent_counts": df.groupby("continent")["location_name"].nunique().to_dict(),
        }
    )
    return df.reset_index(drop=True), report


class DataCleaner:
    """Runs the `clean` stage: raw CSV -> `clean.parquet` + `clean_report.json`, logged as one MLflow run."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.report_path = self.paths.metrics / "clean_report.json"
        self.deviation_path = self.paths.metrics / "daily_batch_deviation.csv"

    def clean(self) -> None:
        """Read the raw CSV and run `clean_dataset` on it (also writes the batch-deviation CSV)."""
        raw = pd.read_csv(self.settings.raw_csv)
        self.clean_df, self.report = clean_dataset(raw, self.settings.clean, self.deviation_path)

    def save(self) -> None:
        """Write the cleaned parquet and the JSON report."""
        self.clean_df.to_parquet(self.paths.clean_path, index=False)
        self.report_path.write_text(json.dumps(self.report, indent=2, default=str))

    def log_to_mlflow(self) -> None:
        """One `clean` run: the clean settings as params, the numeric report entries as metrics."""
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="clean"):
            mlflow.set_tag("stage", "clean")
            mlflow.log_params(self.settings.flat("clean"))
            mlflow.log_metrics({k: v for k, v in self.report.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(self.report_path))

    def run(self) -> None:
        """Execute the full stage."""
        ensure_dirs(self.settings)
        self.clean()
        self.save()
        self.log_to_mlflow()
        print(json.dumps(self.report, indent=2, default=str))


def main() -> None:
    """DVC entry point."""
    DataCleaner().run()


if __name__ == "__main__":
    main()
