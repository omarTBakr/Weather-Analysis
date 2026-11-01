"""Stage `features`: daily panel, lag/rolling/calendar features, time-based split, train-only scaling."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from weather.config import (
    CLEAN_PATH,
    DAILY_PATH,
    FEATURES_PATH,
    GLOBAL_SERIES_PATH,
    METRICS,
    MODELS,
    ensure_dirs,
    load_params,
    setup_mlflow,
)

STATIC = ["country", "continent", "latitude", "longitude", "timezone"]
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
CONTINENT_CODES = {"Africa": 0, "Asia": 1, "Europe": 2, "North America": 3, "South America": 4, "Oceania": 5}


def build_daily(clean: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Regular daily grid per location; short gaps interpolated, long gaps left missing."""
    out = []
    for loc, g in clean.groupby("location_name", sort=True):
        g = g.set_index("date").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        d = g[DAILY_NUMERIC].reindex(idx).interpolate(limit=limit, limit_area="inside")
        d["observed"] = g["temperature_celsius"].reindex(idx).notna()
        for c in STATIC:
            d[c] = g[c].iloc[-1]
        d["condition_text"] = g["condition_text"].reindex(idx)
        d["location_name"] = loc
        out.append(d.rename_axis("date").reset_index())
    daily = pd.concat(out, ignore_index=True)
    return daily.dropna(subset=["temperature_celsius"]).reset_index(drop=True)


def add_features(daily: pd.DataFrame, fp: dict) -> tuple[pd.DataFrame, list[str]]:
    target, h = fp["target"], fp["horizon"]
    frames = []
    for _, g in daily.groupby("location_name", sort=False):
        # re-grid so that shifts are calendar-exact even across long gaps
        g = g.set_index("date").asfreq("D")
        t = g[target]
        f = pd.DataFrame(index=g.index)
        f["temp_t"] = t
        for k in fp["lags"]:
            f[f"temp_lag{k}"] = t.shift(k)
        for w in fp["rolling_windows"]:
            f[f"temp_rollmean{w}"] = t.rolling(w, min_periods=w).mean()
            f[f"temp_rollstd{w}"] = t.rolling(w, min_periods=w).std()
        f["temp_diff1"] = t - t.shift(1)
        for c in fp["exog"]:
            f[c] = g[c]
        f["humidity_diff1"] = g["humidity"] - g["humidity"].shift(1)
        f["pressure_diff1"] = g["pressure_mb"] - g["pressure_mb"].shift(1)
        # observation clock time (diurnal cycle): next snapshot is taken at ~the same local time
        f["hour_sin"] = np.sin(2 * np.pi * g["local_hour"] / 24)
        f["hour_cos"] = np.cos(2 * np.pi * g["local_hour"] / 24)
        f["target"] = t.shift(-h)
        f["target_date"] = f.index + pd.Timedelta(days=h)
        for c in ["location_name", "country", "continent", "latitude", "longitude"]:
            f[c] = g[c].ffill().bfill()
        frames.append(f.rename_axis("date").reset_index())
    feat = pd.concat(frames, ignore_index=True)

    doy = feat["target_date"].dt.day_of_year
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # hemisphere-aware seasonal phase: flips the annual cycle for the southern hemisphere
    feat["season_sin"] = feat["doy_sin"] * np.sign(feat["latitude"]).replace(0, 1)
    feat["season_cos"] = feat["doy_cos"] * np.sign(feat["latitude"]).replace(0, 1)
    feat["abs_latitude"] = feat["latitude"].abs()
    feat["continent_code"] = feat["continent"].map(CONTINENT_CODES).fillna(-1).astype(int)

    feature_cols = [
        c for c in feat.columns if c not in {"date", "target", "target_date", "location_name", "country", "continent"}
    ]
    feat = feat.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    return feat, feature_cols


def time_split(feat: pd.DataFrame, test_days: int, val_days: int) -> pd.DataFrame:
    last = feat["target_date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    feat["split"] = np.select(
        [feat["target_date"] >= test_start, feat["target_date"] >= val_start], ["test", "val"], "train"
    )
    return feat


def build_global_series(daily: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """Mean over a fixed set of well-covered locations so composition changes don't create fake trends."""
    n_days = daily["date"].nunique()
    obs = daily[daily["observed"]]
    cov = obs.groupby("location_name")["date"].nunique() / n_days
    stable = cov[cov >= min_coverage].index
    d = daily[daily["location_name"].isin(stable)].assign(
        hour_sin=lambda x: np.sin(2 * np.pi * x["local_hour"] / 24),
        hour_cos=lambda x: np.cos(2 * np.pi * x["local_hour"] / 24),
    )
    s = d.groupby("date").agg(
        temperature_celsius=("temperature_celsius", "mean"),
        precip_mm=("precip_mm", "mean"),
        humidity=("humidity", "mean"),
        local_hour=("local_hour", "mean"),
        hour_sin=("hour_sin", "mean"),
        hour_cos=("hour_cos", "mean"),
        n_locations=("location_name", "nunique"),
    )
    s = s[s["n_locations"] >= 0.9 * len(stable)].asfreq("D").interpolate(limit=3)
    s.attrs["n_stable_locations"] = len(stable)
    return s.rename_axis("date").reset_index()


def main() -> None:
    ensure_dirs()
    params = load_params()
    fp, sp = params["features"], params["split"]
    clean = pd.read_parquet(CLEAN_PATH)

    daily = build_daily(clean, params["clean"]["interpolate_limit"])
    daily.to_parquet(DAILY_PATH, index=False)

    feat, feature_cols = add_features(daily, fp)
    feat = time_split(feat, sp["test_days"], sp["val_days"])

    # Normalisation: fit on the training window only (no leakage), persisted for linear models / anomaly detection
    scaler = StandardScaler().fit(feat.loc[feat["split"] == "train", feature_cols])
    joblib.dump(scaler, MODELS / "scaler.joblib")
    (MODELS / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    feat.to_parquet(FEATURES_PATH, index=False)

    gs = build_global_series(daily, params["global_series"]["min_coverage"])
    gs.to_parquet(GLOBAL_SERIES_PATH, index=False)

    split_info = {
        s: {
            "rows": int((feat["split"] == s).sum()),
            "start": str(feat.loc[feat["split"] == s, "target_date"].min().date()),
            "end": str(feat.loc[feat["split"] == s, "target_date"].max().date()),
        }
        for s in ["train", "val", "test"]
    }
    report = {
        "daily_rows": len(daily),
        "feature_rows": len(feat),
        "n_features": len(feature_cols),
        "global_series_days": len(gs),
        "global_series_locations": int(gs["n_locations"].median()),
        "split": split_info,
    }
    (METRICS / "features_report.json").write_text(json.dumps(report, indent=2))

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="features"):
        mlflow.set_tag("stage", "features")
        mlflow.log_params({**{f"features.{k}": v for k, v in fp.items()}, **{f"split.{k}": v for k, v in sp.items()}})
        mlflow.log_metrics(
            {
                "feature_rows": len(feat),
                "n_features": len(feature_cols),
                **{f"{s}_rows": v["rows"] for s, v in split_info.items()},
            }
        )
        mlflow.log_artifact(str(MODELS / "feature_columns.json"))
        mlflow.log_artifact(str(METRICS / "features_report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
