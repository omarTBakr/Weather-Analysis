"""Stage `anomaly`: multivariate (Isolation Forest, LOF) and temporal (robust z) anomaly detection."""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from weather.eda import harmonic_decomposition
from weather.plotting import CONTINENT_ORDER, CRITICAL, INK_2, MUTED, SERIES, apply_style, save
from weather.utils.config import ensure_dirs, get_settings, setup_mlflow

S = get_settings()
P = S.paths

OUT = P.figures / "anomaly"
VARS = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "precip_mm",
    "cloud",
    "uv_index",
    "visibility_km",
    "aq_pm25",
    "aq_o3",
]
LABELS = {
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


def local_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation from each city's own climate: temperature vs location-month mean, others vs location mean.

    Without this, every Arctic or desert capital would be an 'anomaly' simply for being itself.
    """
    z = pd.DataFrame(index=df.index)
    month = df["date"].dt.month
    for c in VARS:
        keys = [df["location_name"], month] if c == "temperature_celsius" else [df["location_name"]]
        g = df.groupby(keys)[c]
        sd = g.transform("std").replace(0, np.nan)
        z[c] = ((df[c] - g.transform("mean")) / sd).fillna(0.0)
    return z


def temporal_anomalies(df: pd.DataFrame, thresh: float) -> pd.Series:
    """Robust z-score of temperature against a centred 15-day rolling median (per location)."""
    df = df.sort_values(["location_name", "date"])
    med = df.groupby("location_name")["temperature_celsius"].transform(
        lambda s: s.rolling(15, center=True, min_periods=5).median()
    )
    resid = df["temperature_celsius"] - med
    mad = resid.groupby(df["location_name"]).transform(lambda s: (s - s.median()).abs().median())
    rz = 0.6745 * resid / mad.replace(0, np.nan)
    return rz.reindex(df.index).abs() > thresh, rz


def main() -> None:
    ensure_dirs()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ap = S.anomaly
    seed = S.seed
    df = pd.read_parquet(P.clean_path)
    z = local_standardise(df)

    iso = IsolationForest(n_estimators=300, contamination=ap.contamination, random_state=seed, n_jobs=-1)
    df["iso_flag"] = iso.fit_predict(z) == -1
    df["iso_score"] = -iso.score_samples(z)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(df), size=min(ap.sample_size, len(df)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=ap.lof_neighbors, contamination=ap.contamination, n_jobs=-1)
    df["lof_flag"] = pd.NA
    df.loc[df.index[sample_idx], "lof_flag"] = lof.fit_predict(z.iloc[sample_idx]) == -1
    flag_t, rz = temporal_anomalies(df, ap.z_thresh)
    df["temporal_flag"] = flag_t
    df["temporal_rz"] = rz

    # agreement between detectors (on the LOF sample)
    s = df.iloc[sample_idx]
    iso_s, lof_s = s["iso_flag"].astype(bool), s["lof_flag"].astype(bool)
    jacc = float((iso_s & lof_s).sum() / max((iso_s | lof_s).sum(), 1))

    # global-scale anomalies: residual of the global harmonic model
    gs = pd.read_parquet(P.global_series_path)
    dec = harmonic_decomposition(gs)
    gz = (dec["resid"] - dec["resid"].mean()) / dec["resid"].std()
    g_thr = ap.global_z_thresh
    global_anoms = dec[gz.abs() > g_thr].assign(z=gz[gz.abs() > g_thr])

    # --- figures -----------------------------------------------------------------------------------
    # which variables drive Isolation Forest anomalies
    drv = z[df["iso_flag"]].abs().mean().rename(LABELS).sort_values()
    base = z[~df["iso_flag"]].abs().mean().rename(LABELS).reindex(drv.index)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(drv))
    ax.barh(y + 0.18, base, height=0.34, color=MUTED, label="normal observations")
    ax.barh(y - 0.18, drv, height=0.34, color=SERIES[0], label="Isolation-Forest anomalies")
    ax.set_yticks(y, drv.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |deviation from the city's own climate| (σ)")
    ax.legend(loc="lower right")
    ax.set_title("What makes an observation anomalous")
    save(fig, OUT / "anomaly_drivers.png")

    # anomaly rate by continent and detector
    rates = (
        pd.DataFrame(
            {
                "Isolation Forest": df.groupby("continent")["iso_flag"].mean(),
                "Temporal robust-z": df.groupby("continent")["temporal_flag"].mean(),
                "LOF (sample)": s.groupby("continent")["lof_flag"].apply(lambda x: x.astype(bool).mean()),
            }
        ).reindex(CONTINENT_ORDER)
        * 100
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.26
    for i, col in enumerate(rates.columns):
        ax.bar(np.arange(len(rates)) + (i - 1) * w, rates[col], width=w - 0.02, color=SERIES[i], label=col)
    ax.set_xticks(range(len(rates)), rates.index)
    ax.set_ylabel("% of observations flagged")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Anomaly rate by continent and detector")
    save(fig, OUT / "anomaly_rate_by_continent.png")

    # PCA projection
    pca = PCA(n_components=2, random_state=seed).fit(z.iloc[sample_idx])
    proj = pca.transform(z.iloc[sample_idx])
    f = iso_s.to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(proj[~f, 0], proj[~f, 1], s=3, color=MUTED, alpha=0.25, linewidths=0, label="normal")
    ax.scatter(proj[f, 0], proj[f, 1], s=10, color=CRITICAL, alpha=0.8, linewidths=0, label="Isolation-Forest anomaly")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var.)")
    ax.legend(loc="upper right", markerscale=2)
    ax.set_title("Anomalies in the space of locally standardised weather")
    save(fig, OUT / "anomaly_pca.png")

    # global series with flagged days
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(dec.index, dec["observed"], color=SERIES[0], linewidth=1.1, label="global mean temperature")
    ax.scatter(
        global_anoms.index,
        global_anoms["observed"],
        s=40,
        color=CRITICAL,
        zorder=3,
        edgecolor="#fcfcfb",
        linewidth=1.5,
        label=f"|residual z| > {g_thr}",
    )
    for d, r in global_anoms.iterrows():
        ax.annotate(
            d.strftime("%d %b %Y"),
            (d, r["observed"]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    ax.set_ylabel("°C")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Global-scale anomalies (residual of harmonic model)")
    save(fig, OUT / "global_anomalies.png")

    # batch-level corruption detector (computed in the clean stage)
    bd = pd.read_csv(P.metrics / "daily_batch_deviation.csv", parse_dates=["date"]).set_index("date")[
        "median_abs_dev_c"
    ]
    thr = S.clean.corrupt_day_threshold_c
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(bd.index, bd, color=SERIES[0], linewidth=1)
    ax.axhline(thr, color=CRITICAL, linewidth=1, linestyle="--")
    ax.annotate(
        f"threshold {thr} °C", (bd.index[0], thr), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_2
    )
    for d, v in bd[bd > thr].items():
        ax.scatter([d], [v], s=40, color=CRITICAL, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
        ax.annotate(
            f"{d:%d %b %Y}: corrupted snapshot, nulled + imputed",
            (d, v),
            xytext=(-8, -4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=INK_2,
        )
    ax.set_ylabel("Median |dev.| across cities (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Batch-level anomaly detection: days when most cities deviate at once")
    save(fig, OUT / "batch_corruption_detector.png")

    # example city timeline with temporal anomalies
    top_city = df.groupby("location_name")["temporal_flag"].sum().idxmax()
    c = df[df["location_name"] == top_city].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(c["date"], c["temperature_celsius"], color=SERIES[0], linewidth=1, label="temperature")
    fl = c[c["temporal_flag"]]
    ax.scatter(
        fl["date"],
        fl["temperature_celsius"],
        s=28,
        color=CRITICAL,
        zorder=3,
        edgecolor="#fcfcfb",
        linewidth=1.2,
        label="temporal anomaly",
    )
    ax.set_ylabel("°C")
    ax.legend(loc="upper right")
    ax.set_title(f"Temporal anomalies — {top_city} (most flagged location)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save(fig, OUT / "city_temporal_anomalies.png")

    # monthly count of anomalies
    mon = (
        df.assign(m=df["date"].dt.to_period("M").dt.to_timestamp()).groupby("m")[["iso_flag", "temporal_flag"]].mean()
        * 100
    )
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(mon.index, mon["iso_flag"], color=SERIES[0], marker="o", markersize=3, label="Isolation Forest")
    ax.plot(mon.index, mon["temporal_flag"], color=SERIES[1], marker="o", markersize=3, label="Temporal robust-z")
    ax.set_ylabel("% flagged")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Share of anomalous observations per month")
    save(fig, OUT / "anomaly_rate_monthly.png")

    # --- tables / metrics ---------------------------------------------------------------------------
    top = (
        df[df["iso_flag"]]
        .sort_values("iso_score", ascending=False)[
            [
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
        ]
        .head(50)
    )
    top.to_csv(P.tables / "top_anomalies.csv", index=False)
    t10 = top.head(10)[["date", "location_name", "country", "temperature_celsius", "wind_kph", "precip_mm", "aq_pm25"]]
    t10 = t10.assign(date=t10["date"].dt.strftime("%Y-%m-%d")).rename(
        columns={
            "location_name": "Location",
            "country": "Country",
            "date": "Date",
            "temperature_celsius": "Temp (°C)",
            "wind_kph": "Wind (kph)",
            "precip_mm": "Precip (mm)",
            "aq_pm25": "PM2.5",
        }
    )
    t10.to_latex(
        P.tables / "top_anomalies.tex",
        index=False,
        float_format="%.1f",
        caption="Ten strongest Isolation-Forest anomalies.",
        label="tab:anomalies",
        escape=True,
    )
    global_anoms[["observed", "fitted", "z"]].round(2).to_csv(P.tables / "global_anomalies.csv")

    metrics = {
        "iso_flagged": int(df["iso_flag"].sum()),
        "iso_rate_pct": float(df["iso_flag"].mean() * 100),
        "lof_rate_pct": float(lof_s.mean() * 100),
        "temporal_flagged": int(df["temporal_flag"].sum()),
        "temporal_rate_pct": float(df["temporal_flag"].mean() * 100),
        "iso_lof_jaccard": jacc,
        "iso_temporal_overlap_pct": float(
            (df["iso_flag"] & df["temporal_flag"]).sum() / max(df["iso_flag"].sum(), 1) * 100
        ),
        "iso_vs_clean_outlier_overlap_pct": float(
            (df["iso_flag"] & df["is_outlier_any"]).sum() / max(df["iso_flag"].sum(), 1) * 100
        ),
        "global_anomaly_days": int(len(global_anoms)),
        "most_flagged_location": top_city,
    }
    (P.metrics / "anomaly.json").write_text(json.dumps(metrics, indent=2))
    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="anomaly"):
        mlflow.set_tag("stage", "anomaly")
        mlflow.log_params(ap.model_dump())
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifacts(str(OUT), artifact_path="figures")
        mlflow.log_artifact(str(P.tables / "top_anomalies.csv"))
    print(json.dumps(metrics, indent=2))
    print(global_anoms[["observed", "fitted", "z"]])


if __name__ == "__main__":
    main()
