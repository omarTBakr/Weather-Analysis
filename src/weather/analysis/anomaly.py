"""Stage `anomaly`: multivariate (Isolation Forest, LOF) and temporal (robust z) anomaly detection.

Run with `python -m weather.analysis.anomaly` (the DVC `anomaly` stage).
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from weather.analysis.decomposition import harmonic_decomposition
from weather.enums import AnomalyDetecting, Geography, Palette, VariableLabels
from weather.utils.config import Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.plotting import apply_style, legend_below, save
from weather.utils.timeseries import location_rolling_median, month_start


def local_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation from each city's own climate: temperature vs location-month mean, others vs location mean.

    Without this, every Arctic or desert capital would be an 'anomaly' simply for being itself.
    """
    z = pd.DataFrame(index=df.index)
    month = df["date"].dt.month
    for c in VariableLabels.ANOMALY:
        keys = [df["location_name"], month] if c == "temperature_celsius" else [df["location_name"]]
        g = df.groupby(keys)[c]
        sd = g.transform("std").replace(0, np.nan)
        z[c] = ((df[c] - g.transform("mean")) / sd).fillna(0.0)
    return z


def temporal_anomalies(df: pd.DataFrame, thresh: float) -> tuple[pd.Series, pd.Series]:
    """Robust z-score of temperature against a centred 15-day rolling median (per location, `RollingBaseline`).

    Returns `(is_anomaly, robust_z)`: a boolean flag (|z| > `thresh`) aligned on `df.index`, and the z-scores.
    """
    df = df.sort_values(["location_name", "date"])
    resid = df["temperature_celsius"] - location_rolling_median(df, "temperature_celsius")
    mad = resid.groupby(df["location_name"]).transform(lambda s: (s - s.median()).abs().median())
    rz = AnomalyDetecting.MAD_TO_SIGMA * resid / mad.replace(0, np.nan)
    return rz.reindex(df.index).abs() > thresh, rz


def flag_share(flags: pd.Series, base: pd.Series) -> float:
    """Share (%) of the `base` flags that are also set in `flags` (0 when `base` has none)."""
    return float((flags & base).sum() / max(base.sum(), 1) * 100)


class AnomalyDetector:
    """Runs every detector on the cleaned observations and the global series, then writes figures, tables,
    `anomaly.json` and an `anomaly` MLflow run."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.params = self.settings.anomaly
        self.seed = self.settings.seed
        self.figure_dir = self.paths.figures / "anomaly"

    # detection --------------------------------------------------------------------------------------------
    def load_data(self) -> None:
        """Read the cleaned observations and standardise them against each city's own climate."""
        self.df = pd.read_parquet(self.paths.clean_path)
        self.z = local_standardise(self.df)

    def detect_isolation_forest(self) -> None:
        """Isolation Forest on all observations: `iso_flag` and the anomaly score `iso_score`."""
        iso = IsolationForest(
            n_estimators=AnomalyDetecting.ISOLATION_TREES,
            contamination=self.params.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.df["iso_flag"] = iso.fit_predict(self.z) == -1
        self.df["iso_score"] = -iso.score_samples(self.z)

    def detect_lof(self) -> None:
        """Local Outlier Factor on a random sample (`sample_size` rows); `lof_flag` is NA outside the sample."""
        df = self.df
        rng = np.random.default_rng(self.seed)
        self.sample_idx = rng.choice(len(df), size=min(self.params.sample_size, len(df)), replace=False)
        lof = LocalOutlierFactor(
            n_neighbors=self.params.lof_neighbors, contamination=self.params.contamination, n_jobs=-1
        )
        df["lof_flag"] = pd.NA
        df.loc[df.index[self.sample_idx], "lof_flag"] = lof.fit_predict(self.z.iloc[self.sample_idx]) == -1

    def detect_temporal(self) -> None:
        """Per-city robust z against the rolling median: `temporal_flag` and `temporal_rz`."""
        flag, rz = temporal_anomalies(self.df, self.params.z_thresh)
        self.df["temporal_flag"] = flag
        self.df["temporal_rz"] = rz

    def detector_agreement(self) -> None:
        """Isolation Forest vs LOF flags on the LOF sample, and their Jaccard overlap."""
        self.sample = self.df.iloc[self.sample_idx]
        self.iso_sample = self.sample["iso_flag"].astype(bool)
        self.lof_sample = self.sample["lof_flag"].astype(bool)
        both = (self.iso_sample & self.lof_sample).sum()
        self.jaccard = float(both / max((self.iso_sample | self.lof_sample).sum(), 1))

    def detect_global(self) -> None:
        """Global-scale anomalies: days whose harmonic-model residual has |z| > `global_z_thresh`."""
        self.decomposition = dec = harmonic_decomposition(pd.read_parquet(self.paths.global_series_path))
        gz = (dec["resid"] - dec["resid"].mean()) / dec["resid"].std()
        threshold = self.params.global_z_thresh
        self.global_anomalies = dec[gz.abs() > threshold].assign(z=gz[gz.abs() > threshold])

    # figures ----------------------------------------------------------------------------------------------
    def plot_drivers(self) -> None:
        """Mean |local deviation| per variable: Isolation-Forest anomalies vs normal observations."""
        z, iso = self.z, self.df["iso_flag"]
        drv = z[iso].abs().mean().rename(VariableLabels.ANOMALY).sort_values()
        base = z[~iso].abs().mean().rename(VariableLabels.ANOMALY).reindex(drv.index)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        y = np.arange(len(drv))
        ax.barh(y + 0.18, base, height=0.34, color=Palette.MUTED, label="normal observations")
        ax.barh(y - 0.18, drv, height=0.34, color=Palette.SERIES[0], label="Isolation-Forest anomalies")
        ax.set_yticks(y, drv.index)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean |deviation from the city's own climate| (σ)")
        legend_below(fig, ax, ncol=2)
        ax.set_title("What makes an observation anomalous")
        save(fig, self.figure_dir / "anomaly_drivers.png")

    def plot_rate_by_continent(self) -> None:
        """Grouped bars: share of observations flagged per continent and detector."""
        df = self.df
        rates = (
            pd.DataFrame(
                {
                    "Isolation Forest": df.groupby("continent")["iso_flag"].mean(),
                    "Temporal robust-z": df.groupby("continent")["temporal_flag"].mean(),
                    "LOF (sample)": self.sample.groupby("continent")["lof_flag"].apply(lambda x: x.astype(bool).mean()),
                }
            ).reindex(Geography.CONTINENT_ORDER)
            * 100
        )
        fig, ax = plt.subplots(figsize=(8, 3.4))
        w = 0.26
        for i, col in enumerate(rates.columns):
            ax.bar(np.arange(len(rates)) + (i - 1) * w, rates[col], width=w - 0.02, color=Palette.SERIES[i], label=col)
        ax.set_xticks(range(len(rates)), rates.index)
        ax.set_ylabel("% of observations flagged")
        legend_below(fig, ax, ncol=3)
        ax.grid(axis="x", visible=False)
        ax.set_title("Anomaly rate by continent and detector")
        save(fig, self.figure_dir / "anomaly_rate_by_continent.png")

    def plot_pca(self) -> None:
        """2-D PCA projection of the LOF sample, Isolation-Forest anomalies highlighted."""
        sample_z = self.z.iloc[self.sample_idx]
        pca = PCA(n_components=2, random_state=self.seed).fit(sample_z)
        proj = pca.transform(sample_z)
        f = self.iso_sample.to_numpy()
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(
            proj[~f, 0],
            proj[~f, 1],
            s=3,
            color=Palette.MUTED,
            alpha=0.25,
            linewidths=0,
            label="normal",
            rasterized=True,
        )
        ax.scatter(
            proj[f, 0],
            proj[f, 1],
            s=10,
            color=Palette.CRITICAL,
            alpha=0.8,
            linewidths=0,
            label="Isolation-Forest anomaly",
            rasterized=True,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var.)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var.)")
        legend_below(fig, ax, ncol=2, markerscale=2)
        ax.set_title("Anomalies in the space of locally standardised weather")
        save(fig, self.figure_dir / "anomaly_pca.png")

    def plot_global_anomalies(self) -> None:
        """Global mean temperature with the flagged global-scale anomaly days annotated."""
        dec, anomalies = self.decomposition, self.global_anomalies
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(dec.index, dec["observed"], color=Palette.SERIES[0], linewidth=1.1, label="global mean temperature")
        ax.scatter(
            anomalies.index,
            anomalies["observed"],
            s=40,
            color=Palette.CRITICAL,
            zorder=3,
            edgecolor=Palette.SURFACE,
            linewidth=1.5,
            label=f"|residual z| > {self.params.global_z_thresh}",
        )
        for d, r in anomalies.iterrows():
            ax.annotate(
                d.strftime("%d %b %Y"),
                (d, r["observed"]),
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=8,
                color=Palette.INK_2,
            )
        ax.set_ylabel("°C")
        legend_below(fig, ax, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title("Global-scale anomalies (residual of harmonic model)")
        save(fig, self.figure_dir / "global_anomalies.png")

    def plot_batch_corruption(self) -> None:
        """The clean stage's per-day batch-deviation statistic with its threshold and the nulled days."""
        bd = pd.read_csv(self.paths.metrics / "daily_batch_deviation.csv", parse_dates=["date"]).set_index("date")[
            "median_abs_dev_c"
        ]
        thr = self.settings.clean.corrupt_day_threshold_c
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(bd.index, bd, color=Palette.SERIES[0], linewidth=1)
        ax.axhline(thr, color=Palette.CRITICAL, linewidth=1, linestyle="--")
        ax.annotate(
            f"threshold {thr} °C",
            (bd.index[0], thr),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=Palette.INK_2,
        )
        for d, v in bd[bd > thr].items():
            ax.scatter([d], [v], s=40, color=Palette.CRITICAL, zorder=3, edgecolor=Palette.SURFACE, linewidth=1.5)
            ax.annotate(
                f"{d:%d %b %Y}: corrupted snapshot, nulled + imputed",
                (d, v),
                xytext=(-8, -4),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=Palette.INK_2,
            )
        ax.set_ylabel("Median |dev.| across cities (°C)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title("Batch-level anomaly detection: days when most cities deviate at once")
        save(fig, self.figure_dir / "batch_corruption_detector.png")

    def plot_city_timeline(self) -> None:
        """Temperature of the most-flagged city with its temporal anomalies marked."""
        df = self.df
        self.top_city = df.groupby("location_name")["temporal_flag"].sum().idxmax()
        c = df[df["location_name"] == self.top_city].sort_values("date")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(c["date"], c["temperature_celsius"], color=Palette.SERIES[0], linewidth=1, label="temperature")
        fl = c[c["temporal_flag"]]
        ax.scatter(
            fl["date"],
            fl["temperature_celsius"],
            s=28,
            color=Palette.CRITICAL,
            zorder=3,
            edgecolor=Palette.SURFACE,
            linewidth=1.2,
            label="temporal anomaly",
        )
        ax.set_ylabel("°C")
        legend_below(fig, ax, ncol=2)
        ax.set_title(f"Temporal anomalies — {self.top_city} (most flagged location)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        save(fig, self.figure_dir / "city_temporal_anomalies.png")

    def plot_monthly_rate(self) -> None:
        """Monthly share of observations flagged by Isolation Forest and by the temporal detector."""
        df = self.df
        mon = df.assign(m=month_start(df["date"])).groupby("m")[["iso_flag", "temporal_flag"]].mean() * 100
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(mon.index, mon["iso_flag"], color=Palette.SERIES[0], marker="o", markersize=3, label="Isolation Forest")
        ax.plot(
            mon.index,
            mon["temporal_flag"],
            color=Palette.SERIES[1],
            marker="o",
            markersize=3,
            label="Temporal robust-z",
        )
        ax.set_ylabel("% flagged")
        legend_below(fig, ax, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title("Share of anomalous observations per month")
        save(fig, self.figure_dir / "anomaly_rate_monthly.png")

    # tables / metrics -------------------------------------------------------------------------------------
    def write_tables(self) -> None:
        """Strongest Isolation-Forest anomalies (CSV + LaTeX) and the global anomaly days (CSV)."""
        df, tables = self.df, self.paths.tables
        top = (
            df[df["iso_flag"]]
            .sort_values("iso_score", ascending=False)[AnomalyDetecting.TOP_TABLE_COLUMNS]
            .head(AnomalyDetecting.TOP_TABLE_ROWS)
        )
        top.to_csv(tables / "top_anomalies.csv", index=False)
        latex = top.head(AnomalyDetecting.TOP_LATEX_ROWS)[list(AnomalyDetecting.TOP_LATEX_HEADERS)]
        latex = latex.assign(date=latex["date"].dt.strftime("%Y-%m-%d")).rename(
            columns=AnomalyDetecting.TOP_LATEX_HEADERS
        )
        latex.to_latex(
            tables / "top_anomalies.tex",
            index=False,
            float_format="%.1f",
            caption="Ten strongest Isolation-Forest anomalies.",
            label="tab:anomalies",
            escape=True,
        )
        self.global_anomalies[["observed", "fitted", "z"]].round(2).to_csv(tables / "global_anomalies.csv")

    def summary_metrics(self) -> dict:
        """Flag counts / rates per detector, detector overlaps and the most-flagged location."""
        df = self.df
        return {
            "iso_flagged": int(df["iso_flag"].sum()),
            "iso_rate_pct": float(df["iso_flag"].mean() * 100),
            "lof_rate_pct": float(self.lof_sample.mean() * 100),
            "temporal_flagged": int(df["temporal_flag"].sum()),
            "temporal_rate_pct": float(df["temporal_flag"].mean() * 100),
            "iso_lof_jaccard": self.jaccard,
            "iso_temporal_overlap_pct": flag_share(df["temporal_flag"], df["iso_flag"]),
            "iso_vs_clean_outlier_overlap_pct": flag_share(df["is_outlier_any"], df["iso_flag"]),
            "global_anomaly_days": int(len(self.global_anomalies)),
            "most_flagged_location": self.top_city,
        }

    def log_to_mlflow(self, metrics: dict) -> None:
        """One `anomaly` run: anomaly settings as params, numeric metrics, figures and the top-anomaly table."""
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="anomaly"):
            mlflow.set_tag("stage", "anomaly")
            mlflow.log_params(self.params.model_dump())
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifacts(str(self.figure_dir), artifact_path="figures")
            mlflow.log_artifact(str(self.paths.tables / "top_anomalies.csv"))

    def run(self) -> None:
        """Execute the full stage."""
        ensure_dirs(self.settings)
        apply_style()
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        self.detect_isolation_forest()
        self.detect_lof()
        self.detect_temporal()
        self.detector_agreement()
        self.detect_global()
        self.plot_drivers()
        self.plot_rate_by_continent()
        self.plot_pca()
        self.plot_global_anomalies()
        self.plot_batch_corruption()
        self.plot_city_timeline()
        self.plot_monthly_rate()
        self.write_tables()
        metrics = self.summary_metrics()
        (self.paths.metrics / "anomaly.json").write_text(json.dumps(metrics, indent=2))
        self.log_to_mlflow(metrics)
        print(json.dumps(metrics, indent=2))
        print(self.global_anomalies[["observed", "fitted", "z"]])


def main() -> None:
    """DVC entry point."""
    AnomalyDetector().run()


if __name__ == "__main__":
    main()
