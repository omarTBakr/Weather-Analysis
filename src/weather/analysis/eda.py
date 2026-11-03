"""Stage `eda`: distributions, temperature & precipitation trends, correlations, seasonal decomposition.

Run with `python -m weather.analysis.eda` (the DVC `eda` stage).
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather.analysis.decomposition import harmonic_decomposition
from weather.enums import Calendar, Exploration, Geography, Palette, VariableLabels
from weather.utils.config import Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.plotting import annotate_heatmap, apply_style, legend_below, save
from weather.utils.timeseries import month_start


def _end_labels(ax, series_by_name: dict, colors: dict) -> None:
    """Direct labels at the right edge, nudged apart to avoid collisions."""
    items = sorted(((s.dropna().iloc[-1], n) for n, s in series_by_name.items()), reverse=True)
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * 0.045
    placed = []
    for y, name in items:
        if placed and placed[-1] - y < gap:
            y = placed[-1] - gap
        placed.append(y)
        x = series_by_name[name].dropna().index[-1]
        ax.annotate(
            name, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=Palette.INK_2
        )
        ax.plot([], [], color=colors[name])


class ExploratoryAnalysis:
    """Builds the summary-statistics table and every EDA figure, then logs an `eda` MLflow run."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.figure_dir = self.paths.figures / "eda"
        self.metrics: dict = {}

    # data -------------------------------------------------------------------------------------------------
    def load_data(self) -> None:
        """Read the cleaned observations, the daily panel and the global mean series."""
        self.clean = pd.read_parquet(self.paths.clean_path)
        self.daily = pd.read_parquet(self.paths.daily_path)
        self.global_series = pd.read_parquet(self.paths.global_series_path)

    def write_summary_table(self) -> None:
        """Mean / std / min / median / max of the main variables (CSV + LaTeX)."""
        summary = (
            self.daily[list(VariableLabels.EDA)]
            .describe()
            .T[["mean", "std", "min", "50%", "max"]]
            .round(2)
            .rename(columns={"50%": "median"})
        )
        summary.index = [VariableLabels.EDA[c] for c in summary.index]
        summary.to_csv(self.paths.tables / "summary_statistics.csv")
        summary.to_latex(
            self.paths.tables / "summary_statistics.tex",
            float_format="%.2f",
            caption="Summary statistics of the cleaned daily panel.",
            label="tab:summary",
            escape=True,
        )

    # temperature ------------------------------------------------------------------------------------------
    def plot_temperature_distribution(self) -> None:
        """Violin plot of daily temperature per continent."""
        daily = self.daily
        fig, ax = plt.subplots(figsize=(8, 3.6))
        data = [daily.loc[daily["continent"] == c, "temperature_celsius"] for c in Geography.CONTINENT_ORDER]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], Geography.CONTINENT_ORDER, strict=False):
            body.set_facecolor(Palette.CONTINENT_COLORS[c])
            body.set_edgecolor("none")
            body.set_alpha(0.75)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color(Palette.INK_2)
            parts[k].set_linewidth(0.8)
        ax.set_xticks(range(1, 7), Geography.CONTINENT_ORDER)
        ax.set_ylabel("Daily temperature (°C)")
        ax.set_title("Daily temperature distribution by continent")
        save(fig, self.figure_dir / "temperature_distribution_by_continent.png")

    def plot_global_trend(self) -> None:
        """Global mean temperature with its centred rolling mean."""
        gs = self.global_series
        s = gs.set_index("date")["temperature_celsius"]
        window = Exploration.TREND_ROLLING_DAYS
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(s.index, s, color=Palette.SERIES[0], alpha=0.35, linewidth=1, label="Daily mean")
        ax.plot(
            s.index,
            s.rolling(window, center=True).mean(),
            color=Palette.SERIES[0],
            linewidth=2,
            label=f"{window}-day rolling mean",
        )
        ax.set_ylabel("°C")
        legend_below(fig, ax, ncol=2)
        ax.set_title(f"Global mean temperature ({int(gs['n_locations'].median())} well-covered capitals)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        save(fig, self.figure_dir / "global_temperature_trend.png")

    def plot_monthly_by_continent(self) -> None:
        """Monthly mean temperature per continent (partial first/last months dropped), labelled at the line ends."""
        daily = self.daily
        m = (
            daily.assign(month=month_start(daily["date"]))
            .groupby(["continent", "month"])["temperature_celsius"]
            .mean()
            .unstack(0)
        )
        m = m.iloc[1:-1]  # drop partial first/last months
        fig, ax = plt.subplots(figsize=(9, 4))
        for c in Geography.CONTINENT_ORDER:
            ax.plot(m.index, m[c], color=Palette.CONTINENT_COLORS[c], label=c, marker="o", markersize=3)
        ax.set_xlim(m.index.min(), m.index.max() + pd.Timedelta(days=75))
        _end_labels(ax, {c: m[c] for c in Geography.CONTINENT_ORDER}, Palette.CONTINENT_COLORS)
        legend_below(fig, ax, ncol=6)
        ax.set_ylabel("Monthly mean temperature (°C)")
        ax.set_title("Monthly mean temperature by continent")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        save(fig, self.figure_dir / "monthly_temperature_by_continent.png")

    def plot_decomposition(self, dec: pd.DataFrame) -> None:
        """Four panels of the harmonic decomposition: observed + fit, trends, annual cycle, residual.

        Harmonic regression is used because STL is degenerate with only ~2.4 annual cycles.
        """
        fig, axes = plt.subplots(4, 1, figsize=(9, 6.6), sharex=True)
        axes[0].plot(dec.index, dec["observed"], color=Palette.SERIES[0], linewidth=1.1)
        axes[0].plot(dec.index, dec["fitted"], color=Palette.INK_2, linewidth=0.9, linestyle="--", label="model fit")
        axes[1].plot(
            dec.index,
            dec["trend_raw"],
            color=Palette.SERIES[1],
            label=f"naive trend ({dec.attrs['slope_raw']:+.2f} °C/yr)",
        )
        axes[1].plot(
            dec.index,
            dec["trend_adj"],
            color=Palette.SERIES[0],
            label=f"observation-hour adjusted ({dec.attrs['slope_adj']:+.2f} °C/yr)",
        )
        legend_below(fig, [axes[0], axes[1]], ncol=3)
        axes[2].plot(dec.index, dec["seasonal"], color=Palette.SERIES[0], linewidth=1.2)
        axes[3].plot(dec.index, dec["resid"], color=Palette.MUTED, linewidth=0.9)
        for ax, name in zip(axes, ["Observed", "Trend", "Annual cycle", "Residual"], strict=False):
            ax.set_ylabel(name, fontsize=8.5)
        axes[0].set_title("Decomposition of global mean temperature (trend + annual harmonics + obs. hour)")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        save(fig, self.figure_dir / "global_temperature_decomposition.png")

    def plot_observation_hour_drift(self) -> pd.Series:
        """Monthly median and IQR of the snapshot's local hour (the confounder behind the naive trend).

        Returns the monthly median local hour.
        """
        daily = self.daily
        h = daily.assign(month=month_start(daily["date"])).groupby("month")["local_hour"]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.fill_between(
            h.quantile(0.25).index,
            h.quantile(0.25),
            h.quantile(0.75),
            color=Palette.SERIES[0],
            alpha=0.2,
            linewidth=0,
            label="interquartile range",
        )
        ax.plot(h.median().index, h.median(), color=Palette.SERIES[0], marker="o", markersize=3, label="median")
        ax.set_ylabel("Local hour of snapshot")
        ax.set_ylim(0, 24)
        ax.set_yticks([0, 6, 12, 18, 24])
        legend_below(fig, ax, ncol=2)
        ax.set_title("Local clock time of the daily snapshot drifts from afternoon to early morning")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        save(fig, self.figure_dir / "observation_hour_drift.png")
        return h.median()

    def temperature_figures(self) -> dict:
        """All temperature figures; returns trend, amplitude and observation-hour headline metrics."""
        self.plot_temperature_distribution()
        self.plot_global_trend()
        self.plot_monthly_by_continent()
        dec = harmonic_decomposition(self.global_series)
        self.plot_decomposition(dec)
        median_hour = self.plot_observation_hour_drift()
        return {
            "trend_raw_c_per_year": dec.attrs["slope_raw"],
            "trend_adjusted_c_per_year": dec.attrs["slope_adj"],
            "annual_amplitude_c": float(dec["seasonal"].max() - dec["seasonal"].min()),
            "decomposition_resid_std_c": float(dec["resid"].std()),
            "median_local_hour_first_month": float(median_hour.iloc[0]),
            "median_local_hour_last_month": float(median_hour.iloc[-1]),
        }

    # precipitation ----------------------------------------------------------------------------------------
    def plot_precipitation_distribution(self) -> float:
        """Histogram of wet-observation intensity (log bins); returns the share of dry observations in %."""
        daily = self.daily
        wet = daily.loc[daily["precip_mm"] > 0, "precip_mm"]
        dry_share = float((daily["precip_mm"] == 0).mean() * 100)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        bins = np.logspace(np.log10(max(wet.min(), 0.01)), np.log10(wet.max()), 40)
        ax.hist(wet, bins=bins, color=Palette.SERIES[0], edgecolor=Palette.SURFACE, linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Precipitation on wet observations (mm, log scale)")
        ax.set_ylabel("Count")
        ax.set_title(f"Precipitation intensity — {dry_share:.0f}% of observations record no rain")
        save(fig, self.figure_dir / "precipitation_distribution.png")
        return dry_share

    def plot_precipitation_heatmap(self) -> None:
        """Mean precipitation by continent x calendar month."""
        daily = self.daily
        h = (
            daily.assign(month=daily["date"].dt.month)
            .groupby(["continent", "month"])["precip_mm"]
            .mean()
            .unstack()
            .reindex(Geography.CONTINENT_ORDER)
        )
        fig, ax = plt.subplots(figsize=(9, 3.2))
        im = ax.imshow(h.values, aspect="auto", cmap=Palette.SEQ_CMAP)
        ax.set_xticks(range(12), Calendar.MONTH_ABBREVIATIONS)
        ax.set_yticks(range(len(h)), h.index)
        ax.grid(False)
        vmax = np.nanmax(h.values)
        annotate_heatmap(ax, h.values, ".2f", 7.5, light_text=lambda v: v > vmax * 0.55)
        fig.colorbar(im, ax=ax, label="mm per observation", shrink=0.9)
        ax.set_title("Mean precipitation by continent and calendar month")
        save(fig, self.figure_dir / "precipitation_heatmap_continent_month.png")

    def plot_wettest_capitals(self) -> None:
        """Horizontal bars of the capitals with the highest mean precipitation."""
        n = Exploration.TOP_WETTEST_CAPITALS
        top = self.daily.groupby(["location_name", "country"])["precip_mm"].mean().nlargest(n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.barh([f"{loc}, {c}" for loc, c in top.index], top.values, color=Palette.SERIES[0], height=0.65)
        ax.set_xlabel("Mean precipitation (mm per observation)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{n} wettest capitals")
        save(fig, self.figure_dir / "wettest_capitals.png")

    def precipitation_figures(self) -> dict:
        """All precipitation figures; returns the dry-observation share and the mean precipitation."""
        dry_share = self.plot_precipitation_distribution()
        self.plot_precipitation_heatmap()
        self.plot_wettest_capitals()
        return {"dry_observation_share_pct": dry_share, "mean_precip_mm": float(self.daily["precip_mm"].mean())}

    # other figures ----------------------------------------------------------------------------------------
    def plot_correlation_matrix(self) -> pd.DataFrame:
        """Lower-triangle Spearman correlation heatmap of weather and air-quality variables; returns the matrix."""
        corr = self.daily[list(VariableLabels.EDA)].rename(columns=VariableLabels.EDA).corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        im = ax.imshow(np.where(mask, np.nan, corr.values), cmap=Palette.DIV_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(corr)), corr.columns)
        ax.grid(False)
        annotate_heatmap(ax, corr.values, ".2f", 6.5, light_text=lambda v: abs(v) > 0.6, lower_triangle=True)
        fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
        ax.set_title("Correlation between weather and air-quality variables")
        save(fig, self.figure_dir / "correlation_matrix.png")
        return corr

    def plot_weather_conditions(self) -> None:
        """Bar chart of the most frequent `condition_text` values (share of observations)."""
        clean = self.clean
        top = clean["condition_text"].value_counts().head(Exploration.TOP_CONDITIONS).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top.index, top.values / len(clean) * 100, color=Palette.SERIES[0], height=0.65)
        ax.set_xlabel("Share of observations (%)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Most frequent reported weather conditions")
        save(fig, self.figure_dir / "weather_conditions.png")

    def plot_cleaning_corrections(self) -> None:
        """Per-variable count of invalid values imputed vs outliers capped, read from `clean_report.json`."""
        rep = json.loads((self.paths.metrics / "clean_report.json").read_text())
        inv = pd.Series(rep["invalid_set_to_nan"])
        cap = pd.Series(rep["outliers_capped"])
        df = pd.DataFrame({"Physically invalid → imputed": inv, "Outlier → capped": cap}).fillna(0)
        df = df[(df.sum(axis=1)) > 0].sort_values("Outlier → capped")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        y = np.arange(len(df))
        ax.barh(y - 0.18, df.iloc[:, 0], height=0.34, color=Palette.SERIES[1], label=df.columns[0])
        ax.barh(y + 0.18, df.iloc[:, 1], height=0.34, color=Palette.SERIES[0], label=df.columns[1])
        ax.set_yticks(y, df.index)
        ax.set_xscale("symlog")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Rows affected (symlog scale)")
        legend_below(fig, ax, ncol=2)
        ax.set_title("Data-quality corrections per variable")
        save(fig, self.figure_dir / "cleaning_corrections.png")

    # reporting --------------------------------------------------------------------------------------------
    def log_to_mlflow(self) -> None:
        """One `eda` run with the headline metrics and every figure."""
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="eda"):
            mlflow.set_tag("stage", "eda")
            mlflow.log_metrics(self.metrics)
            mlflow.log_artifacts(str(self.figure_dir), artifact_path="figures")

    def run(self) -> None:
        """Execute the full stage: summary table, all figures, `eda.json` metrics and the MLflow run."""
        ensure_dirs(self.settings)
        apply_style()
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        self.write_summary_table()
        self.metrics = self.temperature_figures()
        self.metrics |= self.precipitation_figures()
        corr = self.plot_correlation_matrix()
        self.plot_weather_conditions()
        self.plot_cleaning_corrections()
        self.metrics["corr_temp_humidity"] = float(corr.loc["Temperature", "Humidity"])
        self.metrics["corr_temp_uv"] = float(corr.loc["Temperature", "UV index"])
        (self.paths.metrics / "eda.json").write_text(json.dumps(self.metrics, indent=2))
        self.log_to_mlflow()
        print(json.dumps(self.metrics, indent=2))


def main() -> None:
    """DVC entry point."""
    ExploratoryAnalysis().run()


if __name__ == "__main__":
    main()
