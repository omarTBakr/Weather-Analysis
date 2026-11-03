"""Geographic patterns: continent/country summaries, hottest and coldest cities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from weather.enums import GeographicInsights, Geography, Palette
from weather.insights.common import InsightAnalysis
from weather.utils.plotting import save


class GeographicAnalysis(InsightAnalysis):
    """Continent-level distributions of key weather variables, continent / country summary tables and the
    hottest / coldest capitals (from the per-city table of `SpatialAnalysis`)."""

    def plot_continent_boxplots(self, daily: pd.DataFrame) -> None:
        """Box plots of temperature, humidity, wind and PM2.5 per continent."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
        for ax, (v, lab) in zip(axes.flat, GeographicInsights.BOXPLOT_VARIABLES, strict=True):
            data = [daily.loc[daily["continent"] == c, v].dropna() for c in Geography.CONTINENT_ORDER]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops={"color": Palette.INK})
            for patch, c in zip(bp["boxes"], Geography.CONTINENT_ORDER, strict=True):
                patch.set_facecolor(Palette.CONTINENT_COLORS[c])
                patch.set_alpha(0.8)
                patch.set_edgecolor(Palette.INK_2)
            ax.set_xticks(range(1, 7), [c.replace(" ", "\n") for c in Geography.CONTINENT_ORDER], fontsize=8)
            ax.set_ylabel(lab)
            ax.grid(axis="x", visible=False)
            if v == "aq_pm25":
                ax.set_yscale("log")
        fig.suptitle(
            "Weather and air quality by continent (whiskers = 1.5 IQR, outliers hidden)",
            x=0.01,
            ha="left",
            fontweight="bold",
        )
        save(fig, self.figure_dir / "geo_continent_boxplots.png")

    def plot_hottest_coldest(self, hot: pd.DataFrame, cold: pd.DataFrame) -> None:
        """Side-by-side bars of the hottest and coldest capitals by mean temperature."""
        n = GeographicInsights.TOP_EXTREMES
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, d, title, color in [
            (axes[0], hot.iloc[::-1], f"{n} hottest capitals", Palette.SERIES[7]),
            (axes[1], cold.iloc[::-1], f"{n} coldest capitals", Palette.SERIES[0]),
        ]:
            ax.barh(d["location_name"] + ", " + d["country"], d["temperature"], color=color, height=0.65)
            ax.set_xlabel("Mean temperature (°C)")
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", visible=False)
        save(fig, self.figure_dir / "geo_hottest_coldest.png")

    def write_continent_summary(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Per-continent weather / air-quality profile (CSV + LaTeX); returns the unrounded table."""
        cont = (
            daily.groupby("continent")
            .agg(
                cities=("location_name", "nunique"),
                temp_mean=("temperature_celsius", "mean"),
                temp_std=("temperature_celsius", "std"),
                precip_mean=("precip_mm", "mean"),
                humidity=("humidity", "mean"),
                wind=("wind_kph", "mean"),
                pm25_median=("aq_pm25", "median"),
                uv=("uv_index", "mean"),
            )
            .reindex(Geography.CONTINENT_ORDER)
        )
        cont.round(2).to_csv(self.paths.tables / "continent_summary.csv")
        cont.rename(columns=GeographicInsights.CONTINENT_LATEX_HEADERS).to_latex(
            self.paths.tables / "continent_summary.tex",
            float_format="%.2f",
            escape=False,
            caption="Weather and air-quality profile per continent.",
            label="tab:continents",
        )
        return cont

    def write_country_summary(self, daily: pd.DataFrame) -> None:
        """Per-country mean temperature, mean precipitation and median PM2.5 (CSV)."""
        country = (
            daily.groupby(["continent", "country"])
            .agg(
                temp_mean=("temperature_celsius", "mean"),
                precip_mean=("precip_mm", "mean"),
                pm25_median=("aq_pm25", "median"),
            )
            .round(2)
        )
        country.to_csv(self.paths.tables / "country_summary.csv")

    def run(self, daily: pd.DataFrame, loc: pd.DataFrame) -> dict:
        """All geographic figures and tables; returns headline metrics."""
        self.plot_continent_boxplots(daily)
        n = GeographicInsights.TOP_EXTREMES
        hot, cold = loc.nlargest(n, "temperature"), loc.nsmallest(n, "temperature")
        self.plot_hottest_coldest(hot, cold)
        cont = self.write_continent_summary(daily)
        self.write_country_summary(daily)
        return {
            "geo_hottest": f"{hot.iloc[0]['location_name']} ({hot.iloc[0]['temperature']:.1f} °C)",
            "geo_coldest": f"{cold.iloc[0]['location_name']} ({cold.iloc[0]['temperature']:.1f} °C)",
            "geo_most_variable_continent": str(cont["temp_std"].idxmax()),
            "geo_highest_pm25_continent": str(cont["pm25_median"].idxmax()),
        }
