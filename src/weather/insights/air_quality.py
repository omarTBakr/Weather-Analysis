"""Air quality vs weather: correlations, response curves, EPA index and PM2.5 drivers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from weather.enums import AirQualityInsights, Geography, Insights, Palette, VariableLabels
from weather.insights.common import InsightAnalysis
from weather.utils.plotting import annotate_heatmap, legend_below, save


class AirQualityAnalysis(InsightAnalysis):
    """Pollutant vs weather correlations, both between cities and within cities (city means removed, which
    isolates day-to-day weather effects from geography), pollution rankings and a PM2.5-driver model."""

    def prepare(self, daily: pd.DataFrame) -> None:
        """Complete-case pollutant/weather rows, and their pooled and within-city Spearman correlations."""
        cols = list(VariableLabels.POLLUTANTS) + list(VariableLabels.WEATHER)
        self.d = d = daily[cols + ["location_name", "continent", "country", "aq_us_epa_index", "date"]].dropna()
        self.between = (
            d[cols].corr(method="spearman").loc[list(VariableLabels.POLLUTANTS), list(VariableLabels.WEATHER)]
        )
        # within-city: remove each city's mean -> isolates day-to-day weather effects from geography
        dm = d[cols] - d.groupby("location_name")[cols].transform("mean")
        self.within = dm.corr(method="spearman").loc[list(VariableLabels.POLLUTANTS), list(VariableLabels.WEATHER)]

    def plot_correlations(self) -> None:
        """Side-by-side pollutant x weather correlation heatmaps: pooled vs within-city."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, m, title in zip(
            axes,
            [self.between, self.within],
            ["Pooled (geography + weather)", "Within-city (day-to-day weather)"],
            strict=True,
        ):
            im = ax.imshow(m.values, cmap=Palette.DIV_CMAP, vmin=-0.6, vmax=0.6, aspect="auto")
            ax.set_xticks(range(m.shape[1]), [VariableLabels.WEATHER[c] for c in m.columns], rotation=45, ha="right")
            ax.set_yticks(range(m.shape[0]), [VariableLabels.POLLUTANTS[c] for c in m.index])
            ax.grid(False)
            annotate_heatmap(ax, m.values, ".2f", 7, light_text=lambda v: abs(v) > 0.45)
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.85, label="Spearman ρ")
        fig.suptitle("Air pollutants vs. weather", x=0.01, ha="left", fontweight="bold")
        save(fig, self.figure_dir / "airquality_weather_correlation.png", tight=False)  # colorbar spans both axes

    def plot_response_curves(self) -> None:
        """Median and IQR of a pollutant within quantile bins of a weather variable (three panels)."""
        d = self.d
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for ax, (x, y, xl, yl) in zip(axes, AirQualityInsights.RESPONSE_CURVES, strict=True):
            b = pd.qcut(d[x], 15, duplicates="drop")
            g = d.groupby(b, observed=True)[y]
            mids = [iv.mid for iv in g.median().index]
            ax.fill_between(mids, g.quantile(0.25), g.quantile(0.75), color=Palette.SERIES[0], alpha=0.18, linewidth=0)
            ax.plot(mids, g.median(), color=Palette.SERIES[0], marker="o", markersize=3)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        axes[0].set_title("Median and IQR within weather bins", fontsize=10)
        save(fig, self.figure_dir / "airquality_response_curves.png")

    def plot_epa_by_continent(self) -> None:
        """Stacked bars of the US-EPA index categories per continent."""
        d = self.d
        epa = (
            pd.crosstab(d["continent"], d["aq_us_epa_index"], normalize="index").reindex(Geography.CONTINENT_ORDER)
            * 100
        )
        cmap = plt.get_cmap(Palette.SEQ_CMAP)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        left = np.zeros(len(epa))
        for i, k in enumerate(epa.columns):
            ax.barh(
                epa.index,
                epa[k],
                left=left,
                color=cmap(0.15 + 0.85 * i / (len(epa.columns) - 1)),
                edgecolor=Palette.SURFACE,
                linewidth=1,
                label=VariableLabels.EPA_INDEX.get(int(k), str(k)),
            )
            left += epa[k].to_numpy()
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of observations")
        ax.grid(axis="y", visible=False)
        legend_below(fig, ax, ncol=6, fontsize=7.5)
        ax.set_title("US-EPA air-quality index by continent")
        save(fig, self.figure_dir / "airquality_epa_by_continent.png")

    def plot_top_pm25(self) -> None:
        """Capitals with the highest median PM2.5, against the WHO 24-hour guideline."""
        n, who = Insights.TOP_RANKED, AirQualityInsights.WHO_PM25_GUIDELINE
        top = self.d.groupby(["location_name", "country"])["aq_pm25"].median().nlargest(n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.barh([f"{a}, {b}" for a, b in top.index], top.values, color=Palette.SERIES[0], height=0.65)
        ax.axvline(who, color=Palette.MUTED, linestyle="--", linewidth=0.9)
        ax.annotate(
            f"WHO 24-h guideline ({who} µg/m³)",
            (who, 0),
            xytext=(4, -2),
            textcoords="offset points",
            fontsize=7.5,
            color=Palette.INK_2,
        )
        ax.set_xlabel("Median PM2.5 (µg/m³)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{n} capitals with the highest median PM2.5")
        save(fig, self.figure_dir / "airquality_top_pm25.png")

    def write_tables(self) -> None:
        """Within-city and pooled correlation matrices (CSV)."""
        self.within.round(3).to_csv(self.paths.tables / "airquality_within_city_corr.csv")
        self.between.round(3).to_csv(self.paths.tables / "airquality_pooled_corr.csv")

    def pm25_drivers(self, daily: pd.DataFrame) -> float:
        """Which weather variables drive PM2.5? LightGBM on log PM2.5 with a time split, ranked by permutation
        importance; draws the figure and returns the test R² (log scale)."""
        import lightgbm as lgb

        s = self.settings
        feats = list(VariableLabels.WEATHER) + ["latitude", "longitude", "month"]
        dd = daily.dropna(subset=["aq_pm25"]).assign(month=lambda x: x["date"].dt.month)
        cut = dd["date"].max() - pd.Timedelta(days=s.split.test_days)
        trn, tst = dd[dd["date"] <= cut], dd[dd["date"] > cut]
        m = lgb.LGBMRegressor(**AirQualityInsights.PM25_MODEL_PARAMS, verbose=-1, random_state=s.seed)
        m.fit(trn[feats], np.log1p(trn["aq_pm25"]))
        y_test = np.log1p(tst["aq_pm25"])
        r2 = float(1 - np.mean((y_test - m.predict(tst[feats])) ** 2) / np.var(y_test))
        n_sample = min(AirQualityInsights.PM25_PERMUTATION_SAMPLE, len(tst))
        pi = permutation_importance(
            m,
            tst[feats].sample(n_sample, random_state=0),
            y_test.sample(n_sample, random_state=0),
            n_repeats=5,
            random_state=0,
            n_jobs=-1,
        )
        imp = pd.Series(pi.importances_mean, index=[VariableLabels.WEATHER.get(f, f.title()) for f in feats])
        imp = imp.sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(
            imp.index,
            imp.values,
            color=[Palette.SERIES[0] if i in VariableLabels.WEATHER.values() else Palette.MUTED for i in imp.index],
            height=0.65,
        )
        ax.set_xlabel("Permutation importance (drop in R², log PM2.5)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"What predicts PM2.5? (LightGBM, test R² = {r2:.2f}; grey = location/season)")
        save(fig, self.figure_dir / "airquality_pm25_drivers.png")
        return r2

    def run(self, daily: pd.DataFrame) -> dict:
        """All air-quality figures and tables; returns headline metrics."""
        self.prepare(daily)
        self.plot_correlations()
        self.plot_response_curves()
        self.plot_epa_by_continent()
        self.plot_top_pm25()
        self.write_tables()
        r2 = self.pm25_drivers(daily)
        d, within = self.d, self.within
        return {
            "aq_within_corr_pm25_wind": float(within.loc["aq_pm25", "wind_kph"]),
            "aq_within_corr_pm25_humidity": float(within.loc["aq_pm25", "humidity"]),
            "aq_within_corr_o3_temp": float(within.loc["aq_o3", "temperature_celsius"]),
            "aq_within_corr_o3_uv": float(within.loc["aq_o3", "uv_index"]),
            "aq_pooled_corr_pm25_humidity": float(self.between.loc["aq_pm25", "humidity"]),
            "aq_pm25_model_test_r2_log": r2,
            "aq_share_good_epa_pct": float((d["aq_us_epa_index"] == 1).mean() * 100),
            "aq_share_cities_median_pm25_above_who": float(
                (d.groupby("location_name")["aq_pm25"].median() > AirQualityInsights.WHO_PM25_GUIDELINE).mean() * 100
            ),
        }
