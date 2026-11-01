"""Air quality vs weather: correlations, response curves, EPA index and PM2.5 drivers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from weather.insights.common import OUT, POLLUTANTS, WEATHER, P, S
from weather.utils.plotting import (
    CONTINENT_ORDER,
    DIV_CMAP,
    INK_2,
    MUTED,
    SEQ_CMAP,
    SERIES,
    legend_below,
    save,
)


def air_quality_analysis(daily: pd.DataFrame) -> dict:
    cols = list(POLLUTANTS) + list(WEATHER)
    d = daily[cols + ["location_name", "continent", "country", "aq_us_epa_index", "date"]].dropna()
    between = d[cols].corr(method="spearman").loc[list(POLLUTANTS), list(WEATHER)]
    # within-city: remove each city's mean -> isolates day-to-day weather effects from geography
    dm = d[cols] - d.groupby("location_name")[cols].transform("mean")
    within = dm.corr(method="spearman").loc[list(POLLUTANTS), list(WEATHER)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, m, title in zip(
        axes, [between, within], ["Pooled (geography + weather)", "Within-city (day-to-day weather)"], strict=True
    ):
        im = ax.imshow(m.values, cmap=DIV_CMAP, vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(m.shape[1]), [WEATHER[c] for c in m.columns], rotation=45, ha="right")
        ax.set_yticks(range(m.shape[0]), [POLLUTANTS[c] for c in m.index])
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{m.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#ffffff" if abs(m.values[i, j]) > 0.45 else "#0b0b0b",
                )
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Spearman ρ")
    fig.suptitle("Air pollutants vs. weather", x=0.01, ha="left", fontweight="bold")
    save(fig, OUT / "airquality_weather_correlation.png", tight=False)  # colorbar spans both axes

    # binned response curves
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (x, y, xl, yl) in zip(
        axes,
        [
            ("wind_kph", "aq_pm25", "Wind (kph)", "PM2.5 (µg/m³)"),
            ("humidity", "aq_pm25", "Humidity (%)", "PM2.5 (µg/m³)"),
            ("temperature_celsius", "aq_o3", "Temperature (°C)", "O3 (µg/m³)"),
        ],
        strict=True,
    ):
        b = pd.qcut(d[x], 15, duplicates="drop")
        g = d.groupby(b, observed=True)[y]
        mids = [iv.mid for iv in g.median().index]
        ax.fill_between(mids, g.quantile(0.25), g.quantile(0.75), color=SERIES[0], alpha=0.18, linewidth=0)
        ax.plot(mids, g.median(), color=SERIES[0], marker="o", markersize=3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].set_title("Median and IQR within weather bins", fontsize=10)
    save(fig, OUT / "airquality_response_curves.png")

    # EPA index composition by continent
    epa = pd.crosstab(d["continent"], d["aq_us_epa_index"], normalize="index").reindex(CONTINENT_ORDER) * 100
    labels = {1: "Good", 2: "Moderate", 3: "Unhealthy (sensitive)", 4: "Unhealthy", 5: "Very unhealthy", 6: "Hazardous"}
    cmap = plt.get_cmap(SEQ_CMAP)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    left = np.zeros(len(epa))
    for i, k in enumerate(epa.columns):
        ax.barh(
            epa.index,
            epa[k],
            left=left,
            color=cmap(0.15 + 0.85 * i / (len(epa.columns) - 1)),
            edgecolor="#fcfcfb",
            linewidth=1,
            label=labels.get(int(k), str(k)),
        )
        left += epa[k].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of observations")
    ax.grid(axis="y", visible=False)
    legend_below(fig, ax, ncol=6, fontsize=7.5)
    ax.set_title("US-EPA air-quality index by continent")
    save(fig, OUT / "airquality_epa_by_continent.png")

    top = d.groupby(["location_name", "country"])["aq_pm25"].median().nlargest(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([f"{a}, {b}" for a, b in top.index], top.values, color=SERIES[0], height=0.65)
    ax.axvline(15, color=MUTED, linestyle="--", linewidth=0.9)
    ax.annotate(
        "WHO 24-h guideline (15 µg/m³)", (15, 0), xytext=(4, -2), textcoords="offset points", fontsize=7.5, color=INK_2
    )
    ax.set_xlabel("Median PM2.5 (µg/m³)")
    ax.grid(axis="y", visible=False)
    ax.set_title("15 capitals with the highest median PM2.5")
    save(fig, OUT / "airquality_top_pm25.png")

    within.round(3).to_csv(P.tables / "airquality_within_city_corr.csv")
    between.round(3).to_csv(P.tables / "airquality_pooled_corr.csv")

    # which weather variables drive PM2.5? (LightGBM + permutation importance, time split)
    import lightgbm as lgb

    feats = list(WEATHER) + ["latitude", "longitude", "month"]
    dd = daily.dropna(subset=["aq_pm25"]).assign(month=lambda x: x["date"].dt.month)
    cut = dd["date"].max() - pd.Timedelta(days=S.split.test_days)
    trn, tst = dd[dd["date"] <= cut], dd[dd["date"] > cut]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=S.seed)
    m.fit(trn[feats], np.log1p(trn["aq_pm25"]))
    r2 = float(1 - np.mean((np.log1p(tst["aq_pm25"]) - m.predict(tst[feats])) ** 2) / np.var(np.log1p(tst["aq_pm25"])))
    pi = permutation_importance(
        m,
        tst[feats].sample(min(15000, len(tst)), random_state=0),
        np.log1p(tst["aq_pm25"]).sample(min(15000, len(tst)), random_state=0),
        n_repeats=5,
        random_state=0,
        n_jobs=-1,
    )
    imp = pd.Series(pi.importances_mean, index=[WEATHER.get(f, f.title()) for f in feats]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(
        imp.index, imp.values, color=[SERIES[0] if i in WEATHER.values() else MUTED for i in imp.index], height=0.65
    )
    ax.set_xlabel("Permutation importance (drop in R², log PM2.5)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What predicts PM2.5? (LightGBM, test R² = {r2:.2f}; grey = location/season)")
    save(fig, OUT / "airquality_pm25_drivers.png")

    return {
        "aq_within_corr_pm25_wind": float(within.loc["aq_pm25", "wind_kph"]),
        "aq_within_corr_pm25_humidity": float(within.loc["aq_pm25", "humidity"]),
        "aq_within_corr_o3_temp": float(within.loc["aq_o3", "temperature_celsius"]),
        "aq_within_corr_o3_uv": float(within.loc["aq_o3", "uv_index"]),
        "aq_pooled_corr_pm25_humidity": float(between.loc["aq_pm25", "humidity"]),
        "aq_pm25_model_test_r2_log": r2,
        "aq_share_good_epa_pct": float((d["aq_us_epa_index"] == 1).mean() * 100),
        "aq_share_cities_median_pm25_above_who": float(
            (d.groupby("location_name")["aq_pm25"].median() > 15).mean() * 100
        ),
    }
