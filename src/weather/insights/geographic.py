"""Geographic patterns: continent/country summaries, hottest and coldest cities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from weather.insights.common import OUT, P
from weather.utils.plotting import (
    CONTINENT_COLORS,
    CONTINENT_ORDER,
    INK_2,
    SERIES,
    save,
)


def geographic_patterns(daily: pd.DataFrame, loc: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    for ax, (v, lab) in zip(
        axes.flat,
        [
            ("temperature_celsius", "Temperature (°C)"),
            ("humidity", "Humidity (%)"),
            ("wind_kph", "Wind (kph)"),
            ("aq_pm25", "PM2.5 (µg/m³, log)"),
        ],
        strict=True,
    ):
        data = [daily.loc[daily["continent"] == c, v].dropna() for c in CONTINENT_ORDER]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops={"color": "#0b0b0b"})
        for patch, c in zip(bp["boxes"], CONTINENT_ORDER, strict=True):
            patch.set_facecolor(CONTINENT_COLORS[c])
            patch.set_alpha(0.8)
            patch.set_edgecolor(INK_2)
        ax.set_xticks(range(1, 7), [c.replace(" ", "\n") for c in CONTINENT_ORDER], fontsize=8)
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
    save(fig, OUT / "geo_continent_boxplots.png")

    hot, cold = loc.nlargest(10, "temperature"), loc.nsmallest(10, "temperature")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, d, title, color in [
        (axes[0], hot.iloc[::-1], "10 hottest capitals", SERIES[7]),
        (axes[1], cold.iloc[::-1], "10 coldest capitals", SERIES[0]),
    ]:
        ax.barh(d["location_name"] + ", " + d["country"], d["temperature"], color=color, height=0.65)
        ax.set_xlabel("Mean temperature (°C)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    save(fig, OUT / "geo_hottest_coldest.png")

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
        .reindex(CONTINENT_ORDER)
    )
    cont.round(2).to_csv(P.tables / "continent_summary.csv")
    cont.rename(
        columns={
            "cities": "Cities",
            "temp_mean": "Temp (°C)",
            "temp_std": "Temp SD",
            "precip_mean": "Precip (mm)",
            "humidity": "Humidity (\\%)",
            "wind": "Wind (kph)",
            "pm25_median": "PM2.5 med.",
            "uv": "UV",
        }
    ).to_latex(
        P.tables / "continent_summary.tex",
        float_format="%.2f",
        escape=False,
        caption="Weather and air-quality profile per continent.",
        label="tab:continents",
    )

    country = (
        daily.groupby(["continent", "country"])
        .agg(
            temp_mean=("temperature_celsius", "mean"),
            precip_mean=("precip_mm", "mean"),
            pm25_median=("aq_pm25", "median"),
        )
        .round(2)
    )
    country.to_csv(P.tables / "country_summary.csv")
    return {
        "geo_hottest": f"{hot.iloc[0]['location_name']} ({hot.iloc[0]['temperature']:.1f} °C)",
        "geo_coldest": f"{cold.iloc[0]['location_name']} ({cold.iloc[0]['temperature']:.1f} °C)",
        "geo_most_variable_continent": str(cont["temp_std"].idxmax()),
        "geo_highest_pm25_continent": str(cont["pm25_median"].idxmax()),
    }
