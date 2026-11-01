"""Climate patterns: per-city trend/seasonality fits, latitude bands and amplitude."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from weather.insights.common import MONTHS, OUT, P
from weather.utils.plotting import (
    CONTINENT_COLORS,
    CONTINENT_ORDER,
    DIV_CMAP,
    INK_2,
    MUTED,
    SERIES,
    legend_below,
    save,
)


def location_trends(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-city OLS: temp ~ trend + 2 annual harmonics (+ 2 diurnal harmonics of the snapshot hour)."""
    rows = []
    for loc, g in daily.groupby("location_name"):
        if len(g) < 500:
            continue
        t = (g["date"] - g["date"].min()).dt.days.to_numpy() / 365.25
        doy = g["date"].dt.day_of_year.to_numpy()
        hr = g["local_hour"].to_numpy()
        X = {"t": t}
        for k in (1, 2):
            X[f"s{k}"], X[f"c{k}"] = np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)
        Xr = sm.add_constant(pd.DataFrame(X))
        for k in (1, 2):
            X[f"hs{k}"], X[f"hc{k}"] = np.sin(2 * np.pi * k * hr / 24), np.cos(2 * np.pi * k * hr / 24)
        Xa = sm.add_constant(pd.DataFrame(X))
        y = g["temperature_celsius"].to_numpy()
        raw, adj = sm.OLS(y, Xr).fit(), sm.OLS(y, Xa).fit()
        amp = np.hypot(adj.params["s1"], adj.params["c1"]) * 2
        diurnal = np.ptp(
            Xa[["hs1", "hc1", "hs2", "hc2"]].to_numpy() @ adj.params[["hs1", "hc1", "hs2", "hc2"]].to_numpy()
        )
        rows.append(
            {
                "location_name": loc,
                "country": g["country"].iloc[0],
                "continent": g["continent"].iloc[0],
                "latitude": g["latitude"].iloc[0],
                "longitude": g["longitude"].iloc[0],
                "mean_temp": y.mean(),
                "slope_raw": raw.params["t"],
                "slope_adj": adj.params["t"],
                "annual_amplitude": amp,
                "diurnal_effect": diurnal,
                "r2_adj": adj.rsquared,
            }
        )
    return pd.DataFrame(rows)


def climate_analysis(daily: pd.DataFrame) -> dict:
    tr = location_trends(daily)
    tr.to_csv(P.tables / "location_climate_trends.csv", index=False)

    # trend: naive vs hour-adjusted, by continent
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(CONTINENT_ORDER))
    for off, col, color, lab in [
        (-0.15, "slope_raw", SERIES[1], "naive"),
        (0.15, "slope_adj", SERIES[0], "observation-hour adjusted"),
    ]:
        g = tr.groupby("continent")[col]
        med, q1, q3 = (
            g.median().reindex(CONTINENT_ORDER),
            g.quantile(0.25).reindex(CONTINENT_ORDER),
            g.quantile(0.75).reindex(CONTINENT_ORDER),
        )
        ax.hlines(y + off, q1, q3, color=color, linewidth=2.5, alpha=0.5)
        ax.scatter(
            med, y + off, color=color, s=40, zorder=3, edgecolor="#fcfcfb", linewidth=1.5, label=f"{lab} (median, IQR)"
        )
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, CONTINENT_ORDER)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Temperature trend across cities (°C per year)")
    legend_below(fig, ax, ncol=2)
    ax.set_title("Short-term temperature trend by continent: naive vs. hour-adjusted")
    save(fig, OUT / "climate_trend_by_continent.png")

    # latitude-band climatology
    bands = pd.cut(
        daily["latitude"],
        [-90, -35, -23.5, 0, 23.5, 35, 50, 90],
        labels=["<35°S", "35–23.5°S", "23.5°S–0", "0–23.5°N", "23.5–35°N", "35–50°N", ">50°N"],
    )
    h = (
        daily.assign(band=bands, m=daily["date"].dt.month)
        .groupby(["band", "m"], observed=True)["temperature_celsius"]
        .mean()
        .unstack()
    )
    h = h.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(h.values, aspect="auto", cmap=DIV_CMAP, vmin=-5, vmax=35)
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(h)), h.index)
    ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#ffffff" if (v > 29 or v < 2) else "#0b0b0b",
            )
    fig.colorbar(im, ax=ax, label="Mean temperature (°C)", shrink=0.9)
    ax.set_title("Seasonal cycle by latitude band — hemispheres in opposite phase")
    save(fig, OUT / "climate_latitude_band_seasonality.png")

    # continentality: annual amplitude vs |latitude|
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for c in CONTINENT_ORDER:
        g = tr[tr["continent"] == c]
        ax.scatter(
            g["latitude"].abs(),
            g["annual_amplitude"],
            s=22,
            color=CONTINENT_COLORS[c],
            label=c,
            edgecolor="#fcfcfb",
            linewidth=0.8,
            alpha=0.9,
        )
    # alternate label sides so neighbouring extremes (e.g. Bishkek / Pyongyang) don't collide
    for i, (_, r) in enumerate(tr.nlargest(4, "annual_amplitude").iterrows()):
        ax.annotate(
            r["location_name"],
            (abs(r["latitude"]), r["annual_amplitude"]),
            xytext=(6, 2) if i % 2 == 0 else (-6, 2),
            ha="left" if i % 2 == 0 else "right",
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_xlabel("|Latitude| (°)")
    ax.set_ylabel("Annual temperature range (°C)")
    legend_below(fig, ax, ncol=6)
    ax.set_title("Seasonal amplitude grows with latitude and continentality")
    save(fig, OUT / "climate_amplitude_vs_latitude.png")

    return {
        "climate_median_slope_raw": float(tr["slope_raw"].median()),
        "climate_median_slope_adj": float(tr["slope_adj"].median()),
        "climate_share_cities_cooling_raw_pct": float((tr["slope_raw"] < 0).mean() * 100),
        "climate_share_cities_cooling_adj_pct": float((tr["slope_adj"] < 0).mean() * 100),
        "climate_median_diurnal_effect_c": float(tr["diurnal_effect"].median()),
        "climate_corr_amplitude_abslat": float(np.corrcoef(tr["latitude"].abs(), tr["annual_amplitude"])[0, 1]),
    }
