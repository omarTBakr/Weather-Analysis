"""Stage `analyses`: climate patterns, air quality vs weather, feature importance, spatial and geographic patterns."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import to_hex
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from weather.config import (
    DAILY_PATH,
    FEATURES_PATH,
    FIGURES,
    MAPS,
    METRICS,
    MODELS,
    TABLES,
    ensure_dirs,
    load_params,
    setup_mlflow,
)
from weather.plotting import (
    CONTINENT_COLORS,
    CONTINENT_ORDER,
    DIV_CMAP,
    INK_2,
    MUTED,
    SEQ_CMAP,
    SERIES,
    apply_style,
    save,
)

OUT = FIGURES / "analyses"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
POLLUTANTS = {"aq_pm25": "PM2.5", "aq_pm10": "PM10", "aq_o3": "O3", "aq_no2": "NO2", "aq_so2": "SO2", "aq_co": "CO"}
WEATHER = {
    "temperature_celsius": "Temperature",
    "humidity": "Humidity",
    "wind_kph": "Wind",
    "pressure_mb": "Pressure",
    "precip_mm": "Precip.",
    "cloud": "Cloud",
    "uv_index": "UV index",
    "visibility_km": "Visibility",
}


# ------------------------------------------------------------------------------------------------ climate
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
    tr.to_csv(TABLES / "location_climate_trends.csv", index=False)

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
    ax.legend(loc="lower right", fontsize=8)
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
    for _, r in tr.nlargest(4, "annual_amplitude").iterrows():
        ax.annotate(
            r["location_name"],
            (abs(r["latitude"]), r["annual_amplitude"]),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_xlabel("|Latitude| (°)")
    ax.set_ylabel("Annual temperature range (°C)")
    ax.legend(loc="upper left", ncol=2, fontsize=7.5)
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


# ------------------------------------------------------------------------------------------- air quality
def air_quality_analysis(daily: pd.DataFrame, params: dict) -> dict:
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
    fig.savefig(OUT / "airquality_weather_correlation.png", bbox_inches="tight")
    plt.close(fig)

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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=6, fontsize=7.5)
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

    within.round(3).to_csv(TABLES / "airquality_within_city_corr.csv")
    between.round(3).to_csv(TABLES / "airquality_pooled_corr.csv")

    # which weather variables drive PM2.5? (LightGBM + permutation importance, time split)
    import lightgbm as lgb

    feats = list(WEATHER) + ["latitude", "longitude", "month"]
    dd = daily.dropna(subset=["aq_pm25"]).assign(month=lambda x: x["date"].dt.month)
    cut = dd["date"].max() - pd.Timedelta(days=params["split"]["test_days"])
    trn, tst = dd[dd["date"] <= cut], dd[dd["date"] > cut]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=params["seed"])
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


# ------------------------------------------------------------------------------------ feature importance
def feature_importance(params: dict) -> dict:
    import shap

    ap = params["analyses"]
    feat = pd.read_parquet(FEATURES_PATH)
    cols = json.loads((MODELS / "feature_columns.json").read_text())
    model = joblib.load(MODELS / "lightgbm.joblib")
    te = feat[feat["split"] == "test"]
    X = te[cols].astype("float64")
    y = te["target"]

    gain = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    gain = gain / gain.sum()
    Xs = X.sample(min(ap["permutation_sample"], len(X)), random_state=0)
    ys = y.loc[Xs.index]
    perm = permutation_importance(
        model, Xs, ys, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=0, n_jobs=-1
    )
    perm = pd.Series(perm.importances_mean, index=cols)
    Xshap = X.sample(min(ap["shap_sample"], len(X)), random_state=1)
    sv = shap.TreeExplainer(model).shap_values(Xshap)
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=cols)
    tr = feat[feat["split"] == "train"].sample(30000, random_state=2)
    mi = pd.Series(mutual_info_regression(tr[cols].astype("float64"), tr["target"], random_state=0), index=cols)

    table = pd.DataFrame(
        {"LightGBM gain": gain, "Permutation (ΔRMSE)": perm, "Mean |SHAP|": shap_imp, "Mutual info.": mi}
    )
    ranks = table.rank(ascending=False)
    table["Mean rank"] = ranks.mean(axis=1)
    table = table.sort_values("Mean rank")
    table.to_csv(TABLES / "feature_importance.csv")
    table.head(15).to_latex(
        TABLES / "feature_importance.tex",
        float_format="%.3f",
        caption="Top-15 features for next-day temperature under four importance techniques " "(sorted by mean rank).",
        label="tab:feature_importance",
        escape=True,
    )

    top = table.head(15).index[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.6), sharey=True)
    for ax, col in zip(axes, ["LightGBM gain", "Permutation (ΔRMSE)", "Mean |SHAP|", "Mutual info."], strict=True):
        v = table.loc[top, col]
        ax.barh(top, v / v.max(), color=SERIES[0], height=0.65)
        ax.set_title(col, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("relative to max", fontsize=8)
    fig.suptitle(
        "Feature importance for next-day temperature — four techniques agree on the leaders",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    save(fig, OUT / "feature_importance_comparison.png")

    # zoom without the dominant persistence features to reveal secondary drivers
    sub = table.drop(index=[c for c in table.index if c.startswith("temp_t") or c.startswith("temp_rollmean")]).head(15)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    v = sub["Mean |SHAP|"].iloc[::-1]
    ax.barh(v.index, v.values, color=SERIES[0], height=0.65)
    ax.set_xlabel("Mean |SHAP| (°C)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Secondary drivers (excluding current temperature & rolling means)")
    save(fig, OUT / "feature_importance_secondary.png")

    shap.summary_plot(sv, Xshap, max_display=15, show=False, plot_size=(8, 5.5))
    plt.gcf().suptitle("SHAP summary — LightGBM next-day temperature model", x=0.01, ha="left", fontweight="bold")
    plt.savefig(OUT / "feature_importance_shap_summary.png", bbox_inches="tight", dpi=160)
    plt.close("all")

    for f in ["hour_cos", "pressure_diff1"]:
        if f in cols:
            shap.dependence_plot(f, sv, Xshap, interaction_index=None, show=False, color=SERIES[0])
            plt.title(f"SHAP dependence — {f}", loc="left", fontweight="bold", fontsize=11)
            plt.savefig(OUT / f"feature_importance_shap_dependence_{f}.png", bbox_inches="tight", dpi=160)
            plt.close("all")
    return {
        "fi_top_feature": table.index[0],
        "fi_top5": ", ".join(table.index[:5]),
        "fi_rank_corr_gain_shap": float(table["LightGBM gain"].corr(table["Mean |SHAP|"], method="spearman")),
        "fi_rank_corr_perm_shap": float(table["Permutation (ΔRMSE)"].corr(table["Mean |SHAP|"], method="spearman")),
    }


# --------------------------------------------------------------------------------------------- spatial
def morans_i(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, k: int = 8, perms: int = 999, seed: int = 0):
    """Moran's I with row-standardised k-nearest-neighbour weights on great-circle distance + permutation p-value."""
    la, lo = np.radians(lat), np.radians(lon)
    d = np.arccos(
        np.clip(np.sin(la[:, None]) * np.sin(la) + np.cos(la[:, None]) * np.cos(la) * np.cos(lo[:, None] - lo), -1, 1)
    )
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    nn = np.argsort(d, axis=1)[:, :k]
    W[np.repeat(np.arange(len(d)), k), nn.ravel()] = 1.0 / k
    z = values - values.mean()

    def stat(zz):
        return len(zz) / W.sum() * (zz @ W @ zz) / (zz @ zz)

    obs = stat(z)
    rng = np.random.default_rng(seed)
    sims = np.array([stat(rng.permutation(z)) for _ in range(perms)])
    p = (np.sum(sims >= obs) + 1) / (perms + 1)
    return float(obs), float(p)


def spatial_analysis(daily: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    loc = (
        daily.groupby("location_name")
        .agg(
            country=("country", "first"),
            continent=("continent", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            temperature=("temperature_celsius", "mean"),
            precip=("precip_mm", "mean"),
            humidity=("humidity", "mean"),
            wind=("wind_kph", "mean"),
            pm25=("aq_pm25", "median"),
            uv=("uv_index", "mean"),
        )
        .reset_index()
    )
    loc.to_csv(TABLES / "location_summary.csv", index=False)

    moran = {
        v: morans_i(loc[v].to_numpy(), loc["latitude"].to_numpy(), loc["longitude"].to_numpy())
        for v in ["temperature", "precip", "humidity", "pm25"]
    }
    pd.DataFrame(moran, index=["Moran's I", "p-value"]).T.to_csv(TABLES / "morans_i.csv")

    # static map-like scatter (lon/lat) for the report
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    for ax, (v, title, cmap, unit) in zip(
        axes.flat,
        [
            ("temperature", "Mean temperature", DIV_CMAP, "°C"),
            ("precip", "Mean precipitation", SEQ_CMAP, "mm/obs"),
            ("humidity", "Mean humidity", SEQ_CMAP, "%"),
            ("pm25", "Median PM2.5", SEQ_CMAP, "µg/m³"),
        ],
        strict=True,
    ):
        vmax = loc[v].quantile(0.97) if v in ("precip", "pm25") else None
        sc = ax.scatter(
            loc["longitude"], loc["latitude"], c=loc[v], cmap=cmap, s=26, edgecolor="#52514e", linewidth=0.3, vmax=vmax
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_title(f"{title}  (Moran's I = {moran[v][0]:.2f}, p = {moran[v][1]:.3f})", fontsize=9.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=unit)
    fig.suptitle("Spatial patterns across 197 capitals", x=0.01, ha="left", fontweight="bold")
    save(fig, OUT / "spatial_maps.png")

    # latitude profile with LOWESS
    low = sm.nonparametric.lowess(loc["temperature"], loc["latitude"], frac=0.3)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(loc["latitude"], loc["temperature"], s=16, color=SERIES[0], alpha=0.7, linewidths=0)
    ax.plot(low[:, 0], low[:, 1], color=SERIES[1], linewidth=2, label="LOWESS fit")
    for _, r in pd.concat([loc.nlargest(2, "temperature"), loc.nsmallest(2, "temperature")]).iterrows():
        ax.annotate(
            r["location_name"],
            (r["latitude"], r["temperature"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Mean temperature (°C)")
    ax.legend(loc="lower center")
    ax.set_title("Temperature falls away from the equator — faster in the north")
    save(fig, OUT / "spatial_latitude_profile.png")

    # interactive maps (plotly + folium)
    import folium
    import plotly.express as px

    for v, title, scale in [
        ("temperature", "Mean temperature (°C)", "RdBu_r"),
        ("pm25", "Median PM2.5 (µg/m³)", "YlOrRd"),
        ("precip", "Mean precipitation (mm/obs)", "Blues"),
    ]:
        figp = px.scatter_geo(
            loc,
            lat="latitude",
            lon="longitude",
            color=v,
            hover_name="location_name",
            hover_data={"country": True, v: ":.2f"},
            color_continuous_scale=scale,
            projection="natural earth",
            title=title,
        )
        figp.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "#52514e"}})
        figp.write_html(MAPS / f"map_{v}.html", include_plotlyjs="cdn")
        try:
            figp.write_image(OUT / f"geo_map_{v}.png", width=1100, height=560, scale=1.5)
        except Exception as exc:  # noqa: BLE001 - static export needs Chrome via kaleido; HTML still produced
            print(f"plotly static export skipped ({exc})")

    fmap = folium.Map(location=[20, 10], zoom_start=2, tiles="OpenStreetMap")
    tmin, tmax = loc["temperature"].min(), loc["temperature"].max()
    for _, r in loc.iterrows():
        color = to_hex(DIV_CMAP((r["temperature"] - tmin) / (tmax - tmin)))
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=6,
            color="#52514e",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{r['location_name']}</b>, {r['country']}<br>Mean temp: {r['temperature']:.1f} °C<br>"
                f"Precip: {r['precip']:.2f} mm/obs<br>Humidity: {r['humidity']:.0f}%<br>"
                f"Median PM2.5: {r['pm25']:.1f} µg/m³",
                max_width=250,
            ),
        ).add_to(fmap)
    fmap.save(str(MAPS / "capitals_weather_map.html"))
    return {f"moran_{k}": v[0] for k, v in moran.items()} | {f"moran_{k}_p": v[1] for k, v in moran.items()}, loc


# ------------------------------------------------------------------------------------------ geographic
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
    cont.round(2).to_csv(TABLES / "continent_summary.csv")
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
        TABLES / "continent_summary.tex",
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
    country.to_csv(TABLES / "country_summary.csv")
    return {
        "geo_hottest": f"{hot.iloc[0]['location_name']} ({hot.iloc[0]['temperature']:.1f} °C)",
        "geo_coldest": f"{cold.iloc[0]['location_name']} ({cold.iloc[0]['temperature']:.1f} °C)",
        "geo_most_variable_continent": str(cont["temp_std"].idxmax()),
        "geo_highest_pm25_continent": str(cont["pm25_median"].idxmax()),
    }


def main() -> None:
    ensure_dirs()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params()
    daily = pd.read_parquet(DAILY_PATH)
    results = {}
    results |= climate_analysis(daily)
    results |= air_quality_analysis(daily, params)
    results |= feature_importance(params)
    spatial, loc = spatial_analysis(daily)
    results |= spatial
    results |= geographic_patterns(daily, loc)
    (METRICS / "analyses.json").write_text(json.dumps(results, indent=2))

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="analyses"):
        mlflow.set_tag("stage", "analyses")
        mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, float)})
        mlflow.set_tags({k: v for k, v in results.items() if isinstance(v, str)})
        mlflow.log_artifacts(str(OUT), artifact_path="figures")
        mlflow.log_artifacts(str(MAPS), artifact_path="maps")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
