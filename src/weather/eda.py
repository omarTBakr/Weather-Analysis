"""Stage `eda`: distributions, temperature & precipitation trends, correlations, seasonal decomposition."""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from weather.config import CLEAN_PATH, DAILY_PATH, FIGURES, GLOBAL_SERIES_PATH, METRICS, TABLES, ensure_dirs, setup_mlflow
from weather.plotting import (
    CONTINENT_COLORS, CONTINENT_ORDER, DIV_CMAP, INK_2, MUTED, SEQ_CMAP, SERIES, apply_style, save,
)

OUT = FIGURES / "eda"
CORR_COLS = {
    "temperature_celsius": "Temperature", "feels_like_celsius": "Feels like", "humidity": "Humidity",
    "pressure_mb": "Pressure", "wind_kph": "Wind", "gust_kph": "Gust", "cloud": "Cloud",
    "precip_mm": "Precip.", "uv_index": "UV index", "visibility_km": "Visibility",
    "aq_pm25": "PM2.5", "aq_pm10": "PM10", "aq_o3": "O3", "aq_no2": "NO2", "aq_co": "CO", "aq_so2": "SO2",
}


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
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK_2)
        ax.plot([], [], color=colors[name])


def temperature_figures(daily: pd.DataFrame, gs: pd.DataFrame) -> None:
    # 1. distribution of daily temperature by continent
    fig, ax = plt.subplots(figsize=(8, 3.6))
    data = [daily.loc[daily["continent"] == c, "temperature_celsius"] for c in CONTINENT_ORDER]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body, c in zip(parts["bodies"], CONTINENT_ORDER):
        body.set_facecolor(CONTINENT_COLORS[c]); body.set_edgecolor("none"); body.set_alpha(0.75)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(INK_2); parts[k].set_linewidth(0.8)
    ax.set_xticks(range(1, 7), CONTINENT_ORDER)
    ax.set_ylabel("Daily temperature (°C)")
    ax.set_title("Daily temperature distribution by continent")
    save(fig, OUT / "temperature_distribution_by_continent.png")

    # 2. global mean temperature trend
    s = gs.set_index("date")["temperature_celsius"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(s.index, s, color=SERIES[0], alpha=0.35, linewidth=1, label="Daily mean")
    ax.plot(s.index, s.rolling(30, center=True).mean(), color=SERIES[0], linewidth=2, label="30-day rolling mean")
    ax.set_ylabel("°C"); ax.legend(loc="upper left", ncol=2)
    ax.set_title(f"Global mean temperature ({int(gs['n_locations'].median())} well-covered capitals)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save(fig, OUT / "global_temperature_trend.png")

    # 3. continent monthly mean temperature
    m = (daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
         .groupby(["continent", "month"])["temperature_celsius"].mean().unstack(0))
    m = m.iloc[1:-1]  # drop partial first/last months
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in CONTINENT_ORDER:
        ax.plot(m.index, m[c], color=CONTINENT_COLORS[c], label=c, marker="o", markersize=3)
    ax.set_xlim(m.index.min(), m.index.max() + pd.Timedelta(days=75))
    _end_labels(ax, {c: m[c] for c in CONTINENT_ORDER}, CONTINENT_COLORS)
    ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.12))
    ax.set_ylabel("Monthly mean temperature (°C)")
    ax.set_title("Monthly mean temperature by continent")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    save(fig, OUT / "monthly_temperature_by_continent.png")

    # 4. harmonic-regression decomposition (STL is degenerate with only ~2.4 annual cycles)
    dec = harmonic_decomposition(gs)
    fig, axes = plt.subplots(4, 1, figsize=(9, 6.6), sharex=True)
    axes[0].plot(dec.index, dec["observed"], color=SERIES[0], linewidth=1.1)
    axes[0].plot(dec.index, dec["fitted"], color=INK_2, linewidth=0.9, linestyle="--", label="model fit")
    axes[0].legend(loc="upper right")
    axes[1].plot(dec.index, dec["trend_raw"], color=SERIES[1], label=f"naive trend ({dec.attrs['slope_raw']:+.2f} °C/yr)")
    axes[1].plot(dec.index, dec["trend_adj"], color=SERIES[0],
                 label=f"observation-hour adjusted ({dec.attrs['slope_adj']:+.2f} °C/yr)")
    axes[1].legend(loc="lower left")
    axes[2].plot(dec.index, dec["seasonal"], color=SERIES[0], linewidth=1.2)
    axes[3].plot(dec.index, dec["resid"], color=MUTED, linewidth=0.9)
    for ax, name in zip(axes, ["Observed", "Trend", "Annual cycle", "Residual"]):
        ax.set_ylabel(name, fontsize=8.5)
    axes[0].set_title("Decomposition of global mean temperature (trend + annual harmonics + obs. hour)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save(fig, OUT / "global_temperature_decomposition.png")

    # 5. observation-time drift (the confounder behind the naive trend)
    h = daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp()).groupby("month")["local_hour"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(h.quantile(0.25).index, h.quantile(0.25), h.quantile(0.75), color=SERIES[0], alpha=0.2,
                    linewidth=0, label="interquartile range")
    ax.plot(h.median().index, h.median(), color=SERIES[0], marker="o", markersize=3, label="median")
    ax.set_ylabel("Local hour of snapshot"); ax.set_ylim(0, 24); ax.set_yticks([0, 6, 12, 18, 24])
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Local clock time of the daily snapshot drifts from afternoon to early morning")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save(fig, OUT / "observation_hour_drift.png")
    return {"trend_raw_c_per_year": dec.attrs["slope_raw"], "trend_adjusted_c_per_year": dec.attrs["slope_adj"],
            "annual_amplitude_c": float(dec["seasonal"].max() - dec["seasonal"].min()),
            "decomposition_resid_std_c": float(dec["resid"].std()),
            "median_local_hour_first_month": float(h.median().iloc[0]),
            "median_local_hour_last_month": float(h.median().iloc[-1])}


def harmonic_decomposition(gs: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    """OLS: y = a + b*t + sum_k annual harmonics [+ mean observation-hour harmonics].

    Returns observed / fitted / trend / seasonal / residual; `attrs` hold naive vs hour-adjusted slopes (degC/yr).
    """
    g = gs.set_index("date").interpolate()
    t = (g.index - g.index[0]).days.to_numpy() / 365.25
    X = pd.DataFrame({"t": t}, index=g.index)
    doy = g.index.day_of_year.to_numpy()
    for k in range(1, order + 1):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    seas_cols = [c for c in X if c != "t"]
    y = g["temperature_celsius"]
    raw = sm.OLS(y, sm.add_constant(X)).fit()
    Xa = X.assign(hour_sin=g["hour_sin"], hour_cos=g["hour_cos"])
    adj = sm.OLS(y, sm.add_constant(Xa)).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
    out = pd.DataFrame(index=g.index)
    out["observed"] = y
    out["fitted"] = adj.fittedvalues
    out["seasonal"] = X[seas_cols] @ adj.params[seas_cols]
    # trend lines drawn at the series level: intercept + slope*t + mean of the other fitted components
    hour_cols = ["hour_sin", "hour_cos"]
    out["trend_adj"] = (adj.params["const"] + adj.params["t"] * t
                        + (Xa[seas_cols + hour_cols] @ adj.params[seas_cols + hour_cols]).mean())
    out["trend_raw"] = raw.params["const"] + raw.params["t"] * t + (X[seas_cols] @ raw.params[seas_cols]).mean()
    out["resid"] = adj.resid
    out.attrs.update(slope_raw=float(raw.params["t"]), slope_adj=float(adj.params["t"]),
                     slope_adj_ci=[float(v) for v in adj.conf_int().loc["t"]])
    return out


def precipitation_figures(daily: pd.DataFrame) -> dict:
    # 5. precipitation distribution: share of dry days + wet-day intensity
    wet = daily.loc[daily["precip_mm"] > 0, "precip_mm"]
    dry_share = float((daily["precip_mm"] == 0).mean() * 100)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.logspace(np.log10(max(wet.min(), 0.01)), np.log10(wet.max()), 40)
    ax.hist(wet, bins=bins, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.6)
    ax.set_xscale("log"); ax.set_xlabel("Precipitation on wet observations (mm, log scale)"); ax.set_ylabel("Count")
    ax.set_title(f"Precipitation intensity — {dry_share:.0f}% of observations record no rain")
    save(fig, OUT / "precipitation_distribution.png")

    # 6. heatmap: mean precipitation by continent x calendar month
    h = (daily.assign(month=daily["date"].dt.month)
         .groupby(["continent", "month"])["precip_mm"].mean().unstack().reindex(CONTINENT_ORDER))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(h.values, aspect="auto", cmap=SEQ_CMAP)
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(h)), h.index); ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > np.nanmax(h.values) * 0.55 else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="mm per observation", shrink=0.9)
    ax.set_title("Mean precipitation by continent and calendar month")
    save(fig, OUT / "precipitation_heatmap_continent_month.png")

    # 7. wettest capitals
    top = daily.groupby(["location_name", "country"])["precip_mm"].mean().nlargest(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([f"{l}, {c}" for l, c in top.index], top.values, color=SERIES[0], height=0.65)
    ax.set_xlabel("Mean precipitation (mm per observation)"); ax.grid(axis="y", visible=False)
    ax.set_title("15 wettest capitals")
    save(fig, OUT / "wettest_capitals.png")
    return {"dry_observation_share_pct": dry_share, "mean_precip_mm": float(daily["precip_mm"].mean())}


def correlation_figure(daily: pd.DataFrame) -> pd.DataFrame:
    corr = daily[list(CORR_COLS)].rename(columns=CORR_COLS).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    im = ax.imshow(np.where(mask, np.nan, corr.values), cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns); ax.grid(False)
    for i in range(len(corr)):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if abs(v) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Correlation between weather and air-quality variables")
    save(fig, OUT / "correlation_matrix.png")
    return corr


def condition_figure(clean: pd.DataFrame) -> None:
    top = clean["condition_text"].value_counts().head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top.values / len(clean) * 100, color=SERIES[0], height=0.65)
    ax.set_xlabel("Share of observations (%)"); ax.grid(axis="y", visible=False)
    ax.set_title("Most frequent reported weather conditions")
    save(fig, OUT / "weather_conditions.png")


def cleaning_figure() -> None:
    rep = json.loads((METRICS / "clean_report.json").read_text())
    inv = pd.Series(rep["invalid_set_to_nan"]); cap = pd.Series(rep["outliers_capped"])
    df = pd.DataFrame({"Physically invalid → imputed": inv, "Outlier → capped": cap}).fillna(0)
    df = df[(df.sum(axis=1)) > 0].sort_values("Outlier → capped")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df.iloc[:, 0], height=0.34, color=SERIES[1], label=df.columns[0])
    ax.barh(y + 0.18, df.iloc[:, 1], height=0.34, color=SERIES[0], label=df.columns[1])
    ax.set_yticks(y, df.index); ax.set_xscale("symlog"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rows affected (symlog scale)"); ax.legend(loc="lower right")
    ax.set_title("Data-quality corrections per variable")
    save(fig, OUT / "cleaning_corrections.png")


def main() -> None:
    ensure_dirs(); apply_style(); OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_parquet(CLEAN_PATH)
    daily = pd.read_parquet(DAILY_PATH)
    gs = pd.read_parquet(GLOBAL_SERIES_PATH)

    summary = daily[list(CORR_COLS)].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
    summary.index = [CORR_COLS[c] for c in summary.index]
    summary.to_csv(TABLES / "summary_statistics.csv")
    summary.to_latex(TABLES / "summary_statistics.tex", float_format="%.2f",
                     caption="Summary statistics of the cleaned daily panel.", label="tab:summary")

    metrics = temperature_figures(daily, gs)
    metrics |= precipitation_figures(daily)
    corr = correlation_figure(daily)
    condition_figure(clean)
    cleaning_figure()
    metrics["corr_temp_humidity"] = float(corr.loc["Temperature", "Humidity"])
    metrics["corr_temp_uv"] = float(corr.loc["Temperature", "UV index"])
    (METRICS / "eda.json").write_text(json.dumps(metrics, indent=2))

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="eda"):
        mlflow.set_tag("stage", "eda")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(OUT), artifact_path="figures")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
