"""Spatial patterns: per-city summaries, Moran's I autocorrelation and interactive maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import to_hex

from weather.enums import Palette, SpatialInsights
from weather.insights.common import InsightAnalysis
from weather.utils.plotting import legend_below, save


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

    def morans_statistic(zz):
        """Moran's I for centred values `zz` under the weight matrix `W`."""
        return len(zz) / W.sum() * (zz @ W @ zz) / (zz @ zz)

    obs = morans_statistic(z)
    rng = np.random.default_rng(seed)
    sims = np.array([morans_statistic(rng.permutation(z)) for _ in range(perms)])
    p = (np.sum(sims >= obs) + 1) / (perms + 1)
    return float(obs), float(p)


def location_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per city: location attributes plus mean weather and median PM2.5."""
    return (
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


class SpatialAnalysis(InsightAnalysis):
    """Per-city climate summary, spatial autocorrelation (Moran's I), static and interactive maps."""

    def compute_morans_i(self) -> None:
        """Moran's I and its permutation p-value for every `SpatialInsights.MORAN_VARIABLES` entry (also written as CSV)."""
        loc = self.locations
        self.moran = {
            v: morans_i(loc[v].to_numpy(), loc["latitude"].to_numpy(), loc["longitude"].to_numpy())
            for v in SpatialInsights.MORAN_VARIABLES
        }
        pd.DataFrame(self.moran, index=["Moran's I", "p-value"]).T.to_csv(self.paths.tables / "morans_i.csv")

    def plot_static_maps(self) -> None:
        """Lon/lat scatter maps (for the report) of four per-city variables, titled with their Moran's I."""
        loc = self.locations
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
        for ax, (v, title, cmap, unit) in zip(axes.flat, SpatialInsights.STATIC_MAPS, strict=True):
            vmax = loc[v].quantile(0.97) if v in ("precip", "pm25") else None
            sc = ax.scatter(
                loc["longitude"],
                loc["latitude"],
                c=loc[v],
                cmap=cmap,
                s=26,
                edgecolor=Palette.INK_2,
                linewidth=0.3,
                vmax=vmax,
            )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 75)
            ax.set_aspect("equal")
            ax.axhline(0, color=Palette.MUTED, linewidth=0.6, linestyle=":")
            ax.set_title(f"{title}  (Moran's I = {self.moran[v][0]:.2f}, p = {self.moran[v][1]:.3f})", fontsize=9.5)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(sc, ax=ax, shrink=0.8, label=unit)
        fig.suptitle("Spatial patterns across 197 capitals", x=0.01, ha="left", fontweight="bold")
        save(fig, self.figure_dir / "spatial_maps.png")

    def plot_latitude_profile(self) -> None:
        """Mean temperature against latitude with a LOWESS fit; the two hottest and coldest cities labelled."""
        loc = self.locations
        low = sm.nonparametric.lowess(loc["temperature"], loc["latitude"], frac=0.3)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.scatter(loc["latitude"], loc["temperature"], s=16, color=Palette.SERIES[0], alpha=0.7, linewidths=0)
        ax.plot(low[:, 0], low[:, 1], color=Palette.SERIES[1], linewidth=2, label="LOWESS fit")
        for _, r in pd.concat([loc.nlargest(2, "temperature"), loc.nsmallest(2, "temperature")]).iterrows():
            ax.annotate(
                r["location_name"],
                (r["latitude"], r["temperature"]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=Palette.INK_2,
            )
        ax.set_xlabel("Latitude (°)")
        ax.set_ylabel("Mean temperature (°C)")
        legend_below(fig, ax, ncol=1)
        ax.set_title("Temperature falls away from the equator — faster in the north")
        save(fig, self.figure_dir / "spatial_latitude_profile.png")

    def write_plotly_maps(self) -> None:
        """Interactive plotly maps (HTML), plus static PNG/PDF exports when kaleido can render them."""
        import plotly.express as px

        loc = self.locations
        for v, title, scale in SpatialInsights.INTERACTIVE_MAPS:
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
            figp.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": Palette.INK_2}})
            figp.write_html(self.paths.maps / f"map_{v}.html", include_plotlyjs="cdn")
            try:
                figp.write_image(self.figure_dir / f"geo_map_{v}.png", width=1100, height=560, scale=5)
                figp.write_image(self.figure_dir / f"geo_map_{v}.pdf", width=1100, height=560)
            except Exception as exc:  # noqa: BLE001 - static export needs Chrome via kaleido; HTML still produced
                print(f"plotly static export skipped ({exc})")

    def write_folium_map(self) -> None:
        """Folium map of the capitals coloured by mean temperature, with a weather summary popup per city."""
        import folium

        loc = self.locations
        fmap = folium.Map(location=[20, 10], zoom_start=2, tiles="OpenStreetMap")
        tmin, tmax = loc["temperature"].min(), loc["temperature"].max()
        for _, r in loc.iterrows():
            color = to_hex(Palette.DIV_CMAP((r["temperature"] - tmin) / (tmax - tmin)))
            folium.CircleMarker(
                [r["latitude"], r["longitude"]],
                radius=6,
                color=Palette.INK_2,
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
        fmap.save(str(self.paths.maps / "capitals_weather_map.html"))

    def run(self, daily: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """Run the analysis. Returns `(metrics, locations)`; `locations` is reused by `GeographicAnalysis`."""
        self.locations = location_summary(daily)
        self.locations.to_csv(self.paths.tables / "location_summary.csv", index=False)
        self.compute_morans_i()
        self.plot_static_maps()
        self.plot_latitude_profile()
        self.write_plotly_maps()
        self.write_folium_map()
        moran = self.moran
        metrics = {f"moran_{k}": v[0] for k, v in moran.items()} | {f"moran_{k}_p": v[1] for k, v in moran.items()}
        return metrics, self.locations
