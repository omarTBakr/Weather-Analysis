"""Stage `train_global`: multi-step forecasting of the global mean temperature series.

Every model is fitted on train+val and forecasts the whole 60-day test window in one shot (true multi-step),
then the best approach is refitted on all data to project the next `future_days` days.
"""

from __future__ import annotations

import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from weather.metrics import regression_metrics
from weather.plotting import INK_2, MUTED, SERIES, apply_style, save
from weather.utils.config import ensure_dirs, get_settings, setup_mlflow

S = get_settings()
P = S.paths

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")
OUT = P.figures / "forecast_global"


def design(index: pd.DatetimeIndex, t0: pd.Timestamp, order: int, hour: pd.DataFrame | None) -> pd.DataFrame:
    """Linear trend + annual Fourier terms (+ mean observation-hour harmonics)."""
    X = pd.DataFrame(index=index)
    X["t"] = (index - t0).days / 365.25
    doy = index.day_of_year
    for k in range(1, order + 1):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    if hour is not None:
        X = X.join(hour)
    return X


def future_hours(hist: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Observation hour is not known in advance -> hold the last 30-day average (snapshot schedule drifts slowly)."""
    last = hist[["hour_sin", "hour_cos"]].iloc[-30:].mean()
    return pd.DataFrame({c: last[c] for c in last.index}, index=index)


class GlobalModels:
    def __init__(self, order: int):
        self.order = order

    def forecast(self, name: str, hist: pd.DataFrame, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray | None]:
        y = hist["temperature_celsius"]
        t0 = hist.index[0]
        h = len(index)
        if name == "naive":
            return np.repeat(y.iloc[-1], h), None
        if name == "seasonal_naive":
            return y.reindex(index - pd.Timedelta(days=365)).interpolate().bfill().ffill().to_numpy(), None
        if name == "holt_winters":
            m = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=365,
                initialization_method="estimated",
            ).fit()
            return m.forecast(h).to_numpy(), None
        if name == "harmonic_regression":
            import statsmodels.api as sm

            X = sm.add_constant(design(hist.index, t0, self.order, hist[["hour_sin", "hour_cos"]]))
            m = sm.OLS(y, X).fit()
            Xf = sm.add_constant(design(index, t0, self.order, future_hours(hist, index)), has_constant="add")
            fc = m.get_prediction(Xf).summary_frame(alpha=0.05)
            return fc["mean"].to_numpy(), fc[["obs_ci_lower", "obs_ci_upper"]].to_numpy()
        if name == "sarimax_fourier":
            # regression with ARIMA(1,0,1) errors; constant lives in the exogenous design
            X = design(hist.index, t0, self.order, hist[["hour_sin", "hour_cos"]]).assign(const=1.0)
            m = SARIMAX(y, exog=X, order=(1, 0, 1), trend="n").fit(disp=False, maxiter=500)
            Xf = design(index, t0, self.order, future_hours(hist, index)).assign(const=1.0)
            fc = m.get_forecast(h, exog=Xf)
            return fc.predicted_mean.to_numpy(), fc.conf_int(alpha=0.05).to_numpy()
        raise ValueError(name)


GLOBAL_MODELS = ["naive", "seasonal_naive", "holt_winters", "harmonic_regression", "sarimax_fourier"]


def main() -> None:
    ensure_dirs()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    gp = S.global_series
    test_days = S.split.test_days
    gs = pd.read_parquet(P.global_series_path).set_index("date").asfreq("D").interpolate()
    hist, test = gs.iloc[:-test_days], gs.iloc[-test_days:]
    gm = GlobalModels(gp.fourier_order)

    preds, results, intervals = {}, {}, {}
    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="train_global"):
        mlflow.set_tags({"stage": "train_global", "framing": "global_series"})
        mlflow.log_params({**gp.model_dump(), "test_days": test_days, "history_days": len(hist)})
        for name in GLOBAL_MODELS:
            with mlflow.start_run(run_name=f"global_{name}", nested=True):
                mlflow.set_tags({"model": name, "framing": "global_series"})
                p, ci = gm.forecast(name, hist, test.index)
                m = regression_metrics(test["temperature_celsius"], p)
                mlflow.log_metrics({f"test_{k}": v for k, v in m.items() if np.isfinite(v)})
                preds[name], results[name] = p, m
                if ci is not None:
                    intervals[name] = ci
                    cover = np.mean(
                        (test["temperature_celsius"].to_numpy() >= ci[:, 0])
                        & (test["temperature_celsius"].to_numpy() <= ci[:, 1])
                    )
                    mlflow.log_metric("test_pi95_coverage", float(cover))
                    results[name]["pi95_coverage"] = float(cover)
                print(f"{name:20s} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}")

        # ensemble of the structural models (simple average)
        members = ["holt_winters", "harmonic_regression", "sarimax_fourier"]
        preds["ensemble_mean"] = np.mean([preds[k] for k in members], axis=0)
        results["ensemble_mean"] = regression_metrics(test["temperature_celsius"], preds["ensemble_mean"])
        mlflow.log_metrics({f"ensemble_test_{k}": v for k, v in results["ensemble_mean"].items() if np.isfinite(v)})
        print(
            f"{'ensemble_mean':20s} MAE {results['ensemble_mean']['mae']:.3f}  RMSE {results['ensemble_mean']['rmse']:.3f}"
        )

        # future projection: refit on all data with the best interval-producing model
        best = min(intervals, key=lambda k: results[k]["rmse"])
        fidx = pd.date_range(gs.index[-1] + pd.Timedelta(days=1), periods=gp.future_days, freq="D")
        fut, fci = gm.forecast(best, gs, fidx)
        future = pd.DataFrame({"date": fidx, "forecast": fut, "lower95": fci[:, 0], "upper95": fci[:, 1]})
        future.to_csv(P.tables / "global_future_forecast.csv", index=False)
        mlflow.set_tag("future_model", best)

        # figures --------------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ctx = gs.iloc[-200:]
        ax.plot(ctx.index, ctx["temperature_celsius"], color=INK_2, linewidth=1.2, label="observed")
        for i, name in enumerate(
            ["seasonal_naive", "holt_winters", "harmonic_regression", "sarimax_fourier", "ensemble_mean"]
        ):
            ax.plot(
                test.index,
                preds[name],
                color=SERIES[i],
                linewidth=1.6,
                linestyle="--" if name == "seasonal_naive" else "-",
                label=f"{name.replace('_', ' ')} (RMSE {results[name]['rmse']:.2f})",
            )
        ax.axvline(test.index[0], color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(
            "forecast origin",
            (test.index[0], ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5)
        ax.set_ylabel("°C")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(f"Global mean temperature — {test_days}-day multi-step forecast vs. observed")
        save(fig, OUT / "global_multistep_forecast.png")

        fig, ax = plt.subplots(figsize=(9, 3.6))
        ctx = gs.iloc[-365:]
        ax.plot(ctx.index, ctx["temperature_celsius"], color=INK_2, linewidth=1.2, label="observed (last 12 months)")
        ax.fill_between(
            fidx, fci[:, 0], fci[:, 1], color=SERIES[0], alpha=0.18, linewidth=0, label="95% prediction interval"
        )
        ax.plot(fidx, fut, color=SERIES[0], linewidth=2, label=f"forecast ({best.replace('_', ' ')})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        ax.set_ylabel("°C")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(f"Projected global mean temperature, next {gp.future_days} days")
        save(fig, OUT / "global_future_forecast.png")
        mlflow.log_artifacts(str(OUT), artifact_path="figures")
        mlflow.log_artifact(str(P.tables / "global_future_forecast.csv"))

    tbl = pd.DataFrame(results).T[["mae", "rmse", "smape", "r2"]]
    tbl.index = [i.replace("_", " ").title() for i in tbl.index]
    tbl.to_csv(P.tables / "global_model_comparison.csv")
    tbl.to_latex(
        P.tables / "global_model_comparison.tex",
        float_format="%.3f",
        caption=f"Global mean temperature: {test_days}-day multi-step forecast accuracy on the test window.",
        label="tab:global_models",
        escape=True,
    )
    (P.metrics / "train_global.json").write_text(json.dumps({"results": results, "future_model": best}, indent=2))


if __name__ == "__main__":
    main()
