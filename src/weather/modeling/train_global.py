"""Stage `train_global`: multi-step forecasting of the global mean temperature series.

Every model is fitted on train+val and forecasts the whole 60-day test window in one shot (true multi-step),
then the best approach is refitted on all data to project the next `future_days` days.

Run with `python -m weather.modeling.train_global` (the DVC `train_global` stage).
"""

from __future__ import annotations

import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather.enums import GlobalForecasting, GlobalModel, Palette
from weather.modeling.global_forecasters import Forecast, GlobalForecaster, build_global_forecasters
from weather.utils.config import Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.metrics import regression_metrics
from weather.utils.mlflow_utils import log_metrics_prefixed
from weather.utils.plotting import apply_style, legend_below, save

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")


class GlobalTrainer:
    """Back-tests every global forecaster on the test window, then projects the future with the best one."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.series_settings = self.settings.global_series
        self.test_days = self.settings.split.test_days
        self.figure_dir = self.paths.figures / "forecast_global"
        self.forecasters = build_global_forecasters(self.series_settings.fourier_order)
        self.predictions: dict[str, np.ndarray] = {}
        self.results: dict[str, dict] = {}
        self.intervals: dict[str, np.ndarray] = {}

    # data -------------------------------------------------------------------------------------------------
    def load_series(self) -> None:
        """Read the global series on a gap-free daily index and hold out the last `test_days` days."""
        self.series = pd.read_parquet(self.paths.global_series_path).set_index("date").asfreq("D").interpolate()
        self.history, self.test = self.series.iloc[: -self.test_days], self.series.iloc[-self.test_days :]

    # evaluation -------------------------------------------------------------------------------------------
    def evaluate_forecaster(self, mlflow, forecaster: GlobalForecaster) -> None:
        """Nested run: forecast the test window, log test metrics and (if available) 95% interval coverage."""
        name = forecaster.name
        with mlflow.start_run(run_name=f"global_{name}", nested=True):
            mlflow.set_tags({"model": name, "framing": "global_series"})
            fc = forecaster.forecast(self.history, self.test.index)
            metrics = regression_metrics(self.test[GlobalForecasting.TARGET], fc.mean)
            log_metrics_prefixed(mlflow, metrics, "test")
            self.predictions[name], self.results[name] = fc.mean, metrics
            if fc.interval is not None:
                self.intervals[name] = fc.interval
                coverage = self._interval_coverage(fc.interval)
                mlflow.log_metric("test_pi95_coverage", coverage)
                self.results[name]["pi95_coverage"] = coverage
            print(f"{name:20s} MAE {metrics['mae']:.3f}  RMSE {metrics['rmse']:.3f}")

    def _interval_coverage(self, interval: np.ndarray) -> float:
        """Share of test days whose observed value falls inside the interval."""
        y = self.test[GlobalForecasting.TARGET].to_numpy()
        return float(np.mean((y >= interval[:, 0]) & (y <= interval[:, 1])))

    def evaluate_ensemble(self, mlflow) -> None:
        """Simple average of the structural models' test forecasts."""
        name = GlobalModel.ENSEMBLE_MEAN
        self.predictions[name] = np.mean([self.predictions[k] for k in GlobalForecasting.ENSEMBLE_MEMBERS], axis=0)
        self.results[name] = regression_metrics(self.test[GlobalForecasting.TARGET], self.predictions[name])
        log_metrics_prefixed(mlflow, self.results[name], "ensemble_test")
        print(f"{name:20s} MAE {self.results[name]['mae']:.3f}  RMSE {self.results[name]['rmse']:.3f}")

    # future projection ------------------------------------------------------------------------------------
    def project_future(self, mlflow) -> None:
        """Refit the lowest-RMSE interval-producing model on ALL data and forecast `future_days` ahead."""
        self.best_model = min(self.intervals, key=lambda k: self.results[k]["rmse"])
        self.future_index = pd.date_range(
            self.series.index[-1] + pd.Timedelta(days=1), periods=self.series_settings.future_days, freq="D"
        )
        self.future: Forecast = self.forecasters[self.best_model].forecast(self.series, self.future_index)
        pd.DataFrame(
            {
                "date": self.future_index,
                "forecast": self.future.mean,
                "lower95": self.future.interval[:, 0],
                "upper95": self.future.interval[:, 1],
            }
        ).to_csv(self.paths.tables / "global_future_forecast.csv", index=False)
        mlflow.set_tag("future_model", self.best_model)

    # figures ----------------------------------------------------------------------------------------------
    def plot_backtest(self) -> None:
        """The last `BACKTEST_CONTEXT_DAYS` observed days with every plotted model's test-window forecast overlaid."""
        fig, ax = plt.subplots(figsize=(9, 3.8))
        context = self.series.iloc[-GlobalForecasting.BACKTEST_CONTEXT_DAYS :]
        ax.plot(context.index, context[GlobalForecasting.TARGET], color=Palette.INK_2, linewidth=1.2, label="observed")
        for i, name in enumerate(GlobalForecasting.PLOTTED_FORECASTS):
            ax.plot(
                self.test.index,
                self.predictions[name],
                color=Palette.SERIES[i],
                linewidth=1.6,
                linestyle="--" if name == GlobalModel.SEASONAL_NAIVE else "-",
                label=f"{name.replace('_', ' ')} (RMSE {self.results[name]['rmse']:.2f})",
            )
        origin = self.test.index[0]
        ax.axvline(origin, color=Palette.MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(
            "forecast origin",
            (origin, ax.get_ylim()[0]),  # bottom of the axis: the forecasts live in the upper half
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=Palette.INK_2,
        )
        legend_below(fig, ax, ncol=3, fontsize=7.5)
        ax.set_ylabel("°C")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(f"Global mean temperature — {self.test_days}-day multi-step forecast vs. observed")
        save(fig, self.figure_dir / "global_multistep_forecast.png")

    def plot_future(self) -> None:
        """The last `FUTURE_CONTEXT_DAYS` (12 months) observed plus the future projection and its 95% prediction interval."""
        fig, ax = plt.subplots(figsize=(9, 3.6))
        context = self.series.iloc[-GlobalForecasting.FUTURE_CONTEXT_DAYS :]
        ax.plot(
            context.index,
            context[GlobalForecasting.TARGET],
            color=Palette.INK_2,
            linewidth=1.2,
            label="observed (last 12 months)",
        )
        lower, upper = self.future.interval[:, 0], self.future.interval[:, 1]
        ax.fill_between(
            self.future_index,
            lower,
            upper,
            color=Palette.SERIES[0],
            alpha=0.18,
            linewidth=0,
            label="95% prediction interval",
        )
        ax.plot(
            self.future_index,
            self.future.mean,
            color=Palette.SERIES[0],
            linewidth=2,
            label=f"forecast ({self.best_model.replace('_', ' ')})",
        )
        legend_below(fig, ax, ncol=3)
        ax.set_ylabel("°C")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(f"Projected global mean temperature, next {self.series_settings.future_days} days")
        save(fig, self.figure_dir / "global_future_forecast.png")

    # reports ----------------------------------------------------------------------------------------------
    def write_reports(self) -> None:
        """Model comparison table (CSV + LaTeX) and `train_global.json` metrics."""
        tables = self.paths.tables
        table = pd.DataFrame(self.results).T[["mae", "rmse", "smape", "r2"]]
        table.index = [i.replace("_", " ").title() for i in table.index]
        table.to_csv(tables / "global_model_comparison.csv")
        table.to_latex(
            tables / "global_model_comparison.tex",
            float_format="%.3f",
            caption=f"Global mean temperature: {self.test_days}-day multi-step forecast accuracy on the test window.",
            label="tab:global_models",
            escape=True,
        )
        (self.paths.metrics / "train_global.json").write_text(
            json.dumps({"results": self.results, "future_model": self.best_model}, indent=2)
        )

    def run(self) -> None:
        """Execute the full stage inside one `train_global` MLflow run."""
        ensure_dirs(self.settings)
        apply_style()
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.load_series()
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="train_global"):
            mlflow.set_tags({"stage": "train_global", "framing": "global_series"})
            mlflow.log_params(
                {**self.series_settings.model_dump(), "test_days": self.test_days, "history_days": len(self.history)}
            )
            for forecaster in self.forecasters.values():
                self.evaluate_forecaster(mlflow, forecaster)
            self.evaluate_ensemble(mlflow)
            self.project_future(mlflow)
            self.plot_backtest()
            self.plot_future()
            mlflow.log_artifacts(str(self.figure_dir), artifact_path="figures")
            mlflow.log_artifact(str(self.paths.tables / "global_future_forecast.csv"))
        self.write_reports()


def main() -> None:
    """DVC entry point."""
    GlobalTrainer().run()


if __name__ == "__main__":
    main()
