"""Stage `ensemble`: combine panel models (mean, inverse-RMSE weights, stacking) and evaluate on the test window.

The combiners are fitted on the validation window only (`weather.modeling.combiners`); the stacking ensemble is
registered in MLflow as a pyfunc model (`weather.modeling.ensemble_model.WeightedEnsemble`).

Run with `python -m weather.modeling.ensemble` (the DVC `ensemble` stage).
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather.enums import Geography, Palette, PanelEnsemble, PanelEnsembling, PanelModel, Split
from weather.modeling.combiners import Combiner, build_combiners, prediction_column
from weather.utils.config import Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.metrics import regression_metrics
from weather.utils.mlflow_utils import log_metrics_prefixed
from weather.utils.plotting import annotate_heatmap, apply_style, legend_below, save


def rmse(observed: pd.Series, predicted: pd.Series) -> float:
    """Root-mean-square error (used per group, where the full `regression_metrics` would be wasteful)."""
    return np.sqrt(np.mean((observed - predicted) ** 2))


class PanelEnsembleStage:
    """Fits the ensembles on the validation window, scores every model on the test window, writes tables and
    figures, and registers the stacking ensemble as a pyfunc model."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.figure_dir = self.paths.figures / "forecast_panel"
        self.config_path = self.paths.models / "ensemble_config.json"
        self.members = PanelEnsembling.MEMBERS
        # every model and ensemble that gets scored, in table order
        self.model_names = [PanelModel.PERSISTENCE, *self.members, *PanelEnsemble]
        self.combiners: dict[PanelEnsemble, Combiner] = build_combiners(self.members)

    @property
    def inverse_rmse(self) -> Combiner:
        """The inverse-RMSE weighted combiner."""
        return self.combiners[PanelEnsemble.WEIGHTED]

    @property
    def stacking(self) -> Combiner:
        """The stacking combiner (the one that is registered and served)."""
        return self.combiners[PanelEnsemble.STACKING]

    # data -------------------------------------------------------------------------------------------------
    def load_predictions(self) -> None:
        """Read the panel models' val/test predictions and split them into the two windows."""
        preds = pd.read_parquet(self.paths.predictions_path)
        self.val = preds[preds["split"] == Split.VAL].copy()
        self.test = preds[preds["split"] == Split.TEST].copy()

    # ensembling -------------------------------------------------------------------------------------------
    def fit_combiners(self) -> None:
        """Fit every combiner on the validation window only."""
        for combiner in self.combiners.values():
            combiner.fit(self.val, self.val["target"])

    def add_ensemble_predictions(self) -> None:
        """Add a `pred_<ensemble>` column per combiner to both windows."""
        for frame in (self.val, self.test):
            for name, combiner in self.combiners.items():
                frame[prediction_column(name)] = combiner.predict(frame)

    # evaluation -------------------------------------------------------------------------------------------
    def _score(self, frame: pd.DataFrame) -> dict[str, dict]:
        """`regression_metrics` of every model / ensemble on `frame`."""
        return {n: regression_metrics(frame["target"], frame[prediction_column(n)]) for n in self.model_names}

    def evaluate(self) -> None:
        """Test (and validation) metrics, best models, skill vs persistence and test RMSE by continent."""
        self.test_metrics = self._score(self.test)
        self.val_metrics = self._score(self.val)
        rmse_of = {n: self.test_metrics[n]["rmse"] for n in self.model_names}
        self.best_single = min(self.members, key=rmse_of.get)
        self.best_overall = min(self.model_names, key=rmse_of.get)
        self.skill = {n: 1 - rmse_of[n] / rmse_of[PanelModel.PERSISTENCE] for n in self.model_names}
        by_continent = self.test.groupby("continent")
        self.continent_rmse = pd.DataFrame(
            {
                PanelEnsembling.DISPLAY_NAMES[n]: by_continent.apply(
                    lambda g, n=n: rmse(g["target"], g[prediction_column(n)]), include_groups=False
                )
                for n in self.model_names
            }
        ).T[Geography.CONTINENT_ORDER]

    # tables -----------------------------------------------------------------------------------------------
    def write_tables(self) -> None:
        """Model comparison (CSV + LaTeX) and RMSE by continent (CSV)."""
        tables = self.paths.tables
        self.comparison = pd.DataFrame(self.test_metrics).T[["mae", "rmse", "smape", "r2", "within_2c"]]
        self.comparison["skill_vs_persistence_pct"] = pd.Series(self.skill) * 100
        self.comparison["val_rmse"] = pd.Series({n: self.val_metrics[n]["rmse"] for n in self.model_names})
        self.comparison.index = [PanelEnsembling.DISPLAY_NAMES[n] for n in self.comparison.index]
        self.comparison.to_csv(tables / "model_comparison.csv")
        self.comparison.rename(columns=PanelEnsembling.LATEX_HEADERS).to_latex(
            tables / "model_comparison.tex",
            float_format="%.3f",
            escape=False,
            caption=f"Next-day temperature forecast accuracy on the {self.settings.split.test_days}-day test window "
            "(all cities). Skill = RMSE reduction relative to persistence.",
            label="tab:model_comparison",
        )
        self.continent_rmse.to_csv(tables / "rmse_by_continent.csv")

    # figures ----------------------------------------------------------------------------------------------
    def plot_rmse_comparison(self) -> None:
        """Horizontal bars of test RMSE: persistence grey, single models blue, ensembles orange."""
        rmse_of = {n: self.test_metrics[n]["rmse"] for n in self.model_names}
        order = sorted(self.model_names, key=rmse_of.get, reverse=True)
        colors = [
            (
                Palette.MUTED
                if n == PanelModel.PERSISTENCE
                else (Palette.SERIES[1] if n in PanelEnsemble else Palette.SERIES[0])
            )
            for n in order
        ]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh(
            [PanelEnsembling.DISPLAY_NAMES[n] for n in order], [rmse_of[n] for n in order], color=colors, height=0.62
        )
        for i, n in enumerate(order):
            ax.annotate(
                f"{rmse_of[n]:.3f}",
                (rmse_of[n], i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=Palette.INK_2,
            )
        ax.set_xlabel("Test RMSE (°C) — lower is better")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(rmse_of.values()) * 1.12)
        ax.set_title("Next-day temperature: single models (blue) vs. ensembles (orange)")
        save(fig, self.figure_dir / "model_comparison_rmse.png")

    def plot_continent_heatmap(self) -> None:
        """Heatmap of test RMSE per model (rows) and continent (columns)."""
        cont = self.continent_rmse
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        im = ax.imshow(cont.values, cmap=Palette.SEQ_CMAP, aspect="auto")
        ax.set_xticks(range(cont.shape[1]), cont.columns)
        ax.set_yticks(range(cont.shape[0]), cont.index)
        ax.grid(False)
        vmax = np.nanmax(cont.values)
        annotate_heatmap(ax, cont.values, ".2f", 7.5, light_text=lambda v: v > vmax * 0.7)
        fig.colorbar(im, ax=ax, label="RMSE (°C)", shrink=0.9)
        ax.set_title("Test RMSE by continent")
        save(fig, self.figure_dir / "rmse_by_continent.png")

    def plot_daily_error(self) -> None:
        """Cross-city RMSE per test day: persistence vs the best model overall."""
        persistence, best = PanelModel.PERSISTENCE, self.best_overall
        daily_err = self.test.groupby("target_date").apply(
            lambda g: pd.Series(
                {
                    persistence: rmse(g["target"], g[prediction_column(persistence)]),
                    best: rmse(g["target"], g[prediction_column(best)]),
                }
            ),
            include_groups=False,
        )
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(
            daily_err.index,
            daily_err[persistence],
            color=Palette.MUTED,
            label=PanelEnsembling.DISPLAY_NAMES[persistence],
        )
        ax.plot(daily_err.index, daily_err[best], color=Palette.SERIES[1], label=PanelEnsembling.DISPLAY_NAMES[best])
        ax.set_ylabel("Cross-city RMSE (°C)")
        legend_below(fig, ax, ncol=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_title("Daily forecast error over the test window")
        save(fig, self.figure_dir / "daily_error_test.png")

    def plot_city_forecasts(self) -> None:
        """Observed vs best-model forecast over the test window for up to four example capitals."""
        present = set(self.test["location_name"])
        cities = [c for c in PanelEnsembling.EXAMPLE_CITIES if c in present][:4]
        best_column = prediction_column(self.best_overall)
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
        for ax, city in zip(axes.flat, cities, strict=False):
            g = self.test[self.test["location_name"] == city].sort_values("target_date")
            ax.plot(g["target_date"], g["target"], color=Palette.INK_2, linewidth=1.3, label="observed")
            ax.plot(g["target_date"], g[best_column], color=Palette.SERIES[1], linewidth=1.5, label="ensemble forecast")
            ax.set_title(city, fontsize=10)
            ax.set_ylabel("°C")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        legend_below(fig, axes.flat[0], ncol=2)
        fig.suptitle("Next-day forecasts on the test window — example capitals", x=0.01, ha="left", fontweight="bold")
        save(fig, self.figure_dir / "city_forecasts.png")

    # serving ----------------------------------------------------------------------------------------------
    def write_serving_config(self) -> None:
        """`ensemble_config.json`: the stacking weights (contributing members only), intercept and feature list."""
        self.feature_columns = json.loads((self.paths.models / "feature_columns.json").read_text())
        self.serving_config = {
            "weights": self.stacking.positive_weights(),
            "intercept": self.stacking.intercept,
            "feature_columns": self.feature_columns,
        }
        self.config_path.write_text(json.dumps(self.serving_config, indent=2))

    def register_pyfunc(self, mlflow) -> None:
        """Log + register the stacking ensemble as a `WeightedEnsemble` pyfunc, then check that the served model
        reproduces the offline stacking predictions on the first test rows."""
        from mlflow.models import infer_signature

        from weather.modeling.ensemble_model import WeightedEnsemble

        paths, columns = self.paths, self.feature_columns
        features = pd.read_parquet(paths.features_path, columns=columns + ["split"])
        example = (
            features[features["split"] == Split.TEST][columns]
            .astype("float64")
            .head(PanelEnsembling.SERVING_CHECK_ROWS)
        )
        artifacts = {"config": str(self.config_path)} | {
            n: str(paths.models / f"{n}.joblib") for n in self.serving_config["weights"]
        }
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=WeightedEnsemble(),
            artifacts=artifacts,
            code_paths=[str(paths.root / "src" / "weather")],
            input_example=example.head(5),
            registered_model_name=f"{self.settings.mlflow.registered_model_prefix}-ensemble",
            signature=infer_signature(example, np.zeros(len(example))),
        )
        served = mlflow.pyfunc.load_model(info.model_uri).predict(example)
        # predictions.parquet keeps the feature table's row order -> the first test rows line up
        offline = self.test[prediction_column(PanelEnsemble.STACKING)].iloc[: len(example)].to_numpy()
        max_diff = float(np.max(np.abs(served - offline)))
        mlflow.log_metric("served_vs_offline_max_abs_diff", max_diff)
        assert (
            max_diff < PanelEnsembling.SERVING_TOLERANCE
        ), f"served ensemble deviates from offline predictions by {max_diff}"

    def log_to_mlflow(self) -> None:
        """One `ensemble` run: best-model tags, ensemble weights as params, ensemble test metrics, figures,
        the comparison table and the registered pyfunc model."""
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="ensemble"):
            mlflow.set_tags({"stage": "ensemble", "best_single": self.best_single, "best_overall": self.best_overall})
            mlflow.log_params(
                {f"w_inv_{k}": round(v, 4) for k, v in self.inverse_rmse.weights.items()}
                | {f"w_stack_{k}": v for k, v in self.stacking.weights.items()}
                | {"stack_intercept": self.stacking.intercept}
            )
            for name in PanelEnsemble:
                log_metrics_prefixed(mlflow, self.test_metrics[name], f"{name}_test")
            mlflow.log_artifacts(str(self.figure_dir), artifact_path="figures")
            mlflow.log_artifact(str(self.paths.tables / "model_comparison.csv"))
            self.register_pyfunc(mlflow)

    # reporting --------------------------------------------------------------------------------------------
    def write_metrics(self) -> None:
        """`ensemble.json`: member validation RMSEs, both weight sets, test metrics, skill and best models."""
        out = {
            "val_rmse_members": self.inverse_rmse.val_rmse,
            "weights_inverse_rmse": self.inverse_rmse.weights,
            "weights_stacking": self.stacking.weights,
            "stack_intercept": self.stacking.intercept,
            "test": self.test_metrics,
            "skill_vs_persistence": self.skill,
            "best_single": self.best_single,
            "best_overall": self.best_overall,
        }
        (self.paths.metrics / "ensemble.json").write_text(json.dumps(out, indent=2))

    def run(self) -> None:
        """Execute the full stage."""
        ensure_dirs(self.settings)
        apply_style()
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.load_predictions()
        self.fit_combiners()
        self.add_ensemble_predictions()
        self.evaluate()
        self.write_tables()
        self.plot_rmse_comparison()
        self.plot_continent_heatmap()
        self.plot_daily_error()
        self.plot_city_forecasts()
        self.write_serving_config()
        self.log_to_mlflow()
        self.write_metrics()
        print(self.comparison.round(3).to_string())
        print("stacking weights:", self.stacking.weights, "intercept", round(self.stacking.intercept, 3))


def main() -> None:
    """DVC entry point."""
    PanelEnsembleStage().run()


if __name__ == "__main__":
    main()
