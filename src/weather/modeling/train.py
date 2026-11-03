"""Stage `train`: next-day temperature models on the city panel, each tracked as a nested MLflow run.

Models: persistence baseline, Ridge (on train-fitted standardisation), Random Forest, XGBoost, LightGBM.
Validation window = early stopping / ensemble weights; test window = untouched final evaluation.

Run with `python -m weather.modeling.train` (the DVC `train` stage).
"""

from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from weather.enums import PanelColumns, PanelModel, PanelTraining, Split
from weather.modeling.baselines import PersistenceBaseline
from weather.utils.config import ModelSettings, Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.metrics import regression_metrics
from weather.utils.mlflow_utils import log_figure, log_metrics_prefixed, residual_figure


class PanelTrainer:
    """Fits every panel model, logs each as a nested MLflow run, and writes predictions + metrics."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.seed = self.settings.seed
        self.model_settings = self.settings.models
        self.paths = self.settings.paths
        self.results: dict = {}

    def load_data(self) -> None:
        """Load the feature table, feature column list and train-fitted scaler; split into train/val/test."""
        paths = self.paths
        self.features = pd.read_parquet(paths.features_path)
        self.feature_columns = json.loads((paths.models / "feature_columns.json").read_text())
        self.scaler = joblib.load(paths.models / "scaler.joblib")
        self.train_df, self.val_df, self.test_df = (self.features[self.features["split"] == s] for s in Split)
        # all-float inputs keep the MLflow signature robust to missing values at serving time
        self.X_train, self.X_val, self.X_test = (
            d[self.feature_columns].astype("float64") for d in (self.train_df, self.val_df, self.test_df)
        )
        self.y_train, self.y_val, self.y_test = (d["target"] for d in (self.train_df, self.val_df, self.test_df))

    def build_models(self) -> dict:
        """All panel models keyed by name, unfitted, configured from the `models` settings section."""
        ms, seed = self.model_settings, self.seed
        return {
            PanelModel.PERSISTENCE: PersistenceBaseline(self.feature_columns),
            PanelModel.RIDGE: Pipeline([("scaler", self.scaler), ("ridge", Ridge(alpha=ms.ridge.alpha))]),
            PanelModel.RANDOM_FOREST: RandomForestRegressor(
                random_state=seed, n_jobs=-1, **ms.random_forest.model_dump()
            ),
            PanelModel.XGBOOST: xgb.XGBRegressor(
                random_state=seed, n_jobs=-1, tree_method="hist", **ms.xgboost.model_dump()
            ),
            PanelModel.LIGHTGBM: lgb.LGBMRegressor(
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
                **{k: v for k, v in ms.lightgbm.model_dump().items() if k != "early_stopping_rounds"},
            ),
        }

    def fit_model(self, name: str, model) -> dict:
        """Fit `model` on the train window; returns extra metrics (e.g. early-stopping best iteration)."""
        X_train, y_train, X_val, y_val = self.X_train, self.y_train, self.X_val, self.y_val
        if name == PanelModel.PERSISTENCE:
            return {}
        if name == PanelModel.XGBOOST:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return {"best_iteration": int(model.best_iteration)}
        if name == PanelModel.LIGHTGBM:
            model.fit(
                X_train,
                y_train,
                eval_X=(X_val,),
                eval_y=(y_val,),
                callbacks=[lgb.early_stopping(self.model_settings.lightgbm.early_stopping_rounds, verbose=False)],
            )
            return {"best_iteration": int(model.best_iteration_)}
        if name == PanelModel.RIDGE:
            # scaler inside the pipeline is already fitted on the train window -> only fit the regressor
            model.named_steps["ridge"].fit(model.named_steps["scaler"].transform(X_train), y_train)
            return {}
        model.fit(X_train, y_train)
        return {}

    def log_model_to_mlflow(self, mlflow, name: str, model, X_sample: pd.DataFrame) -> None:
        """Log `model` with its flavour-specific MLflow API and register it as `<prefix>-<name>`."""
        kwargs = dict(
            name="model",
            signature=infer_signature(X_sample, model.predict(X_sample)),
            input_example=X_sample.iloc[:5],
            registered_model_name=f"{self.settings.mlflow.registered_model_prefix}-{name}",
        )
        if name == PanelModel.XGBOOST:
            mlflow.xgboost.log_model(model, **kwargs)
        elif name == PanelModel.LIGHTGBM:
            mlflow.lightgbm.log_model(model, **kwargs)
        elif name == PanelModel.RANDOM_FOREST:
            # self-trained model -> the tree node storage type is safe to trust for skops serialisation
            mlflow.sklearn.log_model(model, skops_trusted_types=["sklearn.tree._tree.Tree"], **kwargs)
        else:
            mlflow.sklearn.log_model(model, **kwargs)

    def train_and_evaluate(self, mlflow, name: str, model) -> None:
        """One nested run: fit, score on val/test, log metrics + residual plot, persist, record predictions."""
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.set_tags({"stage": "train", "model": name, "framing": "panel"})
            if name in ModelSettings.model_fields:
                mlflow.log_params(getattr(self.model_settings, name).model_dump())
            start = time.time()
            extra = self.fit_model(name, model)
            fit_seconds = time.time() - start
            pred_val, pred_test = model.predict(self.X_val), model.predict(self.X_test)
            val_metrics = regression_metrics(self.y_val, pred_val)
            test_metrics = regression_metrics(self.y_test, pred_test)
            log_metrics_prefixed(mlflow, val_metrics, "val")
            log_metrics_prefixed(mlflow, test_metrics, "test")
            mlflow.log_metrics({"fit_seconds": fit_seconds, **extra})
            log_figure(
                mlflow, residual_figure(self.y_test, pred_test, f"{name} — test window"), f"residuals_{name}.png"
            )
            if name != PanelModel.PERSISTENCE:
                sample = self.X_val.sample(PanelTraining.SIGNATURE_SAMPLE_ROWS, random_state=self.seed)
                self.log_model_to_mlflow(mlflow, name, model, sample)
                joblib.dump(model, self.paths.models / f"{name}.joblib")
            self.predictions.loc[self.val_df.index, f"{PanelColumns.PREDICTION_PREFIX}{name}"] = pred_val
            self.predictions.loc[self.test_df.index, f"{PanelColumns.PREDICTION_PREFIX}{name}"] = pred_test
            self.results[name] = {"val": val_metrics, "test": test_metrics, "fit_seconds": fit_seconds, **extra}
            print(
                f"{name:14s} val RMSE {val_metrics['rmse']:.3f}  test RMSE {test_metrics['rmse']:.3f}"
                f"  MAE {test_metrics['mae']:.3f}  ({fit_seconds:.0f}s)"
            )

    def run(self) -> None:
        """Execute the full stage inside one `train_panel` parent run; write predictions and metrics."""
        ensure_dirs(self.settings)
        self.load_data()
        val_test = self.features["split"].isin([Split.VAL, Split.TEST])
        self.predictions = self.features.loc[val_test, PanelColumns.PREDICTION_ID].copy()
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="train_panel") as parent:
            mlflow.set_tag("stage", "train")
            mlflow.log_params(
                {
                    "n_features": len(self.feature_columns),
                    "train_rows": len(self.train_df),
                    "val_rows": len(self.val_df),
                    "test_rows": len(self.test_df),
                    "horizon_days": self.settings.features.horizon,
                }
            )
            for name, model in self.build_models().items():
                self.train_and_evaluate(mlflow, name, model)
            mlflow.set_tag("parent_run_id", parent.info.run_id)

        self.predictions.to_parquet(self.paths.predictions_path, index=False)
        (self.paths.metrics / "train_panel.json").write_text(json.dumps(self.results, indent=2))


def main() -> None:
    """DVC entry point."""
    PanelTrainer().run()


if __name__ == "__main__":
    main()
