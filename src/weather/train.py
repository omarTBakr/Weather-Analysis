"""Stage `train`: next-day temperature models on the city panel, each tracked as a nested MLflow run.


Models: persistence baseline, Ridge (on train-fitted standardisation), Random Forest, XGBoost, LightGBM.
Validation window = early stopping / ensemble weights; test window = untouched final evaluation.
"""

from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from weather.metrics import regression_metrics
from weather.mlflow_utils import log_figure, log_metrics_prefixed, residual_figure
from weather.utils.config import ModelSettings, ensure_dirs, get_settings, setup_mlflow

S = get_settings()
P = S.paths

KEYS = ["location_name", "country", "continent", "date", "target_date", "split", "target"]


class Persistence:
    """Tomorrow = today. The bar every model must clear."""

    def __init__(self, cols):
        self.i = cols.index("temp_t")

    def predict(self, X):
        return np.asarray(X)[:, self.i]


def build_models(mp: ModelSettings, seed: int, scaler) -> dict:
    return {
        "ridge": Pipeline([("scaler", scaler), ("ridge", Ridge(alpha=mp.ridge.alpha))]),
        "random_forest": RandomForestRegressor(random_state=seed, n_jobs=-1, **mp.random_forest.model_dump()),
        "xgboost": xgb.XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **mp.xgboost.model_dump()),
        "lightgbm": lgb.LGBMRegressor(
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **{k: v for k, v in mp.lightgbm.model_dump().items() if k != "early_stopping_rounds"},
        ),
    }


def fit(name, model, Xtr, ytr, Xva, yva, mp):
    if name == "xgboost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return {"best_iteration": int(model.best_iteration)}
    if name == "lightgbm":
        model.fit(
            Xtr,
            ytr,
            eval_X=(Xva,),
            eval_y=(yva,),
            callbacks=[lgb.early_stopping(mp.lightgbm.early_stopping_rounds, verbose=False)],
        )
        return {"best_iteration": int(model.best_iteration_)}
    if name == "ridge":
        # scaler inside the pipeline is already fitted on the train window -> only fit the regressor
        model.named_steps["ridge"].fit(model.named_steps["scaler"].transform(Xtr), ytr)
        return {}
    model.fit(Xtr, ytr)
    return {}


def log_model(mlflow, name, model, Xs, register: bool):
    sig = infer_signature(Xs, model.predict(Xs))
    kw = dict(
        name="model",
        signature=sig,
        input_example=Xs.iloc[:5],
        registered_model_name=f"{S.mlflow.registered_model_prefix}-{name}" if register else None,
    )
    if name == "xgboost":
        mlflow.xgboost.log_model(model, **kw)
    elif name == "lightgbm":
        mlflow.lightgbm.log_model(model, **kw)
    elif name == "random_forest":
        # self-trained model -> the tree node storage type is safe to trust for skops serialisation
        mlflow.sklearn.log_model(model, skops_trusted_types=["sklearn.tree._tree.Tree"], **kw)
    else:
        mlflow.sklearn.log_model(model, **kw)


def main() -> None:
    ensure_dirs()
    mp = S.models
    seed = S.seed
    feat = pd.read_parquet(P.features_path)
    cols = json.loads((P.models / "feature_columns.json").read_text())
    scaler = joblib.load(P.models / "scaler.joblib")
    tr, va, te = (feat[feat["split"] == s] for s in ("train", "val", "test"))
    # all-float inputs keep the MLflow signature robust to missing values at serving time
    Xtr, Xva, Xte = (d[cols].astype("float64") for d in (tr, va, te))
    ytr, yva, yte = tr["target"], va["target"], te["target"]

    preds = feat.loc[feat["split"].isin(["val", "test"]), KEYS].copy()
    results = {}
    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="train_panel") as parent:
        mlflow.set_tag("stage", "train")
        mlflow.log_params(
            {
                "n_features": len(cols),
                "train_rows": len(tr),
                "val_rows": len(va),
                "test_rows": len(te),
                "horizon_days": S.features.horizon,
            }
        )

        models = {"persistence": Persistence(cols), **build_models(mp, seed, scaler)}
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"stage": "train", "model": name, "framing": "panel"})
                if name in ModelSettings.model_fields:
                    mlflow.log_params(getattr(mp, name).model_dump())
                t0 = time.time()
                extra = {} if name == "persistence" else fit(name, model, Xtr, ytr, Xva, yva, mp)
                fit_s = time.time() - t0
                pva, pte = model.predict(Xva), model.predict(Xte)
                mva, mte = regression_metrics(yva, pva), regression_metrics(yte, pte)
                log_metrics_prefixed(mlflow, mva, "val")
                log_metrics_prefixed(mlflow, mte, "test")
                mlflow.log_metrics({"fit_seconds": fit_s, **extra})
                log_figure(mlflow, residual_figure(yte, pte, f"{name} — test window"), f"residuals_{name}.png")
                if name != "persistence":
                    log_model(mlflow, name, model, Xva.sample(2000, random_state=seed), register=True)
                    joblib.dump(model, P.models / f"{name}.joblib")
                preds.loc[va.index, f"pred_{name}"] = pva
                preds.loc[te.index, f"pred_{name}"] = pte
                results[name] = {"val": mva, "test": mte, "fit_seconds": fit_s, **extra}
                print(
                    f"{name:14s} val RMSE {mva['rmse']:.3f}  test RMSE {mte['rmse']:.3f}  MAE {mte['mae']:.3f}  ({fit_s:.0f}s)"
                )
        mlflow.set_tag("parent_run_id", parent.info.run_id)

    preds.to_parquet(P.predictions_path, index=False)
    (P.metrics / "train_panel.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
