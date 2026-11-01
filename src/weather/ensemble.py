"""Stage `ensemble`: combine panel models (mean, inverse-RMSE weights, stacking) and evaluate on the test window."""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from weather.metrics import regression_metrics
from weather.plotting import CONTINENT_ORDER, INK_2, MUTED, SEQ_CMAP, SERIES, apply_style, save
from weather.utils.config import ensure_dirs, get_settings, setup_mlflow

S = get_settings()
P = S.paths

OUT = P.figures / "forecast_panel"
MEMBERS = ["ridge", "random_forest", "xgboost", "lightgbm"]
PRETTY = {
    "persistence": "Persistence (baseline)",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "ens_mean": "Ensemble: simple mean",
    "ens_weighted": "Ensemble: inverse-RMSE weighted",
    "ens_stacking": "Ensemble: stacking (NNLS)",
}


def main() -> None:
    ensure_dirs()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(P.predictions_path)
    va, te = preds[preds["split"] == "val"], preds[preds["split"] == "test"].copy()

    val_rmse = {m: regression_metrics(va["target"], va[f"pred_{m}"])["rmse"] for m in MEMBERS}
    inv = {m: 1 / r for m, r in val_rmse.items()}
    w_inv = {m: v / sum(inv.values()) for m, v in inv.items()}
    stack = LinearRegression(positive=True).fit(va[[f"pred_{m}" for m in MEMBERS]], va["target"])
    w_stack = {m: float(c) for m, c in zip(MEMBERS, stack.coef_, strict=True)}

    for d in (va, te):
        d.loc[:, "pred_ens_mean"] = d[[f"pred_{m}" for m in MEMBERS]].mean(axis=1)
        d.loc[:, "pred_ens_weighted"] = sum(w_inv[m] * d[f"pred_{m}"] for m in MEMBERS)
        d.loc[:, "pred_ens_stacking"] = stack.predict(d[[f"pred_{m}" for m in MEMBERS]])

    names = ["persistence", *MEMBERS, "ens_mean", "ens_weighted", "ens_stacking"]
    test_metrics = {n: regression_metrics(te["target"], te[f"pred_{n}"]) for n in names}
    val_metrics = {n: regression_metrics(va["target"], va[f"pred_{n}"]) for n in names}
    best_single = min(MEMBERS, key=lambda m: test_metrics[m]["rmse"])
    best_overall = min(names, key=lambda m: test_metrics[m]["rmse"])
    skill = {n: 1 - test_metrics[n]["rmse"] / test_metrics["persistence"]["rmse"] for n in names}

    # comparison table --------------------------------------------------------------------------
    tbl = pd.DataFrame(test_metrics).T[["mae", "rmse", "smape", "r2", "within_2c"]]
    tbl["skill_vs_persistence_pct"] = pd.Series(skill) * 100
    tbl["val_rmse"] = pd.Series({n: val_metrics[n]["rmse"] for n in names})
    tbl.index = [PRETTY[n] for n in tbl.index]
    tbl.to_csv(P.tables / "model_comparison.csv")
    tbl.rename(
        columns={
            "mae": "MAE",
            "rmse": "RMSE",
            "smape": "sMAPE (\\%)",
            "r2": "$R^2$",
            "within_2c": "$\\le$2\\,°C (\\%)",
            "skill_vs_persistence_pct": "Skill (\\%)",
            "val_rmse": "Val RMSE",
        }
    ).to_latex(
        P.tables / "model_comparison.tex",
        float_format="%.3f",
        escape=False,
        caption="Next-day temperature forecast accuracy on the 60-day test window (all cities). "
        "Skill = RMSE reduction relative to persistence.",
        label="tab:model_comparison",
    )

    # per-continent RMSE ---------------------------------------------------------------------------
    cont = pd.DataFrame(
        {
            PRETTY[n]: te.groupby("continent").apply(
                lambda g, n=n: np.sqrt(np.mean((g["target"] - g[f"pred_{n}"]) ** 2)), include_groups=False
            )
            for n in names
        }
    ).T
    cont = cont[CONTINENT_ORDER]
    cont.to_csv(P.tables / "rmse_by_continent.csv")

    # figures ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = sorted(names, key=lambda n: test_metrics[n]["rmse"], reverse=True)
    colors = [MUTED if n == "persistence" else (SERIES[1] if n.startswith("ens") else SERIES[0]) for n in order]
    ax.barh([PRETTY[n] for n in order], [test_metrics[n]["rmse"] for n in order], color=colors, height=0.62)
    for i, n in enumerate(order):
        ax.annotate(
            f"{test_metrics[n]['rmse']:.3f}",
            (test_metrics[n]["rmse"], i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    ax.set_xlabel("Test RMSE (°C) — lower is better")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(test_metrics[n]["rmse"] for n in names) * 1.12)
    ax.set_title("Next-day temperature: single models (blue) vs. ensembles (orange)")
    save(fig, OUT / "model_comparison_rmse.png")

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    im = ax.imshow(cont.values, cmap=SEQ_CMAP, aspect="auto")
    ax.set_xticks(range(cont.shape[1]), cont.columns)
    ax.set_yticks(range(cont.shape[0]), cont.index)
    ax.grid(False)
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            v = cont.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#ffffff" if v > np.nanmax(cont.values) * 0.7 else "#0b0b0b",
            )
    fig.colorbar(im, ax=ax, label="RMSE (°C)", shrink=0.9)
    ax.set_title("Test RMSE by continent")
    save(fig, OUT / "rmse_by_continent.png")

    daily_err = te.groupby("target_date").apply(
        lambda g: pd.Series(
            {
                "persistence": np.sqrt(np.mean((g["target"] - g["pred_persistence"]) ** 2)),
                best_overall: np.sqrt(np.mean((g["target"] - g[f"pred_{best_overall}"]) ** 2)),
            }
        ),
        include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(daily_err.index, daily_err["persistence"], color=MUTED, label=PRETTY["persistence"])
    ax.plot(daily_err.index, daily_err[best_overall], color=SERIES[1], label=PRETTY[best_overall])
    ax.set_ylabel("Cross-city RMSE (°C)")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("Daily forecast error over the test window")
    save(fig, OUT / "daily_error_test.png")

    cities = ["London", "Tokyo", "Nairobi", "Buenos Aires"]
    cities = [c for c in cities if c in set(te["location_name"])][:4]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
    for ax, c in zip(axes.flat, cities, strict=False):
        g = te[te["location_name"] == c].sort_values("target_date")
        ax.plot(g["target_date"], g["target"], color=INK_2, linewidth=1.3, label="observed")
        ax.plot(g["target_date"], g[f"pred_{best_overall}"], color=SERIES[1], linewidth=1.5, label="ensemble forecast")
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("°C")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes.flat[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Next-day forecasts on the test window — example capitals", x=0.01, ha="left", fontweight="bold")
    save(fig, OUT / "city_forecasts.png")

    # MLflow: log + register the stacking ensemble as a pyfunc model --------------------------------
    feature_cols = json.loads((P.models / "feature_columns.json").read_text())
    cfg = {
        "weights": {k: float(v) for k, v in w_stack.items() if v > 0},
        "intercept": float(stack.intercept_),
        "feature_columns": feature_cols,
    }
    (P.models / "ensemble_config.json").write_text(json.dumps(cfg, indent=2))
    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="ensemble"):
        mlflow.set_tags({"stage": "ensemble", "best_single": best_single, "best_overall": best_overall})
        mlflow.log_params(
            {f"w_inv_{k}": round(v, 4) for k, v in w_inv.items()}
            | {f"w_stack_{k}": v for k, v in w_stack.items()}
            | {"stack_intercept": float(stack.intercept_)}
        )
        for n in ["ens_mean", "ens_weighted", "ens_stacking"]:
            mlflow.log_metrics({f"{n}_test_{k}": v for k, v in test_metrics[n].items() if np.isfinite(v)})
        mlflow.log_artifacts(str(OUT), artifact_path="figures")
        mlflow.log_artifact(str(P.tables / "model_comparison.csv"))

        from mlflow.models import infer_signature

        from weather.ensemble_model import WeightedEnsemble

        feat = pd.read_parquet(P.features_path, columns=feature_cols + ["split"])
        example = feat[feat["split"] == "test"][feature_cols].astype("float64").head(200)
        artifacts = {"config": str(P.models / "ensemble_config.json")} | {
            n: str(P.models / f"{n}.joblib") for n in cfg["weights"]
        }
        model = WeightedEnsemble()
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            artifacts=artifacts,
            code_paths=[str(P.root / "src" / "weather")],
            input_example=example.head(5),
            registered_model_name=f"{S.mlflow.registered_model_prefix}-ensemble",
            signature=infer_signature(example, np.zeros(len(example))),
        )
        # sanity: the served model reproduces the offline stacking prediction
        served = mlflow.pyfunc.load_model(info.model_uri).predict(example)
        # predictions.parquet keeps the feature table's row order -> the first 200 test rows line up
        offline = te["pred_ens_stacking"].iloc[: len(example)].to_numpy()
        max_diff = float(np.max(np.abs(served - offline)))
        mlflow.log_metric("served_vs_offline_max_abs_diff", max_diff)
        assert max_diff < 1e-4, f"served ensemble deviates from offline predictions by {max_diff}"

    out = {
        "val_rmse_members": val_rmse,
        "weights_inverse_rmse": w_inv,
        "weights_stacking": w_stack,
        "stack_intercept": float(stack.intercept_),
        "test": test_metrics,
        "skill_vs_persistence": skill,
        "best_single": best_single,
        "best_overall": best_overall,
    }
    (P.metrics / "ensemble.json").write_text(json.dumps(out, indent=2))
    print(tbl.round(3).to_string())
    print("stacking weights:", w_stack, "intercept", round(float(stack.intercept_), 3))


if __name__ == "__main__":
    main()
