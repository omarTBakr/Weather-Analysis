"""Feature importance of the LightGBM panel model: gain, permutation, mutual information, SHAP."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from weather.insights.common import OUT, P, S
from weather.utils.plotting import (
    SERIES,
    save,
)


def feature_importance() -> dict:
    import shap

    ap = S.analyses
    feat = pd.read_parquet(P.features_path)
    cols = json.loads((P.models / "feature_columns.json").read_text())
    model = joblib.load(P.models / "lightgbm.joblib")
    te = feat[feat["split"] == "test"]
    X = te[cols].astype("float64")
    y = te["target"]

    gain = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    gain = gain / gain.sum()
    Xs = X.sample(min(ap.permutation_sample, len(X)), random_state=0)
    ys = y.loc[Xs.index]
    perm = permutation_importance(
        model, Xs, ys, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=0, n_jobs=-1
    )
    perm = pd.Series(perm.importances_mean, index=cols)
    Xshap = X.sample(min(ap.shap_sample, len(X)), random_state=1)
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
    table.to_csv(P.tables / "feature_importance.csv")
    table.head(15).to_latex(
        P.tables / "feature_importance.tex",
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
    for coll in plt.gca().collections:  # thousands of beeswarm dots -> raster layer inside the vector PDF
        coll.set_rasterized(True)
    save(plt.gcf(), OUT / "feature_importance_shap_summary.png", tight=False)

    for f in ["hour_cos", "pressure_diff1"]:
        if f in cols:
            shap.dependence_plot(f, sv, Xshap, interaction_index=None, show=False, color=SERIES[0])
            plt.title(f"SHAP dependence — {f}", loc="left", fontweight="bold", fontsize=11)
            for coll in plt.gca().collections:
                coll.set_rasterized(True)
            save(plt.gcf(), OUT / f"feature_importance_shap_dependence_{f}.png", tight=False)
    return {
        "fi_top_feature": table.index[0],
        "fi_top5": ", ".join(table.index[:5]),
        "fi_rank_corr_gain_shap": float(table["LightGBM gain"].corr(table["Mean |SHAP|"], method="spearman")),
        "fi_rank_corr_perm_shap": float(table["Permutation (ΔRMSE)"].corr(table["Mean |SHAP|"], method="spearman")),
    }
