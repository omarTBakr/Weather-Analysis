"""Feature importance of the LightGBM panel model: gain, permutation, mutual information, SHAP."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from weather.enums import FeatureImportanceInsights, Insights, Palette, PanelModel, Split
from weather.insights.common import InsightAnalysis
from weather.utils.plotting import save


class FeatureImportanceAnalysis(InsightAnalysis):
    """Ranks the LightGBM panel model's inputs by gain, permutation importance, mutual information and SHAP."""

    def load(self) -> None:
        """Feature table, feature column list and the fitted LightGBM model; test-window design matrix."""
        paths = self.paths
        self.features = pd.read_parquet(paths.features_path)
        self.columns = json.loads((paths.models / "feature_columns.json").read_text())
        self.model = joblib.load(paths.models / f"{PanelModel.LIGHTGBM}.joblib")
        test = self.features[self.features["split"] == Split.TEST]
        self.X = test[self.columns].astype("float64")
        self.y = test["target"]

    def compute_importances(self) -> None:
        """The four importance measures, combined into `table` sorted by their mean rank."""
        import shap

        ap, cols, model, X = self.settings.analyses, self.columns, self.model, self.X
        gain = pd.Series(model.booster_.feature_importance("gain"), index=cols)
        gain = gain / gain.sum()
        Xs = X.sample(min(ap.permutation_sample, len(X)), random_state=0)
        ys = self.y.loc[Xs.index]
        perm = permutation_importance(
            model, Xs, ys, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=0, n_jobs=-1
        )
        perm = pd.Series(perm.importances_mean, index=cols)
        self.X_shap = X.sample(min(ap.shap_sample, len(X)), random_state=1)
        self.shap_values = shap.TreeExplainer(model).shap_values(self.X_shap)
        shap_imp = pd.Series(np.abs(self.shap_values).mean(axis=0), index=cols)
        tr = self.features[self.features["split"] == Split.TRAIN].sample(ap.mutual_info_sample, random_state=2)
        mi = pd.Series(mutual_info_regression(tr[cols].astype("float64"), tr["target"], random_state=0), index=cols)

        table = pd.DataFrame(dict(zip(FeatureImportanceInsights.TECHNIQUES, [gain, perm, shap_imp, mi], strict=True)))
        ranks = table.rank(ascending=False)
        table["Mean rank"] = ranks.mean(axis=1)
        self.table = table.sort_values("Mean rank")

    def write_tables(self) -> None:
        """Full comparison table (CSV) and its top rows (LaTeX)."""
        n = Insights.TOP_RANKED
        self.table.to_csv(self.paths.tables / "feature_importance.csv")
        self.table.head(n).to_latex(
            self.paths.tables / "feature_importance.tex",
            float_format="%.3f",
            caption=f"Top-{n} features for next-day temperature under four importance techniques "
            "(sorted by mean rank).",
            label="tab:feature_importance",
            escape=True,
        )

    def plot_comparison(self) -> None:
        """One panel per technique: the top features' importance relative to that technique's maximum."""
        top = self.table.head(Insights.TOP_RANKED).index[::-1]
        fig, axes = plt.subplots(1, 4, figsize=(12, 4.6), sharey=True)
        for ax, col in zip(axes, FeatureImportanceInsights.TECHNIQUES, strict=True):
            v = self.table.loc[top, col]
            ax.barh(top, v / v.max(), color=Palette.SERIES[0], height=0.65)
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
        save(fig, self.figure_dir / "feature_importance_comparison.png")

    def plot_secondary(self) -> None:
        """Zoom without the dominant persistence features to reveal secondary drivers (mean |SHAP|)."""
        table = self.table
        sub = table.drop(
            index=[c for c in table.index if c.startswith(FeatureImportanceInsights.PERSISTENCE_PREFIXES)]
        ).head(Insights.TOP_RANKED)
        fig, ax = plt.subplots(figsize=(7, 4.4))
        v = sub["Mean |SHAP|"].iloc[::-1]
        ax.barh(v.index, v.values, color=Palette.SERIES[0], height=0.65)
        ax.set_xlabel("Mean |SHAP| (°C)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Secondary drivers (excluding current temperature & rolling means)")
        save(fig, self.figure_dir / "feature_importance_secondary.png")

    def plot_shap_summary(self) -> None:
        """SHAP beeswarm of the top features (dots rasterised inside the vector PDF)."""
        import shap

        shap.summary_plot(
            self.shap_values, self.X_shap, max_display=Insights.TOP_RANKED, show=False, plot_size=(8, 5.5)
        )
        plt.gcf().suptitle("SHAP summary — LightGBM next-day temperature model", x=0.01, ha="left", fontweight="bold")
        for coll in plt.gca().collections:  # thousands of beeswarm dots -> raster layer inside the vector PDF
            coll.set_rasterized(True)
        save(plt.gcf(), self.figure_dir / "feature_importance_shap_summary.png", tight=False)

    def plot_shap_dependence(self) -> None:
        """SHAP dependence plot for each feature of `FeatureImportanceInsights.SHAP_DEPENDENCE_FEATURES` that the model uses."""
        import shap

        for f in FeatureImportanceInsights.SHAP_DEPENDENCE_FEATURES:
            if f in self.columns:
                shap.dependence_plot(
                    f, self.shap_values, self.X_shap, interaction_index=None, show=False, color=Palette.SERIES[0]
                )
                plt.title(f"SHAP dependence — {f}", loc="left", fontweight="bold", fontsize=11)
                for coll in plt.gca().collections:
                    coll.set_rasterized(True)
                save(plt.gcf(), self.figure_dir / f"feature_importance_shap_dependence_{f}.png", tight=False)

    def run(self) -> dict:
        """Compute the importances, write tables and figures; return headline metrics."""
        self.load()
        self.compute_importances()
        self.write_tables()
        self.plot_comparison()
        self.plot_secondary()
        self.plot_shap_summary()
        self.plot_shap_dependence()
        table = self.table
        return {
            "fi_top_feature": table.index[0],
            "fi_top5": ", ".join(table.index[:5]),
            "fi_rank_corr_gain_shap": float(table["LightGBM gain"].corr(table["Mean |SHAP|"], method="spearman")),
            "fi_rank_corr_perm_shap": float(table["Permutation (ΔRMSE)"].corr(table["Mean |SHAP|"], method="spearman")),
        }
