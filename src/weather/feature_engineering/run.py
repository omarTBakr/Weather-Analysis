"""Stage `features`: daily panel, lag/rolling/calendar features, time-based split, train-only scaling.

Run with `python -m weather.feature_engineering.run` (the DVC `features` stage). Writes the daily panel,
the feature table, the global series, the fitted scaler + feature column list and `features_report.json`.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from weather.enums import Split
from weather.feature_engineering.daily_panel import build_daily_panel
from weather.feature_engineering.global_series import build_global_series
from weather.feature_engineering.panel_features import build_panel_features
from weather.feature_engineering.splits import assign_time_split
from weather.utils.config import Settings, ensure_dirs, get_settings, setup_mlflow


class FeaturePipeline:
    """Builds and persists every artefact of the `features` stage, then logs a summary run to MLflow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths

    def build(self) -> None:
        """Clean data -> daily panel -> feature table (split + scaled on train only) -> global series."""
        s, paths = self.settings, self.paths
        clean = pd.read_parquet(paths.clean_path)
        self.daily = build_daily_panel(clean, s.clean.interpolate_limit)
        features, self.feature_columns = build_panel_features(self.daily, s.features)
        self.features = assign_time_split(features, s.split.test_days, s.split.val_days)
        self.global_series = build_global_series(self.daily, s.global_series.min_coverage)

    def save(self) -> None:
        """Persist the tables, plus a StandardScaler fitted on the training window only (no leakage)."""
        paths = self.paths
        self.daily.to_parquet(paths.daily_path, index=False)
        # used by linear models / anomaly detection
        train_rows = self.features.loc[self.features["split"] == Split.TRAIN, self.feature_columns]
        joblib.dump(StandardScaler().fit(train_rows), paths.models / "scaler.joblib")
        (paths.models / "feature_columns.json").write_text(json.dumps(self.feature_columns, indent=2))
        self.features.to_parquet(paths.features_path, index=False)
        self.global_series.to_parquet(paths.global_series_path, index=False)

    def report(self) -> dict:
        """Row counts and date range of every split, plus sizes of the panel, feature table and global series."""
        f = self.features
        split_info = {
            s: {
                "rows": int((f["split"] == s).sum()),
                "start": str(f.loc[f["split"] == s, "target_date"].min().date()),
                "end": str(f.loc[f["split"] == s, "target_date"].max().date()),
            }
            for s in Split
        }
        return {
            "daily_rows": len(self.daily),
            "feature_rows": len(f),
            "n_features": len(self.feature_columns),
            "global_series_days": len(self.global_series),
            "global_series_locations": int(self.global_series["n_locations"].median()),
            "split": split_info,
        }

    def log_to_mlflow(self, report: dict) -> None:
        """One `features` run with the feature/split settings as params and the row counts as metrics."""
        s, paths = self.settings, self.paths
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="features"):
            mlflow.set_tag("stage", "features")
            mlflow.log_params(
                {
                    **{f"features.{k}": v for k, v in s.features.model_dump().items()},
                    **{f"split.{k}": v for k, v in s.split.model_dump().items()},
                }
            )
            mlflow.log_metrics(
                {
                    "feature_rows": report["feature_rows"],
                    "n_features": report["n_features"],
                    **{f"{name}_rows": v["rows"] for name, v in report["split"].items()},
                }
            )
            mlflow.log_artifact(str(paths.models / "feature_columns.json"))
            mlflow.log_artifact(str(paths.metrics / "features_report.json"))

    def run(self) -> None:
        """Execute the full stage."""
        ensure_dirs(self.settings)
        self.build()
        self.save()
        report = self.report()
        (self.paths.metrics / "features_report.json").write_text(json.dumps(report, indent=2))
        self.log_to_mlflow(report)
        print(json.dumps(report, indent=2))


def main() -> None:
    """DVC entry point."""
    FeaturePipeline().run()


if __name__ == "__main__":
    main()
