"""Stage `analyses`: climate patterns, air quality vs weather, feature importance, spatial and geographic patterns.

Run with `python -m weather.insights.run` (the DVC `analyses` stage).
"""

from __future__ import annotations

import json

import pandas as pd

from weather.insights.air_quality import AirQualityAnalysis
from weather.insights.climate import ClimateAnalysis
from weather.insights.feature_importance import FeatureImportanceAnalysis
from weather.insights.geographic import GeographicAnalysis
from weather.insights.spatial import SpatialAnalysis
from weather.utils.config import Settings, ensure_dirs, get_settings, setup_mlflow
from weather.utils.plotting import apply_style


class InsightsPipeline:
    """Runs every analysis on the daily panel, writes `analyses.json` and logs an `analyses` MLflow run."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.figure_dir = self.paths.figures / "analyses"

    def run_analyses(self) -> dict:
        """Climate, air quality, feature importance, spatial and geographic analyses; merged headline metrics."""
        s = self.settings
        daily = pd.read_parquet(self.paths.daily_path)
        results = {}
        results |= ClimateAnalysis(s).run(daily)
        results |= AirQualityAnalysis(s).run(daily)
        results |= FeatureImportanceAnalysis(s).run()
        spatial, locations = SpatialAnalysis(s).run(daily)
        results |= spatial
        results |= GeographicAnalysis(s).run(daily, locations)
        return results

    def log_to_mlflow(self, results: dict) -> None:
        """One `analyses` run: float results as metrics, string results as tags, figures and maps as artifacts."""
        mlflow = setup_mlflow(self.settings)
        with mlflow.start_run(run_name="analyses"):
            mlflow.set_tag("stage", "analyses")
            mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, float)})
            mlflow.set_tags({k: v for k, v in results.items() if isinstance(v, str)})
            mlflow.log_artifacts(str(self.figure_dir), artifact_path="figures")
            mlflow.log_artifacts(str(self.paths.maps), artifact_path="maps")

    def run(self) -> None:
        """Execute the full stage."""
        ensure_dirs(self.settings)
        apply_style()
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        results = self.run_analyses()
        (self.paths.metrics / "analyses.json").write_text(json.dumps(results, indent=2))
        self.log_to_mlflow(results)
        print(json.dumps(results, indent=2))


def main() -> None:
    """DVC entry point."""
    InsightsPipeline().run()


if __name__ == "__main__":
    main()
