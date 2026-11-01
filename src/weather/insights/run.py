"""Stage `analyses`: climate patterns, air quality vs weather, feature importance, spatial and geographic patterns."""

from __future__ import annotations

import json

import pandas as pd

from weather.insights.air_quality import air_quality_analysis
from weather.insights.climate import climate_analysis
from weather.insights.common import OUT, P
from weather.insights.feature_importance import feature_importance
from weather.insights.geographic import geographic_patterns
from weather.insights.spatial import spatial_analysis
from weather.utils.config import ensure_dirs, setup_mlflow
from weather.utils.plotting import apply_style


def main() -> None:
    ensure_dirs()
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(P.daily_path)
    results = {}
    results |= climate_analysis(daily)
    results |= air_quality_analysis(daily)
    results |= feature_importance()
    spatial, loc = spatial_analysis(daily)
    results |= spatial
    results |= geographic_patterns(daily, loc)
    (P.metrics / "analyses.json").write_text(json.dumps(results, indent=2))

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="analyses"):
        mlflow.set_tag("stage", "analyses")
        mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, float)})
        mlflow.set_tags({k: v for k, v in results.items() if isinstance(v, str)})
        mlflow.log_artifacts(str(OUT), artifact_path="figures")
        mlflow.log_artifacts(str(P.maps), artifact_path="maps")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
