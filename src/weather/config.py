"""Paths, parameters and MLflow setup shared by every pipeline stage."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
METRICS = REPORTS / "metrics"
MAPS = REPORTS / "maps"

CLEAN_PATH = DATA_INTERIM / "clean.parquet"
DAILY_PATH = DATA_PROCESSED / "daily.parquet"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"
GLOBAL_SERIES_PATH = DATA_PROCESSED / "global_series.parquet"
PREDICTIONS_PATH = DATA_PROCESSED / "predictions.parquet"

MLFLOW_EXPERIMENT = "weather-forecasting"


def load_params() -> dict:
    with open(ROOT / "params.yaml") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS, FIGURES, TABLES, METRICS, MAPS):
        p.mkdir(parents=True, exist_ok=True)


def setup_mlflow():
    """Local SQLite tracking store + file artifact store at the repo root."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow
