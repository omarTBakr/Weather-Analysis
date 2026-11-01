"""Project settings loaded from `.env` into a typed pydantic model.

Usage::

    from weather.utils.config import get_settings

    s = get_settings()
    s.clean.iqr_k            # 3.0
    s.paths.features_path    # <root>/data/processed/features.parquet
    s.models.lightgbm.model_dump()

`get_settings()` builds the object once and caches it; call `get_settings.cache_clear()` to reload.
Values come from (highest priority first): real environment variables, then `.env` at the project root,
then the defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PathSettings(BaseModel):
    """Directory layout (relative entries are resolved against the project root)."""

    root: Path = PROJECT_ROOT
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    def _abs(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    @property
    def data_raw(self) -> Path:
        return self._abs(self.data_dir) / "raw"

    @property
    def data_interim(self) -> Path:
        return self._abs(self.data_dir) / "interim"

    @property
    def data_processed(self) -> Path:
        return self._abs(self.data_dir) / "processed"

    @property
    def models(self) -> Path:
        return self._abs(self.models_dir)

    @property
    def reports(self) -> Path:
        return self._abs(self.reports_dir)

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    @property
    def metrics(self) -> Path:
        return self.reports / "metrics"

    @property
    def maps(self) -> Path:
        return self.reports / "maps"

    @property
    def clean_path(self) -> Path:
        return self.data_interim / "clean.parquet"

    @property
    def daily_path(self) -> Path:
        return self.data_processed / "daily.parquet"

    @property
    def features_path(self) -> Path:
        return self.data_processed / "features.parquet"

    @property
    def global_series_path(self) -> Path:
        return self.data_processed / "global_series.parquet"

    @property
    def predictions_path(self) -> Path:
        return self.data_processed / "predictions.parquet"


class MLflowSettings(BaseModel):
    tracking_uri: str = ""  # empty -> sqlite:///<root>/mlflow.db
    experiment_name: str = "weather-forecasting"
    artifacts_dir: Path = Path("mlartifacts")
    registered_model_prefix: str = "weather-temp"


class DataSettings(BaseModel):
    kaggle_dataset: str = "nelgiriyewithana/global-weather-repository"
    kaggle_file: str = "GlobalWeatherRepository.csv"


class CleanSettings(BaseModel):
    min_obs_per_location: int = Field(365, ge=1)
    iqr_k: float = Field(3.0, gt=0)
    heavy_tail_quantile: float = Field(0.999, gt=0.5, le=1.0)
    interpolate_limit: int = Field(3, ge=0)
    corrupt_day_threshold_c: float = Field(4.0, gt=0)


class FeatureSettings(BaseModel):
    target: str = "temperature_celsius"
    horizon: int = Field(1, ge=1)
    lags: list[int] = [1, 2, 3, 7, 14]
    rolling_windows: list[int] = [3, 7, 14]
    exog: list[str] = [
        "humidity", "pressure_mb", "wind_kph", "cloud", "precip_mm",
        "uv_index", "feels_like_celsius", "visibility_km", "gust_kph",
    ]  # fmt: skip


class SplitSettings(BaseModel):
    test_days: int = Field(60, ge=1)
    val_days: int = Field(60, ge=1)


class GlobalSeriesSettings(BaseModel):
    min_coverage: float = Field(0.95, gt=0, le=1)
    fourier_order: int = Field(4, ge=1)
    future_days: int = Field(90, ge=1)


class AnomalySettings(BaseModel):
    contamination: float = Field(0.01, gt=0, lt=0.5)
    lof_neighbors: int = Field(35, ge=2)
    z_thresh: float = 3.5
    global_z_thresh: float = 3.0
    sample_size: int = Field(60000, ge=100)


class RidgeParams(BaseModel):
    alpha: float = 1.0


class RandomForestParams(BaseModel):
    n_estimators: int = 200
    max_depth: int | None = 16
    min_samples_leaf: int = 5
    max_features: float = 0.5


class XGBoostParams(BaseModel):
    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 7
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 100


class LightGBMParams(BaseModel):
    n_estimators: int = 3000
    learning_rate: float = 0.03
    num_leaves: int = 63
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 100


class ModelSettings(BaseModel):
    ridge: RidgeParams = RidgeParams()
    random_forest: RandomForestParams = RandomForestParams()
    xgboost: XGBoostParams = XGBoostParams()
    lightgbm: LightGBMParams = LightGBMParams()


class AnalysesSettings(BaseModel):
    shap_sample: int = Field(3000, ge=100)
    permutation_sample: int = Field(20000, ge=100)


class Settings(BaseSettings):
    """All project configuration. `BaseSettings` is a pydantic `BaseModel` that also reads env vars / `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    seed: int = 42
    paths: PathSettings = PathSettings()
    mlflow: MLflowSettings = MLflowSettings()
    data: DataSettings = DataSettings()
    clean: CleanSettings = CleanSettings()
    features: FeatureSettings = FeatureSettings()
    split: SplitSettings = SplitSettings()
    global_series: GlobalSeriesSettings = GlobalSeriesSettings()
    anomaly: AnomalySettings = AnomalySettings()
    models: ModelSettings = ModelSettings()
    analyses: AnalysesSettings = AnalysesSettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_csv(self) -> Path:
        return self.paths.data_raw / self.data.kaggle_file

    @property
    def mlflow_tracking_uri(self) -> str:
        return self.mlflow.tracking_uri or f"sqlite:///{self.paths.root / 'mlflow.db'}"

    @property
    def mlflow_artifact_location(self) -> str:
        p = self.mlflow.artifacts_dir
        return (p if p.is_absolute() else self.paths.root / p).as_uri()

    def flat(self, section: str) -> dict:
        """A section as a flat dict, e.g. for `mlflow.log_params`."""
        return getattr(self, section).model_dump()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the settings once (reads `.env`) and return the cached instance on every later call."""
    return Settings()


def ensure_dirs() -> None:
    p = get_settings().paths
    for d in (p.data_raw, p.data_interim, p.data_processed, p.models, p.figures, p.tables, p.metrics, p.maps):
        d.mkdir(parents=True, exist_ok=True)


def setup_mlflow():
    """Point MLflow at the configured tracking store and experiment; returns the `mlflow` module."""
    import mlflow

    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(s.mlflow.experiment_name) is None:
        mlflow.create_experiment(s.mlflow.experiment_name, artifact_location=s.mlflow_artifact_location)
    mlflow.set_experiment(s.mlflow.experiment_name)
    return mlflow
