import pytest

from weather.utils.config import PROJECT_ROOT, Settings, get_settings


def test_settings_load_from_dotenv():
    s = Settings()
    assert (PROJECT_ROOT / ".env").exists()
    assert s.features.lags == [1, 2, 3, 7, 14]  # JSON list parsed from .env
    assert s.models.lightgbm.num_leaves == 63  # nested MODELS__LIGHTGBM__NUM_LEAVES
    assert s.paths.features_path == PROJECT_ROOT / "data" / "processed" / "features.parquet"
    assert s.mlflow_tracking_uri.startswith("sqlite:///")


def test_environment_variables_override_dotenv(monkeypatch):
    monkeypatch.setenv("CLEAN__IQR_K", "2.5")
    monkeypatch.setenv("SPLIT__TEST_DAYS", "30")
    s = Settings()
    assert s.clean.iqr_k == 2.5 and s.split.test_days == 30


def test_validation_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("SPLIT__TEST_DAYS", "0")
    with pytest.raises(ValueError):
        Settings()


def test_get_settings_is_cached():
    get_settings.cache_clear()
    assert get_settings() is get_settings()
