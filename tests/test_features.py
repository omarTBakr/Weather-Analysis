import numpy as np

from weather.data.features import add_features, time_split
from weather.utils.config import FeatureSettings

FP = FeatureSettings(
    target="temperature_celsius", horizon=1, lags=[1, 7], rolling_windows=[3, 7], exog=["humidity", "pressure_mb"]
)


def test_target_is_next_day_temperature(daily_panel):
    feat, cols = add_features(daily_panel, FP)
    src = daily_panel.set_index(["location_name", "date"])["temperature_celsius"]
    for _, r in feat.sample(20, random_state=0).iterrows():
        assert np.isclose(r["target"], src[(r["location_name"], r["target_date"])])
        assert np.isclose(r["temp_t"], src[(r["location_name"], r["date"])])
    assert "target" not in cols and "target_date" not in cols


def test_lags_only_look_backwards(daily_panel):
    feat, _ = add_features(daily_panel, FP)
    src = daily_panel.set_index(["location_name", "date"])["temperature_celsius"]
    r = feat.iloc[30]
    assert np.isclose(r["temp_lag7"], src[(r["location_name"], r["date"] - np.timedelta64(7, "D"))])
    assert not feat[[c for c in feat if c.startswith("temp_")]].isna().any().any()


def test_time_split_is_strictly_ordered(daily_panel):
    feat, _ = add_features(daily_panel, FP)
    feat = time_split(feat, test_days=10, val_days=10)
    tr, va, te = (feat.loc[feat["split"] == s, "target_date"] for s in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert te.nunique() == 10 and va.nunique() == 10
