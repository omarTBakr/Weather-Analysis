import numpy as np

from weather.metrics import regression_metrics


def test_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1 and m["within_2c"] == 100


def test_known_errors():
    m = regression_metrics([0.0, 0.0, 10.0, 10.0], [1.0, -1.0, 13.0, 7.0])
    assert np.isclose(m["mae"], 2.0)
    assert np.isclose(m["rmse"], np.sqrt((1 + 1 + 9 + 9) / 4))
    assert np.isclose(m["mape"], 30.0)  # MAPE ignores |y| < 1
