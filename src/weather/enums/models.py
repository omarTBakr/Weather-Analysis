"""Model names and the constants of each modelling stage."""

from __future__ import annotations

from enum import StrEnum


class PanelModel(StrEnum):
    """Next-day panel models (stage `train`); also the MLflow run / registry suffix and `pred_<name>` column."""

    PERSISTENCE = "persistence"
    RIDGE = "ridge"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class PanelEnsemble(StrEnum):
    """Ensembles of the panel models built in stage `ensemble`."""

    MEAN = "ens_mean"
    WEIGHTED = "ens_weighted"
    STACKING = "ens_stacking"


class GlobalModel(StrEnum):
    """Global-series forecasters (stage `train_global`) plus the averaged `ensemble_mean`."""

    NAIVE = "naive"
    SEASONAL_NAIVE = "seasonal_naive"
    HOLT_WINTERS = "holt_winters"
    HARMONIC_REGRESSION = "harmonic_regression"
    SARIMAX_FOURIER = "sarimax_fourier"
    ENSEMBLE_MEAN = "ensemble_mean"


class PanelTraining:
    """Constants of the `train` stage."""

    SIGNATURE_SAMPLE_ROWS = 2000  # validation rows used to infer the MLflow model signature


class PanelEnsembling:
    """Constants of the `ensemble` stage."""

    MEMBERS = [PanelModel.RIDGE, PanelModel.RANDOM_FOREST, PanelModel.XGBOOST, PanelModel.LIGHTGBM]
    DISPLAY_NAMES = {
        PanelModel.PERSISTENCE: "Persistence (baseline)",
        PanelModel.RIDGE: "Ridge regression",
        PanelModel.RANDOM_FOREST: "Random forest",
        PanelModel.XGBOOST: "XGBoost",
        PanelModel.LIGHTGBM: "LightGBM",
        PanelEnsemble.MEAN: "Ensemble: simple mean",
        PanelEnsemble.WEIGHTED: "Ensemble: inverse-RMSE weighted",
        PanelEnsemble.STACKING: "Ensemble: stacking (NNLS)",
    }
    EXAMPLE_CITIES = ["London", "Tokyo", "Nairobi", "Buenos Aires"]  # panels of the city forecast figure
    SERVING_CHECK_ROWS = 200  # test rows the registered pyfunc must reproduce offline predictions on
    SERVING_TOLERANCE = 1e-4  # max |served - offline| accepted
    # LaTeX headers of the model comparison table
    LATEX_HEADERS = {
        "mae": "MAE",
        "rmse": "RMSE",
        "smape": "sMAPE (\\%)",
        "r2": "$R^2$",
        "within_2c": "$\\le$2\\,°C (\\%)",
        "skill_vs_persistence_pct": "Skill (\\%)",
        "val_rmse": "Val RMSE",
    }


class GlobalForecasting:
    """Constants of the `train_global` stage."""

    TARGET = "temperature_celsius"
    BACKTEST_CONTEXT_DAYS = 200  # observed days drawn before the test window on the back-test figure
    FUTURE_CONTEXT_DAYS = 365  # observed days drawn before the projection on the future figure
    PI_ALPHA = 0.05  # 95% prediction intervals
    # structural models averaged into `ensemble_mean`
    ENSEMBLE_MEMBERS = [GlobalModel.HOLT_WINTERS, GlobalModel.HARMONIC_REGRESSION, GlobalModel.SARIMAX_FOURIER]
    # forecasts drawn on the back-test figure, in colour order
    PLOTTED_FORECASTS = [
        GlobalModel.SEASONAL_NAIVE,
        GlobalModel.HOLT_WINTERS,
        GlobalModel.HARMONIC_REGRESSION,
        GlobalModel.SARIMAX_FOURIER,
        GlobalModel.ENSEMBLE_MEAN,
    ]
