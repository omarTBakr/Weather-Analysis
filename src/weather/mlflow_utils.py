"""Helpers for logging forecasting models to MLflow."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather.plotting import INK_2, MUTED, SERIES, apply_style, save


def log_metrics_prefixed(mlflow, metrics: dict, prefix: str) -> None:
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items() if np.isfinite(v)})


def residual_figure(y_true, y_pred, title: str):
    apply_style()
    r = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    axes[0].hexbin(y_true, y_pred, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    axes[0].plot(lim, lim, color=INK_2, linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Observed (°C)")
    axes[0].set_ylabel("Predicted (°C)")
    axes[1].hist(r, bins=80, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.4)
    axes[1].axvline(0, color=MUTED, linewidth=0.8)
    axes[1].set_xlabel("Residual (pred − obs, °C)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    return fig


def log_figure(mlflow, fig, name: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        save(fig, p)
        mlflow.log_artifact(str(p), artifact_path="plots")


def log_frame(mlflow, df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        df.to_csv(p, index=False)
        mlflow.log_artifact(str(p))
