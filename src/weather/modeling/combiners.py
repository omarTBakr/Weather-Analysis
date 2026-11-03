"""Ways of combining the panel models' predictions into one ensemble forecast.

Every combiner is fitted on the validation window and ends up as the same representation:
`weights` (member name -> weight) plus an `intercept`, i.e. `forecast = intercept + sum_i weight_i * pred_i`.
That representation is what the `ensemble` stage reports, logs and serves.

Each `predict` keeps the exact floating-point arithmetic the stage has always used (row mean, sequential
weighted sum, matrix product), so published metrics stay bit-for-bit reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from weather.enums import PanelColumns, PanelEnsemble
from weather.utils.metrics import regression_metrics


def prediction_column(model: str) -> str:
    """Name of the column holding `model`'s forecasts in `predictions.parquet` (e.g. `pred_ridge`)."""
    return f"{PanelColumns.PREDICTION_PREFIX}{model}"


class Combiner(ABC):
    """Base class: a fitted linear combination of member predictions (`weights` + `intercept`)."""

    name: PanelEnsemble

    def __init__(self, members: list[str]):
        self.members = list(members)
        self.columns = [prediction_column(m) for m in self.members]
        self.weights: dict[str, float] = {}
        self.intercept: float = 0.0

    def member_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The members' prediction columns of `frame`, in member order."""
        return frame[self.columns]

    @abstractmethod
    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Combiner:
        """Set `weights` / `intercept` from the member predictions in `frame` and the observed `target`."""

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """`intercept + sum_i weight_i * pred_i`, summed member by member."""
        return self.intercept + sum(self.weights[m] * frame[c] for m, c in zip(self.members, self.columns, strict=True))

    def positive_weights(self) -> dict[str, float]:
        """Weights of the members that actually contribute (weight > 0); what the served model loads."""
        return {m: float(w) for m, w in self.weights.items() if w > 0}


class MeanCombiner(Combiner):
    """Equal weights `1/n`, no intercept."""

    name = PanelEnsemble.MEAN

    def fit(self, frame, target):
        """Equal weights; nothing is learned from the data."""
        self.weights = {m: 1 / len(self.members) for m in self.members}
        self.intercept = 0.0
        return self

    def predict(self, frame):
        """Row mean of the member predictions (the equal-weight sum)."""
        return self.member_predictions(frame).mean(axis=1)


class InverseRmseCombiner(Combiner):
    """Weights proportional to 1 / validation RMSE, normalised to sum to one; no intercept."""

    name = PanelEnsemble.WEIGHTED

    def fit(self, frame, target):
        """Score each member on the validation window (kept in `val_rmse`) and normalise the inverse RMSEs."""
        self.val_rmse = {
            m: regression_metrics(target, frame[c])["rmse"] for m, c in zip(self.members, self.columns, strict=True)
        }
        inverse = {m: 1 / r for m, r in self.val_rmse.items()}
        self.weights = {m: v / sum(inverse.values()) for m, v in inverse.items()}
        self.intercept = 0.0
        return self


class StackingCombiner(Combiner):
    """Non-negative least squares (linear regression with positive coefficients) on the member predictions."""

    name = PanelEnsemble.STACKING

    def fit(self, frame, target):
        """Fit `LinearRegression(positive=True)`; its coefficients and intercept become the weights."""
        model = LinearRegression(positive=True).fit(self.member_predictions(frame), target)
        self.weights = {m: float(c) for m, c in zip(self.members, model.coef_, strict=True)}
        self.intercept = float(model.intercept_)
        return self

    def predict(self, frame):
        """Matrix product of the member predictions with the weights, plus the intercept."""
        weights = np.array([self.weights[m] for m in self.members])
        return pd.Series(self.member_predictions(frame).to_numpy() @ weights + self.intercept, index=frame.index)


def build_combiners(members: list[str]) -> dict[PanelEnsemble, Combiner]:
    """One unfitted combiner per `PanelEnsemble` member, in enum order."""
    combiners = [MeanCombiner(members), InverseRmseCombiner(members), StackingCombiner(members)]
    return {c.name: c for c in combiners}
