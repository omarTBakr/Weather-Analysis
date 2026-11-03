"""MLflow pyfunc wrapper that serves the weighted ensemble of the panel models."""

from __future__ import annotations

import json

import joblib
import mlflow.pyfunc
import numpy as np


class WeightedEnsemble(mlflow.pyfunc.PythonModel):
    """Serves `intercept + sum(weight_i * model_i.predict(X))` over the persisted panel models.

    The class name is part of already-registered MLflow models, so keep it stable.
    """

    def load_context(self, context):
        """Read weights / intercept / feature columns from the `config` artifact and load each member model."""
        with open(context.artifacts["config"]) as fh:
            cfg = json.load(fh)
        self.weights = cfg["weights"]
        self.intercept = cfg.get("intercept", 0.0)
        self.columns = cfg["feature_columns"]
        self.members = {name: joblib.load(context.artifacts[name]) for name in self.weights}

    def predict(self, context, model_input, params=None):  # noqa: ARG002 (pyfunc signature)
        """Weighted sum of the member predictions on the configured feature columns (cast to float64)."""
        X = model_input[self.columns].astype("float64")
        return self.intercept + np.sum([w * self.members[n].predict(X) for n, w in self.weights.items()], axis=0)
