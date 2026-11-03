"""Naive reference forecasters that the trained models are compared against."""

from __future__ import annotations

import numpy as np


class PersistenceBaseline:
    """Persistence forecast: tomorrow's temperature = today's (`temp_t`). The bar every model must clear.

    Has no parameters to fit; it only exposes `predict` so it plugs into the same loop as the real models.
    """

    def __init__(self, feature_columns: list[str]):
        """Remember the position of `temp_t` in the model input matrix."""
        self.temp_index = feature_columns.index("temp_t")

    def predict(self, X) -> np.ndarray:
        """Return the `temp_t` column of `X` as the forecast."""
        return np.asarray(X)[:, self.temp_index]
