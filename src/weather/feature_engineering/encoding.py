"""Small, reusable encoders shared by the panel and global-series feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather.enums import Geography


def cyclical_encode(values, period: float):
    """Map a periodic quantity (hour of day, day of year, ...) onto the unit circle.

    Returns `(sin, cos)` so that the two ends of the cycle (e.g. 23h and 0h) end up close together,
    which a raw integer encoding cannot express.
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def encode_continent(continent: pd.Series) -> pd.Series:
    """Integer-encode continent names using `Geography.CONTINENT_CODES` (-1 for anything unmapped)."""
    return continent.map(Geography.CONTINENT_CODES).fillna(-1).astype(int)
