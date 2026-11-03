"""Chronological train / validation / test split of the feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather.enums import Split


def assign_time_split(features: pd.DataFrame, test_days: int, val_days: int) -> pd.DataFrame:
    """Label each row `train`, `val` or `test` by its `target_date` (in place; also returned).

    The last `test_days` target dates form the test window, the `val_days` before them the validation window,
    and everything earlier is training data, so no model ever sees the future of the window it is scored on.
    """
    last = features["target_date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    features["split"] = np.select(
        [features["target_date"] >= test_start, features["target_date"] >= val_start],
        [Split.TEST.value, Split.VAL.value],
        Split.TRAIN.value,
    )
    return features
