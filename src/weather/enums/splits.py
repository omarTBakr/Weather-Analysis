"""Chronological data splits."""

from __future__ import annotations

from enum import StrEnum


class Split(StrEnum):
    """Value of the `split` column of the feature table (members compare equal to their plain strings)."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
