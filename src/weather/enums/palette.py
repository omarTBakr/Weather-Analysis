"""Colour tokens shared by every figure."""

from __future__ import annotations

import matplotlib.colors

from weather.enums.geography import Geography


class Palette:
    """Validated categorical order (fixed, never cycled), ink/chrome tokens and colour maps."""

    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    INK_2 = "#52514e"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    AXIS = "#c3c2b7"
    CRITICAL = "#d03b3b"
    SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    )
    DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a5a4", "#e34948", "#a32727"]
    )
    CONTINENT_COLORS = dict(zip(Geography.CONTINENT_ORDER, SERIES, strict=False))
