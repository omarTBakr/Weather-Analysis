"""Base class shared by every analysis of the `analyses` stage (labels live in `weather.enums`)."""

from __future__ import annotations

from weather.utils.config import Settings, get_settings


class InsightAnalysis:
    """One analysis of the `analyses` stage: holds the settings, the paths and the shared figure directory."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.figure_dir = self.paths.figures / "analyses"  # every figure of the `analyses` stage
