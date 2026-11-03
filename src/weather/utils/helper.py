from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path | str) -> Path:
    """Ensure that the given directory path exists, creating parent directories if needed."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
