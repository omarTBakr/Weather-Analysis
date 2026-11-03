"""Stage `download`: fetch GlobalWeatherRepository.csv from Kaggle into data/raw/.

Uses kagglehub's single-file download (the full-archive endpoint has been observed to return
an empty zip for this dataset), with a direct Kaggle REST fallback using ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from weather.enums import KaggleSource
from weather.utils.config import Settings, ensure_dirs, get_settings


class KaggleDownloader:
    """Downloads the raw dataset CSV to `settings.raw_csv`, trying kagglehub first and the REST API second."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.dataset = self.settings.data.kaggle_dataset
        self.filename = self.settings.data.kaggle_file
        self.dest = self.settings.raw_csv

    def download_via_kagglehub(self) -> None:
        """Single-file kagglehub download, rejected if the file is missing or implausibly small."""
        import kagglehub

        src = Path(kagglehub.dataset_download(self.dataset, path=self.filename))
        if src.is_dir():
            src = src / self.filename
        if not src.exists() or src.stat().st_size < KaggleSource.MIN_CSV_BYTES:
            raise RuntimeError(f"kagglehub returned an unusable file: {src}")
        shutil.copy(src, self.dest)

    def download_via_rest(self) -> None:
        """Fallback: stream the file from the Kaggle REST API using `~/.kaggle/kaggle.json` credentials."""
        import base64
        import urllib.request

        creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
        token = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
        url = KaggleSource.REST_DOWNLOAD_URL.format(dataset=self.dataset, filename=self.filename)
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
        with urllib.request.urlopen(req) as resp, open(self.dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)

    def run(self) -> None:
        """Execute the stage: kagglehub first, REST fallback on any failure."""
        ensure_dirs(self.settings)
        try:
            self.download_via_kagglehub()
        except Exception as exc:  # noqa: BLE001
            print(f"kagglehub download failed ({exc}); falling back to Kaggle REST API")
            self.download_via_rest()
        print(f"saved {self.dest} ({self.dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    """DVC entry point."""
    KaggleDownloader().run()


if __name__ == "__main__":
    main()
