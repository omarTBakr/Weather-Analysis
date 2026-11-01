"""Stage `download`: fetch GlobalWeatherRepository.csv from Kaggle into data/raw/.

Uses kagglehub's single-file download (the full-archive endpoint has been observed to return
an empty zip for this dataset), with a direct Kaggle REST fallback using ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from weather.utils.config import ensure_dirs, get_settings

S = get_settings()
P = S.paths


def _rest_download(dataset: str, filename: str, dest: Path) -> None:
    import base64
    import urllib.request

    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    token = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{filename}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def main() -> None:
    ensure_dirs()
    p = S.data
    dest = S.raw_csv
    try:
        import kagglehub

        src = Path(kagglehub.dataset_download(p.kaggle_dataset, path=p.kaggle_file))
        if src.is_dir():
            src = src / p.kaggle_file
        if not src.exists() or src.stat().st_size < 1_000_000:
            raise RuntimeError(f"kagglehub returned an unusable file: {src}")
        shutil.copy(src, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"kagglehub download failed ({exc}); falling back to Kaggle REST API")
        _rest_download(p.kaggle_dataset, p.kaggle_file, dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
