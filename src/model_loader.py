"""Load preserved detector artifacts from this project's local models folder."""

from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def load_model(filename: str, model_dir: Path = MODEL_DIR):
    path = Path(model_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model file is missing: {path}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Model file is unreadable or corrupted: {path}") from exc
