from pathlib import Path
from urllib.parse import urlsplit

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/url_detector.joblib"

if not MODEL.exists():
    raise FileNotFoundError("Train the URL model first.")

model = joblib.load(MODEL)


def inspect_url(url):
    # Parse locally. Do not visit the URL.
    parsed = urlsplit(url)

    if parsed.scheme not in ("http", "https"):
        return {
            "label": "invalid_input",
            "message": "Enter a URL beginning with http:// or https://",
        }

    if not parsed.hostname:
        return {
            "label": "invalid_input",
            "message": "The URL has no valid hostname.",
        }

    label = model.predict([url])[0]
    probabilities = model.predict_proba([url])[0]

    classes = model.named_steps["classifier"].classes_
    scores = dict(zip(classes, probabilities))

    return {
        "label": label,
        "scores": {
            name: round(float(score), 4)
            for name, score in scores.items()
        },
    }


test_url = input("Enter a URL to analyse: ").strip()

if len(test_url) > 4096:
    print("URL is too long.")
else:
    print(inspect_url(test_url))
