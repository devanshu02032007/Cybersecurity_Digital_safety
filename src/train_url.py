from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/raw/urls/urls.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Create the labelled dataset at {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = {"url", "label"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain url and label columns."
        )

    df = df.dropna(subset=["url", "label"]).copy()

    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["url"] != ""]
    df = df[df["label"].isin(["benign", "phishing"])]
    df = df.drop_duplicates(subset=["url"])

    if df["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain benign and phishing examples."
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two records."
        )

    X = df["url"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=1,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("Training phishing URL classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["benign", "phishing"],
    )

    print("\nClassification report:\n")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    joblib.dump(
        model,
        MODEL_DIR / "url_detector.joblib",
    )

    (REPORT_DIR / "url_report.txt").write_text(
        report + "\n\nConfusion matrix:\n" + str(matrix),
        encoding="utf-8",
    )

    print("\nSaved models/url_detector.joblib")


if __name__ == "__main__":
    main()
