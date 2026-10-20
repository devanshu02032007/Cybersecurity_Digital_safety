from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    dataset_files = list(
        DATA_DIR.rglob("SMSSpamCollection")
    )

    if not dataset_files:
        raise FileNotFoundError(
            "SMS dataset not found. Extract the UCI dataset "
            "inside data/raw/."
        )

    df = pd.read_csv(
        dataset_files[0],
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    df = df.dropna()
    df = df.drop_duplicates(
        subset=["label", "message"]
    )

    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str)

    df = df[df["label"].isin(["ham", "spam"])]

    X = df["message"]
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
                ngram_range=(1, 2),
                min_df=2,
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

    print("Training SMS classifier...")
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
        labels=["ham", "spam"],
    )

    print("\nClassification report:\n")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    joblib.dump(
        model,
        MODEL_DIR / "sms_detector.joblib",
    )

    (REPORT_DIR / "sms_report.txt").write_text(
        "Classification report:\n"
        + report
        + "\nConfusion matrix:\n"
        + str(matrix),
        encoding="utf-8",
    )

    print("\nSaved model:")
    print(MODEL_DIR / "sms_detector.joblib")

    print("\nSaved report:")
    print(REPORT_DIR / "sms_report.txt")


if __name__ == "__main__":
    main()
