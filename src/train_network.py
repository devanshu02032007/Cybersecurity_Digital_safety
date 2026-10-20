from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/raw/network"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    files = list(DATA_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "Place a labelled network-flow CSV inside data/raw/network/"
        )

    # Start with one file. Expand to multiple files after validation.
    df = pd.read_csv(files[0], low_memory=False)

    # Normalise column names and whitespace.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    label_matches = [
        column for column in df.columns
        if column.lower() == "label"
    ]

    if not label_matches:
        raise ValueError(
            "Could not find a Label column. Inspect your CSV and "
            "update the script to match its label column."
        )

    label_column = label_matches[0]

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Map all non-benign traffic to suspicious.
    df["target"] = np.where(
        df[label_column].isin(["benign", "normal"]),
        "benign",
        "suspicious",
    )

    # Use numeric features only for this baseline.
    X = df.drop(
        columns=[label_column, "target"],
        errors="ignore",
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove unusable columns.
    X = X.dropna(axis=1, how="all")

    # Avoid retaining constant columns.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    y = df["target"]

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if y.nunique() != 2:
        raise ValueError(
            "The selected dataset must contain benign and attack records."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two usable records."
        )

    # Remove exact duplicate rows before splitting.
    combined = X.copy()
    combined["_target"] = y
    combined = combined.drop_duplicates()

    y = combined.pop("_target")
    X = combined

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=2,
            ),
        ),
    ])

    print("Training network classifier...")
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
        labels=["benign", "suspicious"],
    )

    print("\nClassification report:\n")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    # Store the model and its feature schema together.
    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
    }

    joblib.dump(
        artifact,
        MODEL_DIR / "network_detector.joblib",
    )

    (REPORT_DIR / "network_report.txt").write_text(
        report + "\n\nConfusion matrix:\n" + str(matrix),
        encoding="utf-8",
    )

    print("\nSaved models/network_detector.joblib")


if __name__ == "__main__":
    main()
