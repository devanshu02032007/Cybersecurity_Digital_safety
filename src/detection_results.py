"""Small, explicit result payloads passed from detectors to the advisor."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def single_detection_result(model: Any, value: str, component: str) -> dict[str, Any]:
    """Return only the original prediction label and class scores."""
    prediction = model.predict([value])[0]
    probabilities = model.predict_proba([value])[0]
    classes = model.named_steps["classifier"].classes_
    return {
        "component": component,
        "classification": str(prediction),
        "scores": {
            str(label): float(score)
            for label, score in zip(classes, probabilities)
        },
    }


def network_detection_result(predictions: Iterable[Any]) -> dict[str, Any]:
    """Summarize network labels without forwarding uploaded row data."""
    labels = [str(label) for label in predictions]
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    suspicious = counts.get("suspicious", 0)
    return {
        "component": "Network",
        "classification": "suspicious" if suspicious else "benign",
        "scores": {
            label: count / total
            for label, count in counts.items()
        } if total else {},
        "evidence": {"rows_analyzed": total, "label_counts": counts},
    }
