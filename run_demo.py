from __future__ import annotations

import json
from pathlib import Path

from assistant.core import analyze_input, evaluate_detection_quality


def main() -> None:
    base = Path(__file__).parent
    dataset = json.loads((base / "demo" / "sample_cases.json").read_text())

    print("=== Offline Demo ===")
    for row in dataset:
        result = analyze_input(row["message"], row.get("url"))
        print(f"Input: {row['message'][:60]}...")
        print(f"Risk: {result['risk_level']} ({result['risk_score']}) | Offline: {result['offline_mode']}")

    metrics = evaluate_detection_quality(dataset)
    print("\n=== Detection Quality & Performance ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
