import json
import time
import unittest
from pathlib import Path

from assistant.core import analyze_input, evaluate_detection_quality


class AssistantTests(unittest.TestCase):
    def test_high_risk_detection(self):
        result = analyze_input(
            "URGENT: verify your account and send OTP now!!!",
            "http://192.168.10.5/security-check",
        )
        self.assertIn(result["risk_level"], {"high", "critical"})
        self.assertGreaterEqual(result["risk_score"], 50)

    def test_low_risk_detection(self):
        result = analyze_input(
            "Reminder: your dentist appointment is tomorrow at 9 AM.",
            "https://clinic.example.com/schedule",
        )
        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["cloud_used"], False)

    def test_quality_metrics_include_fp_fn(self):
        dataset = json.loads(
            (Path(__file__).resolve().parents[1] / "demo" / "sample_cases.json").read_text()
        )
        metrics = evaluate_detection_quality(dataset)
        self.assertIn("false_positive", metrics)
        self.assertIn("false_negative", metrics)
        self.assertGreaterEqual(metrics["precision"], 0.70)
        self.assertGreaterEqual(metrics["recall"], 0.70)

    def test_performance_reasonable(self):
        start = time.perf_counter()
        for _ in range(250):
            analyze_input(
                "Your account is suspended, click here to verify password now",
                "https://bit.ly/security-reset",
            )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 4.0)


if __name__ == "__main__":
    unittest.main()
