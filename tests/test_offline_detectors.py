import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from src.advisor import LlamaAdvisor
from src.detection_results import network_detection_result, single_detection_result
from src.model_loader import load_model


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


class OfflineDetectorTests(unittest.TestCase):
    def setUp(self):
        self.block_network = patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("scanner attempted an external connection"),
        )
        self.block_network.start()
        self.addCleanup(self.block_network.stop)

    def test_existing_sms_model_predicts_locally(self):
        model = joblib.load(MODELS / "sms_detector.joblib")
        result = single_detection_result(model, "Please claim your prize now", "SMS")
        self.assertIn(result["classification"], {"ham", "spam"})
        self.assertTrue(result["scores"])

    def test_existing_url_model_predicts_locally_without_visiting_url(self):
        model = joblib.load(MODELS / "url_detector.joblib")
        result = single_detection_result(model, "https://example.invalid/login", "URL")
        self.assertIn(result["classification"], {"benign", "phishing"})

    def test_existing_network_model_preserves_saved_feature_order(self):
        artifact = joblib.load(MODELS / "network_detector.joblib")
        columns = artifact["feature_columns"]
        features = pd.DataFrame([[0.0] * len(columns)], columns=columns)
        prediction = artifact["model"].predict(features)
        summary = network_detection_result(prediction)
        self.assertEqual(summary["evidence"]["rows_analyzed"], 1)

    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Model file is missing"):
                load_model("missing.joblib", Path(tmp))

    def test_corrupt_model_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.joblib"
            broken.write_bytes(b"not a joblib artifact")
            with self.assertRaisesRegex(RuntimeError, "unreadable or corrupted"):
                load_model("broken.joblib", Path(tmp))

    def test_llama_unavailable_uses_deterministic_fallback_with_status(self):
        advisor = LlamaAdvisor(generator=lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError()))
        result = {"component": "SMS", "classification": "spam", "scores": {"spam": 0.9}}
        first = advisor.explain_with_status(result)
        second = advisor.explain_with_status(result)
        self.assertEqual(first, second)
        self.assertFalse(first[1])
        self.assertEqual(first[2], "ImportError")
        self.assertIn("Assessment:", first[0])


if __name__ == "__main__":
    unittest.main()
