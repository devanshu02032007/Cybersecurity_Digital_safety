import unittest

from src.advisor import LlamaAdvisor, MAX_NEW_TOKENS, MAX_RESPONSE_CHARS
from src.detection_results import network_detection_result, single_detection_result


class LlamaAdvisorTests(unittest.TestCase):
    def test_mocked_llama_and_findings_separate(self):
        calls = []

        def mocked_generator(prompt, max_new_tokens):
            calls.append((prompt, max_new_tokens))
            return (
                "Assessment: Treat this message cautiously.\n"
                "Evidence: The detector labeled it spam.\n"
                "Explanation: The score is a model estimate.\n"
                "Recommended actions: Verify through a trusted channel.\n"
                "Limitations: The detector can be wrong."
            )

        result = LlamaAdvisor(generator=mocked_generator).explain({
            "component": "SMS",
            "classification": "spam",
            "scores": {"ham": 0.1, "spam": 0.9},
            "scanned_text": "SECRET_SCANNED_CONTENT",
        })

        self.assertIn("Assessment:", result)
        self.assertIn("Recommended actions:", result)
        self.assertNotIn("SECRET_SCANNED_CONTENT", calls[0][0])
        self.assertEqual(calls[0][1], MAX_NEW_TOKENS)
        self.assertIn('"classification":"spam"', calls[0][0])

    def test_fallback_if_llama_fails(self):
        def broken_generator(prompt, max_new_tokens):
            raise RuntimeError("runner unavailable")

        result = LlamaAdvisor(generator=broken_generator).explain({
            "component": "URL",
            "classification": "phishing",
            "scores": {"phishing": 0.97},
        })

        self.assertIn("Assessment: The URL scanner marked this as phishing", result)
        self.assertIn("Treat it with caution", result)
        self.assertNotIn("General safety advice (AI generated", result)

    def test_empty_mock_output_uses_fallback(self):
        result = LlamaAdvisor(generator=lambda prompt, max_new_tokens: " ").explain({
            "component": "Network",
            "classification": "benign",
        })

        self.assertIsInstance(result, str)
        self.assertIn("does not prove the item is safe", result)

    def test_detector_result_helpers_preserve_labels_and_scores(self):
        class Classifier:
            classes_ = ["ham", "spam"]

        class FakeModel:
            named_steps = {"classifier": Classifier()}

            def predict(self, values):
                return ["spam"]

            def predict_proba(self, values):
                return [[0.15, 0.85]]

        sms = single_detection_result(FakeModel(), "message", "SMS")
        url = single_detection_result(FakeModel(), "url", "URL")
        network = network_detection_result(["benign", "suspicious", "suspicious"])

        self.assertEqual(sms["classification"], "spam")
        self.assertEqual(sms["scores"]["spam"], 0.85)
        self.assertEqual(url["classification"], "spam")
        self.assertEqual(network["classification"], "suspicious")
        self.assertEqual(network["evidence"]["rows_analyzed"], 3)
        self.assertEqual(network["scores"]["suspicious"], 2 / 3)

    def test_llama_response_is_bounded(self):
        result = LlamaAdvisor(
            generator=lambda prompt, max_new_tokens: "x" * (MAX_RESPONSE_CHARS + 10)
        ).explain({"component": "SMS", "classification": "ham"})
        self.assertLessEqual(len(result), MAX_RESPONSE_CHARS)


if __name__ == "__main__":
    unittest.main()
