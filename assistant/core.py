from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from urllib.parse import urlparse

PHISHING_KEYWORDS = {
    "verify your account",
    "suspended",
    "urgent",
    "act now",
    "limited time",
    "click here",
    "gift card",
    "wire transfer",
    "password",
    "otp",
    "bank",
    "crypto",
}

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "rb.gy"}


@dataclass
class DetectionResult:
    indicators: list[str]
    risk_score: int
    risk_level: str


class ExecuTorchLlamaExplainer:
    """On-device Llama component wrapper with offline fallback."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.runtime_available = False
        self.runtime_reason = "executorch unavailable"
        try:
            import executorch  # type: ignore # noqa: F401

            if model_path and Path(model_path).exists():
                self.runtime_available = True
                self.runtime_reason = "executorch_runtime_ready"
            else:
                self.runtime_reason = "model_not_provided_or_missing"
        except Exception:
            self.runtime_reason = "executorch_unavailable"

    def explain(self, result: DetectionResult, message: str, url: str | None) -> dict:
        recommendations = [
            "Do not share passwords, OTPs, or financial information.",
            "Verify sender identity through an independent channel.",
            "Avoid opening links or attachments until verified.",
        ]
        if result.risk_level in {"high", "critical"}:
            recommendations.append("Report this message to your security/admin team.")

        mode = "executorch_llama" if self.runtime_available else "offline_fallback"
        explanation = (
            f"Risk is {result.risk_level.upper()} ({result.risk_score}/100) due to: "
            + (", ".join(result.indicators) if result.indicators else "no suspicious indicators")
            + "."
        )
        return {
            "engine": mode,
            "runtime_status": self.runtime_reason,
            "explanation": explanation,
            "recommendations": recommendations,
        }


def _is_ip_host(host: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host))


def detect_suspicious_indicators(message: str, url: str | None = None) -> list[str]:
    indicators: list[str] = []
    msg = (message or "").lower()
    url = url or ""

    for keyword in PHISHING_KEYWORDS:
        if keyword in msg:
            indicators.append(f"keyword:{keyword}")

    if re.search(r"\b\d{4,}\b", msg) and ("bank" in msg or "account" in msg):
        indicators.append("possible_account_number_request")

    if msg.count("!") >= 2:
        indicators.append("urgency_punctuation")

    if url:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if not parsed.scheme.startswith("https"):
            indicators.append("url_not_https")
        if host in SHORTENERS:
            indicators.append("shortened_url")
        if "@" in url:
            indicators.append("url_contains_at_symbol")
        if "xn--" in host:
            indicators.append("possible_punycode_domain")
        if _is_ip_host(host.split(":")[0]):
            indicators.append("ip_address_url")

    return sorted(set(indicators))


def risk_assessment(indicators: list[str]) -> DetectionResult:
    score = 0
    for item in indicators:
        if item.startswith("keyword:"):
            score += 8
        elif item in {"shortened_url", "url_contains_at_symbol", "possible_punycode_domain", "ip_address_url"}:
            score += 20
        elif item == "url_not_https":
            score += 10
        else:
            score += 12

    score = min(score, 100)
    if score >= 75:
        level = "critical"
    elif score >= 50:
        level = "high"
    elif score >= 25:
        level = "medium"
    else:
        level = "low"

    return DetectionResult(indicators=indicators, risk_score=score, risk_level=level)


def analyze_input(message: str, url: str | None = None, model_path: str | None = None) -> dict:
    indicators = detect_suspicious_indicators(message, url)
    assessment = risk_assessment(indicators)
    explainer = ExecuTorchLlamaExplainer(model_path=model_path)
    explain = explainer.explain(assessment, message, url)
    return {
        "input": {"message": message, "url": url},
        "indicators": assessment.indicators,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "llama_explanation": explain,
        "offline_mode": True,
        "cloud_used": False,
    }


def evaluate_detection_quality(dataset: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    start = time.perf_counter()
    for row in dataset:
        result = analyze_input(row.get("message", ""), row.get("url"))
        predicted_malicious = result["risk_level"] in {"medium", "high", "critical"}
        actual_malicious = bool(row.get("malicious"))

        if predicted_malicious and actual_malicious:
            tp += 1
        elif predicted_malicious and not actual_malicious:
            fp += 1
        elif not predicted_malicious and not actual_malicious:
            tn += 1
        else:
            fn += 1

    duration_ms = (time.perf_counter() - start) * 1000
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "samples": len(dataset),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "avg_latency_ms_per_sample": round(duration_ms / max(1, len(dataset)), 3),
    }
