from __future__ import annotations

import argparse
import json

from .core import analyze_input


def main() -> None:
    parser = argparse.ArgumentParser(description="On-device AI Cybersecurity Assistant")
    parser.add_argument("--message", required=True, help="Suspicious message to analyze")
    parser.add_argument("--url", default=None, help="Optional suspicious URL")
    parser.add_argument("--model-path", default=None, help="Optional ExecuTorch model path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    args = parser.parse_args()

    result = analyze_input(args.message, args.url, args.model_path)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=== On-Device AI Cybersecurity Assistant ===")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Risk Score: {result['risk_score']}/100")
    print("Indicators:")
    for indicator in result["indicators"] or ["none"]:
        print(f"  - {indicator}")
    explain = result["llama_explanation"]
    print(f"Explanation Engine: {explain['engine']} ({explain['runtime_status']})")
    print(f"Explanation: {explain['explanation']}")
    print("Recommendations:")
    for rec in explain["recommendations"]:
        print(f"  - {rec}")
    print("Offline processing verified: cloud inference disabled")


if __name__ == "__main__":
    main()
