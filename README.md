# Cybersecurity_Digital_safety

On-device AI cybersecurity assistant to detect phishing/scam messages and suspicious URLs, score risk, and provide offline safety explanations on Arm-class devices.

## Features

- Submit suspicious message text and optional URL
- Detection engine for phishing/scam indicators
- Risk score (0-100) and risk level (`low`, `medium`, `high`, `critical`)
- On-device Llama component interface through an ExecuTorch wrapper
  - Uses local/offline fallback explanations when ExecuTorch runtime/model is unavailable
- Offline-first execution (`cloud_used=false` in results)

## Quick start

Run CLI analysis:

```bash
python -m assistant.cli --message "URGENT: verify your account now" --url "http://192.168.1.10/login"
```

JSON output:

```bash
python -m assistant.cli --message "Claim your gift card now" --url "https://bit.ly/reward" --json
```

Optional model path for ExecuTorch:

```bash
python -m assistant.cli --message "..." --model-path /path/to/llama.pte
```

## Offline demo

```bash
python run_demo.py
```

This runs sample phishing/benign inputs from `demo/sample_cases.json`, prints risk outputs, and reports detection quality and performance metrics.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Covered test goals:

- detection quality for malicious vs benign examples
- false positives and false negatives in metrics output
- performance sanity check for repeated local inference
- offline processing behavior (`cloud_used=false`)
