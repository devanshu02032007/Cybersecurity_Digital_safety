from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]

model_path = ROOT / "models" / "sms_detector.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        "Train the model first using python src/train_sms.py"
    )

model = joblib.load(model_path)

messages = [
    "Hi, are we meeting at college tomorrow?",
    "Congratulations! You have won a prize. Claim now!",
    "Your account needs attention. Verify your details immediately.",
]

for message in messages:
    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]
    confidence = float(max(probabilities))

    print("\nMessage:", message)
    print("Prediction:", prediction)
    print(f"Model confidence: {confidence:.3f}")
