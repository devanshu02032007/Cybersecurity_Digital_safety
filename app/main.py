from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.advisor import explain_detection
from src.detection_results import (
    network_detection_result,
    single_detection_result,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="On-Device Cybersecurity AI",
    page_icon="🛡️",
    layout="centered",
)

st.title("On-Device Cybersecurity AI")
st.caption(
    "Local SMS, phishing URL, and network-flow analysis"
)


def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    return joblib.load(path)


@st.cache_resource
def get_sms_model():
    return load_model("sms_detector.joblib")


@st.cache_resource
def get_url_model():
    return load_model("url_detector.joblib")


@st.cache_resource
def get_network_model():
    return load_model("network_detector.joblib")


def show_prediction(model, text, component):
    detection = single_detection_result(model, text, component)

    st.subheader("Detection result")
    st.write(f"**Component:** {component}")
    st.write(f"**Classification:** {detection['classification']}")
    st.json({
        label: round(score, 4)
        for label, score in detection["scores"].items()
    })

    st.caption(
        "These are model scores. They are not proof that a threat "
        "is present or absent."
    )

    show_advisory(detection)


def show_advisory(detection_result):
    with st.expander("Plain-language explanation and safety advice", expanded=True):
        try:
            advice = explain_detection(detection_result)
        except Exception:
            # The advisor has its own fallback, but keep the UI resilient too.
            advice = (
                "Assessment: Advice is temporarily unavailable.\n\n"
                "Recommended actions: Treat suspicious items cautiously and "
                "verify unexpected requests through a trusted channel."
            )
        st.write(advice)
        st.caption(
            "Detector label and scores above are the scan evidence. This advice "
            "is a separate explanation and does not replace the detector result."
        )


tab_sms, tab_url, tab_network = st.tabs([
    "SMS Scanner",
    "URL Scanner",
    "Network Scanner",
])


with tab_sms:
    st.header("SMS scam detection")

    message = st.text_area(
        "Enter the SMS message",
        max_chars=10000,
        placeholder="Paste a message to analyse...",
    )

    if st.button("Analyse SMS"):
        if not message.strip():
            st.warning("Enter an SMS message first.")
        else:
            model = get_sms_model()

            if model is None:
                st.error(
                    "SMS model missing. Run "
                    "python src/train_sms.py first."
                )
            else:
                show_prediction(model, message, "SMS")


with tab_url:
    st.header("Phishing URL detection")

    url = st.text_input(
        "Enter the URL",
        max_chars=4096,
        placeholder="https://example.com",
    )

    if st.button("Analyse URL"):
        if not url.strip():
            st.warning("Enter a URL first.")
        else:
            model = get_url_model()

            if model is None:
                st.error(
                    "URL model missing. Run "
                    "python src/train_url.py first."
                )
            elif not url.lower().startswith(("http://", "https://")):
                st.warning(
                    "Enter a URL beginning with http:// or https://."
                )
            else:
                # No HTTP request is made to the submitted URL.
                show_prediction(model, url, "URL")


with tab_network:
    st.header("Network intrusion detection")

    st.write(
        "Upload a labelled network-flow CSV or a CSV containing "
        "the same feature columns used during training."
    )

    uploaded = st.file_uploader(
        "Choose network-flow CSV",
        type=["csv"],
    )

    if st.button("Analyse network flows"):
        if uploaded is None:
            st.warning("Upload a CSV first.")
        else:
            artifact = get_network_model()

            if artifact is None:
                st.error(
                    "Network model missing. Run "
                    "python src/train_network.py first."
                )
            else:
                try:
                    df = pd.read_csv(uploaded)

                    model = artifact["model"]
                    feature_columns = artifact["feature_columns"]

                    missing = [
                        column for column in feature_columns
                        if column not in df.columns
                    ]

                    if missing:
                        st.error(
                            "The CSV is missing required model features: "
                            + ", ".join(missing[:20])
                        )
                    else:
                        features = df[feature_columns].apply(
                            pd.to_numeric,
                            errors="coerce",
                        )

                        features = features.replace(
                            [float("inf"), float("-inf")],
                            float("nan"),
                        )

                        predictions = model.predict(features)

                        result = df.copy()
                        result["prediction"] = predictions

                        st.subheader("Network analysis results")
                        st.dataframe(result.head(100))

                        st.write(
                            "Rows analysed:",
                            len(result),
                        )

                        st.write(
                            "Suspicious rows:",
                            int((result["prediction"] == "suspicious").sum()),
                        )

                        network_detection = network_detection_result(predictions)
                        show_advisory(network_detection)

                        st.download_button(
                            "Download analysis results",
                            data=result.to_csv(index=False).encode("utf-8"),
                            file_name="network_analysis.csv",
                            mime="text/csv",
                        )

                except Exception as exc:
                    # Do not render exception details that might contain row data.
                    st.error("Could not analyse the CSV. Check its format and feature columns.")


st.divider()

st.caption(
    "Prototype: results are advisory, not a guarantee of security. "
    "A benign prediction does not prove that a message, URL, or "
    "network flow is safe."
)
