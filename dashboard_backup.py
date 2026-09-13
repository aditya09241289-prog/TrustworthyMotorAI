import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.fusion import fuse_predictions
from src.explain import explain_sample


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Trustworthy AI for IoT-Connected Electric Motors",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("⚙️ Trustworthy AI for IoT-Connected Electric Motors")

st.caption(
    "Distinguishing genuine physical motor faults from "
    "untrustworthy or potentially manipulated IoT sensor data"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "vibration",
    "temperature",
    "current",
    "rpm"
]


# ============================================================
# CHECK THAT MODELS EXIST
# ============================================================

fault_model_path = ROOT / "models" / "fault_model.joblib"
trust_model_path = ROOT / "models" / "trust_model.joblib"

if not fault_model_path.exists() or not trust_model_path.exists():

    st.warning(
        "Trained models were not found. "
        "Please run the following command first:"
    )

    st.code(
        "python run_pipeline.py"
    )

    st.stop()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

fault_model = joblib.load(fault_model_path)

trust_model = joblib.load(trust_model_path)


# ============================================================
# SENSOR INPUT SECTION
# ============================================================

st.subheader("IoT Motor Sensor Readings")

col1, col2, col3, col4 = st.columns(4)


with col1:
    vibration = st.number_input(
        "Vibration",
        value=1.0
    )


with col2:
    temperature = st.number_input(
        "Temperature",
        value=40.0
    )


with col3:
    current = st.number_input(
        "Current",
        value=10.0
    )


with col4:
    rpm = st.number_input(
        "RPM",
        value=1500.0
    )


# ============================================================
# ANALYSIS BUTTON
# ============================================================

if st.button("Analyze", type="primary"):

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    X = pd.DataFrame(
        [
            [
                vibration,
                temperature,
                current,
                rpm
            ]
        ],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # MOTOR FAULT PREDICTION
    # --------------------------------------------------------

    fault_prediction = fault_model.predict(X)[0]

    fault_probabilities = dict(
        zip(
            fault_model.classes_,
            fault_model.predict_proba(X)[0]
        )
    )


    # --------------------------------------------------------
    # SENSOR TRUST PREDICTION
    # --------------------------------------------------------

    trust_prediction = trust_model.predict(X)[0]

    trust_probabilities = dict(
        zip(
            trust_model.classes_,
            trust_model.predict_proba(X)[0]
        )
    )


    # --------------------------------------------------------
    # CYBER-PHYSICAL FUSION DECISION
    # --------------------------------------------------------

    output = fuse_predictions(
        fault_prediction,
        fault_probabilities,
        trust_prediction,
        trust_probabilities
    )


    # ========================================================
    # FINAL SYSTEM DECISION
    # ========================================================

    st.divider()

    st.subheader("Final Trustworthy AI Decision")

    st.success(output["decision"])

    st.write(
        output["reason"]
    )


    # ========================================================
    # INDIVIDUAL AI MODEL RESULTS
    # ========================================================

    st.divider()

    left, right = st.columns(2)


    # --------------------------------------------------------
    # MOTOR MODEL RESULTS
    # --------------------------------------------------------

    with left:

        st.subheader("⚙️ Motor Fault Prediction")

        st.write(
            f"**Predicted motor condition:** {fault_prediction}"
        )

        st.write(
            "**Fault probability distribution:**"
        )

        formatted_fault_probabilities = {
            key: round(float(value), 3)
            for key, value in fault_probabilities.items()
        }

        st.json(
            formatted_fault_probabilities
        )


    # --------------------------------------------------------
    # IOT TRUST MODEL RESULTS
    # --------------------------------------------------------

    with right:

        st.subheader("🛡️ IoT Sensor Trust Prediction")

        st.write(
            f"**Sensor evidence status:** {trust_prediction}"
        )

        st.write(
            "**Trust probability distribution:**"
        )

        formatted_trust_probabilities = {
            key: round(float(value), 3)
            for key, value in trust_probabilities.items()
        }

        st.json(
            formatted_trust_probabilities
        )


    # ========================================================
    # EXPLAINABILITY SECTION
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Transparent Diagnostic Explanation"
    )

    explanation = explain_sample(
        X.iloc[0].to_dict()
    )

    explanation_df = pd.DataFrame(
        explanation
    )

    st.dataframe(
        explanation_df,
        use_container_width=True
    )