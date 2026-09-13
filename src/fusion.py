from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "paderborn_trust_features.csv"
)

FAULT_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "paderborn_motor_model.joblib"
)

TRUST_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "paderborn_trust_model.joblib"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD MODELS
# ==================================================

def load_models():

    print("\nLoading Motor Condition Model...")

    fault_model_data = joblib.load(
        FAULT_MODEL_PATH
    )

    print("Motor model loaded.")

    print(
        "Loading Sensor Trust Model..."
    )

    trust_model_data = joblib.load(
        TRUST_MODEL_PATH
    )

    print("Trust model loaded.")

    return (
        fault_model_data,
        trust_model_data
    )


# ==================================================
# UNPACK MODEL
# ==================================================

def unpack_model(model_data):

    if isinstance(model_data, dict):

        model = model_data.get(
            "model",
            model_data
        )

        feature_names = model_data.get(
            "feature_names",
            None
        )

    else:

        model = model_data

        feature_names = None

    return (
        model,
        feature_names
    )


# ==================================================
# RECOVER FEATURE NAMES
# ==================================================

def get_model_feature_names(
    model,
    saved_feature_names=None
):

    # ----------------------------------------------
    # Use explicitly saved feature names
    # ----------------------------------------------

    if saved_feature_names is not None:

        return list(
            saved_feature_names
        )

    # ----------------------------------------------
    # Model directly stores feature names
    # ----------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    # ----------------------------------------------
    # Search inside sklearn pipeline
    # ----------------------------------------------

    if hasattr(
        model,
        "steps"
    ):

        for _, step in model.steps:

            if hasattr(
                step,
                "feature_names_in_"
            ):

                return list(
                    step.feature_names_in_
                )

    # ----------------------------------------------
    # No feature names found
    # ----------------------------------------------

    return None


# ==================================================
# PREPARE MODEL FEATURES
# ==================================================

def prepare_features(
    df,
    model,
    saved_feature_names=None
):

    feature_names = (
        get_model_feature_names(
            model,
            saved_feature_names
        )
    )

    # ----------------------------------------------
    # Exact model feature alignment
    # ----------------------------------------------

    if feature_names is not None:

        X = df.reindex(
            columns=feature_names,
            fill_value=0
        )

        return X

    # ----------------------------------------------
    # Fallback
    # ----------------------------------------------

    excluded = [

        "source_file",

        "condition_label",

        "trust_label",

        "attack_type",
    ]

    X = df.drop(
        columns=excluded,
        errors="ignore"
    )

    X = X.select_dtypes(
        include=[np.number]
    )

    return X


# ==================================================
# FUSION DECISION
# ==================================================

def make_fusion_decision(
    fault_prediction,
    fault_confidence,
    trust_prediction,
    trust_confidence
):

    # ----------------------------------------------
    # TRUSTED SENSOR DATA
    # ----------------------------------------------

    if trust_prediction == "trusted":

        if fault_confidence >= 0.80:

            final_status = (
                "HIGH_CONFIDENCE"
            )

            decision = (
                "ACCEPT_MOTOR_PREDICTION"
            )

        else:

            final_status = (
                "MEDIUM_CONFIDENCE"
            )

            decision = (
                "ACCEPT_WITH_CAUTION"
            )

    # ----------------------------------------------
    # TAMPERED SENSOR DATA
    # ----------------------------------------------

    else:

        if trust_confidence >= 0.80:

            final_status = (
                "UNTRUSTWORTHY"
            )

            decision = (
                "REJECT_SENSOR_DATA"
            )

        else:

            final_status = (
                "SUSPICIOUS"
            )

            decision = (
                "REQUIRE_FURTHER_CHECK"
            )

    return (
        final_status,
        decision
    )


# ==================================================
# RUN FUSION
# ==================================================

def run_fusion():

    print(
        "\n" + "=" * 65
    )

    print(
        "TRUSTWORTHY IoT MOTOR AI "
        "- FUSION ENGINE"
    )

    print(
        "=" * 65
    )

    # ----------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------

    df = pd.read_csv(
        FEATURE_PATH
    )

    print(
        f"\nLoaded dataset: "
        f"{df.shape}"
    )

    # ----------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------

    (
        fault_model_data,
        trust_model_data
    ) = load_models()

    (
        fault_model,
        fault_saved_features
    ) = unpack_model(
        fault_model_data
    )

    (
        trust_model,
        trust_saved_features
    ) = unpack_model(
        trust_model_data
    )

    # ----------------------------------------------
    # PREPARE FAULT MODEL FEATURES
    # ----------------------------------------------

    X_fault = prepare_features(

        df,

        fault_model,

        fault_saved_features
    )

    # ----------------------------------------------
    # PREPARE TRUST MODEL FEATURES
    # ----------------------------------------------

    X_trust = prepare_features(

        df,

        trust_model,

        trust_saved_features
    )

    print(
        f"\nFault model features: "
        f"{X_fault.shape[1]}"
    )

    print(
        f"Trust model features: "
        f"{X_trust.shape[1]}"
    )

    # ----------------------------------------------
    # MOTOR CONDITION PREDICTION
    # ----------------------------------------------

    print(
        "\nRunning Motor Condition AI..."
    )

    fault_predictions = (
        fault_model.predict(
            X_fault
        )
    )

    fault_probabilities = (
        fault_model.predict_proba(
            X_fault
        )
    )

    fault_confidences = (
        fault_probabilities.max(
            axis=1
        )
    )

    print(
        "Motor predictions complete."
    )

    # ----------------------------------------------
    # SENSOR TRUST PREDICTION
    # ----------------------------------------------

    print(
        "\nRunning Sensor Trust AI..."
    )

    trust_predictions = (
        trust_model.predict(
            X_trust
        )
    )

    trust_probabilities = (
        trust_model.predict_proba(
            X_trust
        )
    )

    trust_confidences = (
        trust_probabilities.max(
            axis=1
        )
    )

    print(
        "Trust predictions complete."
    )

    # ----------------------------------------------
    # FUSION DECISIONS
    # ----------------------------------------------

    final_statuses = []

    final_decisions = []

    for (
        fault_prediction,
        fault_confidence,
        trust_prediction,
        trust_confidence
    ) in zip(

        fault_predictions,

        fault_confidences,

        trust_predictions,

        trust_confidences
    ):

        (
            status,
            decision
        ) = make_fusion_decision(

            fault_prediction,

            fault_confidence,

            trust_prediction,

            trust_confidence
        )

        final_statuses.append(
            status
        )

        final_decisions.append(
            decision
        )

    # ----------------------------------------------
    # CREATE RESULTS
    # ----------------------------------------------

    results = pd.DataFrame({

        "source_file":
            df["source_file"],

        "true_condition":
            df["condition_label"],

        "motor_prediction":
            fault_predictions,

        "motor_confidence":
            fault_confidences,

        "true_trust_label":
            df["trust_label"],

        "trust_prediction":
            trust_predictions,

        "trust_confidence":
            trust_confidences,

        "final_status":
            final_statuses,

        "final_decision":
            final_decisions,
    })

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    output_path = (
        RESULTS_DIR
        / "fusion_results.csv"
    )

    results.to_csv(

        output_path,

        index=False
    )

    # ----------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "FUSION RESULTS SUMMARY"
    )

    print(
        "=" * 65
    )

    print(
        "\nFinal Status Distribution:\n"
    )

    print(
        results[
            "final_status"
        ]
        .value_counts()
    )

    print(
        "\nFinal Decision Distribution:\n"
    )

    print(
        results[
            "final_decision"
        ]
        .value_counts()
    )

    print(
        "\nFirst 10 Fusion Decisions:\n"
    )

    print(
        results
        .head(10)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------
    # SAVE SUMMARY JSON
    # ----------------------------------------------

    summary = {

        "total_samples":
            int(
                len(results)
            ),

        "fault_model_features":
            int(
                X_fault.shape[1]
            ),

        "trust_model_features":
            int(
                X_trust.shape[1]
            ),

        "status_distribution":

            results[
                "final_status"
            ]
            .value_counts()
            .to_dict(),

        "decision_distribution":

            results[
                "final_decision"
            ]
            .value_counts()
            .to_dict(),
    }

    summary_path = (
        RESULTS_DIR
        / "fusion_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        json.dump(

            summary,

            file,

            indent=2
        )

    print(
        "\n" + "=" * 65
    )

    print(
        "FUSION COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_path}"
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    run_fusion()