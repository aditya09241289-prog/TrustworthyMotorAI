from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "paderborn_features.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "paderborn_motor_model.joblib"
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
# EVALUATION
# ==================================================

def evaluate_model():

    print("\n" + "=" * 60)
    print("STRICTER MOTOR MODEL EVALUATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=[
            "source_file",
            "condition_label",
        ],
        errors="ignore"
    )

    X = X.select_dtypes(
        include=[np.number]
    )

    y = df["condition_label"]

    print(f"\nSamples: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {sorted(y.unique())}")

    # Load trained pipeline
    model = joblib.load(
        MODEL_PATH
    )

    # ------------------------------------------------
    # 5-Fold Stratified Cross Validation
    # ------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print(
        "\nRunning 5-fold stratified "
        "cross-validation..."
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1 Score: "
        f"{macro_f1:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    # ------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------

    classes = sorted(
        y.unique()
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=classes
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    )

    confusion_path = (
        RESULTS_DIR
        / "paderborn_cv_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # ------------------------------------------------
    # Save Metrics
    # ------------------------------------------------

    metrics = {

        "validation_method":
            "5-fold Stratified Cross Validation",

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "macro_f1":
            float(macro_f1),

        "samples":
            int(len(df)),

        "features":
            int(X.shape[1]),

        "classes":
            classes,
    }

    metrics_path = (
        RESULTS_DIR
        / "paderborn_cv_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"\nMetrics saved to:\n"
        f"{metrics_path}"
    )

    print(
        f"\nConfusion matrix saved to:\n"
        f"{confusion_path}"
    )


if __name__ == "__main__":

    evaluate_model()