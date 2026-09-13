from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
    / "paderborn_trust_features.csv"
)

MODEL_PATH = (
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
# TRAIN TRUST MODEL
# ==================================================

def train_trust_model():

    print("\n" + "=" * 60)
    print("TRAINING IoT SENSOR TRUST AI MODEL")
    print("=" * 60)

    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    print("\nTrust distribution:")

    print(
        df["trust_label"]
        .value_counts()
    )

    # ----------------------------------------------
    # Prepare features
    # ----------------------------------------------

    excluded_columns = [

        "source_file",

        "condition_label",

        "trust_label",

        "attack_type",
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    # Keep numeric features only

    X = X.select_dtypes(
        include=[np.number]
    )

    y = df[
        "trust_label"
    ]

    print(
        f"\nNumber of features: "
        f"{X.shape[1]}"
    )

    # ----------------------------------------------
    # Train / Test Split
    # ----------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.25,

            random_state=42,

            stratify=y
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # ----------------------------------------------
    # Train Random Forest
    # ----------------------------------------------

    print(
        "\nTraining Random Forest "
        "Trust Model..."
    )

    model = RandomForestClassifier(

        n_estimators=400,

        random_state=42,

        n_jobs=-1,

        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    macro_f1 = f1_score(

        y_test,

        predictions,

        average="macro"
    )

    print("\n" + "=" * 60)
    print("TRUST MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f} "
        f"({balanced_accuracy * 100:.2f}%)"
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
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ----------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------

    classes = [
        "trusted",
        "tampered",
    ]

    matrix = confusion_matrix(

        y_test,

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
        / "trust_model_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # ----------------------------------------------
    # Feature Importance
    # ----------------------------------------------

    importance_df = pd.DataFrame({

        "feature":
            X.columns,

        "importance":
            model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_path = (
        RESULTS_DIR
        / "trust_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    model_data = {

        "model":
            model,

        "feature_names":
            list(X.columns),

        "classes":
            list(model.classes_)
    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    # ----------------------------------------------
    # Save metrics
    # ----------------------------------------------

    metrics = {

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "training_samples":
            int(len(X_train)),

        "testing_samples":
            int(len(X_test)),

        "features":
            int(X.shape[1]),
    }

    metrics_path = (
        RESULTS_DIR
        / "trust_model_metrics.json"
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

    # ----------------------------------------------
    # Show top features
    # ----------------------------------------------

    print(
        "\nTop 10 important "
        "trust-detection features:\n"
    )

    print(
        importance_df
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("TRUST MODEL TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nModel saved to:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nMetrics saved to:\n"
        f"{metrics_path}"
    )

    print(
        f"\nConfusion matrix saved to:\n"
        f"{confusion_path}"
    )


if __name__ == "__main__":

    train_trust_model()