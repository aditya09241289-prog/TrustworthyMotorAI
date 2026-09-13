from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "paderborn_features.csv"
)

MODELS_DIR = (
    PROJECT_DIR
    / "models"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# TRAIN MODEL
# ==================================================

def train_motor_model():

    print("\n" + "=" * 60)
    print("TRAINING REAL PADERBORN MOTOR CONDITION MODEL")
    print("=" * 60)

    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------

    print(f"\nLoading dataset:\n{DATA_PATH}")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    print("\nCondition distribution:")

    print(
        df["condition_label"]
        .value_counts()
        .sort_index()
    )

    # ----------------------------------------------
    # Separate features and labels
    # ----------------------------------------------

    target_column = "condition_label"

    drop_columns = [
        "source_file",
        target_column,
    ]

    X = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    y = df[target_column]

    # Keep only numeric features
    X = X.select_dtypes(
        include=[np.number]
    )

    print(f"\nNumber of ML features: {X.shape[1]}")

    print("\nClasses:")

    print(
        sorted(
            y.unique()
        )
    )

    # ----------------------------------------------
    # Train/test split
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
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ----------------------------------------------
    # ML pipeline
    # ----------------------------------------------

    model = Pipeline(
        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",

                RandomForestClassifier(

                    n_estimators=400,

                    random_state=42,

                    class_weight="balanced",

                    n_jobs=-1
                )
            ),
        ]
    )

    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    print("\nTraining Random Forest model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ----------------------------------------------
    # Evaluate
    # ----------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nClassification Report:\n")

    print(report)

    # ----------------------------------------------
    # Confusion matrix
    # ----------------------------------------------

    classes = sorted(
        y.unique()
    )

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
        / "paderborn_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # ----------------------------------------------
    # Feature importance
    # ----------------------------------------------

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    feature_importance = pd.DataFrame({

        "feature":
            X.columns,

        "importance":
            classifier.feature_importances_,
    })

    feature_importance = (
        feature_importance
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
        / "paderborn_feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_path,
        index=False
    )

    print("\nTop 10 important features:\n")

    print(
        feature_importance
        .head(10)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    model_path = (
        MODELS_DIR
        / "paderborn_motor_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # ----------------------------------------------
    # Save metadata
    # ----------------------------------------------

    metadata = {

        "accuracy":
            float(accuracy),

        "classes":
            classes,

        "n_features":
            int(X.shape[1]),

        "n_samples":
            int(len(df)),

        "train_samples":
            int(len(X_train)),

        "test_samples":
            int(len(X_test)),

        "feature_names":
            list(X.columns),
    }

    metadata_path = (
        RESULTS_DIR
        / "paderborn_motor_metrics.json"
    )

    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    # ----------------------------------------------
    # Final output
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nModel saved to:\n{model_path}"
    )

    print(
        f"\nMetrics saved to:\n{metadata_path}"
    )

    print(
        f"\nFeature importance saved to:\n{importance_path}"
    )

    print(
        f"\nConfusion matrix saved to:\n{confusion_path}"
    )

    return model, accuracy


# ==================================================
# RUN DIRECTLY
# ==================================================

if __name__ == "__main__":

    train_motor_model()