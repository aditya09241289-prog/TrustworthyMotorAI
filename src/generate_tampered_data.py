from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR / "data" / "processed" / "paderborn_features.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR / "data" / "processed" / "paderborn_trust_features.csv"
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ==================================================
# SENSOR GROUPS
# ==================================================

SENSORS = [
    "vibration_1",
    "phase_current_1",
    "phase_current_2",
    "speed",
    "temp_2_bearing_module",
    "torque",
    "force",
]


def get_sensor_groups(columns):

    groups = {}

    for sensor in SENSORS:

        sensor_columns = [
            col
            for col in columns
            if col.startswith(sensor + "_")
        ]

        if sensor_columns:
            groups[sensor] = sensor_columns

    return groups


# ==================================================
# SENSOR-GROUP BIAS ATTACK
# ==================================================

def bias_attack(row, sensor_groups):

    sensor = rng.choice(list(sensor_groups.keys()))

    columns = sensor_groups[sensor]

    factor = rng.choice([
        rng.uniform(0.50, 0.75),
        rng.uniform(1.30, 1.80),
    ])

    for column in columns:

        row[column] = row[column] * factor

    return row, f"bias_{sensor}"


# ==================================================
# SENSOR-GROUP SCALING ATTACK
# ==================================================

def scaling_attack(row, sensor_groups):

    sensor = rng.choice(list(sensor_groups.keys()))

    columns = sensor_groups[sensor]

    factor = rng.choice([
        rng.uniform(0.10, 0.45),
        rng.uniform(2.00, 4.00),
    ])

    for column in columns:

        row[column] = row[column] * factor

    return row, f"scaling_{sensor}"


# ==================================================
# MULTI-FEATURE SPIKE ATTACK
# ==================================================

def spike_attack(row, sensor_groups):

    sensor = rng.choice(list(sensor_groups.keys()))

    columns = sensor_groups[sensor]

    selected = rng.choice(
        columns,
        size=max(1, len(columns) // 2),
        replace=False
    )

    for column in selected:

        value = row[column]

        magnitude = max(
            abs(value) * rng.uniform(3.0, 8.0),
            1e-6
        )

        row[column] = (
            value
            + rng.choice([-1, 1]) * magnitude
        )

    return row, f"spike_{sensor}"


# ==================================================
# CROSS-SENSOR INCONSISTENCY ATTACK
# ==================================================

def cross_sensor_attack(row, sensor_groups):

    available = list(sensor_groups.keys())

    selected_sensors = rng.choice(
        available,
        size=2,
        replace=False
    )

    sensor_a = selected_sensors[0]
    sensor_b = selected_sensors[1]

    factor_a = rng.uniform(0.20, 0.50)
    factor_b = rng.uniform(2.00, 4.00)

    for column in sensor_groups[sensor_a]:

        row[column] *= factor_a

    for column in sensor_groups[sensor_b]:

        row[column] *= factor_b

    return (
        row,
        f"cross_sensor_{sensor_a}_{sensor_b}"
    )


# ==================================================
# ADD DEVIATION FEATURES
# ==================================================

def add_deviation_features(df, feature_columns):

    trusted_reference = df[
        df["trust_label"] == "trusted"
    ]

    reference_mean = (
        trusted_reference[feature_columns]
        .mean()
    )

    reference_std = (
        trusted_reference[feature_columns]
        .std()
        .replace(0, 1e-9)
    )

    # Overall feature deviation from normal reference

    z_scores = (
        df[feature_columns]
        .subtract(reference_mean)
        .divide(reference_std)
    )

    df["mean_abs_deviation"] = (
        z_scores
        .abs()
        .mean(axis=1)
    )

    df["max_abs_deviation"] = (
        z_scores
        .abs()
        .max(axis=1)
    )

    df["high_deviation_feature_count"] = (
        z_scores
        .abs()
        .gt(3)
        .sum(axis=1)
    )

    return df


# ==================================================
# MAIN DATASET CREATION
# ==================================================

def create_dataset():

    print("\n" + "=" * 60)
    print("GENERATING IMPROVED IoT TRUST DATASET")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    metadata_columns = [
        "source_file",
        "condition_label",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in metadata_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    sensor_groups = get_sensor_groups(
        feature_columns
    )

    print(f"\nReal samples: {len(df)}")
    print(f"Features: {len(feature_columns)}")

    print("\nSensor groups:")

    for sensor, columns in sensor_groups.items():

        print(
            f"{sensor}: {len(columns)} features"
        )

    # ----------------------------------------------
    # TRUSTED DATA
    # ----------------------------------------------

    trusted = df.copy()

    trusted["trust_label"] = "trusted"
    trusted["attack_type"] = "none"

    # ----------------------------------------------
    # TAMPERED DATA
    # ----------------------------------------------

    tampered_rows = []

    attack_types = [
        "bias",
        "scaling",
        "spike",
        "cross_sensor",
    ]

    for _, original_row in df.iterrows():

        row = original_row.copy()

        attack = rng.choice(attack_types)

        if attack == "bias":

            row, attack_name = bias_attack(
                row,
                sensor_groups
            )

        elif attack == "scaling":

            row, attack_name = scaling_attack(
                row,
                sensor_groups
            )

        elif attack == "spike":

            row, attack_name = spike_attack(
                row,
                sensor_groups
            )

        else:

            row, attack_name = cross_sensor_attack(
                row,
                sensor_groups
            )

        row["trust_label"] = "tampered"
        row["attack_type"] = attack_name

        tampered_rows.append(row)

    tampered = pd.DataFrame(
        tampered_rows
    )

    # ----------------------------------------------
    # COMBINE
    # ----------------------------------------------

    combined = pd.concat(
        [trusted, tampered],
        ignore_index=True
    )

    combined = add_deviation_features(
        combined,
        feature_columns
    )

    combined = combined.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    combined.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal shape: "
        f"{combined.shape}"
    )

    print("\nTrust distribution:")

    print(
        combined["trust_label"]
        .value_counts()
    )

    print("\nAttack distribution:")

    print(
        combined["attack_type"]
        .value_counts()
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    create_dataset()