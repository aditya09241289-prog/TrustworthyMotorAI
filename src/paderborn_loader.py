from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


# ==================================================
# PROJECT PATHS
# ==================================================

# This file is located at:
# OUTER/Trustworthy_IoT_Motor_AI/src/paderborn_loader.py
#
# PROJECT_DIR = inner Trustworthy_IoT_Motor_AI folder
# OUTER_DIR   = outer folder containing .venv and raw dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTER_DIR = PROJECT_DIR.parent


# Real Paderborn dataset location
RAW_DIR = (
    OUTER_DIR
    / "data"
    / "raw"
    / "paderborn"
)


# Processed ML dataset stays inside the actual project
PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# CONDITION LABELS
# ==================================================
# We keep the real dataset condition IDs.
# We do not invent labels such as overload_fault.

CONDITION_LABELS = {
    "K001": "K001",
    "K002": "K002",
    "KA01": "KA01",
    "KA03": "KA03",
    "KI01": "KI01",
    "KI03": "KI03",
}


# ==================================================
# SAFE NUMERIC CONVERSION
# ==================================================

def to_numeric_array(data):
    """
    Convert MATLAB channel data into a clean
    one-dimensional NumPy array.
    """

    array = np.asarray(
        data,
        dtype=float
    ).flatten()

    array = array[
        np.isfinite(array)
    ]

    return array


# ==================================================
# SIGNAL FEATURE EXTRACTION
# ==================================================

def extract_signal_features(signal, prefix):
    """
    Extract statistical and frequency-domain
    features from one sensor signal.
    """

    signal = to_numeric_array(signal)

    if len(signal) == 0:
        return {}

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    variance_value = np.var(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    peak_value = np.max(
        np.abs(signal)
    )

    min_value = np.min(signal)

    max_value = np.max(signal)

    median_value = np.median(signal)

    # ----------------------------------------------
    # Crest Factor
    # ----------------------------------------------

    if rms_value != 0:

        crest_factor = (
            peak_value / rms_value
        )

    else:

        crest_factor = 0.0

    # ----------------------------------------------
    # Kurtosis
    # ----------------------------------------------

    if std_value != 0:

        kurtosis_value = np.mean(
            (
                (
                    signal - mean_value
                )
                /
                std_value
            ) ** 4
        )

    else:

        kurtosis_value = 0.0

    # ----------------------------------------------
    # Frequency Domain
    # ----------------------------------------------

    fft_values = np.fft.rfft(
        signal
    )

    power = (
        np.abs(fft_values) ** 2
    )

    spectral_energy = np.sum(
        power
    )

    dominant_frequency_index = int(
        np.argmax(power)
    )

    return {

        f"{prefix}_mean":
            float(mean_value),

        f"{prefix}_std":
            float(std_value),

        f"{prefix}_variance":
            float(variance_value),

        f"{prefix}_rms":
            float(rms_value),

        f"{prefix}_peak":
            float(peak_value),

        f"{prefix}_min":
            float(min_value),

        f"{prefix}_max":
            float(max_value),

        f"{prefix}_median":
            float(median_value),

        f"{prefix}_crest_factor":
            float(crest_factor),

        f"{prefix}_kurtosis":
            float(kurtosis_value),

        f"{prefix}_spectral_energy":
            float(spectral_energy),

        f"{prefix}_dominant_frequency_index":
            dominant_frequency_index,
    }


# ==================================================
# LOAD ONE MATLAB RECORDING
# ==================================================

def load_mat_recording(mat_path):
    """
    Load one Paderborn MATLAB recording and extract
    the named sensor channels.
    """

    data = sio.loadmat(

        mat_path,

        struct_as_record=False,

        squeeze_me=True
    )

    # Ignore MATLAB internal keys
    valid_keys = [

        key

        for key in data.keys()

        if not key.startswith("__")
    ]

    if not valid_keys:

        raise ValueError(

            f"No valid MATLAB data found in:\n"
            f"{mat_path}"
        )

    # The main recording structure
    record = data[
        valid_keys[0]
    ]

    channels = {}

    # ----------------------------------------------
    # Extract named sensor channels from Y
    # ----------------------------------------------

    sensor_data = record.Y

    for channel in sensor_data:

        name = getattr(
            channel,
            "Name",
            None
        )

        # MATLAB sometimes stores empty strings
        # or arrays

        if isinstance(
            name,
            np.ndarray
        ):

            if name.size == 0:

                continue

            try:

                name = str(
                    name.item()
                )

            except Exception:

                name = str(
                    name
                )

        if name is None:

            continue

        if str(name).strip() == "":

            continue

        channel_data = getattr(
            channel,
            "Data",
            None
        )

        if channel_data is None:

            continue

        channels[
            str(name)
        ] = to_numeric_array(
            channel_data
        )

    return channels


# ==================================================
# PROCESS ONE RECORDING
# ==================================================

def process_recording(
    mat_path,
    condition_label
):
    """
    Convert one real Paderborn recording into
    one machine-learning feature row.
    """

    channels = load_mat_recording(
        mat_path
    )

    row = {

        "source_file":
            mat_path.name,

        "condition_label":
            condition_label,
    }

    # These sensors were confirmed directly
    # from the Paderborn .mat structure.

    sensors = [

        "vibration_1",

        "phase_current_1",

        "phase_current_2",

        "speed",

        "temp_2_bearing_module",

        "torque",

        "force",
    ]

    for sensor in sensors:

        if sensor in channels:

            features = (
                extract_signal_features(

                    channels[
                        sensor
                    ],

                    sensor
                )
            )

            row.update(
                features
            )

    return row


# ==================================================
# FIND MATLAB FILES
# ==================================================

def find_mat_files():
    """
    Find all MATLAB files recursively.
    """

    return sorted(

        RAW_DIR.rglob(
            "*.mat"
        )
    )


# ==================================================
# DETERMINE CONDITION LABEL
# ==================================================

def get_condition_label(
    mat_path
):
    """
    Determine the Paderborn condition from
    the file's folder path.
    """

    for part in mat_path.parts:

        if part in CONDITION_LABELS:

            return CONDITION_LABELS[
                part
            ]

    return None


# ==================================================
# BUILD COMPLETE DATASET
# ==================================================

def build_dataset():
    """
    Process all real Paderborn MATLAB files
    into a feature dataset.
    """

    print("\n")

    print("=" * 60)

    print(
        "BUILDING REAL PADERBORN DATASET"
    )

    print("=" * 60)

    print(
        "\nRaw dataset directory:"
    )

    print(
        RAW_DIR
    )

    print(
        "\nProcessed output directory:"
    )

    print(
        PROCESSED_DIR
    )

    print("\n")

    # ----------------------------------------------
    # Check dataset directory
    # ----------------------------------------------

    if not RAW_DIR.exists():

        raise FileNotFoundError(

            "\nPaderborn dataset folder "
            "does not exist:\n"

            f"{RAW_DIR}\n"

            "\nCheck where your extracted "
            "dataset folders are located."
        )

    # ----------------------------------------------
    # Find .mat files
    # ----------------------------------------------

    mat_files = find_mat_files()

    if not mat_files:

        raise FileNotFoundError(

            "\nNo .mat files found inside:\n"

            f"{RAW_DIR}\n"

            "\nMake sure folders such as "
            "K001, K002, KA01, KA03, "
            "KI01 and KI03 are inside "
            "the Paderborn directory."
        )

    print(

        f"Found {len(mat_files)} "
        f"MATLAB recordings.\n"
    )

    rows = []

    # ----------------------------------------------
    # Process recordings
    # ----------------------------------------------

    for index, mat_path in enumerate(

        mat_files,

        start=1
    ):

        condition = (
            get_condition_label(
                mat_path
            )
        )

        if condition is None:

            print(

                f"[SKIPPED {index}/"
                f"{len(mat_files)}] "

                f"Could not determine "
                f"condition:\n"

                f"{mat_path}"
            )

            continue

        try:

            row = process_recording(

                mat_path,

                condition
            )

            rows.append(
                row
            )

            print(

                f"[{index}/"
                f"{len(mat_files)}] "

                f"Processed: "

                f"{mat_path.name} "

                f"→ {condition}"
            )

        except Exception as error:

            print(

                f"[ERROR {index}/"
                f"{len(mat_files)}] "

                f"{mat_path.name}\n"

                f"{error}\n"
            )

    # ----------------------------------------------
    # Validate results
    # ----------------------------------------------

    if not rows:

        raise RuntimeError(

            "\nNo recordings were "
            "successfully processed."
        )

    # ----------------------------------------------
    # Create DataFrame
    # ----------------------------------------------

    dataset = pd.DataFrame(
        rows
    )

    # Remove completely empty columns

    dataset = dataset.dropna(

        axis=1,

        how="all"
    )

    # ----------------------------------------------
    # Save processed dataset
    # ----------------------------------------------

    output_path = (

        PROCESSED_DIR

        / "paderborn_features.csv"
    )

    dataset.to_csv(

        output_path,

        index=False
    )

    # ----------------------------------------------
    # Final report
    # ----------------------------------------------

    print("\n")

    print("=" * 60)

    print(
        "DATASET BUILD COMPLETE"
    )

    print("=" * 60)

    print(

        f"\nSuccessfully processed: "
        f"{len(dataset)} recordings"
    )

    print(

        "\nCondition distribution:\n"
    )

    print(

        dataset[
            "condition_label"
        ]
        .value_counts()
        .sort_index()
    )

    print(

        f"\nDataset shape: "
        f"{dataset.shape}"
    )

    print(

        f"\nSaved to:\n"
        f"{output_path}"
    )

    return dataset


# ==================================================
# RUN SCRIPT DIRECTLY
# ==================================================

if __name__ == "__main__":

    dataset = build_dataset()

    print(
        "\nFirst five rows:\n"
    )

    print(
        dataset.head()
    )