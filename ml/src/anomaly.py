from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.covariance import LedoitWolf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "train.csv"
)

CLASSIFIER_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "best_classifier.joblib"
)

ANOMALY_MODEL_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "anomaly_profiles.joblib"
)


CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]

TARGET = "medicine"


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    df = pd.read_csv(
        TRAIN_PATH
    )

    required_columns = (
        CHANNELS +
        [TARGET]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Training dataset is missing columns: "
            f"{missing}"
        )

    return df


# ============================================================
# LOAD CLASSIFIER PIPELINE
# ============================================================

def load_classifier():

    if not CLASSIFIER_PATH.exists():

        raise FileNotFoundError(
            f"Classifier not found: "
            f"{CLASSIFIER_PATH}\n"
            "Run classification.py first."
        )

    pipeline = joblib.load(
        CLASSIFIER_PATH
    )

    return pipeline


# ============================================================
# BUILD REFERENCE PROFILES
# ============================================================

def build_reference_profiles(
    train_df,
    classifier_pipeline
):

    # --------------------------------------------------------
    # Extract the already-fitted preprocessor
    # --------------------------------------------------------

    preprocessor = (
        classifier_pipeline
        .named_steps["preprocessor"]
    )

    X_raw = train_df[
        CHANNELS
    ]

    y = train_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Transform using EXACT SAME preprocessing
    # used by the classifier
    # --------------------------------------------------------

    X_features = preprocessor.transform(
        X_raw
    )

    X_features = np.asarray(
        X_features,
        dtype=float
    )

    profiles = {}

    medicines = sorted(
        y.unique()
    )

    print(
        "\nBuilding anomaly reference profiles..."
    )

    for medicine in medicines:

        mask = (
            y.values == medicine
        )

        X_medicine = (
            X_features[mask]
        )

        print(
            f"{medicine}: "
            f"{len(X_medicine)} samples"
        )

        if len(X_medicine) < 3:

            print(
                f"Skipping {medicine}: "
                "not enough samples."
            )

            continue

        # ----------------------------------------------------
        # Robust covariance estimation
        # ----------------------------------------------------

        covariance = LedoitWolf()

        covariance.fit(
            X_medicine
        )

        profiles[medicine] = {

            "mean":
                covariance.location_,

            "precision":
                covariance.precision_,

            "n_samples":
                len(X_medicine)
        }

    return profiles


# ============================================================
# SAVE PROFILES
# ============================================================

def save_profiles(profiles):

    ANOMALY_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        profiles,
        ANOMALY_MODEL_PATH
    )

    print(
        "\nAnomaly profiles saved to:"
    )

    print(
        ANOMALY_MODEL_PATH
    )


# ============================================================
# MAHALANOBIS DISTANCE
# ============================================================

def mahalanobis_distance(
    X,
    mean,
    precision
):

    difference = (
        X - mean
    )

    distance_squared = np.sum(
        (difference @ precision)
        * difference,
        axis=1
    )

    distance_squared = np.maximum(
        distance_squared,
        0
    )

    return np.sqrt(
        distance_squared
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_df = (
        load_training_data()
    )

    classifier = (
        load_classifier()
    )

    profiles = (
        build_reference_profiles(
            train_df,
            classifier
        )
    )

    save_profiles(
        profiles
    )

    print(
        "\nMedicines with reference profiles:"
    )

    for medicine in profiles:

        print(
            f"  - {medicine}"
        )