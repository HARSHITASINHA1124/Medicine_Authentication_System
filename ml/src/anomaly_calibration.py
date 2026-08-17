from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)

CLASSIFIER_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "best_classifier.joblib"
)

ANOMALY_PROFILE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_profiles.joblib"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_thresholds.joblib"
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

# Percentage of genuine validation samples
# allowed to fall outside the reference boundary.
CONTAMINATION = 0.05


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    return train_df, test_df


# ============================================================
# MAHALANOBIS DISTANCE
# ============================================================

def calculate_distance(
    X,
    mean,
    precision
):

    difference = X - mean

    distance_squared = np.sum(
        (difference @ precision)
        * difference,
        axis=1
    )

    # Numerical safety
    distance_squared = np.maximum(
        distance_squared,
        0
    )

    return np.sqrt(
        distance_squared
    )


# ============================================================
# CALIBRATE THRESHOLDS
# ============================================================

def calibrate_thresholds(
    train_df,
    test_df,
    classifier_pipeline,
    profiles
):

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the EXACT SAME preprocessor that was fitted
    # inside best_classifier.joblib.
    # --------------------------------------------------------

    preprocessor = (
        classifier_pipeline
        .named_steps["preprocessor"]
    )

    X_test_raw = test_df[
        CHANNELS
    ]

    y_test = test_df[
        TARGET
    ]

    X_test_features = (
        preprocessor.transform(
            X_test_raw
        )
    )

    X_test_features = np.asarray(
        X_test_features,
        dtype=float
    )

    thresholds = {}

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ANOMALY THRESHOLD CALIBRATION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Calibrate separately for every medicine
    # --------------------------------------------------------

    for medicine in sorted(
        profiles.keys()
    ):

        # Genuine validation samples belonging
        # to this medicine
        mask = (
            y_test.values
            == medicine
        )

        X_medicine = (
            X_test_features[mask]
        )

        if len(X_medicine) < 2:

            print(
                f"\nSkipping {medicine}: "
                f"only {len(X_medicine)} "
                "validation sample(s)."
            )

            continue

        profile = profiles[
            medicine
        ]

        distances = calculate_distance(
            X_medicine,
            profile["mean"],
            profile["precision"]
        )

        # ----------------------------------------------------
        # Threshold
        #
        # 95th percentile means approximately 95%
        # of genuine validation observations are
        # expected to fall below this threshold.
        # ----------------------------------------------------

        threshold = np.percentile(
            distances,
            100 * (1 - CONTAMINATION)
        )

        thresholds[medicine] = {
            "threshold": float(
                threshold
            ),

            "n_validation_samples":
                int(len(distances)),

            "distance_mean":
                float(np.mean(distances)),

            "distance_std":
                float(np.std(distances)),

            "distance_min":
                float(np.min(distances)),

            "distance_max":
                float(np.max(distances))
        }

        print(
            f"\nMedicine: {medicine}"
        )

        print(
            f"Validation samples: "
            f"{len(distances)}"
        )

        print(
            f"Mean distance: "
            f"{np.mean(distances):.4f}"
        )

        print(
            f"Threshold: "
            f"{threshold:.4f}"
        )

    return thresholds


# ============================================================
# SAVE THRESHOLDS
# ============================================================

def save_thresholds(
    thresholds
):

    THRESHOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        thresholds,
        THRESHOLD_PATH
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Thresholds saved to:"
    )

    print(
        THRESHOLD_PATH
    )

    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, test_df = (
        load_data()
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    if not CLASSIFIER_PATH.exists():

        raise FileNotFoundError(
            "best_classifier.joblib "
            "not found.\n"
            "Run classification.py first."
        )

    classifier_pipeline = (
        joblib.load(
            CLASSIFIER_PATH
        )
    )

    # --------------------------------------------------------
    # Load anomaly reference profiles
    # --------------------------------------------------------

    if not ANOMALY_PROFILE_PATH.exists():

        raise FileNotFoundError(
            "anomaly_profiles.joblib "
            "not found.\n"
            "Run anomaly.py first."
        )

    profiles = (
        joblib.load(
            ANOMALY_PROFILE_PATH
        )
    )

    # --------------------------------------------------------
    # Calibrate
    # --------------------------------------------------------

    thresholds = (
        calibrate_thresholds(
            train_df,
            test_df,
            classifier_pipeline,
            profiles
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_thresholds(
        thresholds
    )