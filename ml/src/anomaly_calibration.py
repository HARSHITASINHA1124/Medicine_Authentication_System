from pathlib import Path

import numpy as np
import pandas as pd
import joblib


try:
    from .config import (
        CHANNELS,
        TARGET,
        TRAIN_PATH,
        TEST_PATH,
        MODEL_DIR,
        RESULTS_DIR,
        ANOMALY_GENUINE_PERCENTILE,
        ANOMALY_COUNTERFEIT_PERCENTILE
    )

except ImportError:

    from config import (
        CHANNELS,
        TARGET,
        TRAIN_PATH,
        TEST_PATH,
        MODEL_DIR,
        RESULTS_DIR,
        ANOMALY_GENUINE_PERCENTILE,
        ANOMALY_COUNTERFEIT_PERCENTILE
    )


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(MODEL_DIR)
RESULTS_DIR = Path(RESULTS_DIR)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANOMALY_MODEL_PATH = (
    MODEL_DIR /
    "anomaly_model.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR /
    "anomaly_thresholds.joblib"
)

RESULT_PATH = (
    RESULTS_DIR /
    "anomaly_calibration_scores.csv"
)


# ============================================================
# LOAD ANOMALY MODEL
# ============================================================

def load_anomaly_model():

    if not ANOMALY_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Anomaly model not found:\n"
            f"{ANOMALY_MODEL_PATH}\n\n"
            "Run anomaly_training.py first."
        )

    package = joblib.load(
        ANOMALY_MODEL_PATH
    )

    if "medicine_profiles" not in package:

        raise ValueError(
            "The anomaly model is not medicine-aware.\n"
            "Rerun anomaly_training.py."
        )

    print(
        "\nMedicine profiles:"
    )

    for medicine in package[
        "medicine_profiles"
    ]:

        print(
            f"  {medicine}"
        )

    return package


# ============================================================
# CALCULATE THRESHOLDS
# ============================================================

def calculate_thresholds(
    package
):

    profiles = package[
        "medicine_profiles"
    ]

    thresholds = {}

    print(
        "\n" + "=" * 60
    )

    print(
        "MEDICINE-AWARE ANOMALY THRESHOLDS"
    )

    print(
        "=" * 60
    )

    for medicine, profile in profiles.items():

        scores = np.asarray(
            profile[
                "training_scores"
            ],
            dtype=float
        )

        scores = scores[
            np.isfinite(scores)
        ]

        if len(scores) < 3:

            print(
                f"\nSkipping {medicine}: "
                f"only {len(scores)} scores."
            )

            continue

        genuine_threshold = np.percentile(
            scores,
            ANOMALY_GENUINE_PERCENTILE
        )

        counterfeit_threshold = np.percentile(
            scores,
            ANOMALY_COUNTERFEIT_PERCENTILE
        )

        thresholds[
            medicine
        ] = {

            "genuine_threshold":
                float(
                    genuine_threshold
                ),

            "counterfeit_threshold":
                float(
                    counterfeit_threshold
                ),

            "genuine_percentile":
                ANOMALY_GENUINE_PERCENTILE,

            "counterfeit_percentile":
                ANOMALY_COUNTERFEIT_PERCENTILE,

            "n_reference_samples":
                int(
                    len(scores)
                )
        }

        print(
            f"\n{medicine}"
        )

        print(
            "  Reference samples:",
            len(scores)
        )

        print(
            "  Genuine boundary:",
            f"{genuine_threshold:.4f}"
        )

        print(
            "  Counterfeit boundary:",
            f"{counterfeit_threshold:.4f}"
        )

    if not thresholds:

        raise ValueError(
            "No medicine thresholds could be created."
        )

    return thresholds


# ============================================================
# SAVE THRESHOLDS
# ============================================================

def save_thresholds(
    thresholds
):

    package = {

        "version":
            "2.0",

        "description":
            "Medicine-specific anomaly "
            "decision boundaries",

        "profiles":
            thresholds
    }

    joblib.dump(
        package,
        THRESHOLD_PATH
    )

    print(
        "\nThresholds saved to:"
    )

    print(
        THRESHOLD_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    package = (
        load_anomaly_model()
    )

    thresholds = (
        calculate_thresholds(
            package
        )
    )

    save_thresholds(
        thresholds
    )

    print(
        "\nAnomaly calibration completed."
    )


if __name__ == "__main__":

    main()