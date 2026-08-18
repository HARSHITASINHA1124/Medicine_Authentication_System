from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIG
# ============================================================

try:
    from .config import (
        CHANNELS,
        TARGET,
        MODEL_DIR,
        CLASSIFICATION_LOW_THRESHOLD,
        CLASSIFICATION_HIGH_THRESHOLD
    )

except ImportError:

    from config import (
        CHANNELS,
        TARGET,
        MODEL_DIR,
        CLASSIFICATION_LOW_THRESHOLD,
        CLASSIFICATION_HIGH_THRESHOLD
    )


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = Path(
    MODEL_DIR
)

CLASSIFIER_PATH = (
    MODEL_DIR /
    "best_classifier.joblib"
)

ANOMALY_MODEL_PATH = (
    MODEL_DIR /
    "anomaly_model.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR /
    "anomaly_thresholds.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not CLASSIFIER_PATH.exists():

        raise FileNotFoundError(
            f"Classifier not found:\n"
            f"{CLASSIFIER_PATH}"
        )

    if not ANOMALY_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Anomaly model not found:\n"
            f"{ANOMALY_MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Anomaly thresholds not found:\n"
            f"{THRESHOLD_PATH}"
        )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    anomaly_model = joblib.load(
        ANOMALY_MODEL_PATH
    )

    thresholds = joblib.load(
        THRESHOLD_PATH
    )

    return (
        classifier,
        anomaly_model,
        thresholds
    )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(
    sample
):

    missing = [
        channel
        for channel in CHANNELS
        if channel not in sample
    ]

    if missing:

        raise ValueError(
            "Missing AS7262 channels: "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        [[
            sample[channel]
            for channel in CHANNELS
        ]],
        columns=CHANNELS
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_sample(
    classifier,
    X
):

    predicted_medicine = (
        classifier.predict(X)[0]
    )

    if hasattr(
        classifier,
        "predict_proba"
    ):

        probabilities = (
            classifier.predict_proba(X)[0]
        )

        classes = (
            classifier.classes_
        )

        probability_map = {

            str(cls):
                float(probability)

            for cls, probability
            in zip(
                classes,
                probabilities
            )
        }

        confidence = float(
            np.max(probabilities)
        )

    else:

        probability_map = {}

        confidence = None

    return (
        str(predicted_medicine),
        confidence,
        probability_map
    )


# ============================================================
# CLASSIFICATION STATUS
# ============================================================

def get_classification_status(
    confidence
):

    if confidence is None:

        return "UNKNOWN"

    if (
        confidence
        < CLASSIFICATION_LOW_THRESHOLD
    ):

        return "LOW"

    if (
        confidence
        < CLASSIFICATION_HIGH_THRESHOLD
    ):

        return "MEDIUM"

    return "HIGH"


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    anomaly_model,
    X,
    medicine
):

    profiles = (
        anomaly_model[
            "medicine_profiles"
        ]
    )

    if medicine not in profiles:

        raise ValueError(
            f"No reference profile exists "
            f"for medicine '{medicine}'."
        )

    profile = profiles[
        medicine
    ]

    preprocessor = (
        anomaly_model[
            "preprocessor"
        ]
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    X_features = (
        preprocessor.transform(
            X
        )
    )

    feature_names = (
        anomaly_model[
            "feature_names"
        ]
    )

    X_features = pd.DataFrame(
        X_features,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    X_features = X_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_features = X_features.fillna(0)

    # --------------------------------------------------------
    # Medicine-specific scaler
    # --------------------------------------------------------

    scaler = profile[
        "scaler"
    ]

    X_scaled = scaler.transform(
        X_features
    )

    # --------------------------------------------------------
    # Mahalanobis distance
    # --------------------------------------------------------

    covariance = profile[
        "covariance"
    ]

    score = covariance.mahalanobis(
        X_scaled
    )[0]

    return float(score)


# ============================================================
# ANOMALY DECISION
# ============================================================

def get_anomaly_status(
    score,
    threshold_profile
):

    genuine_threshold = (
        threshold_profile[
            "genuine_threshold"
        ]
    )

    counterfeit_threshold = (
        threshold_profile[
            "counterfeit_threshold"
        ]
    )

    if score <= genuine_threshold:

        return "GENUINE"

    elif score <= counterfeit_threshold:

        return "SUSPICIOUS"

    else:

        return "COUNTERFEIT"


# ============================================================
# FINAL DECISION
# ============================================================

def make_final_decision(
    medicine,
    confidence,
    classification_status,
    anomaly_status,
    known_medicines
):

    # ========================================================
    # UNKNOWN MEDICINE
    # ========================================================

    if medicine not in known_medicines:

        return "SUSPICIOUS"


    # ========================================================
    # CLASSIFICATION FAILED
    # ========================================================

    if confidence is None:

        return "SUSPICIOUS"


    # ========================================================
    # CLASSIFICATION TOO LOW
    # ========================================================

    if (
        classification_status
        == "LOW"
    ):

        return "SUSPICIOUS"


    # ========================================================
    # CLASSIFICATION MEDIUM
    #
    # We don't trust it enough to declare genuine.
    # ========================================================

    if (
        classification_status
        == "MEDIUM"
    ):

        return "SUSPICIOUS"


    # ========================================================
    # CLASSIFICATION HIGH
    #
    # Anomaly detection now determines authenticity.
    # ========================================================

    if anomaly_status == "GENUINE":

        return "GENUINE"

    if anomaly_status == "SUSPICIOUS":

        return "SUSPICIOUS"

    if anomaly_status == "COUNTERFEIT":

        return "COUNTERFEIT"


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    return "SUSPICIOUS"


# ============================================================
# AUTHENTICATE
# ============================================================

def authenticate_sample(
    sample,
    classifier=None,
    anomaly_model=None,
    thresholds=None
):

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    if (
        classifier is None
        or anomaly_model is None
        or thresholds is None
    ):

        (
            classifier,
            anomaly_model,
            thresholds
        ) = load_models()

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    X = prepare_input(
        sample
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    (
        medicine,
        confidence,
        probability_map
    ) = classify_sample(
        classifier,
        X
    )

    classification_status = (
        get_classification_status(
            confidence
        )
    )

    # ========================================================
    # CHECK DATABASE
    # ========================================================

    known_medicines = set(
        anomaly_model[
            "medicine_profiles"
        ].keys()
    )

    # --------------------------------------------------------
    # Unknown medicine
    # --------------------------------------------------------

    if medicine not in known_medicines:

        return {

            "medicine":
                medicine,

            "classification_confidence":
                confidence,

            "classification_status":
                classification_status,

            "classification_probabilities":
                probability_map,

            "anomaly_score":
                None,

            "anomaly_status":
                "NOT_EVALUATED",

            "final_status":
                "SUSPICIOUS",

            "reason":
                "Predicted medicine is not present "
                "in the reference database."
        }

    # ========================================================
    # CLASSIFICATION TOO LOW
    #
    # Don't trust the predicted medicine enough to run
    # medicine-specific anomaly detection.
    # ========================================================

    if (
        confidence is None
        or confidence
        < CLASSIFICATION_HIGH_THRESHOLD
    ):

        return {

            "medicine":
                medicine,

            "classification_confidence":
                confidence,

            "classification_status":
                classification_status,

            "classification_probabilities":
                probability_map,

            "anomaly_score":
                None,

            "anomaly_status":
                "NOT_EVALUATED",

            "final_status":
                "SUSPICIOUS",

            "reason":
                "Classification confidence is not "
                "high enough for authenticity confirmation."
        }

    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    anomaly_score = (
        calculate_anomaly_score(
            anomaly_model,
            X,
            medicine
        )
    )

    # --------------------------------------------------------
    # Get medicine-specific threshold
    # --------------------------------------------------------

    medicine_thresholds = (
        thresholds[
            "profiles"
        ]
    )

    if medicine not in medicine_thresholds:

        return {

            "medicine":
                medicine,

            "classification_confidence":
                confidence,

            "classification_status":
                classification_status,

            "classification_probabilities":
                probability_map,

            "anomaly_score":
                anomaly_score,

            "anomaly_status":
                "NOT_EVALUATED",

            "final_status":
                "SUSPICIOUS",

            "reason":
                "No calibrated anomaly threshold "
                "exists for this medicine."
        }

    # --------------------------------------------------------
    # Anomaly classification
    # --------------------------------------------------------

    anomaly_status = (
        get_anomaly_status(
            anomaly_score,
            medicine_thresholds[
                medicine
            ]
        )
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    final_status = (
        make_final_decision(
            medicine,
            confidence,
            classification_status,
            anomaly_status,
            known_medicines
        )
    )

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    if final_status == "GENUINE":

        reason = (
            "High classification confidence and "
            "normal anomaly score."
        )

    elif final_status == "SUSPICIOUS":

        reason = (
            "The sample requires further verification "
            "because either classification confidence "
            "or anomaly score is outside the genuine range."
        )

    else:

        reason = (
            "High classification confidence but the "
            "sample is far outside the genuine reference "
            "distribution."
        )

    return {

        "medicine":
            medicine,

        "classification_confidence":
            confidence,

        "classification_status":
            classification_status,

        "classification_probabilities":
            probability_map,

        "anomaly_score":
            anomaly_score,

        "anomaly_status":
            anomaly_status,

        "final_status":
            final_status,

        "reason":
            reason
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print(
        "\n" + "=" * 60
    )

    print(
        "MEDICINE AUTHENTICATION RESULT"
    )

    print(
        "=" * 60
    )

    print(
        "\nMedicine:",
        result["medicine"]
    )

    confidence = (
        result[
            "classification_confidence"
        ]
    )

    print(
        "\nClassification confidence:"
    )

    if confidence is None:

        print("N/A")

    else:

        print(
            f"{confidence * 100:.2f}%"
        )

    print(
        "Classification status:",
        result[
            "classification_status"
        ]
    )

    print(
        "\nClassification probabilities:"
    )

    for medicine, probability in (
        result[
            "classification_probabilities"
        ].items()
    ):

        print(
            f"  {medicine}: "
            f"{probability * 100:.2f}%"
        )

    print(
        "\nAnomaly score:",
        result["anomaly_score"]
    )

    print(
        "Anomaly status:",
        result["anomaly_status"]
    )

    print(
        "\nReason:"
    )

    print(
        result["reason"]
    )

    print(
        "\n" + "-" * 60
    )

    print(
        "FINAL RESULT:",
        result["final_status"]
    )

    print(
        "=" * 60
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    (
        classifier,
        anomaly_model,
        thresholds
    ) = load_models()

    # --------------------------------------------------------
    # Temporary test reading
    #
    # Replace with real AS7262 hardware values later.
    # --------------------------------------------------------

    sample = {

        "ch450": 1000,

        "ch500": 1200,

        "ch550": 1400,

        "ch570": 1500,

        "ch600": 1300,

        "ch650": 900
    }

    result = authenticate_sample(
        sample,
        classifier,
        anomaly_model,
        thresholds
    )

    print_result(
        result
    )