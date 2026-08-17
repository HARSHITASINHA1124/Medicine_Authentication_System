"""
ml_inference.py
===============

Production ML inference pipeline for the Raspberry Pi.

Input:
    10 AS7262 readings for one physical medicine scan.

Flow:
    10 readings
        ↓
    validation
        ↓
    median aggregation
        ↓
    classification
        ↓
    predicted medicine + confidence
        ↓
    anomaly detection
        ↓
    medicine-specific thresholds
        ↓
    final decision
        ↓
    explainability

Output:
{
    "medicine": "...",
    "classification_confidence": 0.94,
    "anomaly_score": 3.21,
    "classification_status": "HIGH",
    "anomaly_status": "GENUINE",
    "final_status": "GENUINE",
    ...
}

This file performs inference only.
It does NOT train models.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATH CONFIGURATION
# ============================================================

ML_DIR = Path(__file__).resolve().parent

SRC_DIR = ML_DIR / "src"
MODEL_DIR = ML_DIR / "models"

# The saved classifier contains the custom
# AS7262Preprocessor class from src/preprocessing.py.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# MODEL FILES
# ============================================================

CLASSIFIER_PATH = (
    MODEL_DIR / "best_classifier.joblib"
)

ANOMALY_MODEL_PATH = (
    MODEL_DIR / "anomaly_model.joblib"
)

THRESHOLDS_PATH = (
    MODEL_DIR / "anomaly_thresholds.joblib"
)


# ============================================================
# AS7262 CONFIGURATION
# ============================================================

CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]

EXPECTED_READINGS = 10


# ============================================================
# CLASSIFICATION THRESHOLDS
# ============================================================

LOW_CLASSIFICATION_CONFIDENCE = 0.60

HIGH_CLASSIFICATION_CONFIDENCE = 0.80


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

_classifier = None
_anomaly_model = None
_thresholds = None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    """
    Load all trained model artifacts.

    Models are loaded once and reused.
    """

    global _classifier
    global _anomaly_model
    global _thresholds

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    if not CLASSIFIER_PATH.exists():

        raise FileNotFoundError(
            f"Classifier model not found:\n"
            f"{CLASSIFIER_PATH}"
        )

    _classifier = joblib.load(
        CLASSIFIER_PATH
    )

    # --------------------------------------------------------
    # ANOMALY MODEL
    # --------------------------------------------------------

    if not ANOMALY_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Anomaly model not found:\n"
            f"{ANOMALY_MODEL_PATH}"
        )

    _anomaly_model = joblib.load(
        ANOMALY_MODEL_PATH
    )

    # --------------------------------------------------------
    # ANOMALY THRESHOLDS
    # --------------------------------------------------------

    if not THRESHOLDS_PATH.exists():

        raise FileNotFoundError(
            f"Anomaly threshold file not found:\n"
            f"{THRESHOLDS_PATH}"
        )

    _thresholds = joblib.load(
        THRESHOLDS_PATH
    )

    # --------------------------------------------------------
    # Validate threshold structure
    # --------------------------------------------------------

    if not isinstance(
        _thresholds,
        dict
    ):

        raise ValueError(
            "anomaly_thresholds.joblib must contain "
            "a dictionary."
        )

    if "profiles" not in _thresholds:

        raise ValueError(
            "anomaly_thresholds.joblib does not contain "
            "a 'profiles' section."
        )

    if not isinstance(
        _thresholds["profiles"],
        dict
    ):

        raise ValueError(
            "'profiles' inside anomaly_thresholds.joblib "
            "must be a dictionary."
        )

    print(
        "ML models loaded successfully."
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_reading(
    reading
):
    """
    Validate one AS7262 reading.

    Expected:

    {
        "ch450": value,
        "ch500": value,
        "ch550": value,
        "ch570": value,
        "ch600": value,
        "ch650": value
    }
    """

    if not isinstance(
        reading,
        dict
    ):

        raise TypeError(
            "Each reading must be a dictionary."
        )

    missing = [
        channel
        for channel in CHANNELS
        if channel not in reading
    ]

    if missing:

        raise ValueError(
            "Missing AS7262 channels: "
            + ", ".join(missing)
        )

    for channel in CHANNELS:

        try:

            value = float(
                reading[channel]
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"{channel} must contain "
                f"a numeric value."
            )

        if not np.isfinite(value):

            raise ValueError(
                f"{channel} contains "
                f"an invalid value."
            )


def validate_readings(
    readings
):
    """
    Validate the complete scan.
    """

    if not isinstance(
        readings,
        list
    ):

        raise TypeError(
            "readings must be a list."
        )

    if len(readings) != EXPECTED_READINGS:

        raise ValueError(
            f"Expected {EXPECTED_READINGS} "
            f"readings but received "
            f"{len(readings)}."
        )

    for reading in readings:

        validate_reading(
            reading
        )


# ============================================================
# AGGREGATE 10 READINGS
# ============================================================

def aggregate_readings(
    readings
):
    """
    Convert 10 AS7262 measurements into
    one representative six-channel scan.

    Median is used because it is less sensitive
    to occasional sensor spikes than the mean.
    """

    validate_readings(
        readings
    )

    df = pd.DataFrame(
        readings,
        columns=CHANNELS
    )

    # --------------------------------------------------------
    # Median spectrum
    # --------------------------------------------------------

    median_values = (
        df[CHANNELS]
        .median()
    )

    spectrum = {
        channel: float(
            median_values[channel]
        )
        for channel in CHANNELS
    }

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    mean_values = (
        df[CHANNELS]
        .mean()
    )

    mean_spectrum = {
        channel: float(
            mean_values[channel]
        )
        for channel in CHANNELS
    }

    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    std_values = (
        df[CHANNELS]
        .std(
            ddof=1
        )
    )

    std_spectrum = {
        channel: float(
            std_values[channel]
        )
        for channel in CHANNELS
    }

    # --------------------------------------------------------
    # Coefficient of variation
    # --------------------------------------------------------

    cv_spectrum = {}

    for channel in CHANNELS:

        median = abs(
            spectrum[channel]
        )

        std = std_spectrum[channel]

        if median > 0:

            cv = std / median

        else:

            cv = np.inf

        cv_spectrum[channel] = float(
            cv
        )

    finite_cvs = [
        value
        for value in cv_spectrum.values()
        if np.isfinite(value)
    ]

    if finite_cvs:

        max_cv = max(
            finite_cvs
        )

    else:

        max_cv = np.inf

    # --------------------------------------------------------
    # Quality indicator
    # --------------------------------------------------------

    scan_quality = {

        "number_of_readings":
            len(readings),

        "stable":
            bool(max_cv <= 0.10),

        "max_coefficient_of_variation":
            float(max_cv),

        "mean":
            mean_spectrum,

        "standard_deviation":
            std_spectrum,

        "coefficient_of_variation":
            cv_spectrum
    }

    return (
        spectrum,
        scan_quality
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(
    spectrum
):
    """
    Convert the aggregated six-channel spectrum
    into a DataFrame.

    The saved classifier pipeline contains the
    preprocessing and feature engineering steps.
    """

    validate_reading(
        spectrum
    )

    X = pd.DataFrame(
        [[
            spectrum[channel]
            for channel in CHANNELS
        ]],
        columns=CHANNELS
    )

    return X


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_sample(
    X
):
    """
    Predict medicine and classification confidence.
    """

    if _classifier is None:

        load_models()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = (
        _classifier.predict(X)
    )

    predicted_medicine = str(
        prediction[0]
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = {}
    confidence = None

    if hasattr(
        _classifier,
        "predict_proba"
    ):

        probability_array = (
            _classifier
            .predict_proba(X)[0]
        )

        classes = (
            _classifier.classes_
        )

        probabilities = {
            str(class_name): float(
                probability
            )
            for class_name, probability
            in zip(
                classes,
                probability_array
            )
        }

        confidence = float(
            np.max(
                probability_array
            )
        )

    # --------------------------------------------------------
    # Classification status
    # --------------------------------------------------------

    if confidence is None:

        status = "UNKNOWN"

    elif (
        confidence
        >= HIGH_CLASSIFICATION_CONFIDENCE
    ):

        status = "HIGH"

    elif (
        confidence
        >= LOW_CLASSIFICATION_CONFIDENCE
    ):

        status = "MEDIUM"

    else:

        status = "LOW"

    return {

        "predicted_medicine":
            predicted_medicine,

        "confidence":
            confidence,

        "status":
            status,

        "probabilities":
            probabilities
    }


# ============================================================
# MEDICINE-AWARE THRESHOLD LOOKUP
# ============================================================

def get_medicine_thresholds(
    medicine
):
    """
    Read medicine-specific thresholds.

    Actual artifact structure:

    {
        "version": "2.0",
        "description": "...",
        "profiles": {
            "MDMA": {
                "genuine_threshold": ...,
                "counterfeit_threshold": ...
            }
        }
    }

    Returns None if no profile exists.
    """

    if _thresholds is None:

        load_models()

    profiles = _thresholds.get(
        "profiles",
        {}
    )

    if medicine not in profiles:

        return None

    profile = profiles[
        medicine
    ]

    genuine_threshold = (
        profile.get(
            "genuine_threshold"
        )
    )

    counterfeit_threshold = (
        profile.get(
            "counterfeit_threshold"
        )
    )

    if (
        genuine_threshold is None
        or counterfeit_threshold is None
    ):

        raise ValueError(
            f"Incomplete anomaly threshold "
            f"profile for medicine '{medicine}'."
        )

    return {

        "genuine_threshold":
            float(genuine_threshold),

        "counterfeit_threshold":
            float(counterfeit_threshold),

        "genuine_percentile":
            profile.get(
                "genuine_percentile"
            ),

        "counterfeit_percentile":
            profile.get(
                "counterfeit_percentile"
            ),

        "n_reference_samples":
            profile.get(
                "n_reference_samples"
            )
    }


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    X
):
    """
    Calculate anomaly score.

    Supports common sklearn anomaly models:

        decision_function()
        score_samples()

    For sklearn's IsolationForest-style models,
    larger decision_function values indicate more
    normal observations.

    Therefore the score is inverted so that:

        LOW score  = more normal
        HIGH score = more anomalous

    This matches the project's threshold convention.
    """

    if _anomaly_model is None:

        load_models()

    # --------------------------------------------------------
    # decision_function
    # --------------------------------------------------------

    if hasattr(
        _anomaly_model,
        "decision_function"
    ):

        raw_score = (
            _anomaly_model
            .decision_function(X)
        )

        raw_score = float(
            np.asarray(
                raw_score
            ).reshape(-1)[0]
        )

        anomaly_score = -raw_score

    # --------------------------------------------------------
    # score_samples
    # --------------------------------------------------------

    elif hasattr(
        _anomaly_model,
        "score_samples"
    ):

        raw_score = (
            _anomaly_model
            .score_samples(X)
        )

        raw_score = float(
            np.asarray(
                raw_score
            ).reshape(-1)[0]
        )

        anomaly_score = -raw_score

    else:

        raise ValueError(
            "The anomaly model does not provide "
            "decision_function() or score_samples()."
        )

    if not np.isfinite(
        anomaly_score
    ):

        raise ValueError(
            "Anomaly model returned an invalid score."
        )

    return float(
        anomaly_score
    )


# ============================================================
# ANOMALY STATUS
# ============================================================

def determine_anomaly_status(
    anomaly_score,
    thresholds
):
    """
    Convert anomaly score into:

        GENUINE
        SUSPICIOUS
        COUNTERFEIT
    """

    genuine_threshold = (
        thresholds[
            "genuine_threshold"
        ]
    )

    counterfeit_threshold = (
        thresholds[
            "counterfeit_threshold"
        ]
    )

    # --------------------------------------------------------
    # Genuine
    # --------------------------------------------------------

    if (
        anomaly_score
        <= genuine_threshold
    ):

        return "GENUINE"

    # --------------------------------------------------------
    # Suspicious
    # --------------------------------------------------------

    if (
        anomaly_score
        < counterfeit_threshold
    ):

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # Counterfeit
    # --------------------------------------------------------

    return "COUNTERFEIT"


# ============================================================
# FINAL DECISION
# ============================================================

def determine_final_status(
    classification,
    anomaly_status
):
    """
    Combine classification confidence and anomaly status.

    Logic:

    LOW classification confidence
        -> SUSPICIOUS

    classification sufficient +
    anomaly GENUINE
        -> GENUINE

    anomaly SUSPICIOUS
        -> SUSPICIOUS

    anomaly COUNTERFEIT
        -> COUNTERFEIT

    No anomaly reference
        -> SUSPICIOUS
    """

    confidence = (
        classification[
            "confidence"
        ]
    )

    # --------------------------------------------------------
    # No classification confidence
    # --------------------------------------------------------

    if confidence is None:

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # Very low classification confidence
    # --------------------------------------------------------

    if (
        confidence
        < LOW_CLASSIFICATION_CONFIDENCE
    ):

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # No anomaly reference
    # --------------------------------------------------------

    if anomaly_status == "NO_REFERENCE":

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # Very high anomaly
    # --------------------------------------------------------

    if anomaly_status == "COUNTERFEIT":

        return "COUNTERFEIT"

    # --------------------------------------------------------
    # Moderate anomaly
    # --------------------------------------------------------

    if anomaly_status == "SUSPICIOUS":

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # Normal anomaly
    # --------------------------------------------------------

    if anomaly_status == "GENUINE":

        return "GENUINE"

    return "SUSPICIOUS"


# ============================================================
# CHANNEL EXPLAINABILITY
# ============================================================

def calculate_channel_statistics(
    readings
):
    """
    Calculate statistics for all ten readings.
    """

    df = pd.DataFrame(
        readings,
        columns=CHANNELS
    )

    result = {}

    for channel in CHANNELS:

        values = (
            df[channel]
            .astype(float)
        )

        median = float(
            values.median()
        )

        mean = float(
            values.mean()
        )

        std = float(
            values.std(
                ddof=1
            )
        )

        if abs(median) > 0:

            cv = std / abs(median)

        else:

            cv = np.inf

        result[channel] = {

            "median":
                median,

            "mean":
                mean,

            "std":
                std,

            "coefficient_of_variation":
                float(cv)
        }

    return result


# ============================================================
# EXPLAINABILITY
# ============================================================

def create_explainability(
    readings,
    spectrum,
    classification,
    anomaly_score,
    anomaly_status,
    thresholds
):
    """
    Generate explanation information.
    """

    medicine = (
        classification[
            "predicted_medicine"
        ]
    )

    confidence = (
        classification[
            "confidence"
        ]
    )

    # --------------------------------------------------------
    # Classification explanation
    # --------------------------------------------------------

    if confidence is None:

        classification_reason = (
            "The classifier did not provide "
            "a confidence score."
        )

    elif confidence < LOW_CLASSIFICATION_CONFIDENCE:

        classification_reason = (
            "The classifier has low confidence "
            f"in identifying the medicine as "
            f"{medicine}."
        )

    elif confidence < HIGH_CLASSIFICATION_CONFIDENCE:

        classification_reason = (
            f"The classifier identified the "
            f"sample as {medicine}, but confidence "
            f"is moderate."
        )

    else:

        classification_reason = (
            f"The classifier identified the "
            f"sample as {medicine} with high "
            f"confidence."
        )

    # --------------------------------------------------------
    # Anomaly explanation
    # --------------------------------------------------------

    if anomaly_status == "NO_REFERENCE":

        anomaly_reason = (
            f"No trained anomaly reference profile "
            f"is available for {medicine}."
        )

    elif anomaly_status == "GENUINE":

        anomaly_reason = (
            f"The anomaly score ({anomaly_score:.4f}) "
            f"is within the genuine range for "
            f"{medicine}."
        )

    elif anomaly_status == "SUSPICIOUS":

        anomaly_reason = (
            f"The anomaly score ({anomaly_score:.4f}) "
            f"is above the genuine boundary but "
            f"below the counterfeit boundary."
        )

    else:

        anomaly_reason = (
            f"The anomaly score ({anomaly_score:.4f}) "
            f"has crossed the counterfeit boundary "
            f"for {medicine}."
        )

    # --------------------------------------------------------
    # Channel statistics
    # --------------------------------------------------------

    channel_statistics = (
        calculate_channel_statistics(
            readings
        )
    )

    return {

        "classification": {

            "predicted_medicine":
                medicine,

            "confidence":
                confidence,

            "status":
                classification[
                    "status"
                ],

            "probabilities":
                classification[
                    "probabilities"
                ],

            "reason":
                classification_reason
        },

        "anomaly": {

            "score":
                anomaly_score,

            "status":
                anomaly_status,

            "thresholds":
                thresholds,

            "reason":
                anomaly_reason
        },

        "spectral_reading":
            spectrum,

        "channel_statistics":
            channel_statistics
    }


# ============================================================
# AUTHENTICATE ONE SCAN
# ============================================================

def authenticate_scan(
    readings
):
    """
    Main authentication function.

    Input:
        10 AS7262 readings.

    Output:
        Complete authentication result.
    """

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    if (
        _classifier is None
        or _anomaly_model is None
        or _thresholds is None
    ):

        load_models()

    # --------------------------------------------------------
    # STEP 1
    # Aggregate 10 readings
    # --------------------------------------------------------

    (
        spectrum,
        scan_quality
    ) = aggregate_readings(
        readings
    )

    # --------------------------------------------------------
    # STEP 2
    # Create classifier input
    # --------------------------------------------------------

    X = create_model_input(
        spectrum
    )

    # --------------------------------------------------------
    # STEP 3
    # Classification
    # --------------------------------------------------------

    classification = classify_sample(
        X
    )

    medicine = (
        classification[
            "predicted_medicine"
        ]
    )

    # --------------------------------------------------------
    # STEP 4
    # Medicine-aware threshold
    # --------------------------------------------------------

    thresholds = (
        get_medicine_thresholds(
            medicine
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # Handle missing anomaly reference
    # --------------------------------------------------------

    if thresholds is None:

        anomaly_score = None

        anomaly_status = (
            "NO_REFERENCE"
        )

        final_status = (
            "SUSPICIOUS"
        )

        explainability = (
            create_explainability(
                readings,
                spectrum,
                classification,
                anomaly_score,
                anomaly_status,
                None
            )
        )

        return {

            "medicine":
                medicine,

            "classification_confidence":
                classification[
                    "confidence"
                ],

            "anomaly_score":
                None,

            "classification_status":
                classification[
                    "status"
                ],

            "anomaly_status":
                anomaly_status,

            "final_status":
                final_status,

            "spectrum":
                spectrum,

            "scan_quality":
                scan_quality,

            "explainability":
                explainability
        }

    # --------------------------------------------------------
    # STEP 6
    # Anomaly score
    # --------------------------------------------------------

    anomaly_score = (
        calculate_anomaly_score(
            X
        )
    )

    # --------------------------------------------------------
    # STEP 7
    # Anomaly decision
    # --------------------------------------------------------

    anomaly_status = (
        determine_anomaly_status(
            anomaly_score,
            thresholds
        )
    )

    # --------------------------------------------------------
    # STEP 8
    # Final decision
    # --------------------------------------------------------

    final_status = (
        determine_final_status(
            classification,
            anomaly_status
        )
    )

    # --------------------------------------------------------
    # STEP 9
    # Explainability
    # --------------------------------------------------------

    explainability = (
        create_explainability(
            readings,
            spectrum,
            classification,
            anomaly_score,
            anomaly_status,
            thresholds
        )
    )

    # --------------------------------------------------------
    # STEP 10
    # Final response
    # --------------------------------------------------------

    return {

        "medicine":
            medicine,

        "classification_confidence":
            classification[
                "confidence"
            ],

        "anomaly_score":
            anomaly_score,

        "classification_status":
            classification[
                "status"
            ],

        "anomaly_status":
            anomaly_status,

        "final_status":
            final_status,

        "spectrum":
            spectrum,

        "scan_quality":
            scan_quality,

        "explainability":
            explainability
    }


# ============================================================
# DEMO INPUT
# ============================================================

def create_demo_readings():
    """
    Ten example AS7262 readings.

    Replace these with the actual Raspberry Pi sensor
    readings when connecting the hardware.
    """

    base = {

        "ch450": 1230,
        "ch500": 1450,
        "ch550": 1780,
        "ch570": 1900,
        "ch600": 1650,
        "ch650": 1100
    }

    readings = []

    for i in range(
        EXPECTED_READINGS
    ):

        offset = (
            (i % 3) - 1
        )

        reading = {
            channel:
                float(
                    value + offset
                )
            for channel, value
            in base.items()
        }

        readings.append(
            reading
        )

    return readings


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nRunning Raspberry Pi "
        "ML inference test..."
    )

    try:

        # ----------------------------------------------------
        # Create example 10-reading scan
        # ----------------------------------------------------

        readings = (
            create_demo_readings()
        )

        # ----------------------------------------------------
        # Authenticate
        # ----------------------------------------------------

        result = authenticate_scan(
            readings
        )

        # ----------------------------------------------------
        # Print compact result
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "AUTHENTICATION RESULT"
        )

        print(
            "=" * 60
        )

        print(
            json.dumps(
                result,
                indent=4,
                default=str
            )
        )

    except Exception as error:

        print(
            "\nInference failed:"
        )

        print(
            str(error)
        )

        raise