"""
Raspberry Pi ML Inference Pipeline
==================================

Production inference pipeline for the Medicine Authentication System.

Flow:

    10 AS7262 readings
            |
            v
    Scan aggregation
            |
            v
    Classification
            |
            +----> medicine
            |
            +----> classification confidence
            |
            v
    Medicine-aware anomaly detection
            |
            v
    Mahalanobis anomaly score
            |
            v
    Medicine-specific thresholds
            |
            v
    Final decision
            |
            +----> GENUINE
            +----> SUSPICIOUS
            +----> COUNTERFEIT
            |
            v
    Explainability
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
MODELS_DIR = ML_DIR / "models"


CLASSIFIER_PATH = (
    MODELS_DIR /
    "best_classifier.joblib"
)

ANOMALY_MODEL_PATH = (
    MODELS_DIR /
    "anomaly_model.joblib"
)

ANOMALY_THRESHOLDS_PATH = (
    MODELS_DIR /
    "anomaly_thresholds.joblib"
)


# ============================================================
# MAKE CUSTOM PREPROCESSOR IMPORTABLE
# ============================================================

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# IMPORTANT:
#
# The saved joblib files contain references to:
#
#     preprocessing.AS7262Preprocessor
#
# Therefore preprocessing must be importable BEFORE
# joblib.load() is called.

try:

    import preprocessing

except ImportError as exc:

    raise ImportError(
        "Could not import preprocessing.py. "
        f"Expected it at: {SRC_DIR / 'preprocessing.py'}"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]


# ------------------------------------------------------------
# Number of readings returned by Raspberry Pi per scan
# ------------------------------------------------------------

READINGS_PER_SCAN = 10


# ------------------------------------------------------------
# Classification confidence thresholds
# ------------------------------------------------------------

CLASSIFICATION_HIGH_THRESHOLD = 0.80
CLASSIFICATION_MEDIUM_THRESHOLD = 0.60


# ------------------------------------------------------------
# Unknown / very-low classification confidence
# ------------------------------------------------------------

CLASSIFICATION_UNKNOWN_THRESHOLD = 0.40


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load all ML artifacts required for inference.
    """

    # --------------------------------------------------------
    # Check classifier
    # --------------------------------------------------------

    if not CLASSIFIER_PATH.exists():

        raise FileNotFoundError(
            "Classifier model not found:\n"
            f"{CLASSIFIER_PATH}"
        )

    # --------------------------------------------------------
    # Check anomaly model
    # --------------------------------------------------------

    if not ANOMALY_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Anomaly model not found:\n"
            f"{ANOMALY_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check thresholds
    # --------------------------------------------------------

    if not ANOMALY_THRESHOLDS_PATH.exists():

        raise FileNotFoundError(
            "Anomaly threshold file not found:\n"
            f"{ANOMALY_THRESHOLDS_PATH}"
        )

    print(
        "Loading ML models..."
    )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    anomaly_model = joblib.load(
        ANOMALY_MODEL_PATH
    )

    anomaly_thresholds = joblib.load(
            ANOMALY_THRESHOLDS_PATH
        )

    # --------------------------------------------------------
    # Validate anomaly model
    # --------------------------------------------------------

    if not isinstance(
        anomaly_model,
        dict
    ):

        raise ValueError(
            "anomaly_model.joblib must contain "
            "the medicine-aware anomaly model dictionary."
        )

    if anomaly_model.get(
        "model_type"
    ) != "medicine_aware_mahalanobis":

        raise ValueError(
            "Unsupported anomaly model type: "
            f"{anomaly_model.get('model_type')}"
        )

    print(
        "ML models loaded successfully."
    )

    return (
        classifier,
        anomaly_model,
        anomaly_thresholds
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_reading(
    reading
):
    """
    Validate one AS7262 reading.
    """

    if not isinstance(
        reading,
        dict
    ):

        raise ValueError(
            "Each reading must be a dictionary."
        )

    missing = [
        channel
        for channel in CHANNELS
        if channel not in reading
    ]

    if missing:

        raise ValueError(
            "Reading is missing AS7262 channels: "
            f"{missing}"
        )

    values = []

    for channel in CHANNELS:

        value = reading[channel]

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"Invalid value for {channel}: "
                f"{value}"
            )

        if not np.isfinite(value):

            raise ValueError(
                f"Invalid non-finite value for "
                f"{channel}: {value}"
            )

        values.append(value)

    return values


# ============================================================
# SCAN AGGREGATION
# ============================================================

def aggregate_scan(
    readings
):
    """
    Convert 10 Raspberry Pi readings into one scan.

    Input:

        [
            {
                "ch450": ...,
                "ch500": ...,
                ...
            },
            ...
        ]

    Output:

        {
            "ch450": average,
            "ch500": average,
            ...
        }

    Median is also calculated internally to detect
    unusually noisy readings.
    """

    if not readings:

        raise ValueError(
            "No readings supplied."
        )

    validated = []

    for reading in readings:

        validated.append(
            validate_reading(
                reading
            )
        )

    X = np.asarray(
        validated,
        dtype=float
    )

    means = np.mean(
        X,
        axis=0
    )

    medians = np.median(
        X,
        axis=0
    )

    stds = np.std(
        X,
        axis=0
    )

    aggregated = {
        channel: float(
            means[index]
        )

        for index, channel
        in enumerate(CHANNELS)
    }

    # --------------------------------------------------------
    # Measurement stability
    # --------------------------------------------------------

    channel_cv = []

    for index in range(
        len(CHANNELS)
    ):

        mean_value = abs(
            means[index]
        )

        std_value = stds[index]

        if mean_value > 1e-12:

            cv = (
                std_value /
                mean_value
            )

        else:

            cv = 0.0

        channel_cv.append(
            cv
        )

    stability_cv = float(
        np.mean(channel_cv)
    )

    return {
        "aggregated_reading": aggregated,
        "mean": means,
        "median": medians,
        "std": stds,
        "stability_cv": stability_cv,
        "n_readings": len(readings)
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_scan(
    aggregated_reading,
    classifier
):
    """
    Classify the aggregated spectral scan.

    Returns:

        medicine
        confidence
        probabilities
        classification_status
    """

    X = pd.DataFrame(
        [
            [
                aggregated_reading[
                    channel
                ]

                for channel in CHANNELS
            ]
        ],
        columns=CHANNELS
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = classifier.predict(
        X
    )

    medicine = str(
        prediction[0]
    )

    # --------------------------------------------------------
    # Classification probability
    # --------------------------------------------------------

    probabilities = {}

    if hasattr(
        classifier,
        "predict_proba"
    ):

        probability_array = (
            classifier.predict_proba(X)
        )

        classes = classifier.classes_

        probabilities = {
            str(classes[index]):
            float(
                probability_array[
                    0,
                    index
                ]
            )

            for index in range(
                len(classes)
            )
        }

        confidence = float(
            np.max(
                probability_array[0]
            )
        )

    else:

        confidence = None

    # --------------------------------------------------------
    # Classification status
    # --------------------------------------------------------

    if confidence is None:

        classification_status = (
            "UNKNOWN"
        )

    elif (
        confidence >=
        CLASSIFICATION_HIGH_THRESHOLD
    ):

        classification_status = (
            "HIGH"
        )

    elif (
        confidence >=
        CLASSIFICATION_MEDIUM_THRESHOLD
    ):

        classification_status = (
            "MEDIUM"
        )

    else:

        classification_status = (
            "LOW"
        )

    return {
        "medicine": medicine,
        "confidence": confidence,
        "probabilities": probabilities,
        "classification_status":
            classification_status
    }


# ============================================================
# ENGINEERED FEATURES
# ============================================================

def create_engineered_features(
    aggregated_reading,
    anomaly_model
):
    """
    Use the EXACT preprocessor saved inside
    anomaly_model.joblib.

    This is important because the anomaly thresholds
    were generated using this preprocessing pipeline.
    """

    preprocessor = anomaly_model.get(
        "preprocessor"
    )

    if preprocessor is None:

        raise ValueError(
            "Anomaly model does not contain "
            "a saved preprocessor."
        )

    X_raw = pd.DataFrame(
        [
            [
                aggregated_reading[
                    channel
                ]

                for channel in CHANNELS
            ]
        ],
        columns=CHANNELS
    )

    X_engineered = (
        preprocessor.transform(
            X_raw
        )
    )

    feature_names = anomaly_model.get(
        "feature_names"
    )

    if feature_names is None:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    if X_engineered.shape[1] != len(
        feature_names
    ):

        raise ValueError(
            "Engineered feature count does not "
            "match the anomaly model."
        )

    return (
        X_engineered,
        list(feature_names)
    )


# ============================================================
# MAHALANOBIS ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    X_engineered,
    medicine,
    anomaly_model
):
    """
    Calculate the medicine-aware Mahalanobis distance.

    The anomaly model contains:

        medicine_profiles[medicine]
            |
            +-- StandardScaler
            |
            +-- EmpiricalCovariance

    The score is:

        sqrt(
            (x - mean)^T
            precision
            (x - mean)
        )
    """

    if not isinstance(
        anomaly_model,
        dict
    ):

        raise ValueError(
            "Anomaly model must be a dictionary."
        )

    profiles = anomaly_model.get(
        "medicine_profiles",
        {}
    )

    if medicine not in profiles:

        raise ValueError(
            f"No anomaly profile found for "
            f"medicine '{medicine}'. "
            f"Available profiles: "
            f"{list(profiles.keys())}"
        )

    profile = profiles[
        medicine
    ]

    scaler = profile.get(
        "scaler"
    )

    covariance = profile.get(
        "covariance"
    )

    if scaler is None:

        raise ValueError(
            f"No scaler found for medicine "
            f"'{medicine}'."
        )

    if covariance is None:

        raise ValueError(
            f"No covariance model found for "
            f"medicine '{medicine}'."
        )

    X = np.asarray(
        X_engineered,
        dtype=float
    )

    if X.ndim == 1:

        X = X.reshape(
            1,
            -1
        )

    # --------------------------------------------------------
    # Medicine-specific scaling
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Mahalanobis distance
    # --------------------------------------------------------

    if hasattr(
        covariance,
        "mahalanobis"
    ):

        distances_squared = (
            covariance.mahalanobis(
                X_scaled
            )
        )

    else:

        mean = covariance.location_

        precision = (
            covariance.precision_
        )

        centered = (
            X_scaled - mean
        )

        distances_squared = (
            np.einsum(
                "ij,jk,ik->i",
                centered,
                precision,
                centered
            )
        )

    distances_squared = np.maximum(
        distances_squared,
        0
    )

    scores = np.sqrt(
        distances_squared
    )

    if len(scores) == 1:

        return float(
            scores[0]
        )

    return scores


# ============================================================
# THRESHOLD LOOKUP
# ============================================================

def get_anomaly_thresholds(
    medicine,
    anomaly_thresholds
):
    """
    Get medicine-specific anomaly thresholds.
    """

    # --------------------------------------------------------
    # Current threshold format:
    #
    # {
    #   "version": "2.0",
    #   "profiles": {
    #       "MDMA": {
    #           "genuine_threshold": ...,
    #           "counterfeit_threshold": ...
    #       }
    #   }
    # }
    # --------------------------------------------------------

    if not isinstance(
        anomaly_thresholds,
        dict
    ):

        raise ValueError(
            "Invalid anomaly threshold object."
        )

    profiles = anomaly_thresholds.get(
        "profiles",
        {}
    )

    if medicine not in profiles:

        raise ValueError(
            f"No anomaly thresholds found "
            f"for medicine '{medicine}'. "
            f"Available medicines: "
            f"{list(profiles.keys())}"
        )

    profile = profiles[
        medicine
    ]

    genuine_threshold = profile.get(
        "genuine_threshold"
    )

    counterfeit_threshold = profile.get(
        "counterfeit_threshold"
    )

    if genuine_threshold is None:

        raise ValueError(
            f"Missing genuine threshold for "
            f"medicine '{medicine}'."
        )

    if counterfeit_threshold is None:

        raise ValueError(
            f"Missing counterfeit threshold for "
            f"medicine '{medicine}'."
        )

    genuine_threshold = float(
        genuine_threshold
    )

    counterfeit_threshold = float(
        counterfeit_threshold
    )

    if (
        counterfeit_threshold <
        genuine_threshold
    ):

        raise ValueError(
            f"Invalid thresholds for "
            f"medicine '{medicine}': "
            "counterfeit threshold is below "
            "genuine threshold."
        )

    return {
        "genuine_threshold":
            genuine_threshold,

        "counterfeit_threshold":
            counterfeit_threshold,

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

    Logic:

        score <= genuine threshold
            -> GENUINE

        genuine < score < counterfeit
            -> SUSPICIOUS

        score >= counterfeit threshold
            -> COUNTERFEIT
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

    if (
        anomaly_score <=
        genuine_threshold
    ):

        return "GENUINE"

    elif (
        anomaly_score <
        counterfeit_threshold
    ):

        return "SUSPICIOUS"

    else:

        return "COUNTERFEIT"


# ============================================================
# FINAL DECISION
# ============================================================

def determine_final_status(
    classification_confidence,
    classification_status,
    anomaly_status
):
    """
    Combine classification confidence and anomaly result.

    Main decision logic:

    1. Very-low classification confidence
       -> SUSPICIOUS

    2. HIGH classification + GENUINE anomaly
       -> GENUINE

    3. HIGH classification + SUSPICIOUS anomaly
       -> SUSPICIOUS

    4. HIGH classification + COUNTERFEIT anomaly
       -> COUNTERFEIT

    5. MEDIUM classification
       -> cannot be confidently authenticated
       -> SUSPICIOUS unless anomaly is strongly counterfeit

    6. LOW classification
       -> SUSPICIOUS

    The anomaly layer is therefore allowed to catch a
    counterfeit even when the classifier recognizes the
    medicine with high confidence.
    """

    # --------------------------------------------------------
    # Classification confidence unavailable
    # --------------------------------------------------------

    if classification_confidence is None:

        if anomaly_status == "COUNTERFEIT":

            return "COUNTERFEIT"

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # Very low classification confidence
    # --------------------------------------------------------

    if (
        classification_confidence <
        CLASSIFICATION_UNKNOWN_THRESHOLD
    ):

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # HIGH classification confidence
    # --------------------------------------------------------

    if (
        classification_status ==
        "HIGH"
    ):

        if anomaly_status == "GENUINE":

            return "GENUINE"

        if anomaly_status == "SUSPICIOUS":

            return "SUSPICIOUS"

        if anomaly_status == "COUNTERFEIT":

            return "COUNTERFEIT"

    # --------------------------------------------------------
    # MEDIUM classification confidence
    # --------------------------------------------------------

    if (
        classification_status ==
        "MEDIUM"
    ):

        if anomaly_status == "COUNTERFEIT":

            return "COUNTERFEIT"

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # LOW classification confidence
    # --------------------------------------------------------

    if (
        classification_status ==
        "LOW"
    ):

        if anomaly_status == "COUNTERFEIT":

            return "COUNTERFEIT"

        return "SUSPICIOUS"

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    return "SUSPICIOUS"


# ============================================================
# EXPLAINABILITY
# ============================================================

def generate_explainability(
    aggregated_reading,
    classification_result,
    anomaly_score,
    anomaly_status,
    thresholds,
    feature_values,
    feature_names,
    stability_cv
):
    """
    Generate human-readable explanation data.

    This is intended to be returned to the backend/frontend
    so the UI can explain why a medicine was classified
    as genuine, suspicious, or counterfeit.
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
    # Distance from thresholds
    # --------------------------------------------------------

    distance_from_genuine = (
        anomaly_score -
        genuine_threshold
    )

    distance_from_counterfeit = (
        anomaly_score -
        counterfeit_threshold
    )

    # --------------------------------------------------------
    # Anomaly explanation
    # --------------------------------------------------------

    if anomaly_status == "GENUINE":

        anomaly_explanation = (
            "The spectral pattern is within the "
            "expected genuine range for the "
            "predicted medicine."
        )

    elif anomaly_status == "SUSPICIOUS":

        anomaly_explanation = (
            "The spectral pattern is outside the "
            "normal genuine range but has not "
            "crossed the counterfeit decision boundary."
        )

    else:

        anomaly_explanation = (
            "The spectral pattern is far enough "
            "from the expected genuine distribution "
            "to cross the counterfeit decision boundary."
        )

    # --------------------------------------------------------
    # Classification explanation
    # --------------------------------------------------------

    confidence = (
        classification_result[
            "confidence"
        ]
    )

    medicine = (
        classification_result[
            "medicine"
        ]
    )

    if confidence is None:

        classification_explanation = (
            "The classifier did not provide "
            "a probability confidence."
        )

    elif (
        confidence >=
        CLASSIFICATION_HIGH_THRESHOLD
    ):

        classification_explanation = (
            f"The classifier strongly identifies "
            f"the sample as {medicine}."
        )

    elif (
        confidence >=
        CLASSIFICATION_MEDIUM_THRESHOLD
    ):

        classification_explanation = (
            f"The classifier identifies "
            f"{medicine}, but confidence is moderate."
        )

    else:

        classification_explanation = (
            "The classifier has low confidence "
            "in the predicted medicine."
        )

    # --------------------------------------------------------
    # Feature contribution proxy
    # --------------------------------------------------------
    #
    # We do NOT claim these are true model-specific
    # SHAP contributions.
    #
    # Instead, we report the largest absolute engineered
    # feature values after transformation as diagnostic
    # information.

    feature_values = np.asarray(
        feature_values,
        dtype=float
    ).reshape(-1)

    feature_pairs = []

    for index, value in enumerate(
        feature_values
    ):

        if index >= len(
            feature_names
        ):

            break

        feature_pairs.append({
            "feature":
                str(
                    feature_names[index]
                ),

            "value":
                float(value),

            "absolute_value":
                float(
                    abs(value)
                )
        })

    feature_pairs.sort(
        key=lambda item:
            item["absolute_value"],
        reverse=True
    )

    top_features = (
        feature_pairs[:5]
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    if stability_cv < 0.05:

        stability_status = "STABLE"

    elif stability_cv < 0.10:

        stability_status = "MODERATE"

    else:

        stability_status = "NOISY"

    return {

        "classification": {

            "medicine":
                medicine,

            "confidence":
                confidence,

            "status":
                classification_result[
                    "classification_status"
                ],

            "explanation":
                classification_explanation
        },

        "anomaly": {

            "score":
                float(
                    anomaly_score
                ),

            "genuine_threshold":
                genuine_threshold,

            "counterfeit_threshold":
                counterfeit_threshold,

            "status":
                anomaly_status,

            "distance_from_genuine":
                float(
                    distance_from_genuine
                ),

            "distance_from_counterfeit":
                float(
                    distance_from_counterfeit
                ),

            "explanation":
                anomaly_explanation
        },

        "measurement": {

            "stability_cv":
                float(
                    stability_cv
                ),

            "stability_status":
                stability_status
        },

        "top_engineered_features":
            top_features
    }


# ============================================================
# MAIN AUTHENTICATION FUNCTION
# ============================================================

def authenticate_scan(
    readings,
    classifier=None,
    anomaly_model=None,
    anomaly_thresholds=None
):
    """
    Authenticate one Raspberry Pi scan.

    Input:
        10 AS7262 readings

    Output:

        {
            "medicine": "...",
            "classification_confidence": 0.94,
            "anomaly_score": 3.21,
            "classification_status": "HIGH",
            "anomaly_status": "GENUINE",
            "final_status": "GENUINE",
            "explainability": {...}
        }
    """

    # --------------------------------------------------------
    # Load models if not supplied
    # --------------------------------------------------------

    if (
        classifier is None
        or anomaly_model is None
        or anomaly_thresholds is None
    ):

        (
            classifier,
            anomaly_model,
            anomaly_thresholds
        ) = load_models()

    # --------------------------------------------------------
    # Validate number of readings
    # --------------------------------------------------------

    if len(readings) == 0:

        raise ValueError(
            "No AS7262 readings supplied."
        )

    if len(readings) != READINGS_PER_SCAN:

        raise ValueError(
            f"Expected exactly "
            f"{READINGS_PER_SCAN} readings "
            f"for one scan, but received "
            f"{len(readings)}."
        )

    # --------------------------------------------------------
    # Aggregate the 10 readings
    # --------------------------------------------------------

    aggregation = aggregate_scan(
        readings
    )

    aggregated_reading = (
        aggregation[
            "aggregated_reading"
        ]
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = classify_scan(
        aggregated_reading,
        classifier
    )

    medicine = (
        classification[
            "medicine"
        ]
    )

    classification_confidence = (
        classification[
            "confidence"
        ]
    )

    classification_status = (
        classification[
            "classification_status"
        ]
    )

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    (
        engineered,
        feature_names
    ) = create_engineered_features(
        aggregated_reading,
        anomaly_model
    )

    # --------------------------------------------------------
    # Anomaly score
    # --------------------------------------------------------

    anomaly_score = (
        calculate_anomaly_score(
            engineered,
            medicine,
            anomaly_model
        )
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = (
        get_anomaly_thresholds(
            medicine,
            anomaly_thresholds
        )
    )

    # --------------------------------------------------------
    # Anomaly status
    # --------------------------------------------------------

    anomaly_status = (
        determine_anomaly_status(
            anomaly_score,
            thresholds
        )
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    final_status = (
        determine_final_status(
            classification_confidence,
            classification_status,
            anomaly_status
        )
    )

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    explainability = (
        generate_explainability(
            aggregated_reading,
            classification,
            anomaly_score,
            anomaly_status,
            thresholds,
            engineered,
            feature_names,
            aggregation[
                "stability_cv"
            ]
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "medicine":
            medicine,

        "classification_confidence":
            (
                None
                if classification_confidence
                is None

                else float(
                    classification_confidence
                )
            ),

        "anomaly_score":
            float(
                anomaly_score
            ),

        "classification_status":
            classification_status,

        "anomaly_status":
            anomaly_status,

        "final_status":
            final_status,

        "explainability":
            explainability,

        "scan_data": {

            "n_readings":
                aggregation[
                    "n_readings"
                ],

            "aggregated_reading":
                aggregated_reading,

            "channel_std":
                {
                    channel:
                        float(
                            aggregation[
                                "std"
                            ][index]
                        )

                    for index, channel
                    in enumerate(CHANNELS)
                },

            "stability_cv":
                float(
                    aggregation[
                        "stability_cv"
                    ]
                )
        },

        "classification_probabilities":
            classification[
                "probabilities"
            ]
    }

    return result


# ============================================================
# SAFE JSON CONVERSION
# ============================================================

def make_json_serializable(
    obj
):
    """
    Convert numpy values to normal Python values.
    """

    if isinstance(
        obj,
        dict
    ):

        return {
            key:
                make_json_serializable(
                    value
                )

            for key, value
            in obj.items()
        }

    if isinstance(
        obj,
        list
    ):

        return [
            make_json_serializable(
                value
            )

            for value in obj
        ]

    if isinstance(
        obj,
        tuple
    ):

        return [
            make_json_serializable(
                value
            )

            for value in obj
        ]

    if isinstance(
        obj,
        np.integer
    ):

        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):

        return float(obj)

    if isinstance(
        obj,
        np.ndarray
    ):

        return obj.tolist()

    return obj


# ============================================================
# TEST DATA
# ============================================================

def create_test_readings():
    """
    Generate 10 example readings for testing.

    Replace these values with actual Raspberry Pi
    AS7262 readings when hardware is available.
    """

    base = {

        "ch450": 1000.0,
        "ch500": 1200.0,
        "ch550": 1400.0,
        "ch570": 1500.0,
        "ch600": 1300.0,
        "ch650": 900.0
    }

    readings = []

    for i in range(
        READINGS_PER_SCAN
    ):

        # Small deterministic variation
        # to simulate repeated sensor readings.

        factor = (
            1.0 +
            (i - 4.5) * 0.002
        )

        reading = {
            channel:
                value * factor

            for channel, value
            in base.items()
        }

        readings.append(
            reading
        )

    return readings


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nRunning Raspberry Pi ML inference test..."
    )

    try:

        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        (
            classifier,
            anomaly_model,
            anomaly_thresholds
        ) = load_models()

        # ----------------------------------------------------
        # Generate test readings
        # ----------------------------------------------------

        readings = (
            create_test_readings()
        )

        print(
            f"\nReceived "
            f"{len(readings)} AS7262 readings."
        )

        # ----------------------------------------------------
        # Authenticate
        # ----------------------------------------------------

        result = authenticate_scan(
            readings,
            classifier,
            anomaly_model,
            anomaly_thresholds
        )

        # ----------------------------------------------------
        # JSON-safe result
        # ----------------------------------------------------

        result = (
            make_json_serializable(
                result
            )
        )

        print(
            "\n" + "=" * 60
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
                indent=4
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