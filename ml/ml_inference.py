
"""
Medicine Authentication System
================================

Production ML inference pipeline.

Flow:

    10 AS7262 readings
            |
            v
    Scan aggregation
            |
            v
    Medicine classification
            |
            v
    Saved preprocessing
            |
            v
    Medicine-aware Mahalanobis anomaly detection
            |
            v
    Medicine-specific thresholds
            |
            v
    Final authentication decision
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
    MODELS_DIR / "best_classifier.joblib"
)

ANOMALY_MODEL_PATH = (
    MODELS_DIR / "anomaly_model.joblib"
)

ANOMALY_THRESHOLDS_PATH = (
    MODELS_DIR / "anomaly_thresholds.joblib"
)


# ============================================================
# MAKE CUSTOM PREPROCESSOR IMPORTABLE
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Saved joblib models may contain references to:
#
#     preprocessing.AS7262Preprocessor
#
# Therefore preprocessing.py must be importable before
# joblib.load() is called.

try:
    from src import preprocessing
    sys.modules.setdefault("preprocessing", preprocessing)

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

READINGS_PER_SCAN = 10


# Classification confidence thresholds

CLASSIFICATION_HIGH_THRESHOLD = 0.80
CLASSIFICATION_MEDIUM_THRESHOLD = 0.60
CLASSIFICATION_UNKNOWN_THRESHOLD = 0.40


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load all ML artifacts required for inference.

    Returns:
        classifier
        anomaly_model
        anomaly_thresholds
    """

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            "Classifier model not found:\n"
            f"{CLASSIFIER_PATH}"
        )

    if not ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Anomaly model not found:\n"
            f"{ANOMALY_MODEL_PATH}"
        )

    if not ANOMALY_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            "Anomaly threshold file not found:\n"
            f"{ANOMALY_THRESHOLDS_PATH}"
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

    return (
        classifier,
        anomaly_model,
        anomaly_thresholds
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_reading(reading):
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

def aggregate_scan(readings):
    """
    Convert 10 Raspberry Pi readings into one scan.

    Mean values are used as the aggregated spectrum.

    Also calculates:

        median
        channel standard deviation
        coefficient of variation
        measurement stability
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
                aggregated_reading[channel]
                for channel in CHANNELS
            ]
        ],
        columns=CHANNELS
    )

    prediction = classifier.predict(
        X
    )

    medicine = str(
        prediction[0]
    )

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
    Use the exact preprocessing pipeline saved
    inside anomaly_model.joblib.
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
                aggregated_reading[channel]
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
    Calculate medicine-aware Mahalanobis distance.

    The anomaly model contains:

        medicine_profiles[medicine]
            |
            +-- StandardScaler
            |
            +-- covariance model
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

    Expected structure:

        {
            "version": "2.0",
            "profiles": {
                "MedicineName": {
                    "genuine_threshold": ...,
                    "counterfeit_threshold": ...
                }
            }
        }
    """

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

    Rules:

        HIGH + GENUINE
            -> GENUINE

        HIGH + SUSPICIOUS
            -> SUSPICIOUS

        HIGH + COUNTERFEIT
            -> COUNTERFEIT

        MEDIUM
            -> SUSPICIOUS unless counterfeit

        LOW
            -> SUSPICIOUS unless counterfeit

        Very-low confidence
            -> SUSPICIOUS unless counterfeit
    """

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

    These feature values are diagnostic feature
    magnitudes. They are NOT claimed to be SHAP
    contributions.
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
    # Feature diagnostic information
    # --------------------------------------------------------

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
    # Measurement stability
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
        readings = list of exactly 10 AS7262 readings

    Output:
        dictionary containing:

            medicine
            classification confidence
            anomaly score
            classification status
            anomaly status
            final status
            explainability
            scan data
            classification probabilities
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
    # Validate readings
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
    # Aggregate 10 readings
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
    # Feature preprocessing
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
    # Medicine-specific thresholds
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
    # Final authentication decision
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

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "medicine":
            medicine,

        "classification_confidence":
            (
                None
                if classification_confidence is None

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

def make_json_serializable(obj):
    """
    Convert NumPy values into normal Python values.

    Useful when authenticate_scan() output is passed
    to Flask/FastAPI/Node.js or json.dumps().
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

