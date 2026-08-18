from pathlib import Path


# ============================================================
# BASE DIRECTORIES
# ============================================================

# ml/
BASE_DIR = Path(__file__).resolve().parent.parent

# ml/data/
DATA_DIR = BASE_DIR / "data"

# ml/data/raw/
RAW_DATA_DIR = DATA_DIR / "raw"

# ml/data/processed/
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ml/models/
MODEL_DIR = BASE_DIR / "models"

# ml/results/
RESULTS_DIR = BASE_DIR / "results"


# Create directories automatically
PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET PATHS
# ============================================================

# Pipeline-test dataset created from ASD data.
#
# IMPORTANT:
# This is only being used to verify that the ML pipeline
# works correctly.
#
# The final project will use the real AS7262 hardware dataset.

DATASET_PATH = (
    RAW_DATA_DIR
    / "medicine_spectral_raw.csv"
)


# Three-way split

TRAIN_PATH = (
    PROCESSED_DATA_DIR
    / "train.csv"
)

VALIDATION_PATH = (
    PROCESSED_DATA_DIR
    / "validation.csv"
)

TEST_PATH = (
    PROCESSED_DATA_DIR
    / "test.csv"
)


# ============================================================
# MODEL PATHS
# ============================================================

CLASSIFIER_PATH = (
    MODEL_DIR
    / "best_classifier.joblib"
)

ANOMALY_PROFILE_PATH = (
    MODEL_DIR
    / "anomaly_profiles.joblib"
)

ANOMALY_THRESHOLD_PATH = (
    MODEL_DIR
    / "anomaly_thresholds.joblib"
)


# ============================================================
# RESULT PATHS
# ============================================================

CLASSIFICATION_RESULTS_PATH = (
    RESULTS_DIR
    / "classification_results.csv"
)

TEST_RESULTS_PATH = (
    RESULTS_DIR
    / "test_results.csv"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

XAI_RESULTS_PATH = (
    RESULTS_DIR
    / "xai_results.csv"
)


# ============================================================
# AS7262 FEATURES
# ============================================================

CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]

FEATURES = CHANNELS 

# ============================================================
# DATASET COLUMNS
# ============================================================

TARGET = "medicine"

GROUP = "sample_id"


# Optional metadata columns.
#
# These columns are NOT ML features.

METADATA_COLUMNS = [
    "sample_id",
    "dataset_source",
    "medicine",
    "status",
    "batch_id"
]


# ============================================================
# RANDOMNESS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# DATA SPLIT
# ============================================================

# Approximately:
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#
# The actual split is performed at sample/group level.

TRAIN_SIZE = 0.70

VALIDATION_SIZE = 0.15

TEST_SIZE = 0.15


# ============================================================
# CROSS VALIDATION
# ============================================================

N_CV_SPLITS = 3

# ============================================================
# CLASSIFICATION CONFIDENCE
# ============================================================

CLASSIFICATION_LOW_THRESHOLD = 0.50


CLASSIFICATION_HIGH_THRESHOLD = 0.70


# ============================================================
# ANOMALY CONFIGURATION
# ============================================================

ANOMALY_GENUINE_PERCENTILE = 95

ANOMALY_COUNTERFEIT_PERCENTILE = 99

# ============================================================
# CLASSIFIER CONFIGURATION
# ============================================================

# Minimum classifier confidence required before
# the anomaly stage is trusted.

CLASSIFIER_CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# AUTHENTICATION RESULT LABELS
# ============================================================

GENUINE = "GENUINE"

SUSPICIOUS = "SUSPICIOUS"

COUNTERFEIT = "COUNTERFEIT"