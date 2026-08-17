

from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "medicines_as7262_pipeline_test.csv"
)


# --------------------------------------------------
# AS7262 channels
# --------------------------------------------------

CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650",
]

FEATURES = CHANNELS

# --------------------------------------------------
# Dataset columns
# --------------------------------------------------

TARGET = "medicine"

GROUP = "sample_id"

METADATA_COLUMNS = [
    "sample_id",
    "medicine",
    "batch_id",
    "manufacturer",
    "measurement_id",
]


# --------------------------------------------------
# Classification
# --------------------------------------------------

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_SPLITS = 5


# --------------------------------------------------
# Pipeline-test classes
# --------------------------------------------------

MEDICINES = [
    "cocaine",
    "MDMA",
    "2C-B",
    "FA",
    "2C-B isomer",
    "FMA",
]