import pandas as pd

FEATURES = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]

TARGET = "medicine"


def load_data(path):
    df = pd.read_csv('ml\data\processed\medicines_as7262.csv')

    # Rename current ASD dataset column if necessary
    if "Sample Code" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={
            "Sample Code": "sample_id"
        })

    # For the current ASD dataset, create missing metadata columns
    # so that the same pipeline can later accept hardware data.
    if "batch_id" not in df.columns:
        df["batch_id"] = df["sample_id"]

    if "manufacturer" not in df.columns:
        df["manufacturer"] = "unknown"

    if "measurement_id" not in df.columns:
        df["measurement_id"] = 1

    required = [
        "sample_id",
        "medicine",
        "batch_id",
        "manufacturer",
        "measurement_id",
        *FEATURES
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df