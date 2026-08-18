# ml/src/data_loader.py

import pandas as pd
from pathlib import Path
from config import DATASET_PATH 


def load_data(path=None):
    """
    Load the AS7262-compatible dataset.

    The same loader will be used for:
    - ASD pipeline-test data
    - real AS7262 hardware data
    """
    file_path = Path(path) if path else Path(DATASET_PATH)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Loaded dataset: {file_path}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())