import pandas as pd

from .config import FEATURES, MEDICINES


def validate_data(df):

    print("\n========== DATA VALIDATION ==========")

    # ----------------------------------
    # Basic information
    # ----------------------------------

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ----------------------------------
    # Required medicines
    # ----------------------------------

    print("\nMedicine counts:")

    print(
        df["medicine"].value_counts()
    )

    invalid_medicines = set(df["medicine"].dropna()) - set(MEDICINES)

    if invalid_medicines:
        print(
            f"\nWARNING: Unknown medicines found: "
            f"{invalid_medicines}"
        )

    # ----------------------------------
    # Missing values
    # ----------------------------------

    print("\nMissing values:")

    missing = df[FEATURES].isna().sum()

    print(missing)

    if missing.sum() > 0:
        print("\nWARNING: Missing spectral values detected.")

    # ----------------------------------
    # Numeric conversion
    # ----------------------------------

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # ----------------------------------
    # Check non-numeric values
    # ----------------------------------

    numeric_missing = df[FEATURES].isna().sum()

    if numeric_missing.sum() > 0:

        print(
            "\nWARNING: Some spectral values "
            "could not be converted to numbers."
        )

    # ----------------------------------
    # Duplicate rows
    # ----------------------------------

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicates}")

    # ----------------------------------
    # Sample counts
    # ----------------------------------

    print("\nUnique samples:")

    print(
        df["sample_id"].nunique()
    )

    # ----------------------------------
    # Spectral statistics
    # ----------------------------------

    print("\nSpectral statistics:")

    print(
        df[FEATURES].describe().round(4)
    )

    print("\n=====================================\n")

    return df