import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/raw/medicine_spectral_raw.csv"

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

RANDOM_STATE = 42
N_SPLITS = 3


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    return df


# --------------------------------------------------
# STRATIFIED GROUP SPLIT
# --------------------------------------------------

def split_data(df):

    required_columns = [
        "sample_id",
        "medicine"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"{column} column not found in dataset."
            )

    # sample_id = physical tablet/sample
    # medicine = class we want to preserve
    groups = df["sample_id"]
    labels = df["medicine"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Take the first valid fold
    for train_indices, test_indices in splitter.split(
        df,
        y=labels,
        groups=groups
    ):

        train_df = df.iloc[
            train_indices
        ].reset_index(drop=True)

        test_df = df.iloc[
            test_indices
        ].reset_index(drop=True)

        # Make sure every medicine is present
        # in both train and test.
        train_classes = set(
            train_df["medicine"]
        )

        test_classes = set(
            test_df["medicine"]
        )

        all_classes = set(
            df["medicine"]
        )

        if (
            train_classes == all_classes
            and test_classes == all_classes
        ):
            return train_df, test_df

    raise ValueError(
        "Could not create a split containing "
        "every medicine in both train and test. "
        "More samples/tablets per medicine are required."
    )


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

def save_data(train_df, test_df):

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print(
        "\nTrain medicines:"
    )

    print(
        train_df["medicine"].value_counts()
    )

    print(
        "\nTest medicines:"
    )

    print(
        test_df["medicine"].value_counts()
    )

    # Verify that no tablet occurs in both sets.
    train_samples = set(
        train_df["sample_id"]
    )

    test_samples = set(
        test_df["sample_id"]
    )

    overlap = train_samples.intersection(
        test_samples
    )

    print(
        "\nSample overlap:",
        len(overlap)
    )

    if overlap:
        raise ValueError(
            "Data leakage detected: "
            "same sample appears in train and test."
        )

    print(
        "\nTrain saved to:",
        TRAIN_PATH
    )

    print(
        "Test saved to:",
        TEST_PATH
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    train_df, test_df = split_data(df)

    save_data(
        train_df,
        test_df
    )