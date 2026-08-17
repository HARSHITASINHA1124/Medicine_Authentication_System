from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler


try:
    from .config import (
        CHANNELS,
        TARGET,
        GROUP,
        TRAIN_PATH,
        MODEL_DIR
    )

    from .preprocessing import AS7262Preprocessor

except ImportError:  # pragma: no cover

    from config import (
        CHANNELS,
        TARGET,
        GROUP,
        TRAIN_PATH,
        MODEL_DIR
    )

    from preprocessing import AS7262Preprocessor


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(MODEL_DIR)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANOMALY_MODEL_PATH = (
    MODEL_DIR /
    "anomaly_model.joblib"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    train_path = Path(TRAIN_PATH)

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{train_path}"
        )

    df = pd.read_csv(
        train_path
    )

    required_columns = (
        CHANNELS +
        [
            TARGET,
            GROUP
        ]
    )

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column "
                f"'{column}' in training dataset."
            )

    print(
        "\n" + "=" * 60
    )

    print(
        "ANOMALY TRAINING DATA"
    )

    print(
        "=" * 60
    )

    print(
        "\nDataset shape:",
        df.shape
    )

    print(
        "\nObservations per medicine:"
    )

    print(
        df[TARGET].value_counts()
    )

    print(
        "\nPhysical samples per medicine:"
    )

    print(
        df.groupby(TARGET)[GROUP]
        .nunique()
    )

    return df


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor():

    return AS7262Preprocessor(

        use_normalized=True,

        use_ratios=True,

        use_differences=True,

        use_slopes=True,

        use_standard_scaling=False
    )


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

def create_engineered_features(
    df,
    preprocessor
):

    X = df[
        CHANNELS
    ]

    X_features = (
        preprocessor.fit_transform(
            X
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    X_features = pd.DataFrame(
        X_features,
        columns=feature_names,
        index=df.index
    )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    X_features = X_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Use column medians calculated from the training data.
    X_features = X_features.fillna(
        X_features.median()
    )

    return X_features


# ============================================================
# BUILD MEDICINE REFERENCE PROFILE
# ============================================================

def build_medicine_profile(
    X,
    medicine
):

    print(
        f"\nBuilding reference profile: "
        f"{medicine}"
    )

    print(
        f"Number of observations: "
        f"{len(X)}"
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Covariance estimation
    # --------------------------------------------------------

    covariance = EmpiricalCovariance(
        assume_centered=False
    )

    covariance.fit(
        X_scaled
    )

    # --------------------------------------------------------
    # Reference anomaly scores
    # --------------------------------------------------------

    scores = covariance.mahalanobis(
        X_scaled
    )

    print(
        f"Mean score: "
        f"{np.mean(scores):.4f}"
    )

    print(
        f"Maximum score: "
        f"{np.max(scores):.4f}"
    )

    # --------------------------------------------------------
    # Store medicine-specific profile
    # --------------------------------------------------------

    profile = {

        "medicine":
            medicine,

        "scaler":
            scaler,

        "covariance":
            covariance,

        "training_scores":
            scores,

        "n_samples":
            len(X),

        "feature_names":
            list(X.columns)
    }

    return profile


# ============================================================
# TRAIN MEDICINE-AWARE ANOMALY MODEL
# ============================================================

def train_medicine_aware_model(
    df,
    X_features
):

    profiles = {}

    medicines = sorted(
        df[TARGET]
        .dropna()
        .unique()
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING MEDICINE-AWARE REFERENCE MODEL"
    )

    print(
        "=" * 60
    )

    for medicine in medicines:

        # ----------------------------------------------------
        # Select only this medicine
        # ----------------------------------------------------

        mask = (
            df[TARGET]
            == medicine
        )

        X_medicine = (
            X_features.loc[mask]
        )

        # ----------------------------------------------------
        # Require enough observations
        # ----------------------------------------------------

        if len(X_medicine) < 3:

            print(
                f"\nSkipping {medicine}: "
                f"only {len(X_medicine)} observations."
            )

            continue

        # ----------------------------------------------------
        # Build reference profile
        # ----------------------------------------------------

        profile = (
            build_medicine_profile(
                X_medicine,
                medicine
            )
        )

        profiles[
            medicine
        ] = profile

    if not profiles:

        raise ValueError(
            "No medicine has enough observations "
            "to build an anomaly reference profile."
        )

    return profiles


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    profiles,
    preprocessor,
    feature_names
):

    model_package = {

        # ----------------------------------------------------
        # Version
        # ----------------------------------------------------

        "model_type":
            "medicine_aware_mahalanobis",

        "version":
            "1.0",

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        "preprocessor":
            preprocessor,

        "channels":
            CHANNELS,

        "feature_names":
            list(feature_names),

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        "target":
            TARGET,

        # ----------------------------------------------------
        # Medicine-specific profiles
        # ----------------------------------------------------

        "medicine_profiles":
            profiles
    }

    joblib.dump(
        model_package,
        ANOMALY_MODEL_PATH
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "ANOMALY MODEL SAVED"
    )

    print(
        "=" * 60
    )

    print(
        "\nModel path:"
    )

    print(
        ANOMALY_MODEL_PATH
    )

    print(
        "\nMedicines included:"
    )

    for medicine in profiles:

        print(
            f"  - {medicine}"
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_model_summary(
    profiles
):

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL SUMMARY"
    )

    print(
        "=" * 60
    )

    for medicine, profile in profiles.items():

        scores = (
            profile[
                "training_scores"
            ]
        )

        print(
            f"\n{medicine}"
        )

        print(
            f"  Samples : "
            f"{profile['n_samples']}"
        )

        print(
            f"  Mean    : "
            f"{np.mean(scores):.4f}"
        )

        print(
            f"  Median  : "
            f"{np.median(scores):.4f}"
        )

        print(
            f"  Maximum : "
            f"{np.max(scores):.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nStarting medicine-aware "
        "anomaly-model training..."
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # Create ONE preprocessing object
    # --------------------------------------------------------

    preprocessor = (
        create_preprocessor()
    )

    # --------------------------------------------------------
    # Engineer features
    # --------------------------------------------------------

    X_features = (
        create_engineered_features(
            df,
            preprocessor
        )
    )

    print(
        "\nEngineered feature count:",
        X_features.shape[1]
    )

    print(
        "\nEngineered features:"
    )

    for feature in X_features.columns:

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # Train medicine-aware profiles
    # --------------------------------------------------------

    profiles = (
        train_medicine_aware_model(
            df,
            X_features
        )
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_model_summary(
        profiles
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        profiles,
        preprocessor,
        X_features.columns
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MEDICINE-AWARE ANOMALY TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()