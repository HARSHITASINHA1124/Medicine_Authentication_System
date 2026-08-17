import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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

MEDICINE_COLUMN = "medicine"


# ============================================================
# AUTHENTICATION ENGINE
# ============================================================

class MedicineAuthenticator:

    def __init__(self, threshold=3.0):

        self.threshold = threshold

        self.scaler = StandardScaler()

        # Reference spectral profile for each medicine
        self.reference_profiles = {}

        self.fitted_ = False


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    def _validate_input(self, df):

        missing = [
            column
            for column in CHANNELS + [MEDICINE_COLUMN]
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )


    # ========================================================
    # FIT AUTHENTICATION PROFILES
    # ========================================================

    def fit(self, df):

        self._validate_input(df)

        X = df[CHANNELS].astype(float)

        # Scale spectral measurements
        X_scaled = self.scaler.fit_transform(X)

        scaled_df = pd.DataFrame(
            X_scaled,
            columns=CHANNELS,
            index=df.index
        )

        # Create one reference profile for each medicine
        for medicine in df[MEDICINE_COLUMN].unique():

            medicine_data = scaled_df[
                df[MEDICINE_COLUMN] == medicine
            ]

            self.reference_profiles[medicine] = (
                medicine_data.mean(axis=0).values
            )

        self.fitted_ = True

        return self


    # ========================================================
    # CALCULATE AUTHENTICATION SCORE
    # ========================================================

    def calculate_score(self, sample):

        if not self.fitted_:

            raise RuntimeError(
                "Authenticator must be fitted before "
                "calculating authentication scores."
            )

        if isinstance(sample, pd.DataFrame):

            X = sample[CHANNELS].astype(float)

        else:

            X = pd.DataFrame(
                [sample],
                columns=CHANNELS
            )

        X_scaled = self.scaler.transform(X)

        scores = {}

        for medicine, reference in (
            self.reference_profiles.items()
        ):

            distance = pairwise_distances(
                X_scaled,
                reference.reshape(1, -1),
                metric="euclidean"
            ).flatten()[0]

            scores[medicine] = distance

        return scores


    # ========================================================
    # AUTHENTICATE SAMPLE
    # ========================================================

    def authenticate(
        self,
        sample,
        predicted_medicine
    ):

        scores = self.calculate_score(sample)

        if predicted_medicine not in scores:

            raise ValueError(
                f"No reference profile found for "
                f"{predicted_medicine}"
            )

        score = scores[predicted_medicine]

        if score <= self.threshold:

            status = "genuine"

        else:

            status = "suspicious"

        return {
            "predicted_medicine": predicted_medicine,
            "authentication_score": float(score),
            "threshold": self.threshold,
            "status": status
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    print(
        "MedicineAuthenticator is ready."
    )

    print(
        "Authentication requires genuine "
        "reference hardware data before "
        "the threshold can be calibrated."
    )