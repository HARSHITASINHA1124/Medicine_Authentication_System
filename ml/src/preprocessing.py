import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# ============================================================
# AS7262 CONFIGURATION
# ============================================================

WAVELENGTHS = np.array([
    450,
    500,
    550,
    570,
    600,
    650
], dtype=float)

CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]


# ============================================================
# PREPROCESSING + FEATURE ENGINEERING
# ============================================================

class AS7262Preprocessor(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        use_normalized=True,
        use_ratios=True,
        use_differences=True,
        use_slopes=True,
        use_standard_scaling=True
    ):

        self.use_normalized = use_normalized
        self.use_ratios = use_ratios
        self.use_differences = use_differences
        self.use_slopes = use_slopes
        self.use_standard_scaling = use_standard_scaling

        self.scaler = StandardScaler()

        self.feature_names_ = None
        self.fitted_ = False


    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _validate_input(self, X):

        if isinstance(X, pd.DataFrame):

            missing = [
                c for c in CHANNELS
                if c not in X.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing AS7262 channels: {missing}"
                )

            values = X[CHANNELS].values

        else:

            values = np.asarray(X)

            if values.ndim == 1:

                values = values.reshape(1, -1)

            if values.shape[1] != 6:

                raise ValueError(
                    "AS7262 input must contain exactly "
                    "6 channels: "
                    f"{CHANNELS}"
                )

        values = values.astype(float)

        if not np.isfinite(values).all():

            raise ValueError(
                "Input contains NaN or infinite values."
            )

        return values


    # ========================================================
    # NORMALIZED SPECTRUM
    # ========================================================

    def _normalized_spectrum(self, X):

        totals = X.sum(axis=1)

        if np.any(totals <= 0):

            raise ValueError(
                "Spectral intensity sum must be greater than zero."
            )

        return X / totals[:, None]


    # ========================================================
    # SPECTRAL RATIOS
    # ========================================================

    def _spectral_ratios(self, X):

        epsilon = 1e-10

        ratios = np.column_stack([

            X[:, 0] / (X[:, 1] + epsilon),

            X[:, 1] / (X[:, 2] + epsilon),

            X[:, 2] / (X[:, 3] + epsilon),

            X[:, 3] / (X[:, 4] + epsilon),

            X[:, 4] / (X[:, 5] + epsilon),

            X[:, 0] / (X[:, 5] + epsilon)

        ])

        return ratios


    # ========================================================
    # ADJACENT DIFFERENCES
    # ========================================================

    def _spectral_differences(self, X):

        return np.column_stack([

            X[:, 1] - X[:, 0],

            X[:, 2] - X[:, 1],

            X[:, 3] - X[:, 2],

            X[:, 4] - X[:, 3],

            X[:, 5] - X[:, 4]

        ])


    # ========================================================
    # SPECTRAL SLOPES
    # ========================================================

    def _spectral_slopes(self, X):

        wavelength_differences = np.diff(
            WAVELENGTHS
        )

        intensity_differences = np.diff(
            X,
            axis=1
        )

        return (
            intensity_differences
            / wavelength_differences
        )


    # ========================================================
    # FEATURE GENERATION
    # ========================================================

    def _create_features(self, X):

        feature_blocks = []
        feature_names = []


        # ----------------------------------------------------
        # 1. NORMALIZED SPECTRAL INTENSITIES
        # ----------------------------------------------------

        if self.use_normalized:

            normalized = self._normalized_spectrum(X)

            feature_blocks.append(
                normalized
            )

            feature_names.extend([
                "norm_450",
                "norm_500",
                "norm_550",
                "norm_570",
                "norm_600",
                "norm_650"
            ])

        else:

            normalized = X


        # ----------------------------------------------------
        # 2. SPECTRAL RATIOS
        # ----------------------------------------------------

        if self.use_ratios:

            ratios = self._spectral_ratios(
                normalized
            )

            feature_blocks.append(
                ratios
            )

            feature_names.extend([
                "ratio_450_500",
                "ratio_500_550",
                "ratio_550_570",
                "ratio_570_600",
                "ratio_600_650",
                "ratio_450_650"
            ])


        # ----------------------------------------------------
        # 3. ADJACENT DIFFERENCES
        # ----------------------------------------------------

        if self.use_differences:

            differences = self._spectral_differences(
                normalized
            )

            feature_blocks.append(
                differences
            )

            feature_names.extend([
                "diff_450_500",
                "diff_500_550",
                "diff_550_570",
                "diff_570_600",
                "diff_600_650"
            ])


        # ----------------------------------------------------
        # 4. SPECTRAL SLOPES
        # ----------------------------------------------------

        if self.use_slopes:

            slopes = self._spectral_slopes(
                normalized
            )

            feature_blocks.append(
                slopes
            )

            feature_names.extend([
                "slope_450_500",
                "slope_500_550",
                "slope_550_570",
                "slope_570_600",
                "slope_600_650"
            ])


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        features = np.column_stack(
            feature_blocks
        )

        self.feature_names_ = feature_names

        return features


    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X, y=None):

        X = self._validate_input(X)

        features = self._create_features(X)

        if self.use_standard_scaling:

            self.scaler.fit(
                features
            )

        self.fitted_ = True

        return self


    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(self, X):

        if not self.fitted_:

            raise RuntimeError(
                "AS7262Preprocessor must be fitted "
                "before calling transform()."
            )

        X = self._validate_input(X)

        features = self._create_features(X)

        if self.use_standard_scaling:

            features = self.scaler.transform(
                features
            )

        return features


    # ========================================================
    # FIT + TRANSFORM
    # ========================================================

    def fit_transform(
        self,
        X,
        y=None
    ):

        self.fit(X, y)

        return self.transform(X)


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    def get_feature_names_out(
        self,
        input_features=None
    ):

        if self.feature_names_ is None:

            raise RuntimeError(
                "Preprocessor has not been fitted yet."
            )

        return np.array(
            self.feature_names_
        )