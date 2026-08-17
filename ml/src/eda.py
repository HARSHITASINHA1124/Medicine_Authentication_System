import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif

from .config import FEATURES
from .data_loader import load_data
from .preprocessing import AS7262Preprocessor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/medicines_as7262.csv"

OUTPUT_DIR = "eda_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def dataset_summary(df):

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nMedicines:")
    print(
        df["medicine"].value_counts()
    )

    print("\nUnique samples:")
    print(
        df["sample_id"].nunique()
    )

    print("\nMissing values:")
    print(
        df[FEATURES].isna().sum()
    )

    print("\nBasic statistics:")
    print(
        df[FEATURES].describe().round(4)
    )


# ============================================================
# RAW CHANNEL DISTRIBUTIONS
# ============================================================

def plot_channel_distributions(df):

    for feature in FEATURES:

        plt.figure(figsize=(8, 5))

        for medicine in sorted(
            df["medicine"].dropna().unique()
        ):

            values = df.loc[
                df["medicine"] == medicine,
                feature
            ]

            plt.hist(
                values,
                alpha=0.5,
                label=medicine,
                bins=15
            )

        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of {feature}"
        )
        plt.legend()
        plt.tight_layout()

        path = os.path.join(
            OUTPUT_DIR,
            f"{feature}_distribution.png"
        )

        plt.savefig(path)
        plt.close()


# ============================================================
# MEAN SPECTRAL PROFILE
# ============================================================

def plot_mean_spectral_profiles(df):

    wavelengths = np.array([
        450,
        500,
        550,
        570,
        600,
        650
    ])

    plt.figure(figsize=(9, 6))

    for medicine in sorted(
        df["medicine"].dropna().unique()
    ):

        subset = df[
            df["medicine"] == medicine
        ]

        means = [
            subset[
                f"ch{wavelength}"
            ].mean()

            for wavelength in wavelengths
        ]

        stds = [
            subset[
                f"ch{wavelength}"
            ].std()

            for wavelength in wavelengths
        ]

        means = np.array(means)
        stds = np.array(stds)

        plt.plot(
            wavelengths,
            means,
            marker="o",
            label=medicine
        )

        plt.fill_between(
            wavelengths,
            means - stds,
            means + stds,
            alpha=0.15
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral response")
    plt.title(
        "Mean AS7262 Spectral Profiles"
    )
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "mean_spectral_profiles.png"
    )

    plt.savefig(path)
    plt.close()


# ============================================================
# NORMALIZED SPECTRAL PROFILES
# ============================================================

def plot_normalized_profiles(df):

    wavelengths = np.array([
        450,
        500,
        550,
        570,
        600,
        650
    ])

    X = df[FEATURES].values

    totals = X.sum(axis=1)

    totals[
        totals == 0
    ] = 1

    normalized = (
        X / totals[:, None]
    )

    normalized_df = pd.DataFrame(
        normalized,
        columns=FEATURES
    )

    normalized_df[
        "medicine"
    ] = df[
        "medicine"
    ].values

    plt.figure(figsize=(9, 6))

    for medicine in sorted(
        normalized_df[
            "medicine"
        ].unique()
    ):

        subset = normalized_df[
            normalized_df[
                "medicine"
            ] == medicine
        ]

        means = [
            subset[
                f"ch{w}"
            ].mean()

            for w in wavelengths
        ]

        plt.plot(
            wavelengths,
            means,
            marker="o",
            label=medicine
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(
        "Normalized spectral response"
    )

    plt.title(
        "Normalized AS7262 Spectral Profiles"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "normalized_spectral_profiles.png"
    )

    plt.savefig(path)
    plt.close()


# ============================================================
# CORRELATION MATRIX
# ============================================================

def calculate_correlation(df):

    correlation = df[
        FEATURES
    ].corr()

    print("\n" + "=" * 60)
    print("CHANNEL CORRELATION")
    print("=" * 60)

    print(
        correlation.round(3)
    )

    correlation.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "channel_correlation.csv"
        )
    )

    return correlation


def plot_correlation(correlation):

    plt.figure(figsize=(8, 7))

    plt.imshow(
        correlation,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "AS7262 Channel Correlation"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "channel_correlation.png"
    )

    plt.savefig(path)
    plt.close()


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

def create_engineered_features(df):

    X = df[FEATURES]

    preprocessor = AS7262Preprocessor(
        use_normalized=True,
        use_ratios=True,
        use_differences=True,
        use_slopes=True,
        use_standard_scaling=False
    )

    engineered = (
        preprocessor.fit_transform(X)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    engineered_df = pd.DataFrame(
        engineered,
        columns=feature_names,
        index=df.index
    )

    engineered_df[
        "medicine"
    ] = df[
        "medicine"
    ].values

    return (
        engineered_df,
        preprocessor
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def calculate_feature_statistics(
    engineered_df
):

    print("\n" + "=" * 60)
    print("FEATURE SEPARATION ANALYSIS")
    print("=" * 60)

    feature_names = [
        column
        for column in engineered_df.columns
        if column != "medicine"
    ]

    medicines = (
        engineered_df[
            "medicine"
        ].unique()
    )

    results = []

    for feature in feature_names:

        groups = []

        for medicine in medicines:

            values = engineered_df.loc[
                engineered_df[
                    "medicine"
                ] == medicine,
                feature
            ].dropna()

            groups.append(values)

        # ----------------------------------------------------
        # ANOVA
        # ----------------------------------------------------

        if len(groups) >= 2:

            try:

                statistic, p_value = (
                    f_oneway(*groups)
                )

            except Exception:

                statistic = np.nan
                p_value = np.nan

        else:

            statistic = np.nan
            p_value = np.nan

        # ----------------------------------------------------
        # Variance
        # ----------------------------------------------------

        variance = (
            engineered_df[
                feature
            ].var()
        )

        results.append({
            "feature": feature,
            "anova_f": statistic,
            "anova_p": p_value,
            "variance": variance
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="anova_p",
        ascending=True
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_statistics.csv"
        ),
        index=False
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def calculate_mutual_information(
    engineered_df
):

    print("\n" + "=" * 60)
    print("MUTUAL INFORMATION")
    print("=" * 60)

    feature_names = [
        column
        for column in engineered_df.columns
        if column != "medicine"
    ]

    X = engineered_df[
        feature_names
    ]

    y = engineered_df[
        "medicine"
    ]

    # Convert class names to integer labels
    y_encoded = pd.factorize(y)[0]

    # Replace invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    scores = mutual_info_classif(
        X,
        y_encoded,
        random_state=42
    )

    results = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": scores
    })

    results = results.sort_values(
        by="mutual_information",
        ascending=False
    )

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mutual_information.csv"
        ),
        index=False
    )

    print(
        results.to_string(
            index=False
        )
    )

    return results


# ============================================================
# FEATURE CORRELATION
# ============================================================

def calculate_feature_correlation(
    engineered_df
):

    feature_names = [
        column
        for column in engineered_df.columns
        if column != "medicine"
    ]

    correlation = engineered_df[
        feature_names
    ].corr()

    correlation.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "engineered_feature_correlation.csv"
        )
    )

    return correlation


# ============================================================
# FEATURE SELECTION RECOMMENDATION
# ============================================================

def generate_feature_recommendation(
    statistics_df,
    mutual_information_df
):

    merged = pd.merge(
        statistics_df,
        mutual_information_df,
        on="feature"
    )

    # --------------------------------------------------------
    # Rank each feature separately
    # --------------------------------------------------------

    merged[
        "anova_rank"
    ] = merged[
        "anova_p"
    ].rank(
        ascending=True
    )

    merged[
        "mi_rank"
    ] = merged[
        "mutual_information"
    ].rank(
        ascending=False
    )

    # Lower combined rank = better
    merged[
        "combined_rank"
    ] = (
        merged["anova_rank"]
        +
        merged["mi_rank"]
    )

    merged = merged.sort_values(
        "combined_rank"
    )

    merged.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_ranking.csv"
        ),
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE RANKING")
    print("=" * 60)

    print(
        merged[
            [
                "feature",
                "anova_p",
                "mutual_information",
                "combined_rank"
            ]
        ].to_string(
            index=False
        )
    )

    return merged


# ============================================================
# MAIN EDA PIPELINE
# ============================================================

def run_eda():

    print(
        "\nStarting AS7262 EDA..."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    dataset_summary(df)

    # --------------------------------------------------------
    # Raw feature distributions
    # --------------------------------------------------------

    plot_channel_distributions(
        df
    )

    # --------------------------------------------------------
    # Spectral profiles
    # --------------------------------------------------------

    plot_mean_spectral_profiles(
        df
    )

    # --------------------------------------------------------
    # Normalized profiles
    # --------------------------------------------------------

    plot_normalized_profiles(
        df
    )

    # --------------------------------------------------------
    # Raw channel correlation
    # --------------------------------------------------------

    correlation = calculate_correlation(
        df
    )

    plot_correlation(
        correlation
    )

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    engineered_df, _ = (
        create_engineered_features(
            df
        )
    )

    print("\nEngineered features:")

    print(
        list(
            engineered_df.columns[
                :-1
            ]
        )
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    statistics = (
        calculate_feature_statistics(
            engineered_df
        )
    )

    # --------------------------------------------------------
    # Mutual information
    # --------------------------------------------------------

    mutual_information = (
        calculate_mutual_information(
            engineered_df
        )
    )

    # --------------------------------------------------------
    # Feature correlation
    # --------------------------------------------------------

    calculate_feature_correlation(
        engineered_df
    )

    # --------------------------------------------------------
    # Feature ranking
    # --------------------------------------------------------

    ranking = (
        generate_feature_recommendation(
            statistics,
            mutual_information
        )
    )

    print("\nEDA completed.")

    print(
        f"Results saved in: {OUTPUT_DIR}/"
    )

    return (
        df,
        engineered_df,
        ranking
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_eda()