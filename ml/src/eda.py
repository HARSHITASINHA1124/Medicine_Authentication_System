import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif


try:
    from .config import (
        DATASET_PATH,
        CHANNELS,
        TARGET,
        GROUP
    )

    from .data_loader import load_data

    from .preprocessing import AS7262Preprocessor

except ImportError:  # pragma: no cover

    from config import (
        DATASET_PATH,
        CHANNELS,
        TARGET,
        GROUP
    )

    from data_loader import load_data

    from preprocessing import AS7262Preprocessor


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path comes directly from config.py
DATA_PATH = DATASET_PATH

# EDA output directory
OUTPUT_DIR = BASE_DIR / "eda_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def dataset_summary(df):

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Dataset path: {DATA_PATH}"
    )

    print(
        f"Number of rows: {len(df)}"
    )

    print(
        f"Number of columns: {len(df.columns)}"
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    print("\nMedicines:")

    print(
        df[TARGET].value_counts()
    )

    print("\nUnique samples:")

    print(
        df[GROUP].nunique()
    )

    print("\nMissing values:")

    print(
        df[CHANNELS].isna().sum()
    )

    print("\nBasic statistics:")

    print(
        df[CHANNELS]
        .describe()
        .round(4)
    )


# ============================================================
# RAW CHANNEL DISTRIBUTIONS
# ============================================================

def plot_channel_distributions(df):

    for feature in CHANNELS:

        plt.figure(
            figsize=(8, 5)
        )

        for medicine in sorted(
            df[TARGET]
            .dropna()
            .unique()
        ):

            values = df.loc[
                df[TARGET] == medicine,
                feature
            ]

            plt.hist(
                values,
                alpha=0.5,
                label=medicine,
                bins=15
            )

        plt.xlabel(feature)

        plt.ylabel(
            "Frequency"
        )

        plt.title(
            f"Distribution of {feature}"
        )

        plt.legend()

        plt.tight_layout()

        path = (
            OUTPUT_DIR
            / f"{feature}_distribution.png"
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

    plt.figure(
        figsize=(9, 6)
    )

    for medicine in sorted(
        df[TARGET]
        .dropna()
        .unique()
    ):

        subset = df[
            df[TARGET] == medicine
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

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Spectral response"
    )

    plt.title(
        "Mean AS7262 Spectral Profiles"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "mean_spectral_profiles.png"
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

    X = df[
        CHANNELS
    ].values

    totals = X.sum(
        axis=1
    )

    totals[
        totals == 0
    ] = 1

    normalized = (
        X / totals[:, None]
    )

    normalized_df = pd.DataFrame(
        normalized,
        columns=CHANNELS,
        index=df.index
    )

    normalized_df[
        TARGET
    ] = df[
        TARGET
    ].values

    plt.figure(
        figsize=(9, 6)
    )

    for medicine in sorted(
        normalized_df[
            TARGET
        ].unique()
    ):

        subset = normalized_df[
            normalized_df[
                TARGET
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

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Normalized spectral response"
    )

    plt.title(
        "Normalized AS7262 Spectral Profiles"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "normalized_spectral_profiles.png"
    )

    plt.savefig(path)

    plt.close()


# ============================================================
# CORRELATION MATRIX
# ============================================================

def calculate_correlation(df):

    correlation = df[
        CHANNELS
    ].corr()

    print(
        "\n" + "=" * 60
    )

    print(
        "CHANNEL CORRELATION"
    )

    print(
        "=" * 60
    )

    print(
        correlation.round(3)
    )

    correlation.to_csv(
        OUTPUT_DIR
        / "channel_correlation.csv"
    )

    return correlation


def plot_correlation(correlation):

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        correlation,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(
            len(correlation.columns)
        ),
        correlation.columns,
        rotation=45
    )

    plt.yticks(
        range(
            len(correlation.columns)
        ),
        correlation.columns
    )

    plt.title(
        "AS7262 Channel Correlation"
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "channel_correlation.png"
    )

    plt.savefig(path)

    plt.close()


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

def create_engineered_features(df):

    X = df[
        CHANNELS
    ]

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
        TARGET
    ] = df[
        TARGET
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

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE SEPARATION ANALYSIS"
    )

    print(
        "=" * 60
    )

    feature_names = [
        column
        for column in engineered_df.columns
        if column != TARGET
    ]

    medicines = (
        engineered_df[
            TARGET
        ].unique()
    )

    results = []

    for feature in feature_names:

        groups = []

        for medicine in medicines:

            values = engineered_df.loc[
                engineered_df[
                    TARGET
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
        OUTPUT_DIR
        / "feature_statistics.csv",
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

    print(
        "\n" + "=" * 60
    )

    print(
        "MUTUAL INFORMATION"
    )

    print(
        "=" * 60
    )

    feature_names = [
        column
        for column in engineered_df.columns
        if column != TARGET
    ]

    X = engineered_df[
        feature_names
    ]

    y = engineered_df[
        TARGET
    ]

    # Convert medicine names into integer labels
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
        OUTPUT_DIR
        / "mutual_information.csv",
        index=False
    )

    print(
        results.to_string(
            index=False
        )
    )

    return results


# ============================================================
# ENGINEERED FEATURE CORRELATION
# ============================================================

def calculate_feature_correlation(
    engineered_df
):

    feature_names = [
        column
        for column in engineered_df.columns
        if column != TARGET
    ]

    correlation = engineered_df[
        feature_names
    ].corr()

    correlation.to_csv(
        OUTPUT_DIR
        / "engineered_feature_correlation.csv"
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
    # Rank features
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
        OUTPUT_DIR
        / "feature_ranking.csv",
        index=False
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE RANKING"
    )

    print(
        "=" * 60
    )

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

    print(
        f"\nUsing dataset:"
        f"\n{DATA_PATH}"
    )

    # --------------------------------------------------------
    # Load dataset from config.py DATASET_PATH
    # --------------------------------------------------------

    df = load_data(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Basic summary
    # --------------------------------------------------------

    dataset_summary(
        df
    )

    # --------------------------------------------------------
    # Raw channel distributions
    # --------------------------------------------------------

    plot_channel_distributions(
        df
    )

    # --------------------------------------------------------
    # Mean spectral profiles
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

    correlation = (
        calculate_correlation(
            df
        )
    )

    plot_correlation(
        correlation
    )

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    (
        engineered_df,
        _
    ) = create_engineered_features(
        df
    )

    print(
        "\nEngineered features:"
    )

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
    # Engineered feature correlation
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

    print(
        "\nEDA completed."
    )

    print(
        f"\nResults saved in:"
        f"\n{OUTPUT_DIR}"
    )

    return (
        df,
        engineered_df,
        ranking
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_eda()