from pathlib import Path

import pandas as pd

try:
    from .config import (
        TEST_PATH,
        RESULTS_DIR,
        CHANNELS,
        TARGET
    )

    from .authentication import (
        load_models,
        authenticate_sample
    )

except ImportError:

    from config import (
        TEST_PATH,
        RESULTS_DIR,
        CHANNELS,
        TARGET
    )

    from authentication import (
        load_models,
        authenticate_sample
    )


# ============================================================
# PATH
# ============================================================

TEST_PATH = Path(
    TEST_PATH
)

RESULTS_DIR = Path(
    RESULTS_DIR
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    RESULTS_DIR /
    "authentication_test_results.csv"
)


# ============================================================
# RUN TEST DATA
# ============================================================

def run_test():

    print(
        "\n" + "=" * 60
    )

    print(
        "FULL AUTHENTICATION PIPELINE TEST"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        classifier,
        anomaly_model,
        thresholds
    ) = load_models()

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        TEST_PATH
    )

    print(
        "\nTest dataset:",
        TEST_PATH
    )

    print(
        "Rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process every sample
    # --------------------------------------------------------

    for index, row in df.iterrows():

        sample = {

            channel:
                row[channel]

            for channel in CHANNELS
        }

        try:

            result = authenticate_sample(
                sample,
                classifier,
                anomaly_model,
                thresholds
            )

            output = {

                "row_index":
                    index,

                "actual_medicine":
                    row[TARGET],

                "predicted_medicine":
                    result[
                        "medicine"
                    ],

                "classification_confidence":
                    result[
                        "classification_confidence"
                    ],

                "classification_status":
                    result[
                        "classification_status"
                    ],

                "anomaly_score":
                    result[
                        "anomaly_score"
                    ],

                "anomaly_status":
                    result[
                        "anomaly_status"
                    ],

                "final_status":
                    result[
                        "final_status"
                    ],

                "reason":
                    result[
                        "reason"
                    ]
            }

        except Exception as error:

            output = {

                "row_index":
                    index,

                "actual_medicine":
                    row[TARGET],

                "predicted_medicine":
                    None,

                "classification_confidence":
                    None,

                "classification_status":
                    "ERROR",

                "anomaly_score":
                    None,

                "anomaly_status":
                    "ERROR",

                "final_status":
                    "ERROR",

                "reason":
                    str(error)
            }

        results.append(
            output
        )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "AUTHENTICATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        "\nFinal decisions:"
    )

    print(
        results_df[
            "final_status"
        ].value_counts()
    )

    print(
        "\nClassification status:"
    )

    print(
        results_df[
            "classification_status"
        ].value_counts()
    )

    print(
        "\nAnomaly status:"
    )

    print(
        results_df[
            "anomaly_status"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Classification accuracy
    # --------------------------------------------------------

    valid = results_df[
        results_df[
            "predicted_medicine"
        ].notna()
    ]

    if len(valid) > 0:

        accuracy = (
            valid[
                "actual_medicine"
            ]
            ==
            valid[
                "predicted_medicine"
            ]
        ).mean()

        print(
            "\nClassification accuracy "
            "on test data:",
            f"{accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nDetailed results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_test()