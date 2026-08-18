from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

try:
    from .config import (
        CHANNELS,
        TARGET,
        MODEL_DIR,
        RESULTS_DIR
    )
except ImportError:
    from config import (
        CHANNELS,
        TARGET,
        MODEL_DIR,
        RESULTS_DIR
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    Path(MODEL_DIR)
    / "best_classifier.joblib"
)

OUTPUT_PATH = (
    Path(RESULTS_DIR)
    / "classification_test_predictions.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Classifier model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run classification.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Loaded classifier:"
    )

    print(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_PATH}"
        )

    df = pd.read_csv(
        TEST_PATH
    )

    required_columns = (
        CHANNELS
        + [TARGET]
    )

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: "
                f"{column}"
            )

    print(
        "\nTest dataset shape:",
        df.shape
    )

    return df


# ============================================================
# TEST CLASSIFIER
# ============================================================

def evaluate_classifier(
    model,
    test_df
):

    X_test = test_df[
        CHANNELS
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probability / confidence
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        confidence = (
            probabilities.max(
                axis=1
            )
        )

    else:

        confidence = None

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL TEST SET CLASSIFICATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    classes = model.classes_

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm_df
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    results_df = test_df.copy()

    results_df[
        "predicted_medicine"
    ] = y_pred

    results_df[
        "classification_correct"
    ] = (
        results_df[
            "medicine"
        ]
        ==
        results_df[
            "predicted_medicine"
        ]
    )

    if confidence is not None:

        results_df[
            "classification_confidence"
        ] = confidence

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    Path(
        RESULTS_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nTest predictions saved to:"
    )

    print(
        OUTPUT_PATH
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model = load_model()

    test_df = load_test_data()

    evaluate_classifier(
        model,
        test_df
    )

    print(
        "\nClassification test completed."
    )