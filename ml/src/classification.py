from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from .preprocessing import AS7262Preprocessor
except ImportError:  # pragma: no cover - direct script execution
    from preprocessing import AS7262Preprocessor


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

CHANNELS = [
    "ch450",
    "ch500",
    "ch550",
    "ch570",
    "ch600",
    "ch650"
]

TARGET = "medicine"
GROUP = "sample_id"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def normalize_sample_column(df):

    if "sample_id" not in df.columns and "sample_code" in df.columns:
        df = df.rename(columns={"sample_code": "sample_id"})

    if "sample_id" not in df.columns:
        raise ValueError(
            "Training/test data is missing 'sample_id'. "
            "Expected the split to include one row per sample/tablet."
        )

    return df


def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = normalize_sample_column(train_df)
    test_df = normalize_sample_column(test_df)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, test_df


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

def create_feature_configs():

    return {

        # 6 normalized features
        "normalized": {
            "use_normalized": True,
            "use_ratios": False,
            "use_differences": False,
            "use_slopes": False
        },

        # 6 normalized + 6 ratios = 12
        "normalized_ratios": {
            "use_normalized": True,
            "use_ratios": True,
            "use_differences": False,
            "use_slopes": False
        },

        # 6 normalized + 6 ratios + 5 differences = 17
        "normalized_ratios_differences": {
            "use_normalized": True,
            "use_ratios": True,
            "use_differences": True,
            "use_slopes": False
        },

        # All 22 engineered features
        "all_features": {
            "use_normalized": True,
            "use_ratios": True,
            "use_differences": True,
            "use_slopes": True
        }
    }


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def create_models():

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_pipeline(feature_config, model):

    preprocessor = AS7262Preprocessor(
        use_normalized=feature_config["use_normalized"],
        use_ratios=feature_config["use_ratios"],
        use_differences=feature_config["use_differences"],
        use_slopes=feature_config["use_slopes"],
        use_standard_scaling=True
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline


# ============================================================
# CROSS-VALIDATION
# ============================================================

def compare_models(train_df):

    X = train_df[CHANNELS]
    y = train_df[TARGET]
    groups = train_df[GROUP]

    feature_configs = create_feature_configs()
    models = create_models()

    # Number of different samples available per medicine
    group_counts = (
        train_df
        .groupby(TARGET)[GROUP]
        .nunique()
    )

    print("\nSamples per medicine:")
    print(group_counts)

    min_groups = group_counts.min()

    if min_groups < 2:

        raise ValueError(
            "Not enough different samples per medicine "
            "for stratified group cross-validation. "
            "Each class needs at least 2 unique sample_id values. "
            f"Current minimum is {min_groups}."
        )

    n_splits = min(
        5,
        int(min_groups)
    )

    print(
        f"\nUsing {n_splits}-fold "
        "Stratified Group Cross-Validation."
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for feature_name, feature_config in feature_configs.items():

        print(
            f"\n{'=' * 60}"
        )
        print(
            f"FEATURE SET: {feature_name}"
        )
        print(
            f"{'=' * 60}"
        )

        for model_name, model in models.items():

            print(
                f"\nTesting: "
                f"{feature_name} + {model_name}"
            )

            pipeline = create_pipeline(
                feature_config,
                model
            )

            scores = cross_validate(
                pipeline,
                X,
                y,
                groups=groups,
                cv=cv,
                scoring=[
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro"
                ],
                n_jobs=-1
            )

            accuracy = scores[
                "test_accuracy"
            ].mean()

            precision = scores[
                "test_precision_macro"
            ].mean()

            recall = scores[
                "test_recall_macro"
            ].mean()

            f1 = scores[
                "test_f1_macro"
            ].mean()

            print(
                f"Accuracy : {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall   : {recall:.4f}"
            )

            print(
                f"F1       : {f1:.4f}"
            )

            results.append({
                "feature_set": feature_name,
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    return pd.DataFrame(results)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    results = results.sort_values(
        by="f1",
        ascending=False
    )

    output_path = (
        "data/processed/"
        "classification_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to: {output_path}"
    )

    print(
        "\nMODEL RANKING:"
    )

    print(
        results.to_string(index=False)
    )

    return results


# ============================================================
# TRAIN BEST MODEL
# ============================================================

def train_best_model(
    train_df,
    best_feature_set,
    best_model_name
):

    feature_config = create_feature_configs()[
        best_feature_set
    ]

    model = create_models()[
        best_model_name
    ]

    pipeline = create_pipeline(
        feature_config,
        model
    )

    X_train = train_df[CHANNELS]
    y_train = train_df[TARGET]

    pipeline.fit(
        X_train,
        y_train
    )

    model_path = (
        "data/processed/"
        "best_classifier.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        f"\nBest classifier saved to: "
        f"{model_path}"
    )

    return pipeline


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_df, test_df = load_data()

    # --------------------------------------------------------
    # Compare feature sets + classifiers
    # --------------------------------------------------------

    results = compare_models(
        train_df
    )

    results = save_results(
        results
    )

    # --------------------------------------------------------
    # Select best combination using CV F1 score
    # --------------------------------------------------------

    best_row = results.iloc[0]

    best_feature_set = (
        best_row["feature_set"]
    )

    best_model_name = (
        best_row["model"]
    )

    print(
        "\nBEST COMBINATION"
    )

    print(
        "Feature set:",
        best_feature_set
    )

    print(
        "Model:",
        best_model_name
    )

    print(
        "CV F1:",
        f"{best_row['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Train best model on complete training data
    # --------------------------------------------------------

    best_pipeline = train_best_model(
        train_df,
        best_feature_set,
        best_model_name
    )