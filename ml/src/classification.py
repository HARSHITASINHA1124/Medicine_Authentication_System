from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


try:
    from .config import (
        CHANNELS,
        TARGET,
        GROUP,
        RANDOM_STATE,
        N_CV_SPLITS,
        TRAIN_PATH,
        TEST_PATH,
        MODEL_DIR,
        RESULTS_DIR
    )

    from .preprocessing import AS7262Preprocessor

except ImportError:  # pragma: no cover

    from config import (
        CHANNELS,
        TARGET,
        GROUP,
        RANDOM_STATE,
        N_CV_SPLITS,
        TRAIN_PATH,
        TEST_PATH,
        MODEL_DIR,
        RESULTS_DIR
    )

    from preprocessing import AS7262Preprocessor


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR = Path(MODEL_DIR)
RESULTS_DIR = Path(RESULTS_DIR)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    # --------------------------------------------------------
    # Backward compatibility
    # --------------------------------------------------------

    if (
        GROUP not in train_df.columns
        and "sample_code" in train_df.columns
    ):

        train_df = train_df.rename(
            columns={
                "sample_code": GROUP
            }
        )

    if (
        GROUP not in test_df.columns
        and "sample_code" in test_df.columns
    ):

        test_df = test_df.rename(
            columns={
                "sample_code": GROUP
            }
        )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required = (
        CHANNELS
        + [TARGET, GROUP]
    )

    for column in required:

        if column not in train_df.columns:

            raise ValueError(
                f"Column '{column}' missing "
                "from train.csv."
            )

        if column not in test_df.columns:

            raise ValueError(
                f"Column '{column}' missing "
                "from test.csv."
            )

    print("\n" + "=" * 60)
    print("CLASSIFICATION DATA")
    print("=" * 60)

    print(
        "\nTrain shape:",
        train_df.shape
    )

    print(
        "Test shape:",
        test_df.shape
    )

    print(
        "\nTrain samples per medicine:"
    )

    print(
        train_df[TARGET].value_counts()
    )

    print(
        "\nTest samples per medicine:"
    )

    print(
        test_df[TARGET].value_counts()
    )

    return train_df, test_df


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

def create_feature_configs():

    return {

        "normalized": {

            "use_normalized": True,
            "use_ratios": False,
            "use_differences": False,
            "use_slopes": False
        },

        "normalized_ratios": {

            "use_normalized": True,
            "use_ratios": True,
            "use_differences": False,
            "use_slopes": False
        },

        "normalized_ratios_differences": {

            "use_normalized": True,
            "use_ratios": True,
            "use_differences": True,
            "use_slopes": False
        },

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

        "Logistic Regression":

            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            ),

        "SVM":

            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ),

        "Random Forest":

            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            )
    }


# ============================================================
# CREATE ML PIPELINE
# ============================================================

def create_pipeline(
    feature_config,
    model
):

    preprocessor = AS7262Preprocessor(

        use_normalized=
        feature_config[
            "use_normalized"
        ],

        use_ratios=
        feature_config[
            "use_ratios"
        ],

        use_differences=
        feature_config[
            "use_differences"
        ],

        use_slopes=
        feature_config[
            "use_slopes"
        ],

        use_standard_scaling=True
    )

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            model
        )
    ])

    return pipeline


# ============================================================
# DETERMINE CV SPLITS
# ============================================================

def determine_cv_splits(
    train_df
):

    group_counts = (
        train_df
        .groupby(TARGET)[GROUP]
        .nunique()
    )

    print(
        "\nPhysical samples per medicine:"
    )

    print(
        group_counts
    )

    min_groups = int(
        group_counts.min()
    )

    n_splits = min(
        N_CV_SPLITS,
        min_groups
    )

    if n_splits < 2:

        raise ValueError(
            "At least 2 physical samples "
            "per medicine are required "
            "for cross-validation."
        )

    print(
        f"\nUsing {n_splits}-fold "
        "StratifiedGroupKFold."
    )

    return n_splits


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    train_df
):

    X = train_df[
        CHANNELS
    ]

    y = train_df[
        TARGET
    ]

    groups = train_df[
        GROUP
    ]

    feature_configs = (
        create_feature_configs()
    )

    models = (
        create_models()
    )

    n_splits = determine_cv_splits(
        train_df
    )

    cv = StratifiedGroupKFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    results = []

    # --------------------------------------------------------
    # Feature configurations
    # --------------------------------------------------------

    for (
        feature_name,
        feature_config
    ) in feature_configs.items():

        print(
            "\n" + "=" * 60
        )

        print(
            "FEATURE SET:",
            feature_name
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        for (
            model_name,
            model
        ) in models.items():

            print(
                f"\nTesting: "
                f"{feature_name} + "
                f"{model_name}"
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

                n_jobs=-1,

                error_score="raise"
            )

            accuracy = (
                scores[
                    "test_accuracy"
                ].mean()
            )

            precision = (
                scores[
                    "test_precision_macro"
                ].mean()
            )

            recall = (
                scores[
                    "test_recall_macro"
                ].mean()
            )

            f1 = (
                scores[
                    "test_f1_macro"
                ].mean()
            )

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

                "feature_set":
                    feature_name,

                "model":
                    model_name,

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

    results = results.sort_values(

        by="f1",

        ascending=False
    )

    output_path = (
        RESULTS_DIR
        / "classification_results.csv"
    )

    results.to_csv(

        output_path,

        index=False
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "CLASSIFICATION MODEL RANKING"
    )

    print(
        "=" * 60
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    return results


# ============================================================
# TRAIN BEST CLASSIFIER
# ============================================================

def train_best_model(

    train_df,

    best_feature_set,

    best_model_name
):

    feature_config = (
        create_feature_configs()
        [best_feature_set]
    )

    model = (
        create_models()
        [best_model_name]
    )

    pipeline = create_pipeline(

        feature_config,

        model
    )

    X_train = train_df[
        CHANNELS
    ]

    y_train = train_df[
        TARGET
    ]

    print(
        "\nTraining final classifier..."
    )

    pipeline.fit(
        X_train,
        y_train
    )

    model_path = (
        MODEL_DIR
        / "best_classifier.joblib"
    )

    joblib.dump(

        pipeline,

        model_path
    )

    print(
        "\nBest classifier saved to:"
    )

    print(
        model_path
    )

    return pipeline


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df, test_df = (
        load_data()
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    results = compare_models(
        train_df
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    results = save_results(
        results
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_row = (
        results.iloc[0]
    )

    best_feature_set = (
        best_row[
            "feature_set"
        ]
    )

    best_model_name = (
        best_row[
            "model"
        ]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "BEST CLASSIFICATION MODEL"
    )

    print(
        "=" * 60
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
    # Train final classifier
    # --------------------------------------------------------

    best_pipeline = train_best_model(

        train_df,

        best_feature_set,

        best_model_name
    )

    print(
        "\nClassification training completed."
    )