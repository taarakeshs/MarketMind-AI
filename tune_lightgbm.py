from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.validation import chronological_split
from src.features import FEATURE_COLUMNS
from src.models import create_lightgbm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)

RESULTS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lightgbm_tuning_results.csv"
)


# ============================================================
# PARAMETER SEARCH SPACE
# ============================================================

PARAMETER_GRID = [

    {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 7,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.10,
        "max_depth": 5,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": -1,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

]


# ============================================================
# MODEL CREATION
# ============================================================


def create_tuned_model(params):

    return create_lightgbm().set_params(
        **params
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["ticker", "date"]
    ).copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train, validation, test = (
        chronological_split(df)
    )

    print()
    print("TIME SERIES SPLIT")
    print("-" * 50)

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    # --------------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------------

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target"
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        "target"
    ]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        "target"
    ]

    results = []

    best_auc = -1

    best_params = None

    best_model = None

    # ========================================================
    # HYPERPARAMETER EXPERIMENTS
    # ========================================================

    for index, params in enumerate(
        PARAMETER_GRID,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"EXPERIMENT {index}"
            f"/{len(PARAMETER_GRID)}"
        )
        print("=" * 70)

        print(
            "Parameters:"
        )

        print(params)

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = create_tuned_model(
            params
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print()
        print("Training...")

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        validation_probability = (
            model
            .predict_proba(
                X_validation
            )[:, 1]
        )

        validation_prediction = (
            validation_probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        validation_accuracy = (
            accuracy_score(
                y_validation,
                validation_prediction,
            )
        )

        validation_f1 = (
            f1_score(
                y_validation,
                validation_prediction,
            )
        )

        validation_auc = (
            roc_auc_score(
                y_validation,
                validation_probability,
            )
        )

        print()
        print("Validation Results")
        print("-" * 50)

        print(
            f"Accuracy : "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"F1       : "
            f"{validation_f1:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{validation_auc:.4f}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            **params,
            "validation_accuracy":
                validation_accuracy,
            "validation_f1":
                validation_f1,
            "validation_auc":
                validation_auc,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if validation_auc > best_auc:

            best_auc = validation_auc

            best_params = params.copy()

            best_model = model

            print()
            print(
                "🔥 NEW BEST MODEL"
            )

            print(
                f"Validation ROC-AUC: "
                f"{best_auc:.4f}"
            )

    # ========================================================
    # SAVE ALL TUNING RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "validation_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # ========================================================
    # BEST PARAMETERS
    # ========================================================

    print()
    print("=" * 70)
    print("BEST LIGHTGBM CONFIGURATION")
    print("=" * 70)

    print()

    for key, value in best_params.items():

        print(
            f"{key:20s}: {value}"
        )

    print()
    print(
        f"Best validation ROC-AUC: "
        f"{best_auc:.4f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_probability = (
        best_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    test_prediction = (
        test_probability >= 0.5
    ).astype(int)

    test_accuracy = (
        accuracy_score(
            y_test,
            test_prediction,
        )
    )

    test_f1 = (
        f1_score(
            y_test,
            test_prediction,
        )
    )

    test_auc = (
        roc_auc_score(
            y_test,
            test_probability,
        )
    )

    print()
    print(
        f"Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test F1       : "
        f"{test_f1:.4f}"
    )

    print(
        f"Test ROC-AUC  : "
        f"{test_auc:.4f}"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    best_model_path = (
        BASE_DIR
        / "models"
        / "lightgbm_tuned.joblib"
    )

    import joblib

    joblib.dump(
        best_model,
        best_model_path,
    )

    print()
    print("Best model saved to:")
    print(best_model_path)

    print()
    print("Tuning results saved to:")
    print(RESULTS_FILE)

    print()
    print("=" * 70)
    print("LIGHTGBM OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()