from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.features import FEATURE_COLUMNS
from src.validation import chronological_split
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

IMPORTANCE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lightgbm_feature_importance.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_selection_results.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE SELECTION EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
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
    # Load feature importance
    # --------------------------------------------------------

    importance_df = pd.read_csv(
        IMPORTANCE_FILE
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Time-series split
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
    # Target
    # --------------------------------------------------------

    target_column = "target"

    # --------------------------------------------------------
    # Feature groups
    # --------------------------------------------------------

    feature_groups = {

        "all_53": FEATURE_COLUMNS,

        "top_30": (
            importance_df
            .head(30)["feature"]
            .tolist()
        ),

        "top_20": (
            importance_df
            .head(20)["feature"]
            .tolist()
        ),
    }

    results = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for group_name, features in feature_groups.items():

        print()
        print("=" * 70)
        print(
            f"EXPERIMENT: {group_name.upper()}"
        )
        print("=" * 70)

        print(
            f"Number of features: {len(features)}"
        )

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        X_train = train[features]
        y_train = train[target_column]

        X_validation = validation[features]
        y_validation = validation[target_column]

        X_test = test[features]
        y_test = test[target_column]

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = create_lightgbm()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print()
        print("Training LightGBM...")

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_prediction = (
            validation_probability >= 0.5
        ).astype(int)

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

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_probability = (
            model.predict_proba(
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

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()
        print("RESULTS")
        print("-" * 50)

        print(
            f"Validation Accuracy : "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Validation F1       : "
            f"{validation_f1:.4f}"
        )

        print(
            f"Validation ROC-AUC  : "
            f"{validation_auc:.4f}"
        )

        print()

        print(
            f"Test Accuracy       : "
            f"{test_accuracy:.4f}"
        )

        print(
            f"Test F1             : "
            f"{test_f1:.4f}"
        )

        print(
            f"Test ROC-AUC        : "
            f"{test_auc:.4f}"
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "feature_set": group_name,
                "feature_count": len(features),
                "validation_accuracy":
                    validation_accuracy,
                "validation_f1":
                    validation_f1,
                "validation_auc":
                    validation_auc,
                "test_accuracy":
                    test_accuracy,
                "test_f1":
                    test_f1,
                "test_auc":
                    test_auc,
            }
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("FEATURE SELECTION COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Results saved to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()