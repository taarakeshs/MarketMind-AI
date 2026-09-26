from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS
from src.regression_features import (
    create_regression_target,
)

from src.validation import (
    chronological_split,
)

from src.return_models import (
    create_random_forest_regressor,
    create_xgboost_regressor,
    create_lightgbm_regressor,
    train_regressor,
    evaluate_regressor,
    save_regression_model,
    LIGHTGBM_AVAILABLE,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)


RESULT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "return_model_results.csv"
)


def main():

    print("=" * 70)
    print("STOCK RETURN REGRESSION")
    print("=" * 70)

    print()
    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    print(
        f"Rows before target: "
        f"{len(df):,}"
    )

    print(
        f"Stocks: "
        f"{df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df = create_regression_target(
        df
    )

    print()

    print(
        f"Rows after target: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # TIME SERIES SPLIT
    # --------------------------------------------------------

    train, validation, test = (
        chronological_split(df)
    )

    print()
    print("=" * 70)
    print("TIME SERIES SPLIT")
    print("=" * 70)

    print()

    print(
        f"Train      : "
        f"{len(train):,}"
    )

    print(
        f"Validation : "
        f"{len(validation):,}"
    )

    print(
        f"Test       : "
        f"{len(test):,}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X_train = train[
        FEATURE_COLUMNS
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target_return_1d"
    ]

    y_validation = validation[
        "target_return_1d"
    ]

    y_test = test[
        "target_return_1d"
    ]

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "random_forest":
            create_random_forest_regressor(),

        "xgboost":
            create_xgboost_regressor(),
    }

    if LIGHTGBM_AVAILABLE:

        models[
            "lightgbm"
        ] = create_lightgbm_regressor()

    results = []

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for name, model in models.items():

        print()
        print("=" * 70)

        print(
            f"TRAINING: "
            f"{name.upper()}"
        )

        print("=" * 70)

        print()

        print("Training...")

        model = train_regressor(
            model,
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_result = (
            evaluate_regressor(
                model,
                X_validation,
                y_validation,
            )
        )

        print()

        print("VALIDATION RESULTS")
        print("-" * 50)

        print(
            f"MAE                 : "
            f"{validation_result['mae']:.6f}"
        )

        print(
            f"RMSE                : "
            f"{validation_result['rmse']:.6f}"
        )

        print(
            f"R²                  : "
            f"{validation_result['r2']:.6f}"
        )

        print(
            f"Directional Accuracy: "
            f"{validation_result['directional_accuracy']:.4f}"
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_result = (
            evaluate_regressor(
                model,
                X_test,
                y_test,
            )
        )

        print()

        print("TEST RESULTS")
        print("-" * 50)

        print(
            f"MAE                 : "
            f"{test_result['mae']:.6f}"
        )

        print(
            f"RMSE                : "
            f"{test_result['rmse']:.6f}"
        )

        print(
            f"R²                  : "
            f"{test_result['r2']:.6f}"
        )

        print(
            f"Directional Accuracy: "
            f"{test_result['directional_accuracy']:.4f}"
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_regression_model(
            model,
            f"{name}_return",
        )

        results.append({

            "model": name,

            "validation_mae":
                validation_result["mae"],

            "validation_rmse":
                validation_result["rmse"],

            "validation_r2":
                validation_result["r2"],

            "validation_directional_accuracy":
                validation_result[
                    "directional_accuracy"
                ],

            "test_mae":
                test_result["mae"],

            "test_rmse":
                test_result["rmse"],

            "test_r2":
                test_result["r2"],

            "test_directional_accuracy":
                test_result[
                    "directional_accuracy"
                ],
        })

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("RETURN MODEL COMPARISON")
    print("=" * 70)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print("=" * 70)
    print("RETURN REGRESSION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()