from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS
from src.validation import chronological_split
from src.models import (
    create_random_forest,
    create_xgboost,
    create_lightgbm,
    train_model,
    evaluate_model,
    save_model,
    LIGHTGBM_AVAILABLE,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)


def prepare_data():

    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["ticker", "date"]
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

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
    # -------------------------------------------------
    # SAFETY CHECK
    # -------------------------------------------------

    for dataset_name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        feature_data = dataset[
            FEATURE_COLUMNS
        ]

        inf_count = (
            feature_data
            .isin([float("inf"), float("-inf")])
            .sum()
            .sum()
        )

        nan_count = (
            feature_data
            .isna()
            .sum()
            .sum()
        )

        print(
            f"{dataset_name}: "
            f"NaN={nan_count}, "
            f"Infinity={inf_count}"
        )

        if inf_count > 0:
            raise ValueError(
                f"{dataset_name} contains "
                f"{inf_count} infinite values."
            )

        if nan_count > 0:
            raise ValueError(
                f"{dataset_name} contains "
                f"{nan_count} NaN values."
            )
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

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def print_metrics(
    name,
    metrics,
):

    print()
    print(
        f"===== {name} ====="
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )


def main():

    print("=" * 70)
    print("STOCK MARKET ML TRAINING")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_data()

    models = {

        "random_forest":
            create_random_forest(),

        "xgboost":
            create_xgboost(),
    }

    if LIGHTGBM_AVAILABLE:

        models["lightgbm"] = (
            create_lightgbm()
        )

    else:

        print(
            "LightGBM unavailable."
        )

    results = []

    for name, model in models.items():

        print()
        print("=" * 70)
        print(
            f"TRAINING: {name.upper()}"
        )
        print("=" * 70)

        model = train_model(
            model,
            X_train,
            y_train,
        )

        validation_metrics = (
            evaluate_model(
                model,
                X_validation,
                y_validation,
            )
        )

        test_metrics = (
            evaluate_model(
                model,
                X_test,
                y_test,
            )
        )

        print()
        print(
            f"{name.upper()} "
            "VALIDATION"
        )

        print_metrics(
            name,
            validation_metrics,
        )

        print()
        print(
            f"{name.upper()} "
            "TEST"
        )

        print_metrics(
            name,
            test_metrics,
        )

        model_path = save_model(
            model,
            name,
        )

        print()
        print(
            f"Saved model: "
            f"{model_path}"
        )

        results.append(
            {
                "model": name,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_f1":
                    validation_metrics[
                        "f1"
                    ],

                "validation_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "test_f1":
                    test_metrics[
                        "f1"
                    ],

                "test_auc":
                    test_metrics[
                        "roc_auc"
                    ],
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to:\n"
        f"{results_path}"
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()