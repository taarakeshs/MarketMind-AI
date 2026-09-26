from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


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
    / "walk_forward_results.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "random_forest_walk_forward.joblib"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return",

    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",

    "ema_5",
    "ema_10",
    "ema_20",
    "ema_50",

    "rsi",

    "macd",
    "macd_signal",
    "macd_histogram",

    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_position",

    "atr",
    "atr_percent",

    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_50d",

    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_60",

    "high_low_range",
    "open_close_range",
    "close_to_high",
    "close_to_low",

    "volume_change",
    "volume_ratio",
    "volume_sma_5",
    "volume_sma_20",

    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",

    "ema_5_ema_20_ratio",
]


# ============================================================
# MODEL
# ============================================================

def create_model():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["date", "ticker"]
    ).copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # AVAILABLE DATES
    # --------------------------------------------------------

    dates = sorted(
        df["date"].unique()
    )

    n_dates = len(dates)

    print(
        f"Trading dates: {n_dates}"
    )

    # --------------------------------------------------------
    # WALK-FORWARD CONFIGURATION
    # --------------------------------------------------------

    n_folds = 4

    initial_train_ratio = 0.50

    test_ratio = 0.10

    initial_train_end = int(
        n_dates * initial_train_ratio
    )

    test_size = int(
        n_dates * test_ratio
    )

    results = []

    # ========================================================
    # FOLDS
    # ========================================================

    for fold in range(n_folds):

        print()
        print("=" * 70)
        print(
            f"WALK-FORWARD FOLD {fold + 1}/{n_folds}"
        )
        print("=" * 70)

        train_start = 0

        train_end = (
            initial_train_end
            + fold * test_size
        )

        test_start = train_end

        test_end = (
            test_start
            + test_size
        )

        if test_end > n_dates:

            print(
                "Not enough future dates."
            )

            break

        train_dates = dates[
            train_start:train_end
        ]

        test_dates = dates[
            test_start:test_end
        ]

        train = df[
            df["date"].isin(train_dates)
        ].copy()

        test = df[
            df["date"].isin(test_dates)
        ].copy()

        print()
        print(
            f"Training period:"
        )

        print(
            f"{train_dates[0]} → "
            f"{train_dates[-1]}"
        )

        print(
            f"Training rows: "
            f"{len(train):,}"
        )

        print()

        print(
            f"Test period:"
        )

        print(
            f"{test_dates[0]} → "
            f"{test_dates[-1]}"
        )

        print(
            f"Test rows: "
            f"{len(test):,}"
        )

        # ----------------------------------------------------
        # PREPARE DATA
        # ----------------------------------------------------

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train = train[
            "target"
        ]

        X_test = test[
            FEATURE_COLUMNS
        ]

        y_test = test[
            "target"
        ]

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print()
        print("Training Random Forest...")

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        probability = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            prediction,
        )

        precision = precision_score(
            y_test,
            prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            prediction,
            zero_division=0,
        )

        auc = roc_auc_score(
            y_test,
            probability,
        )

        print()
        print("RESULTS")
        print("-" * 50)

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1        : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {auc:.4f}"
        )

        results.append(
            {
                "fold": fold + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "train_rows": len(train),
                "test_rows": len(test),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
            }
        )

        # ----------------------------------------------------
        # Save latest model
        # ----------------------------------------------------

        joblib.dump(
            model,
            MODEL_FILE,
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # AVERAGES
    # ========================================================

    if len(results_df) > 0:

        print()
        print("=" * 70)
        print("AVERAGE WALK-FORWARD PERFORMANCE")
        print("=" * 70)

        print(
            f"Average Accuracy : "
            f"{results_df['accuracy'].mean():.4f}"
        )

        print(
            f"Average Precision: "
            f"{results_df['precision'].mean():.4f}"
        )

        print(
            f"Average Recall   : "
            f"{results_df['recall'].mean():.4f}"
        )

        print(
            f"Average F1       : "
            f"{results_df['f1'].mean():.4f}"
        )

        print(
            f"Average ROC-AUC  : "
            f"{results_df['roc_auc'].mean():.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("Results saved to:")
    print(RESULTS_FILE)

    print()
    print("Model saved to:")
    print(MODEL_FILE)

    print()
    print("=" * 70)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()