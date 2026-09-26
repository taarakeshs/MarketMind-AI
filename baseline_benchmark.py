from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.validation import chronological_split


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
    / "baseline_benchmark_results.csv"
)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("BASELINE STRATEGY BENCHMARK")
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
        ["ticker", "date"]
    ).copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
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
    # ONLY TEST SET
    # --------------------------------------------------------
    #
    # We evaluate baselines on the same untouched
    # test period used by our ML models.
    #

    test = test.copy()

    y_test = test["target"].astype(int)

    results = []

    # ========================================================
    # 1. ALWAYS UP
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE 1: ALWAYS UP")
    print("=" * 70)

    always_up = np.ones(
        len(test),
        dtype=int,
    )

    always_up_accuracy = accuracy_score(
        y_test,
        always_up,
    )

    always_up_f1 = f1_score(
        y_test,
        always_up,
    )

    print(
        f"Accuracy : {always_up_accuracy:.4f}"
    )

    print(
        f"F1       : {always_up_f1:.4f}"
    )

    results.append(
        {
            "model": "always_up",
            "accuracy": always_up_accuracy,
            "f1": always_up_f1,
            "roc_auc": np.nan,
        }
    )

    # ========================================================
    # 2. PREVIOUS-DAY MOMENTUM
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE 2: PREVIOUS-DAY MOMENTUM")
    print("=" * 70)

    momentum_prediction = (
        test["return_1d"] > 0
    ).astype(int)

    momentum_accuracy = accuracy_score(
        y_test,
        momentum_prediction,
    )

    momentum_f1 = f1_score(
        y_test,
        momentum_prediction,
    )

    # A binary prediction can also be used as
    # a probability-like ranking score.
    momentum_auc = roc_auc_score(
        y_test,
        momentum_prediction,
    )

    print(
        f"Accuracy : {momentum_accuracy:.4f}"
    )

    print(
        f"F1       : {momentum_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {momentum_auc:.4f}"
    )

    results.append(
        {
            "model": "previous_day_momentum",
            "accuracy": momentum_accuracy,
            "f1": momentum_f1,
            "roc_auc": momentum_auc,
        }
    )

    # ========================================================
    # 3. SMA 20 TREND
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE 3: SMA20 TREND")
    print("=" * 70)

    sma_prediction = (
        test["close"] > test["sma_20"]
    ).astype(int)

    sma_accuracy = accuracy_score(
        y_test,
        sma_prediction,
    )

    sma_f1 = f1_score(
        y_test,
        sma_prediction,
    )

    sma_auc = roc_auc_score(
        y_test,
        sma_prediction,
    )

    print(
        f"Accuracy : {sma_accuracy:.4f}"
    )

    print(
        f"F1       : {sma_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {sma_auc:.4f}"
    )

    results.append(
        {
            "model": "sma20_trend",
            "accuracy": sma_accuracy,
            "f1": sma_f1,
            "roc_auc": sma_auc,
        }
    )

    # ========================================================
    # 4. RSI STRATEGY
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE 4: RSI TREND")
    print("=" * 70)

    # Simple interpretation:
    #
    # RSI >= 50 → UP
    # RSI < 50  → DOWN

    rsi_prediction = (
        test["rsi"] >= 50
    ).astype(int)

    rsi_accuracy = accuracy_score(
        y_test,
        rsi_prediction,
    )

    rsi_f1 = f1_score(
        y_test,
        rsi_prediction,
    )

    rsi_auc = roc_auc_score(
        y_test,
        rsi_prediction,
    )

    print(
        f"Accuracy : {rsi_accuracy:.4f}"
    )

    print(
        f"F1       : {rsi_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {rsi_auc:.4f}"
    )

    results.append(
        {
            "model": "rsi_50",
            "accuracy": rsi_accuracy,
            "f1": rsi_f1,
            "roc_auc": rsi_auc,
        }
    )

    # ========================================================
    # 5. MACD HISTOGRAM
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE 5: MACD")
    print("=" * 70)

    macd_prediction = (
        test["macd_histogram"] > 0
    ).astype(int)

    macd_accuracy = accuracy_score(
        y_test,
        macd_prediction,
    )

    macd_f1 = f1_score(
        y_test,
        macd_prediction,
    )

    macd_auc = roc_auc_score(
        y_test,
        macd_prediction,
    )

    print(
        f"Accuracy : {macd_accuracy:.4f}"
    )

    print(
        f"F1       : {macd_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {macd_auc:.4f}"
    )

    results.append(
        {
            "model": "macd",
            "accuracy": macd_accuracy,
            "f1": macd_f1,
            "roc_auc": macd_auc,
        }
    )

    # ========================================================
    # 6. RANDOM BASELINE
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE 6: RANDOM")
    print("=" * 70)

    rng = np.random.default_rng(
        seed=42
    )

    random_prediction = rng.integers(
        0,
        2,
        size=len(test),
    )

    random_accuracy = accuracy_score(
        y_test,
        random_prediction,
    )

    random_f1 = f1_score(
        y_test,
        random_prediction,
    )

    random_auc = roc_auc_score(
        y_test,
        random_prediction,
    )

    print(
        f"Accuracy : {random_accuracy:.4f}"
    )

    print(
        f"F1       : {random_f1:.4f}"
    )

    print(
        f"ROC-AUC  : {random_auc:.4f}"
    )

    results.append(
        {
            "model": "random",
            "accuracy": random_accuracy,
            "f1": random_f1,
            "roc_auc": random_auc,
        }
    )

    # ========================================================
    # LOAD ML MODEL RESULTS
    # ========================================================

    model_results_file = (
        BASE_DIR
        / "data"
        / "processed"
        / "model_results.csv"
    )

    if model_results_file.exists():

        print()
        print("=" * 70)
        print("ADDING ML MODEL RESULTS")
        print("=" * 70)

        model_results = pd.read_csv(
            model_results_file
        )

        for _, row in model_results.iterrows():

            results.append(
                {
                    "model": row["model"],
                    "accuracy":
                        row["test_accuracy"],
                    "f1":
                        row["test_f1"],
                    "roc_auc":
                        row["test_auc"],
                }
            )

    # ========================================================
    # COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "roc_auc",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("BASELINE + ML MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("Results saved to:")
    print(RESULTS_FILE)

    print()
    print("=" * 70)
    print("BASELINE BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()