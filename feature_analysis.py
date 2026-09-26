from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.features import FEATURE_COLUMNS


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

MODEL_DIR = (
    BASE_DIR
    / "models"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def analyze_model(
    model_name: str,
    model_file: str,
):
    print()
    print("=" * 70)
    print(f"FEATURE IMPORTANCE: {model_name.upper()}")
    print("=" * 70)

    model_path = MODEL_DIR / model_file

    print()
    print("Loading model...")
    print(model_path)

    model = joblib.load(model_path)

    # --------------------------------------------------------
    # Load feature data
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    print()
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # --------------------------------------------------------
    # Extract feature importance
    # --------------------------------------------------------

    if not hasattr(model, "feature_importances_"):
        print(
            f"{model_name} does not provide "
            "feature_importances_."
        )
        return

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importance,
        }
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
    # Print top features
    # --------------------------------------------------------

    print()
    print("TOP 20 FEATURES")
    print("-" * 70)

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"{model_name}_feature_importance.csv"
    )

    importance_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("Saved:")
    print(output_file)

    # --------------------------------------------------------
    # Plot top 20
    # --------------------------------------------------------

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{model_name.upper()} - Top 20 Features"
    )

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / f"{model_name}_feature_importance.png"
    )

    plt.savefig(
        plot_file,
        dpi=150,
    )

    plt.close()

    print()
    print("Plot saved:")
    print(plot_file)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("STOCK MARKET FEATURE ANALYSIS")
    print("=" * 70)

    analyze_model(
        "random_forest",
        "random_forest.joblib",
    )

    analyze_model(
        "xgboost",
        "xgboost.joblib",
    )

    analyze_model(
        "lightgbm",
        "lightgbm.joblib",
    )

    print()
    print("=" * 70)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()