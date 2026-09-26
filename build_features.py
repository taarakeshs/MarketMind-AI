from pathlib import Path

import pandas as pd

from src.features import (
    create_market_features,
    FEATURE_COLUMNS,
)


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "market_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)


def main():

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    print()
    print("Loading market data...")

    market_data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    print(
        f"Input rows: {len(market_data):,}"
    )

    print(
        f"Stocks: {market_data['ticker'].nunique()}"
    )

    print()
    print("Generating features...")

    features = create_market_features(
        market_data
    )

    print()
    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Feature columns: {len(FEATURE_COLUMNS)}"
    )

    print()
    print("Target distribution:")

    print(
        features["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print()
    print("Saving dataset...")

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()