from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)


def main():

    print("=" * 70)
    print("FEATURE LEAKAGE AUDIT")
    print("=" * 70)

    print()
    print("Loading dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Stocks : {df['ticker'].nunique()}"
    )

    print()
    print("=" * 70)
    print("ALL DATASET COLUMNS")
    print("=" * 70)

    for i, column in enumerate(df.columns, 1):

        print(
            f"{i:3}. {column}"
        )

    # ----------------------------------------------------------
    # OBVIOUSLY DANGEROUS COLUMN NAMES
    # ----------------------------------------------------------

    dangerous_words = [
        "target",
        "future",
        "forward",
        "tomorrow",
        "next",
        "lead",
        "future_return",
    ]

    dangerous_columns = []

    for column in df.columns:

        column_lower = column.lower()

        for word in dangerous_words:

            if word in column_lower:

                dangerous_columns.append(
                    column
                )

                break

    print()
    print("=" * 70)
    print("POTENTIALLY DANGEROUS FEATURES")
    print("=" * 70)

    if dangerous_columns:

        for column in dangerous_columns:

            print(
                f"⚠️ {column}"
            )

    else:

        print(
            "No obviously dangerous column names found."
        )

    # ----------------------------------------------------------
    # TARGET
    # ----------------------------------------------------------

    if "target" in df.columns:

        print()
        print(
            "Target column detected: target"
        )

    # ----------------------------------------------------------
    # NUMERIC FEATURES
    # ----------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    excluded = {
        "target"
    }

    candidate_features = [
        c
        for c in numeric_columns
        if c not in excluded
    ]

    print()
    print("=" * 70)
    print("MODEL CANDIDATE FEATURES")
    print("=" * 70)

    print(
        f"Numeric columns : {len(numeric_columns)}"
    )

    print(
        f"Candidate features: {len(candidate_features)}"
    )

    # ----------------------------------------------------------
    # CHECK INF / NAN
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("DATA QUALITY")
    print("=" * 70)

    for column in candidate_features:

        nan_count = df[column].isna().sum()

        inf_count = np.isinf(
            df[column].to_numpy(
                dtype=float
            )
        ).sum()

        if nan_count > 0 or inf_count > 0:

            print(
                f"{column:30}"
                f"NaN={nan_count:<8}"
                f"Inf={inf_count}"
            )

    # ----------------------------------------------------------
    # CHECK EXTREME VALUES
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTREME FEATURE VALUES")
    print("=" * 70)

    for column in candidate_features:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if series.notna().sum() == 0:

            continue

        maximum = series.max()

        minimum = series.min()

        if (
            abs(maximum) > 10
            or abs(minimum) > 10
        ):

            print(
                f"{column:30}"
                f"min={minimum:.6f} "
                f"max={maximum:.6f}"
            )

    # ----------------------------------------------------------
    # TARGET DISTRIBUTION
    # ----------------------------------------------------------

    if "target" in df.columns:

        print()
        print("=" * 70)
        print("TARGET DISTRIBUTION")
        print("=" * 70)

        print(
            df["target"]
            .value_counts(
                normalize=True
            )
        )

    print()
    print("=" * 70)
    print("LEAKAGE AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()