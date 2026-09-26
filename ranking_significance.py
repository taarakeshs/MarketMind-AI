from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ranking_significance.csv"
)

RANDOM_SEED = 42
N_PERMUTATIONS = 100


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["date", "ticker"]
    ).copy()

    print(
        f"Rows : {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    return df


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(df):

    df = df.copy()

    # We predict the NEXT day's return.
    df["future_return"] = (
        df.groupby("ticker")["close"]
        .shift(-1)
        / df["close"]
        - 1.0
    )

    df = df.dropna(
        subset=["future_return"]
    ).copy()

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_features(df):

    excluded = {
        "date",
        "ticker",
        "target",
        "next_close",
        "next_return",
        "future_return",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            features.append(column)

    return features


# ============================================================
# TIME SERIES SPLIT
# ============================================================

def split_data(df):

    dates = sorted(
        df["date"].unique()
    )

    split_index = int(
        len(dates) * 0.85
    )

    train_dates = dates[
        :split_index
    ]

    test_dates = dates[
        split_index:
    ]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

    return train, test


# ============================================================
# TRAIN RANKING MODEL
# ============================================================

def train_model(
    train,
    features,
):

    X_train = (
        train[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    y_train = train[
        "future_return"
    ]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print()
    print("Training ranking model...")

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test,
    features,
):

    test = test.copy()

    X_test = (
        test[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    test["predicted_return"] = (
        model.predict(X_test)
    )

    return test


# ============================================================
# TOP-N PORTFOLIO RETURN
# ============================================================

def calculate_top_n_return(
    df,
    top_n,
):

    daily_returns = []

    for date, day in df.groupby(
        "date"
    ):

        day = day.sort_values(
            "predicted_return",
            ascending=False,
        )

        selected = day.head(
            top_n
        )

        if len(selected) == 0:
            continue

        portfolio_return = (
            selected["future_return"]
            .mean()
        )

        daily_returns.append(
            portfolio_return
        )

    if not daily_returns:
        return np.nan

    return (
        np.prod(
            1 + np.array(
                daily_returns
            )
        )
        - 1
    )


# ============================================================
# RANDOM PORTFOLIO RETURN
# ============================================================

def calculate_random_return(
    df,
    top_n,
    rng,
):

    daily_returns = []

    for date, day in df.groupby(
        "date"
    ):

        if len(day) < top_n:
            continue

        selected = day.sample(
            n=top_n,
            random_state=int(
                rng.integers(
                    0,
                    1_000_000_000,
                )
            ),
        )

        portfolio_return = (
            selected["future_return"]
            .mean()
        )

        daily_returns.append(
            portfolio_return
        )

    if not daily_returns:
        return np.nan

    return (
        np.prod(
            1 + np.array(
                daily_returns
            )
        )
        - 1
    )


# ============================================================
# SIGNIFICANCE TEST
# ============================================================

def run_significance_test(
    test,
    top_n,
):

    ml_return = (
        calculate_top_n_return(
            test,
            top_n,
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    random_returns = []

    print(
        f"Testing Top {top_n}..."
    )

    for _ in range(
        N_PERMUTATIONS
    ):

        random_return = (
            calculate_random_return(
                test,
                top_n,
                rng,
            )
        )

        if not np.isnan(
            random_return
        ):
            random_returns.append(
                random_return
            )

    random_returns = np.array(
        random_returns
    )

    random_mean = (
        random_returns.mean()
    )

    random_std = (
        random_returns.std()
    )

    random_best = (
        random_returns.max()
    )

    random_worst = (
        random_returns.min()
    )

    percentile = (
        np.mean(
            random_returns
            <= ml_return
        )
    )

    return {
        "top_n": top_n,
        "ml_return": ml_return,
        "random_mean_return": random_mean,
        "random_std_return": random_std,
        "random_best_return": random_best,
        "random_worst_return": random_worst,
        "percentile": percentile,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RANKING STATISTICAL SIGNIFICANCE TEST")
    print("=" * 70)

    df = load_data()

    df = prepare_target(df)

    features = get_features(df)

    print()
    print(
        f"Candidate features: {len(features)}"
    )

    train, test = split_data(
        df
    )

    print()
    print(
        f"Test dates: "
        f"{test['date'].nunique()}"
    )

    print(
        f"Test rows : "
        f"{len(test):,}"
    )

    model = train_model(
        train,
        features,
    )

    test = generate_predictions(
        model,
        test,
        features,
    )

    print()
    print("=" * 70)
    print("RUNNING RANDOM PERMUTATION TEST")
    print("=" * 70)

    results = []

    for top_n in [
        5,
        10,
        20,
        50,
    ]:

        result = (
            run_significance_test(
                test,
                top_n,
            )
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("Interpretation:")

    for _, row in (
        results_df.iterrows()
    ):

        print(
            f"Top {int(row['top_n']):>2}: "
            f"ML percentile = "
            f"{row['percentile']:.3f}"
        )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Results saved to:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("SIGNIFICANCE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()