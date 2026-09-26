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
    / "ranking_robustness.csv"
)

RANDOM_SEED = 42

TOP_N_VALUES = [5, 10, 20, 50]

N_PERIODS = 4


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
        f"Rows  : {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    return df


# ============================================================
# CREATE FUTURE RETURN
# ============================================================

def prepare_target(df):

    df = df.copy()

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
# FEATURES
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
# TIME PERIODS
# ============================================================

def create_periods(df):

    dates = sorted(
        df["date"].unique()
    )

    total_dates = len(dates)

    period_size = (
        total_dates // N_PERIODS
    )

    periods = []

    for i in range(N_PERIODS):

        start_index = (
            i * period_size
        )

        if i == N_PERIODS - 1:

            end_index = total_dates

        else:

            end_index = (
                (i + 1)
                * period_size
            )

        period_dates = dates[
            start_index:end_index
        ]

        if len(period_dates) == 0:
            continue

        periods.append(
            (
                period_dates[0],
                period_dates[-1],
            )
        )

    return periods


# ============================================================
# TRAIN MODEL
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

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    df,
    features,
):

    df = df.copy()

    X = (
        df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    df["predicted_return"] = (
        model.predict(X)
    )

    return df


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    daily_returns,
):

    daily_returns = np.asarray(
        daily_returns,
        dtype=float,
    )

    if len(daily_returns) == 0:

        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "winning_days": np.nan,
        }

    equity = np.cumprod(
        1 + daily_returns
    )

    total_return = (
        equity[-1] - 1
    )

    periods = len(
        daily_returns
    )

    annualized_return = (
        (1 + total_return)
        ** (252 / periods)
        - 1
    )

    volatility = (
        np.std(
            daily_returns,
            ddof=1,
        )
        * np.sqrt(252)
    )

    if np.std(
        daily_returns,
        ddof=1,
    ) > 0:

        sharpe = (
            np.mean(daily_returns)
            / np.std(
                daily_returns,
                ddof=1,
            )
            * np.sqrt(252)
        )

    else:

        sharpe = np.nan

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / running_max
        - 1
    )

    max_drawdown = (
        np.min(drawdown)
    )

    winning_days = np.mean(
        daily_returns > 0
    )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_days": winning_days,
    }


# ============================================================
# LONG-ONLY RANKING
# ============================================================

def evaluate_long_only(
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

        if len(selected) < top_n:
            continue

        portfolio_return = (
            selected[
                "future_return"
            ].mean()
        )

        daily_returns.append(
            portfolio_return
        )

    return calculate_metrics(
        daily_returns
    )


# ============================================================
# LONG-SHORT RANKING
# ============================================================

def evaluate_long_short(
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

        if len(day) < (
            top_n * 2
        ):
            continue

        long_stocks = day.head(
            top_n
        )

        short_stocks = day.tail(
            top_n
        )

        long_return = (
            long_stocks[
                "future_return"
            ].mean()
        )

        short_return = (
            short_stocks[
                "future_return"
            ].mean()
        )

        portfolio_return = (
            long_return
            - short_return
        )

        daily_returns.append(
            portfolio_return
        )

    return calculate_metrics(
        daily_returns
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CROSS-SECTIONAL RANKING ROBUSTNESS TEST")
    print("=" * 70)

    df = load_data()

    df = prepare_target(df)

    features = get_features(df)

    print()
    print(
        f"Candidate features: "
        f"{len(features)}"
    )

    periods = create_periods(
        df
    )

    print()
    print(
        f"Evaluation periods: "
        f"{len(periods)}"
    )

    all_results = []

    # --------------------------------------------------------
    # WALK THROUGH PERIODS
    # --------------------------------------------------------

    for period_number, (
        period_start,
        period_end,
    ) in enumerate(
        periods,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"PERIOD {period_number}/"
            f"{len(periods)}"
        )
        print("=" * 70)

        print(
            f"Period: "
            f"{period_start} → "
            f"{period_end}"
        )

        period_df = df[
            (
                df["date"]
                >= period_start
            )
            &
            (
                df["date"]
                <= period_end
            )
        ].copy()

        period_dates = sorted(
            period_df[
                "date"
            ].unique()
        )

        # Use first 60% for training
        # and final 40% for testing.

        split_index = int(
            len(period_dates)
            * 0.60
        )

        train_dates = (
            period_dates[
                :split_index
            ]
        )

        test_dates = (
            period_dates[
                split_index:
            ]
        )

        train = period_df[
            period_df["date"].isin(
                train_dates
            )
        ].copy()

        test = period_df[
            period_df["date"].isin(
                test_dates
            )
        ].copy()

        print(
            f"Training rows: "
            f"{len(train):,}"
        )

        print(
            f"Testing rows : "
            f"{len(test):,}"
        )

        if len(train) == 0 or len(
            test
        ) == 0:

            continue

        print()
        print(
            "Training Random Forest..."
        )

        model = train_model(
            train,
            features,
        )

        test = predict(
            model,
            test,
            features,
        )

        # ----------------------------------------------------
        # LONG ONLY
        # ----------------------------------------------------

        for top_n in TOP_N_VALUES:

            metrics = (
                evaluate_long_only(
                    test,
                    top_n,
                )
            )

            all_results.append(
                {
                    "period": period_number,
                    "strategy": "long_only",
                    "top_n": top_n,
                    **metrics,
                }
            )

            print()
            print(
                f"Top {top_n} "
                "LONG-ONLY"
            )

            print(
                f"Return : "
                f"{metrics['total_return']:.4%}"
            )

            print(
                f"Sharpe : "
                f"{metrics['sharpe']:.4f}"
            )

            print(
                f"Drawdown: "
                f"{metrics['max_drawdown']:.4%}"
            )

        # ----------------------------------------------------
        # LONG SHORT
        # ----------------------------------------------------

        for top_n in TOP_N_VALUES:

            metrics = (
                evaluate_long_short(
                    test,
                    top_n,
                )
            )

            all_results.append(
                {
                    "period": period_number,
                    "strategy": "long_short",
                    "top_n": top_n,
                    **metrics,
                }
            )

            print()
            print(
                f"Top {top_n} "
                "LONG-SHORT"
            )

            print(
                f"Return : "
                f"{metrics['total_return']:.4%}"
            )

            print(
                f"Sharpe : "
                f"{metrics['sharpe']:.4f}"
            )

            print(
                f"Drawdown: "
                f"{metrics['max_drawdown']:.4%}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    print()
    print("=" * 70)
    print("RANKING ROBUSTNESS RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("AVERAGE PERFORMANCE")
    print("=" * 70)

    summary = (
        results_df
        .groupby(
            [
                "strategy",
                "top_n",
            ]
        )
        [
            [
                "total_return",
                "annualized_return",
                "volatility",
                "sharpe",
                "max_drawdown",
                "winning_days",
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # STABILITY
    # ========================================================

    print()
    print("=" * 70)
    print("PROFITABLE PERIOD ANALYSIS")
    print("=" * 70)

    for (
        strategy,
        top_n,
    ), group in results_df.groupby(
        [
            "strategy",
            "top_n",
        ]
    ):

        profitable = (
            group[
                "total_return"
            ] > 0
        ).sum()

        total = len(group)

        print(
            f"{strategy:12s} "
            f"Top {top_n:2d}: "
            f"{profitable}/{total} "
            "profitable periods"
        )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print(
        "RANKING ROBUSTNESS TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()