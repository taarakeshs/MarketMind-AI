from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
    / "ranking_benchmark.csv"
)


TOP_N_VALUES = [5, 10, 20, 50]

RANDOM_SEED = 42


def create_future_return(df):

    df = df.copy()

    df = df.sort_values(
        ["ticker", "date"]
    )

    df["future_return"] = (
        df.groupby("ticker")["close"]
        .shift(-1)
        / df["close"]
        - 1
    )

    return df


def main():

    print("=" * 70)
    print("RANKING STRATEGY BENCHMARK")
    print("=" * 70)

    print()
    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = create_future_return(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=["future_return"]
    ).copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    dates = sorted(
        df["date"].unique()
    )

    # Use the same final 15% test period
    test_start = int(
        len(dates) * 0.85
    )

    test_dates = dates[
        test_start:
    ]

    test = df[
        df["date"].isin(test_dates)
    ].copy()

    print(
        f"Test rows: {len(test):,}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    for top_n in TOP_N_VALUES:

        print()
        print("=" * 70)
        print(
            f"RANDOM TOP {top_n}"
        )
        print("=" * 70)

        daily_returns = []

        for date, group in test.groupby(
            "date"
        ):

            if len(group) < top_n:
                continue

            selected_indices = rng.choice(
                len(group),
                size=top_n,
                replace=False,
            )

            selected = group.iloc[
                selected_indices
            ]

            daily_return = (
                selected["future_return"]
                .mean()
            )

            daily_returns.append(
                daily_return
            )

        daily_returns = pd.Series(
            daily_returns
        )

        total_return = (
            (1 + daily_returns)
            .prod()
            - 1
        )

        volatility = (
            daily_returns.std()
            * np.sqrt(252)
        )

        if daily_returns.std() > 0:

            sharpe = (
                daily_returns.mean()
                / daily_returns.std()
                * np.sqrt(252)
            )

        else:

            sharpe = np.nan

        cumulative = (
            1 + daily_returns
        ).cumprod()

        running_max = (
            cumulative.cummax()
        )

        drawdown = (
            cumulative
            / running_max
            - 1
        )

        max_drawdown = (
            drawdown.min()
        )

        winning_days = (
            daily_returns > 0
        ).mean()

        print(
            f"Total return : "
            f"{total_return:.4%}"
        )

        print(
            f"Sharpe       : "
            f"{sharpe:.4f}"
        )

        print(
            f"Max drawdown : "
            f"{max_drawdown:.4%}"
        )

        print(
            f"Winning days : "
            f"{winning_days:.4%}"
        )

        results.append(
            {
                "strategy":
                    f"random_top_{top_n}",
                "top_n": top_n,
                "total_return":
                    total_return,
                "sharpe": sharpe,
                "max_drawdown":
                    max_drawdown,
                "winning_days":
                    winning_days,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("RANDOM BENCHMARK RESULTS")
    print("=" * 70)

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
        f"Results saved to:\n{RESULT_FILE}"
    )

    print()
    print("=" * 70)
    print("RANKING BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()