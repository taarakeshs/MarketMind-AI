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

BACKTEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "backtest_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "strategy_comparison.csv"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    daily_returns,
):

    daily_returns = (
        pd.Series(daily_returns)
        .fillna(0)
    )

    total_return = (
        1 + daily_returns
    ).prod() - 1

    if len(daily_returns) > 0:

        annualized_return = (
            1 + total_return
        ) ** (
            252 / len(daily_returns)
        ) - 1

    else:

        annualized_return = 0

    volatility = (
        daily_returns.std()
        * np.sqrt(252)
    )

    if volatility != 0:

        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    equity = (
        1 + daily_returns
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    max_drawdown = drawdown.min()

    win_rate = (
        daily_returns > 0
    ).mean()

    return {
        "total_return": total_return,
        "annualized_return":
            annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown":
            max_drawdown,
        "winning_days":
            win_rate,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STRATEGY BENCHMARK COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading market data...")

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
    # NEXT DAY RETURN
    # --------------------------------------------------------

    df["next_return"] = (
        df.groupby("ticker")["close"]
        .shift(-1)
        / df["close"]
        - 1
    )

    df = df.dropna(
        subset=["next_return"]
    )

    # --------------------------------------------------------
    # TEST PERIOD
    # --------------------------------------------------------

    backtest = pd.read_csv(
        BACKTEST_FILE,
        parse_dates=["date"],
    )

    test_dates = sorted(
        backtest["date"].unique()
    )

    df = df[
        df["date"].isin(test_dates)
    ].copy()

    print()
    print(
        f"Test rows: {len(df):,}"
    )

    # ========================================================
    # STRATEGY 1
    # ========================================================

    print()
    print("=" * 70)
    print("STRATEGY 1: EQUAL-WEIGHT BUY & HOLD")
    print("=" * 70)

    market_daily = (
        df.groupby("date")
        ["next_return"]
        .mean()
    )

    market_metrics = (
        calculate_metrics(
            market_daily
        )
    )

    # ========================================================
    # STRATEGY 2
    # ========================================================

    print()
    print("=" * 70)
    print("STRATEGY 2: MOMENTUM")
    print("=" * 70)

    df["momentum_signal"] = (
        df["return_1d"] > 0
    ).astype(int)

    df["momentum_return"] = (
        df["momentum_signal"]
        * df["next_return"]
    )

    momentum_daily = (
        df.groupby("date")
        ["momentum_return"]
        .mean()
    )

    momentum_metrics = (
        calculate_metrics(
            momentum_daily
        )
    )

    # ========================================================
    # STRATEGY 3
    # ========================================================

    print()
    print("=" * 70)
    print("STRATEGY 3: ML STRATEGY")
    print("=" * 70)

    ml_daily = (
        backtest
        .groupby("date")
        ["strategy_return"]
        .mean()
    )

    ml_metrics = (
        calculate_metrics(
            ml_daily
        )
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    results = pd.DataFrame(
        [
            {
                "strategy":
                    "Equal Weight Buy & Hold",
                **market_metrics,
            },
            {
                "strategy":
                    "Momentum",
                **momentum_metrics,
            },
            {
                "strategy":
                    "ML Random Forest",
                **ml_metrics,
            },
        ]
    )

    print()
    print("=" * 70)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 70)

    print()

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()