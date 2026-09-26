from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

SIGNAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "enhanced_signals.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "enhanced_backtest.csv"
)


TRANSACTION_COST = 0.0005


def sharpe_ratio(returns):

    returns = pd.Series(
        returns
    ).dropna()

    if len(returns) == 0:
        return 0.0

    std = returns.std()

    if std == 0:
        return 0.0

    return (
        returns.mean()
        / std
        * np.sqrt(252)
    )


def max_drawdown(returns):

    equity = (
        1 + pd.Series(
            returns
        ).fillna(0)
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    return drawdown.min()


def main():

    print("=" * 70)
    print("ENHANCED MULTI-FACTOR BACKTEST")
    print("=" * 70)

    print()
    print("Loading enhanced signals...")

    df = pd.read_csv(
        SIGNAL_FILE,
        parse_dates=["date"],
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # NEXT DAY RETURN
    # --------------------------------------------------------

    df = df.sort_values(
        ["ticker", "date"]
    ).copy()

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
    # POSITION
    # --------------------------------------------------------

    # Direction comes from the signal.
    #
    # Position size determines how strongly
    # we participate.

    df["position"] = (
        df["signal"]
        * df["position_size"]
    )

    # --------------------------------------------------------
    # RAW STRATEGY RETURN
    # --------------------------------------------------------

    df["strategy_return"] = (
        df["position"]
        * df["next_return"]
    )

    # --------------------------------------------------------
    # TRANSACTION COST
    # --------------------------------------------------------

    df["strategy_return"] = (
        df["strategy_return"]
        - np.where(
            df["signal"] != 0,
            TRANSACTION_COST
            * df["position_size"],
            0,
        )
    )

    # --------------------------------------------------------
    # DAILY PORTFOLIO RETURN
    # --------------------------------------------------------

    daily = (
        df.groupby("date")
        ["strategy_return"]
        .mean()
        .reset_index()
    )

    daily = daily.sort_values(
        "date"
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    total_return = (
        1 + daily[
            "strategy_return"
        ]
    ).prod() - 1

    annualized_return = (
        1 + total_return
    ) ** (
        252 / len(daily)
    ) - 1

    sharpe = sharpe_ratio(
        daily[
            "strategy_return"
        ]
    )

    drawdown = max_drawdown(
        daily[
            "strategy_return"
        ]
    )

    winning_days = (
        daily[
            "strategy_return"
        ] > 0
    ).mean()

    # --------------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------------

    active = (
        df["signal"] != 0
    )

    active_returns = df.loc[
        active,
        "strategy_return"
    ]

    trades = len(
        active_returns
    )

    if trades > 0:

        win_rate = (
            active_returns > 0
        ).mean()

        average_trade = (
            active_returns.mean()
        )

    else:

        win_rate = 0

        average_trade = 0

    # --------------------------------------------------------
    # SIGNAL COUNTS
    # --------------------------------------------------------

    buy_count = int(
        (
            df["signal"] == 1
        ).sum()
    )

    sell_count = int(
        (
            df["signal"] == -1
        ).sum()
    )

    hold_count = int(
        (
            df["signal"] == 0
        ).sum()
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ENHANCED BACKTEST RESULTS")
    print("=" * 70)

    print()

    print(
        f"Total Return      : "
        f"{total_return:.2%}"
    )

    print(
        f"Annualized Return : "
        f"{annualized_return:.2%}"
    )

    print(
        f"Sharpe Ratio      : "
        f"{sharpe:.4f}"
    )

    print(
        f"Maximum Drawdown  : "
        f"{drawdown:.2%}"
    )

    print(
        f"Winning Days      : "
        f"{winning_days:.2%}"
    )

    print()

    print(
        f"Active Trades     : "
        f"{trades:,}"
    )

    print(
        f"Trade Win Rate    : "
        f"{win_rate:.2%}"
    )

    print(
        f"Average Trade     : "
        f"{average_trade:.6f}"
    )

    print()

    print("=" * 70)
    print("SIGNAL DISTRIBUTION")
    print("=" * 70)

    print()

    print(
        f"BUY  : {buy_count:,}"
    )

    print(
        f"HOLD : {hold_count:,}"
    )

    print(
        f"SELL : {sell_count:,}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = df[
        [
            "date",
            "ticker",
            "probability_up",
            "composite_score",
            "signal",
            "position_size",
            "next_return",
            "strategy_return",
        ]
    ]

    output.to_csv(
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
    print("ENHANCED BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()