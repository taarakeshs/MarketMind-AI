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
    / "threshold_analysis.csv"
)

TRANSACTION_COST = 0.0005


THRESHOLDS = [
    0.50,
    0.52,
    0.54,
    0.55,
    0.56,
    0.58,
    0.60,
]


def calculate_metrics(
    df,
    threshold,
):

    data = df.copy()

    probability = data[
        "probability_up"
    ]

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    data["signal"] = np.where(
        probability >= threshold,
        1,
        np.where(
            probability <= (1 - threshold),
            -1,
            0,
        ),
    )

    # --------------------------------------------------------
    # NEXT DAY RETURN
    # --------------------------------------------------------

    data = data.sort_values(
        ["ticker", "date"]
    )

    data["next_return"] = (
        data.groupby("ticker")["close"]
        .shift(-1)
        / data["close"]
        - 1
    )

    data = data.dropna(
        subset=["next_return"]
    )

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    data["position"] = (
        data["signal"]
        * 0.02
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    data["strategy_return"] = (
        data["position"]
        * data["next_return"]
    )

    # Transaction costs

    data["strategy_return"] -= np.where(
        data["signal"] != 0,
        TRANSACTION_COST
        * 0.02,
        0,
    )

    # --------------------------------------------------------
    # DAILY RETURNS
    # --------------------------------------------------------

    daily = (
        data.groupby("date")[
            "strategy_return"
        ]
        .mean()
    )

    if len(daily) == 0:

        return {
            "threshold": threshold,
            "buy_signals": 0,
            "sell_signals": 0,
            "active_signals": 0,
            "total_return": 0,
            "annualized_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "average_trade": 0,
        }

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    total_return = (
        (1 + daily).prod()
        - 1
    )

    annualized_return = (
        (1 + total_return)
        ** (252 / len(daily))
        - 1
    )

    std = daily.std()

    if std > 0:

        sharpe = (
            daily.mean()
            / std
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    equity = (
        1 + daily
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # TRADE METRICS
    # --------------------------------------------------------

    active = data[
        data["signal"] != 0
    ]

    buy_signals = (
        data["signal"] == 1
    ).sum()

    sell_signals = (
        data["signal"] == -1
    ).sum()

    active_signals = len(active)

    if active_signals > 0:

        trade_returns = (
            active["strategy_return"]
        )

        win_rate = (
            trade_returns > 0
        ).mean()

        average_trade = (
            trade_returns.mean()
        )

    else:

        win_rate = 0

        average_trade = 0

    return {

        "threshold": threshold,

        "buy_signals": int(
            buy_signals
        ),

        "sell_signals": int(
            sell_signals
        ),

        "active_signals": int(
            active_signals
        ),

        "total_return": total_return,

        "annualized_return":
            annualized_return,

        "sharpe": sharpe,

        "max_drawdown":
            max_drawdown,

        "win_rate": win_rate,

        "average_trade":
            average_trade,
    }


def main():

    print("=" * 70)
    print("SIGNAL THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

    print()
    print("Loading signals...")

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

    print()

    results = []

    for i, threshold in enumerate(
        THRESHOLDS,
        start=1,
    ):

        print("=" * 70)

        print(
            f"EXPERIMENT {i}/{len(THRESHOLDS)}"
        )

        print("=" * 70)

        print()

        print(
            f"BUY threshold  : "
            f"{threshold:.2f}"
        )

        print(
            f"SELL threshold : "
            f"{1 - threshold:.2f}"
        )

        result = calculate_metrics(
            df,
            threshold,
        )

        results.append(
            result
        )

        print()

        print(
            f"Buy signals    : "
            f"{result['buy_signals']:,}"
        )

        print(
            f"Sell signals   : "
            f"{result['sell_signals']:,}"
        )

        print(
            f"Total return   : "
            f"{result['total_return']:.2%}"
        )

        print(
            f"Sharpe         : "
            f"{result['sharpe']:.4f}"
        )

        print(
            f"Max drawdown   : "
            f"{result['max_drawdown']:.2%}"
        )

        print(
            f"Win rate       : "
            f"{result['win_rate']:.2%}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # BEST STRATEGIES
    # --------------------------------------------------------

    best_return = (
        results_df.loc[
            results_df[
                "total_return"
            ].idxmax()
        ]
    )

    best_sharpe = (
        results_df.loc[
            results_df[
                "sharpe"
            ].idxmax()
        ]
    )

    print()
    print("=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    print()

    print(
        "Best Return Threshold : "
        f"{best_return['threshold']:.2f}"
    )

    print(
        "Return                : "
        f"{best_return['total_return']:.2%}"
    )

    print()

    print(
        "Best Sharpe Threshold : "
        f"{best_sharpe['threshold']:.2f}"
    )

    print(
        "Sharpe                : "
        f"{best_sharpe['sharpe']:.4f}"
    )

    results_df.to_csv(
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
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()