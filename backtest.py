from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


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
    / "backtest_results.csv"
)

DAILY_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "backtest_daily_returns.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return",

    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",

    "ema_5",
    "ema_10",
    "ema_20",
    "ema_50",

    "rsi",

    "macd",
    "macd_signal",
    "macd_histogram",

    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_position",

    "atr",
    "atr_percent",

    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_50d",

    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_60",

    "high_low_range",
    "open_close_range",
    "close_to_high",
    "close_to_low",

    "volume_change",
    "volume_ratio",
    "volume_sma_5",
    "volume_sma_20",

    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",

    "ema_5_ema_20_ratio",
]


# ============================================================
# MODEL
# ============================================================

def create_model():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


# ============================================================
# METRICS
# ============================================================

def calculate_sharpe(returns):

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


def calculate_max_drawdown(returns):

    returns = pd.Series(
        returns
    ).fillna(0)

    equity = (
        1 + returns
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    return drawdown.min()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WALK-FORWARD TRADING BACKTEST")
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
        ["date", "ticker"]
    ).copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # We use NEXT day's actual return.
    #
    # The model only sees information available
    # at today's close.
    #
    # Therefore there is no look-ahead in the signal.
    # --------------------------------------------------------

    df["next_return"] = (
        df.groupby("ticker")["close"]
        .shift(-1)
        / df["close"]
        - 1
    )

    df = df.dropna(
        subset=[
            "next_return"
        ]
    ).copy()

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    dates = sorted(
        df["date"].unique()
    )

    n_dates = len(dates)

    print(
        f"Trading dates: {n_dates}"
    )

    # --------------------------------------------------------
    # WALK-FORWARD CONFIG
    # --------------------------------------------------------

    n_folds = 4

    initial_train_ratio = 0.50

    test_ratio = 0.10

    initial_train_end = int(
        n_dates
        * initial_train_ratio
    )

    test_size = int(
        n_dates
        * test_ratio
    )

    # --------------------------------------------------------
    # SIGNAL THRESHOLDS
    # --------------------------------------------------------

    BUY_THRESHOLD = 0.55

    SELL_THRESHOLD = 0.45

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    all_predictions = []

    fold_results = []

    # ========================================================
    # WALK-FORWARD LOOP
    # ========================================================

    for fold in range(n_folds):

        print()
        print("=" * 70)

        print(
            f"BACKTEST FOLD "
            f"{fold + 1}/{n_folds}"
        )

        print("=" * 70)

        train_end = (
            initial_train_end
            + fold * test_size
        )

        test_start = train_end

        test_end = (
            test_start
            + test_size
        )

        if test_end > n_dates:

            print(
                "Not enough dates."
            )

            break

        train_dates = dates[
            :train_end
        ]

        test_dates = dates[
            test_start:test_end
        ]

        train = df[
            df["date"].isin(
                train_dates
            )
        ].copy()

        test = df[
            df["date"].isin(
                test_dates
            )
        ].copy()

        print()
        print(
            f"Training: "
            f"{train_dates[0]} → "
            f"{train_dates[-1]}"
        )

        print(
            f"Testing : "
            f"{test_dates[0]} → "
            f"{test_dates[-1]}"
        )

        print(
            f"Train rows: "
            f"{len(train):,}"
        )

        print(
            f"Test rows : "
            f"{len(test):,}"
        )

        # ----------------------------------------------------
        # PREPARE MODEL DATA
        # ----------------------------------------------------

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train = train[
            "target"
        ]

        X_test = test[
            FEATURE_COLUMNS
        ]

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print()
        print(
            "Training Random Forest..."
        )

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # PREDICT PROBABILITIES
        # ----------------------------------------------------

        test["probability_up"] = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # CREATE SIGNAL
        # ----------------------------------------------------

        test["signal"] = np.where(
            test[
                "probability_up"
            ] >= BUY_THRESHOLD,
            1,
            np.where(
                test[
                    "probability_up"
                ] <= SELL_THRESHOLD,
                -1,
                0,
            ),
        )

        # ----------------------------------------------------
        # STRATEGY RETURN
        # ----------------------------------------------------

        test["strategy_return"] = (
            test["signal"]
            * test["next_return"]
        )

        # ----------------------------------------------------
        # TRANSACTION COST
        # ----------------------------------------------------

        # Simple 0.05% cost whenever
        # the strategy takes a position.

        TRANSACTION_COST = 0.0005

        test["strategy_return"] = (
            test["strategy_return"]
            - np.where(
                test["signal"] != 0,
                TRANSACTION_COST,
                0,
            )
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        all_predictions.append(
            test[
                [
                    "date",
                    "ticker",
                    "close",
                    "probability_up",
                    "signal",
                    "next_return",
                    "strategy_return",
                ]
            ]
        )

        # ----------------------------------------------------
        # FOLD METRICS
        # ----------------------------------------------------

        active = (
            test["signal"] != 0
        )

        active_count = int(
            active.sum()
        )

        if active_count > 0:

            active_returns = test.loc[
                active,
                "strategy_return"
            ]

            win_rate = (
                active_returns > 0
            ).mean()

            average_return = (
                active_returns.mean()
            )

        else:

            win_rate = 0

            average_return = 0

        print()
        print(
            "FOLD PERFORMANCE"
        )

        print("-" * 50)

        print(
            f"Active trades : "
            f"{active_count:,}"
        )

        print(
            f"Win rate      : "
            f"{win_rate:.4f}"
        )

        print(
            f"Average trade : "
            f"{average_return:.6f}"
        )

        fold_results.append(
            {
                "fold": fold + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "trades": active_count,
                "win_rate": win_rate,
                "average_trade_return":
                    average_return,
            }
        )

    # ========================================================
    # COMBINE PREDICTIONS
    # ========================================================

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ========================================================
    # DAILY PORTFOLIO
    # ========================================================

    daily = (
        predictions
        .groupby("date")
        ["strategy_return"]
        .mean()
        .reset_index()
    )

    daily = daily.sort_values(
        "date"
    )

    # ========================================================
    # METRICS
    # ========================================================

    total_return = (
        1
        + daily[
            "strategy_return"
        ]
    ).prod() - 1

    annualized_return = (
        1 + total_return
    ) ** (
        252
        / len(daily)
    ) - 1

    sharpe = calculate_sharpe(
        daily[
            "strategy_return"
        ]
    )

    max_drawdown = (
        calculate_max_drawdown(
            daily[
                "strategy_return"
            ]
        )
    )

    winning_days = (
        daily[
            "strategy_return"
        ] > 0
    ).mean()

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL BACKTEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Total Return       : "
        f"{total_return:.2%}"
    )

    print(
        f"Annualized Return  : "
        f"{annualized_return:.2%}"
    )

    print(
        f"Sharpe Ratio       : "
        f"{sharpe:.4f}"
    )

    print(
        f"Maximum Drawdown   : "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Winning Days       : "
        f"{winning_days:.2%}"
    )

    print()
    print(
        f"Total predictions  : "
        f"{len(predictions):,}"
    )

    print(
        f"Trading days       : "
        f"{len(daily):,}"
    )

    # ========================================================
    # BUY / SELL / HOLD COUNTS
    # ========================================================

    print()
    print("SIGNAL DISTRIBUTION")
    print("-" * 50)

    print(
        "BUY  :",
        int(
            (
                predictions["signal"]
                == 1
            ).sum()
        ),
    )

    print(
        "HOLD :",
        int(
            (
                predictions["signal"]
                == 0
            ).sum()
        ),
    )

    print(
        "SELL :",
        int(
            (
                predictions["signal"]
                == -1
            ).sum()
        ),
    )

    # ========================================================
    # SAVE
    # ========================================================

    predictions.to_csv(
        RESULTS_FILE,
        index=False,
    )

    daily.to_csv(
        DAILY_FILE,
        index=False,
    )

    print()
    print(
        "Trade-level results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print(
        "Daily returns saved to:"
    )

    print(
        DAILY_FILE
    )

    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()