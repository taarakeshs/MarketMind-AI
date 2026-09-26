from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


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

RESULT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ranking_results.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "ranking_random_forest.joblib"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N_VALUES = [5, 10, 20, 50]

RANDOM_STATE = 42


# ============================================================
# FEATURE SELECTION
# ============================================================

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return",

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

    "volume_change",
    "volume_ratio",

    "high_low_range",
    "open_close_range",
    "close_to_high",
    "close_to_low",

    "price_to_sma_20",
    "price_to_sma_200",
    "ema_5_ema_20_ratio",
]


# ============================================================
# CREATE FUTURE RETURN TARGET
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:

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


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):

    dates = sorted(
        df["date"].unique()
    )

    n_dates = len(dates)

    train_end = int(
        n_dates * train_ratio
    )

    validation_end = int(
        n_dates
        * (train_ratio + validation_ratio)
    )

    train_dates = dates[:train_end]

    validation_dates = dates[
        train_end:validation_end
    ]

    test_dates = dates[
        validation_end:
    ]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    validation = df[
        df["date"].isin(validation_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

    return train, validation, test


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CROSS-SECTIONAL STOCK RANKING")
    print("=" * 70)

    print()
    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["ticker", "date"]
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['ticker'].nunique()}"
    )

    print(
        f"Trading dates: {df['date'].nunique()}"
    )

    # --------------------------------------------------------
    # CREATE TARGET
    # --------------------------------------------------------

    print()
    print("Creating future return target...")

    df = create_target(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    required_columns = (
        FEATURE_COLUMNS
        + ["future_return"]
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

    print(
        f"Rows after target creation: {len(df):,}"
    )

    # --------------------------------------------------------
    # TIME SERIES SPLIT
    # --------------------------------------------------------

    train, validation, test = (
        chronological_split(df)
    )

    print()
    print("=" * 70)
    print("TIME SERIES SPLIT")
    print("=" * 70)

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    print()
    print(
        f"Train period: "
        f"{train['date'].min().date()} "
        f"→ "
        f"{train['date'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{validation['date'].min().date()} "
        f"→ "
        f"{validation['date'].max().date()}"
    )

    print(
        f"Test period: "
        f"{test['date'].min().date()} "
        f"→ "
        f"{test['date'].max().date()}"
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "future_return"
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        "future_return"
    ]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        "future_return"
    ]

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING RANDOM FOREST RANKING MODEL")
    print("=" * 70)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print()
    print("Training...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_predictions = (
        model.predict(X_validation)
    )

    validation_mae = (
        mean_absolute_error(
            y_validation,
            validation_predictions,
        )
    )

    validation_r2 = (
        r2_score(
            y_validation,
            validation_predictions,
        )
    )

    print()
    print("VALIDATION RESULTS")
    print("-" * 50)

    print(
        f"MAE : {validation_mae:.6f}"
    )

    print(
        f"R²  : {validation_r2:.6f}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = (
        model.predict(X_test)
    )

    test["predicted_return"] = (
        test_predictions
    )

    test_mae = (
        mean_absolute_error(
            y_test,
            test_predictions,
        )
    )

    test_r2 = (
        r2_score(
            y_test,
            test_predictions,
        )
    )

    print()
    print("TEST RESULTS")
    print("-" * 50)

    print(
        f"MAE : {test_mae:.6f}"
    )

    print(
        f"R²  : {test_r2:.6f}"
    )

    # --------------------------------------------------------
    # CROSS-SECTIONAL RANKING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CROSS-SECTIONAL RANKING")
    print("=" * 70)

    test["rank"] = (
        test.groupby("date")[
            "predicted_return"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    test["actual_rank"] = (
        test.groupby("date")[
            "future_return"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    # --------------------------------------------------------
    # TOP-N PORTFOLIO EXPERIMENT
    # --------------------------------------------------------

    results = []

    for top_n in TOP_N_VALUES:

        print()
        print("=" * 70)
        print(
            f"TOP {top_n} PORTFOLIO"
        )
        print("=" * 70)

        selected = test[
            test["rank"] <= top_n
        ].copy()

        daily_returns = (
            selected.groupby("date")[
                "future_return"
            ]
            .mean()
        )

        total_return = (
            (1 + daily_returns)
            .prod()
            - 1
        )

        trading_days = len(
            daily_returns
        )

        if trading_days > 0:

            annualized_return = (
                (1 + total_return)
                ** (
                    252
                    / trading_days
                )
                - 1
            )

        else:

            annualized_return = np.nan

        volatility = (
            daily_returns.std()
            * np.sqrt(252)
        )

        if volatility > 0:

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

        average_trade = (
            selected["future_return"]
            .mean()
        )

        print(
            f"Stocks per day : {top_n}"
        )

        print(
            f"Total return   : "
            f"{total_return:.4%}"
        )

        print(
            f"Annualized     : "
            f"{annualized_return:.4%}"
        )

        print(
            f"Volatility     : "
            f"{volatility:.4%}"
        )

        print(
            f"Sharpe         : "
            f"{sharpe:.4f}"
        )

        print(
            f"Max drawdown   : "
            f"{max_drawdown:.4%}"
        )

        print(
            f"Winning days   : "
            f"{winning_days:.4%}"
        )

        print(
            f"Average return : "
            f"{average_trade:.6%}"
        )

        results.append(
            {
                "strategy": f"top_{top_n}",
                "top_n": top_n,
                "total_return": total_return,
                "annualized_return":
                    annualized_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "max_drawdown":
                    max_drawdown,
                "winning_days":
                    winning_days,
                "average_trade":
                    average_trade,
            }
        )

    # --------------------------------------------------------
    # LONG-SHORT PORTFOLIO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LONG-SHORT RANKING STRATEGY")
    print("=" * 70)

    test["reverse_rank"] = (
        test.groupby("date")[
            "predicted_return"
        ]
        .rank(
            ascending=True,
            method="first",
        )
    )

    long_positions = test[
        test["rank"] <= 10
    ]

    short_positions = test[
        test["reverse_rank"] <= 10
    ]

    long_returns = (
        long_positions
        .groupby("date")[
            "future_return"
        ]
        .mean()
    )

    short_returns = (
        short_positions
        .groupby("date")[
            "future_return"
        ]
        .mean()
    )

    common_dates = (
        long_returns.index
        .intersection(
            short_returns.index
        )
    )

    long_short_returns = (
        long_returns.loc[common_dates]
        - short_returns.loc[common_dates]
    )

    ls_total_return = (
        (1 + long_short_returns)
        .prod()
        - 1
    )

    ls_volatility = (
        long_short_returns.std()
        * np.sqrt(252)
    )

    if long_short_returns.std() > 0:

        ls_sharpe = (
            long_short_returns.mean()
            / long_short_returns.std()
            * np.sqrt(252)
        )

    else:

        ls_sharpe = np.nan

    ls_cumulative = (
        1 + long_short_returns
    ).cumprod()

    ls_running_max = (
        ls_cumulative.cummax()
    )

    ls_drawdown = (
        ls_cumulative
        / ls_running_max
        - 1
    )

    ls_max_drawdown = (
        ls_drawdown.min()
    )

    print(
        f"Long stocks  : 10"
    )

    print(
        f"Short stocks : 10"
    )

    print(
        f"Total return : "
        f"{ls_total_return:.4%}"
    )

    print(
        f"Volatility   : "
        f"{ls_volatility:.4%}"
    )

    print(
        f"Sharpe       : "
        f"{ls_sharpe:.4f}"
    )

    print(
        f"Max drawdown : "
        f"{ls_max_drawdown:.4%}"
    )

    results.append(
        {
            "strategy": "long_short_top10",
            "top_n": 10,
            "total_return":
                ls_total_return,
            "annualized_return": np.nan,
            "volatility":
                ls_volatility,
            "sharpe":
                ls_sharpe,
            "max_drawdown":
                ls_max_drawdown,
            "winning_days":
                (long_short_returns > 0).mean(),
            "average_trade":
                long_short_returns.mean(),
        }
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("RANKING COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print()
    print(
        f"Model saved to:\n{MODEL_FILE}"
    )

    print()
    print(
        f"Results saved to:\n{RESULT_FILE}"
    )

    print()
    print("=" * 70)
    print("CROSS-SECTIONAL RANKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()