"""
REGIME-ADAPTIVE ML PORTFOLIO STRATEGY

Strategy:
    BULL          -> ML Top 20
    BEAR          -> ML Top 50
    SIDEWAYS      -> ML Top 50
    BEAR_HIGH_VOL -> ML Top 50

The strategy is compared against:
    1. Equal Weight Buy & Hold
    2. Fixed ML Top 20
    3. Fixed ML Top 50
    4. Regime-Adaptive ML

Includes:
    - Time-series split
    - Market regime detection
    - Random Forest ranking model
    - Transaction costs
    - Portfolio returns
    - Sharpe ratio
    - Maximum drawdown
    - Winning days
    - Turnover
    - Regime-wise performance
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "processed",
    "features.csv"
)

OUTPUT_PATH = os.path.join(
    "data",
    "processed",
    "regime_adaptive_results.csv"
)

MODEL_PATH = os.path.join(
    "models",
    "regime_adaptive_ranking.joblib"
)

TRANSACTION_COST = 0.001

TOP_N_BULL = 20
TOP_N_BEAR = 50
TOP_N_SIDEWAYS = 50
TOP_N_HIGH_VOL = 50

TRAIN_RATIO = 0.70

RANDOM_STATE = 42


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def max_drawdown(returns):

    cumulative = (1 + returns).cumprod()

    peak = cumulative.cummax()

    drawdown = cumulative / peak - 1

    return drawdown.min()


def calculate_metrics(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "winning_days": 0,
        }

    total_return = (1 + returns).prod() - 1

    annualized_return = (
        (1 + total_return) ** (252 / len(returns)) - 1
    )

    volatility = returns.std() * np.sqrt(252)

    if returns.std() > 0:
        sharpe = (
            returns.mean() / returns.std()
        ) * np.sqrt(252)
    else:
        sharpe = 0

    winning_days = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "winning_days": winning_days,
    }


# ============================================================
# MARKET REGIME DETECTION
# ============================================================

def calculate_market_regimes(df):

    print("\nCalculating market regimes...")

    market = (
        df.groupby("date")["return_1d"]
        .mean()
        .sort_index()
    )

    market_20 = market.rolling(20).mean()

    market_vol = (
        market
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    regime_df = pd.DataFrame({
        "market_return": market,
        "market_20d": market_20,
        "market_volatility": market_vol,
    })

    # Thresholds are intentionally simple.
    # The purpose is to classify broad market conditions.

    regime_df["regime"] = "SIDEWAYS"

    bull_condition = (
        (regime_df["market_20d"] > 0)
        & (regime_df["market_volatility"] < 0.40)
    )

    bear_condition = (
        (regime_df["market_20d"] < 0)
        & (regime_df["market_volatility"] < 0.40)
    )

    high_vol_bear_condition = (
        (regime_df["market_20d"] < 0)
        & (regime_df["market_volatility"] >= 0.40)
    )

    regime_df.loc[
        bull_condition,
        "regime"
    ] = "BULL"

    regime_df.loc[
        bear_condition,
        "regime"
    ] = "BEAR"

    regime_df.loc[
        high_vol_bear_condition,
        "regime"
    ] = "BEAR_HIGH_VOL"

    return regime_df.reset_index()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REGIME-ADAPTIVE ML PORTFOLIO STRATEGY")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading feature dataset...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    print(f"Rows   : {len(df):,}")
    print(f"Stocks : {df['ticker'].nunique():,}")
    print(f"Dates  : {df['date'].nunique():,}")


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if "next_return" not in df.columns:

        if "next_close" in df.columns:

            df["next_return"] = (
                df["next_close"] / df["close"]
            ) - 1

        else:

            raise ValueError(
                "next_return or next_close not found."
            )


    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=["next_return"]
    )


    # --------------------------------------------------------
    # MARKET REGIMES
    # --------------------------------------------------------

    regime_df = calculate_market_regimes(df)

    df = df.merge(
        regime_df[
            [
                "date",
                "regime",
                "market_return",
                "market_volatility",
            ]
        ],
        on="date",
        how="left"
    )

    df = df.dropna(
        subset=["regime"]
    )

    print("\nREGIME DISTRIBUTION")
    print("-" * 60)

    print(
        regime_df["regime"]
        .value_counts()
    )


    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    excluded = {
        "date",
        "ticker",
        "next_close",
        "next_return",
        "target",
        "regime",
        "market_return",
        "market_volatility",
    }

    feature_columns = []

    for col in df.columns:

        if col in excluded:
            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            feature_columns.append(col)

    print(
        f"\nCandidate features: "
        f"{len(feature_columns)}"
    )


    # --------------------------------------------------------
    # TIME SERIES SPLIT
    # --------------------------------------------------------

    dates = sorted(
        df["date"].unique()
    )

    split_index = int(
        len(dates) * TRAIN_RATIO
    )

    train_end = dates[split_index]

    train = df[
        df["date"] < train_end
    ].copy()

    test = df[
        df["date"] >= train_end
    ].copy()

    print("\nTIME SERIES SPLIT")
    print("-" * 60)

    print(
        f"Train : {train['date'].min().date()} "
        f"→ {train['date'].max().date()}"
    )

    print(
        f"Test  : {test['date'].min().date()} "
        f"→ {test['date'].max().date()}"
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows : {len(test):,}"
    )


    # --------------------------------------------------------
    # TRAIN RANKING MODEL
    # --------------------------------------------------------

    print("\nTraining ranking model...")

    X_train = train[
        feature_columns
    ]

    y_train = train[
        "next_return"
    ]

    X_test = test[
        feature_columns
    ]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model trained successfully.")

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to:\n{MODEL_PATH}"
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    test["predicted_return"] = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # STRATEGY CONFIGURATION
    # --------------------------------------------------------

    def get_top_n(regime):

        if regime == "BULL":
            return TOP_N_BULL

        if regime == "BEAR":
            return TOP_N_BEAR

        if regime == "BEAR_HIGH_VOL":
            return TOP_N_HIGH_VOL

        return TOP_N_SIDEWAYS


    # --------------------------------------------------------
    # DAILY PORTFOLIO SIMULATION
    # --------------------------------------------------------

    strategies = {
        "Equal Weight": [],
        "Fixed Top 20": [],
        "Fixed Top 50": [],
        "Regime Adaptive": [],
    }

    dates_test = sorted(
        test["date"].unique()
    )

    previous_holdings = {
        "Fixed Top 20": set(),
        "Fixed Top 50": set(),
        "Regime Adaptive": set(),
    }

    turnover_records = []

    regime_records = []


    for current_date in dates_test:

        daily = test[
            test["date"] == current_date
        ].copy()

        if daily.empty:
            continue


        # ----------------------------------------------------
        # EQUAL WEIGHT
        # ----------------------------------------------------

        equal_return = daily[
            "next_return"
        ].mean()

        strategies[
            "Equal Weight"
        ].append(
            {
                "date": current_date,
                "return": equal_return,
            }
        )


        # ----------------------------------------------------
        # RANK STOCKS
        # ----------------------------------------------------

        daily = daily.sort_values(
            "predicted_return",
            ascending=False
        )


        # ----------------------------------------------------
        # FIXED TOP 20
        # ----------------------------------------------------

        top20 = daily.head(
            20
        )

        top20_tickers = set(
            top20["ticker"]
        )

        turnover20 = len(
            top20_tickers
            ^ previous_holdings[
                "Fixed Top 20"
            ]
        ) / max(
            len(top20_tickers),
            1
        )

        return20 = top20[
            "next_return"
        ].mean()

        return20 -= (
            turnover20
            * TRANSACTION_COST
        )

        strategies[
            "Fixed Top 20"
        ].append(
            {
                "date": current_date,
                "return": return20,
            }
        )

        previous_holdings[
            "Fixed Top 20"
        ] = top20_tickers


        # ----------------------------------------------------
        # FIXED TOP 50
        # ----------------------------------------------------

        top50 = daily.head(
            50
        )

        top50_tickers = set(
            top50["ticker"]
        )

        turnover50 = len(
            top50_tickers
            ^ previous_holdings[
                "Fixed Top 50"
            ]
        ) / max(
            len(top50_tickers),
            1
        )

        return50 = top50[
            "next_return"
        ].mean()

        return50 -= (
            turnover50
            * TRANSACTION_COST
        )

        strategies[
            "Fixed Top 50"
        ].append(
            {
                "date": current_date,
                "return": return50,
            }
        )

        previous_holdings[
            "Fixed Top 50"
        ] = top50_tickers


        # ----------------------------------------------------
        # REGIME ADAPTIVE
        # ----------------------------------------------------

        regime = daily[
            "regime"
        ].iloc[0]

        top_n = get_top_n(
            regime
        )

        selected = daily.head(
            top_n
        )

        selected_tickers = set(
            selected["ticker"]
        )

        turnover_adaptive = len(
            selected_tickers
            ^ previous_holdings[
                "Regime Adaptive"
            ]
        ) / max(
            len(selected_tickers),
            1
        )

        adaptive_return = selected[
            "next_return"
        ].mean()

        adaptive_return -= (
            turnover_adaptive
            * TRANSACTION_COST
        )

        strategies[
            "Regime Adaptive"
        ].append(
            {
                "date": current_date,
                "return": adaptive_return,
            }
        )

        previous_holdings[
            "Regime Adaptive"
        ] = selected_tickers


        turnover_records.append(
            {
                "date": current_date,
                "regime": regime,
                "top_n": top_n,
                "turnover": turnover_adaptive,
            }
        )


        regime_records.append(
            {
                "date": current_date,
                "regime": regime,
                "return": adaptive_return,
            }
        )


    # --------------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 70)

    results = []

    for strategy_name, values in strategies.items():

        returns = pd.Series(
            [
                x["return"]
                for x in values
            ]
        )

        metrics = calculate_metrics(
            returns
        )

        results.append(
            {
                "strategy": strategy_name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # REGIME PERFORMANCE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("REGIME-ADAPTIVE PERFORMANCE BY REGIME")
    print("=" * 70)

    regime_returns = pd.DataFrame(
        regime_records
    )

    regime_results = []

    for regime, group in regime_returns.groupby(
        "regime"
    ):

        metrics = calculate_metrics(
            group["return"]
        )

        regime_results.append(
            {
                "regime": regime,
                "days": len(group),
                **metrics,
            }
        )

    regime_results_df = pd.DataFrame(
        regime_results
    )

    print(
        regime_results_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # TURNOVER
    # --------------------------------------------------------

    turnover_df = pd.DataFrame(
        turnover_records
    )

    average_turnover = (
        turnover_df["turnover"]
        .mean()
    )

    print("\n")
    print("=" * 70)
    print("TRADING STATISTICS")
    print("=" * 70)

    print(
        f"Average daily turnover : "
        f"{average_turnover:.4f}"
    )

    print(
        f"Transaction cost       : "
        f"{TRANSACTION_COST:.4f}"
    )

    print(
        f"Trading days           : "
        f"{len(dates_test)}"
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    regime_output = os.path.join(
        "data",
        "processed",
        "regime_adaptive_by_regime.csv"
    )

    regime_results_df.to_csv(
        regime_output,
        index=False
    )

    turnover_output = os.path.join(
        "data",
        "processed",
        "regime_adaptive_turnover.csv"
    )

    turnover_df.to_csv(
        turnover_output,
        index=False
    )


    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nRegime results saved to:")
    print(regime_output)

    print("\nTurnover results saved to:")
    print(turnover_output)

    print("\n")
    print("=" * 70)
    print("REGIME-ADAPTIVE STRATEGY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()