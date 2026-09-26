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

FEATURE_PATH = os.path.join(
    "data",
    "processed",
    "features.csv"
)

OUTPUT_PATH = os.path.join(
    "data",
    "processed",
    "transaction_cost_regime_adaptive_results.csv"
)

MODEL_PATH = os.path.join(
    "models",
    "transaction_cost_regime_model.joblib"
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.70

# Regime-specific portfolio sizes
TOP_N_BULL = 20
TOP_N_BEAR = 50
TOP_N_HIGH_VOL = 50
TOP_N_SIDEWAYS = 20

# Transaction costs to test
TRANSACTION_COSTS = [
    0.0000,   # 0.00%
    0.0005,   # 0.05%
    0.0010,   # 0.10%
    0.0015,   # 0.15%
    0.0020,   # 0.20%
    0.0030,   # 0.30%
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "winning_days": np.nan,
        }

    cumulative = (1 + returns).cumprod()

    total_return = cumulative.iloc[-1] - 1

    years = len(returns) / 252

    if years > 0:
        annualized_return = (
            cumulative.iloc[-1] ** (1 / years)
        ) - 1
    else:
        annualized_return = np.nan

    volatility = returns.std() * np.sqrt(252)

    if returns.std() > 0:
        sharpe = (
            returns.mean() / returns.std()
        ) * np.sqrt(252)
    else:
        sharpe = np.nan

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    winning_days = (
        returns > 0
    ).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_days": winning_days,
    }


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(df):

    print("\nCalculating market returns...")

    market = (
        df.groupby("date")["return_1d"]
        .mean()
        .reset_index()
    )

    market = market.sort_values("date")

    market["market_return"] = market[
        "return_1d"
    ]

    market["market_volatility"] = (
        market["market_return"]
        .rolling(20)
        .std()
    )

    market["market_ma_20"] = (
        market["market_return"]
        .rolling(20)
        .mean()
    )

    # 20-day cumulative market return
    market["market_momentum"] = (
        (1 + market["market_return"])
        .rolling(20)
        .apply(np.prod, raw=True)
        - 1
    )

    # Volatility threshold
    volatility_threshold = (
        market["market_volatility"]
        .rolling(60)
        .median()
    )

    conditions = [

        # High volatility bear
        (
            (market["market_momentum"] < -0.03)
            &
            (
                market["market_volatility"]
                > volatility_threshold * 1.5
            )
        ),

        # Bear
        (
            market["market_momentum"] < -0.03
        ),

        # Bull
        (
            market["market_momentum"] > 0.03
        ),
    ]

    choices = [
        "BEAR_HIGH_VOL",
        "BEAR",
        "BULL",
    ]

    market["regime"] = np.select(
        conditions,
        choices,
        default="SIDEWAYS"
    )

    return market[
        [
            "date",
            "market_return",
            "market_volatility",
            "regime",
        ]
    ]


# ============================================================
# TOP N SELECTION
# ============================================================

def get_top_n(regime):

    if regime == "BULL":
        return TOP_N_BULL

    if regime == "BEAR":
        return TOP_N_BEAR

    if regime == "BEAR_HIGH_VOL":
        return TOP_N_HIGH_VOL

    return TOP_N_SIDEWAYS


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REGIME-ADAPTIVE TRANSACTION COST ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["date"]
    )

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Stocks : {df['ticker'].nunique():,}"
    )

    print(
        f"Dates  : {df['date'].nunique():,}"
    )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    regime_df = calculate_market_regime(
        df
    )

    print("\nREGIME DISTRIBUTION")
    print("-" * 60)

    print(
        regime_df["regime"]
        .value_counts()
    )

    # Merge regime information
    df = df.merge(
        regime_df,
        on="date",
        how="left"
    )

    df = df.dropna(
        subset=["regime"]
    )

    print(
        f"\nRows after regime merge: "
        f"{len(df):,}"
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
    # REMOVE MISSING VALUES
    # --------------------------------------------------------

    required_columns = (
        feature_columns
        + [
            "next_return",
            "regime",
            "ticker",
            "date",
        ]
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

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
        f"Train : "
        f"{train['date'].min().date()} "
        f"→ "
        f"{train['date'].max().date()}"
    )

    print(
        f"Test  : "
        f"{test['date'].min().date()} "
        f"→ "
        f"{test['date'].max().date()}"
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows : {len(test):,}"
    )

    # --------------------------------------------------------
    # TRAIN MODEL
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

    print(
        "Model trained successfully."
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

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

    test["predicted_return"] = (
        model.predict(X_test)
    )

    # --------------------------------------------------------
    # RUN TRANSACTION COST EXPERIMENT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RUNNING TRANSACTION COST EXPERIMENT")
    print("=" * 70)

    all_results = []

    dates_test = sorted(
        test["date"].unique()
    )

    for transaction_cost in TRANSACTION_COSTS:

        print("\n")
        print("-" * 70)

        print(
            f"TRANSACTION COST: "
            f"{transaction_cost:.4f}"
        )

        previous_holdings = set()

        daily_returns = []

        turnover_records = []

        regime_records = []

        for current_date in dates_test:

            daily = test[
                test["date"] == current_date
            ].copy()

            if daily.empty:
                continue

            # ------------------------------------------------
            # RANK STOCKS
            # ------------------------------------------------

            daily = daily.sort_values(
                "predicted_return",
                ascending=False
            )

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

            # ------------------------------------------------
            # TURNOVER
            # ------------------------------------------------

            if len(previous_holdings) == 0:

                turnover = 1.0

            else:

                changed_positions = (
                    selected_tickers
                    ^ previous_holdings
                )

                turnover = (
                    len(changed_positions)
                    / max(
                        len(selected_tickers),
                        1
                    )
                )

            # ------------------------------------------------
            # GROSS RETURN
            # ------------------------------------------------

            gross_return = selected[
                "next_return"
            ].mean()

            # ------------------------------------------------
            # TRANSACTION COST
            # ------------------------------------------------

            cost = (
                turnover
                * transaction_cost
            )

            net_return = (
                gross_return
                - cost
            )

            daily_returns.append(
                net_return
            )

            turnover_records.append(
                turnover
            )

            regime_records.append(
                {
                    "date": current_date,
                    "regime": regime,
                    "top_n": top_n,
                    "gross_return": gross_return,
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "net_return": net_return,
                }
            )

            previous_holdings = (
                selected_tickers
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(
            daily_returns
        )

        average_turnover = (
            np.mean(turnover_records)
            if turnover_records
            else np.nan
        )

        all_results.append(
            {
                "transaction_cost":
                    transaction_cost,

                "cost_percent":
                    transaction_cost * 100,

                **metrics,

                "average_turnover":
                    average_turnover,

                "trading_days":
                    len(daily_returns),
            }
        )

        print(
            f"Total Return : "
            f"{metrics['total_return']:.4%}"
        )

        print(
            f"Annualized   : "
            f"{metrics['annualized_return']:.4%}"
        )

        print(
            f"Volatility   : "
            f"{metrics['volatility']:.4%}"
        )

        print(
            f"Sharpe       : "
            f"{metrics['sharpe']:.4f}"
        )

        print(
            f"Max Drawdown : "
            f"{metrics['max_drawdown']:.4%}"
        )

        print(
            f"Win Rate     : "
            f"{metrics['winning_days']:.2%}"
        )

        print(
            f"Avg Turnover : "
            f"{average_turnover:.4f}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    print("\n")
    print("=" * 70)
    print("TRANSACTION COST COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # REGIME BREAKDOWN AT EACH COST
    # --------------------------------------------------------

    regime_results = []

    for transaction_cost in TRANSACTION_COSTS:

        previous_holdings = set()

        records = []

        for current_date in dates_test:

            daily = test[
                test["date"] == current_date
            ].copy()

            if daily.empty:
                continue

            daily = daily.sort_values(
                "predicted_return",
                ascending=False
            )

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

            if len(previous_holdings) == 0:

                turnover = 1.0

            else:

                turnover = (
                    len(
                        selected_tickers
                        ^ previous_holdings
                    )
                    / max(
                        len(selected_tickers),
                        1
                    )
                )

            gross_return = selected[
                "next_return"
            ].mean()

            net_return = (
                gross_return
                - turnover
                * transaction_cost
            )

            records.append(
                {
                    "date": current_date,
                    "regime": regime,
                    "return": net_return,
                }
            )

            previous_holdings = (
                selected_tickers
            )

        regime_df_results = pd.DataFrame(
            records
        )

        for regime, group in (
            regime_df_results
            .groupby("regime")
        ):

            metrics = calculate_metrics(
                group["return"]
            )

            regime_results.append(
                {
                    "transaction_cost":
                        transaction_cost,

                    "regime":
                        regime,

                    "days":
                        len(group),

                    **metrics,
                }
            )

    regime_results_df = pd.DataFrame(
        regime_results
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    regime_output = os.path.join(
        "data",
        "processed",
        "transaction_cost_regime_by_regime.csv"
    )

    regime_results_df.to_csv(
        regime_output,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nOverall results:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nRegime results:\n"
        f"{regime_output}"
    )

    print("\n")
    print("=" * 70)
    print(
        "TRANSACTION COST REGIME "
        "ANALYSIS COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()