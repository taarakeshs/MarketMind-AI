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

MODEL_DIR = "models"

OUTPUT_DIR = os.path.join(
    "data",
    "processed"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "turnover_control_model.joblib"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "turnover_control_results.csv"
)

DAILY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "turnover_control_daily.csv"
)

REGIME_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "turnover_control_by_regime.csv"
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.70


# ============================================================
# TRANSACTION COSTS
# ============================================================

TRANSACTION_COSTS = [
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
]


# ============================================================
# STRATEGY CONFIGURATION
# ============================================================

TOP_N_BULL = 20
TOP_N_BEAR = 50
TOP_N_HIGH_VOL = 50
TOP_N_SIDEWAYS = 20


# ============================================================
# BUFFER CONFIGURATION
# ============================================================

BUFFER_SIZES = [
    0,
    5,
    10,
    15,
    20,
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

    trading_days = len(returns)

    if trading_days > 1:

        annualized_return = (
            cumulative.iloc[-1]
            ** (252 / trading_days)
        ) - 1

    else:

        annualized_return = np.nan

    volatility = (
        returns.std()
        * np.sqrt(252)
    )

    if (
        volatility > 0
        and not np.isnan(volatility)
    ):

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = np.nan

    running_max = cumulative.cummax()

    drawdown = (
        cumulative
        / running_max
        - 1
    )

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

def calculate_market_regimes(df):

    print("\nCalculating market returns...")

    market_returns = (
        df.groupby("date")["return_1d"]
        .mean()
        .rename("market_return")
    )

    market_volatility = (
        market_returns
        .rolling(20)
        .std()
        .rename("market_volatility")
    )

    regime_df = pd.concat(
        [
            market_returns,
            market_volatility,
        ],
        axis=1
    ).reset_index()

    return_rolling = (
        regime_df["market_return"]
        .rolling(20)
        .mean()
    )

    volatility_median = (
        regime_df["market_volatility"]
        .median()
    )

    regimes = []

    for i in range(len(regime_df)):

        rolling_return = (
            return_rolling.iloc[i]
        )

        volatility = (
            regime_df
            .loc[i, "market_volatility"]
        )

        if pd.isna(rolling_return):

            regimes.append("SIDEWAYS")

        elif (
            rolling_return < 0
            and volatility > volatility_median
        ):

            regimes.append(
                "BEAR_HIGH_VOL"
            )

        elif rolling_return < 0:

            regimes.append("BEAR")

        elif rolling_return > 0:

            regimes.append("BULL")

        else:

            regimes.append("SIDEWAYS")

    regime_df["regime"] = regimes

    return regime_df


# ============================================================
# REGIME TOP N
# ============================================================

def get_top_n(regime):

    if regime == "BULL":

        return TOP_N_BULL

    elif regime == "BEAR":

        return TOP_N_BEAR

    elif regime == "BEAR_HIGH_VOL":

        return TOP_N_HIGH_VOL

    else:

        return TOP_N_SIDEWAYS


# ============================================================
# CORRECT BUFFER PORTFOLIO SELECTION
# ============================================================

def select_with_buffer(
    daily,
    top_n,
    previous_holdings,
    buffer_size
):
    """
    Select portfolio using a true ranking buffer.

    Logic:

    1. Rank all stocks by predicted return.
    2. New stocks must enter the normal Top-N.
    3. Existing holdings are allowed to survive while
       they remain within Top-N + buffer.
    4. Existing holdings are preferred over new stocks
       when replacing positions.
    5. Final portfolio always contains exactly Top-N
       stocks when enough stocks are available.

    Example:

        top_n = 20
        buffer = 10

        Existing stock ranked 25:
            KEEP

        Existing stock ranked 31:
            REMOVE

    This prevents unnecessary turnover.
    """

    daily = daily.copy()

    # --------------------------------------------------------
    # Rank stocks
    # --------------------------------------------------------

    daily = daily.sort_values(
        "predicted_return",
        ascending=False
    ).reset_index(drop=True)

    daily["rank"] = (
        np.arange(len(daily)) + 1
    )

    # --------------------------------------------------------
    # Initial portfolio
    # --------------------------------------------------------

    if not previous_holdings:

        return daily.head(top_n).copy()

    # --------------------------------------------------------
    # Existing holdings that qualify for the buffer
    # --------------------------------------------------------

    buffer_limit = (
        top_n + buffer_size
    )

    existing_buffer = daily[
        (
            daily["ticker"].isin(
                previous_holdings
            )
        )
        &
        (
            daily["rank"] <= buffer_limit
        )
    ].copy()

    # --------------------------------------------------------
    # Normal Top-N candidates
    # --------------------------------------------------------

    normal_top = daily[
        daily["rank"] <= top_n
    ].copy()

    # --------------------------------------------------------
    # Start portfolio with existing holdings
    # that are protected by the buffer.
    #
    # This is the key correction.
    # --------------------------------------------------------

    selected_tickers = set()

    selected_rows = []

    # Existing stocks are retained first.
    #
    # Sort them by predicted return so that if the
    # buffer contains more stocks than the portfolio
    # capacity, the strongest ones survive.

    existing_buffer = (
        existing_buffer
        .sort_values(
            "predicted_return",
            ascending=False
        )
    )

    for _, row in existing_buffer.iterrows():

        ticker = row["ticker"]

        if ticker not in selected_tickers:

            selected_tickers.add(ticker)

            selected_rows.append(row)

        if len(selected_rows) >= top_n:

            break

    # --------------------------------------------------------
    # Fill remaining positions with best-ranked stocks
    # --------------------------------------------------------

    if len(selected_rows) < top_n:

        remaining = daily[
            ~daily["ticker"].isin(
                selected_tickers
            )
        ].copy()

        remaining = remaining.sort_values(
            "predicted_return",
            ascending=False
        )

        required = (
            top_n
            - len(selected_rows)
        )

        additions = remaining.head(
            required
        )

        for _, row in additions.iterrows():

            ticker = row["ticker"]

            if ticker not in selected_tickers:

                selected_tickers.add(
                    ticker
                )

                selected_rows.append(
                    row
                )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    selected = pd.DataFrame(
        selected_rows
    )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if len(selected) < top_n:

        fallback = daily[
            ~daily["ticker"].isin(
                selected_tickers
            )
        ].head(
            top_n - len(selected)
        )

        selected = pd.concat(
            [
                selected,
                fallback
            ],
            ignore_index=True
        )

    return selected.head(top_n).copy()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TURNOVER CONTROL / RANKING BUFFER BACKTEST"
    )
    print("=" * 70)

    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Stocks : {df['ticker'].nunique():,}"
    )

    print(
        f"Dates  : {df['date'].nunique():,}"
    )

    # ========================================================
    # MARKET REGIMES
    # ========================================================

    regime_df = calculate_market_regimes(
        df
    )

    print("\nREGIME DISTRIBUTION")
    print("-" * 60)

    print(
        regime_df["regime"].value_counts()
    )

    # ========================================================
    # MERGE REGIMES
    # ========================================================

    df = df.merge(
        regime_df[
            [
                "date",
                "market_return",
                "market_volatility",
                "regime",
            ]
        ],
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

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

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

            feature_columns.append(
                col
            )

    print(
        f"\nCandidate features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # REMOVE ROWS WITH MISSING MODEL FEATURES
    # ========================================================

    required_columns = (
        feature_columns
        + ["next_return"]
    )

    before_drop = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    after_drop = len(df)

    print(
        f"Rows removed due to missing "
        f"features/target: "
        f"{before_drop - after_drop:,}"
    )

    # ========================================================
    # TIME SERIES SPLIT
    # ========================================================

    dates = sorted(
        df["date"].unique()
    )

    split_index = int(
        len(dates)
        * TRAIN_RATIO
    )

    train_end = dates[
        split_index
    ]

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
        f"-> "
        f"{train['date'].max().date()}"
    )

    print(
        f"Test  : "
        f"{test['date'].min().date()} "
        f"-> "
        f"{test['date'].max().date()}"
    )

    print(
        f"Train rows: "
        f"{len(train):,}"
    )

    print(
        f"Test rows : "
        f"{len(test):,}"
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining ranking model..."
    )

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

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\nGenerating predictions..."
    )

    test["predicted_return"] = (
        model.predict(X_test)
    )

    # ========================================================
    # EXPERIMENT STORAGE
    # ========================================================

    all_results = []

    daily_results = []

    regime_results = []

    # ========================================================
    # EXPERIMENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "RUNNING BUFFER / TURNOVER EXPERIMENT"
    )
    print("=" * 70)

    dates_test = sorted(
        test["date"].unique()
    )

    for buffer_size in BUFFER_SIZES:

        print("\n")
        print("-" * 70)

        print(
            f"BUFFER SIZE: {buffer_size}"
        )

        print("-" * 70)

        for transaction_cost in TRANSACTION_COSTS:

            print(
                f"\nTransaction cost: "
                f"{transaction_cost:.4f}"
            )

            previous_holdings = set()

            daily_returns = []

            turnover_values = []

            # =================================================
            # DAILY SIMULATION
            # =================================================

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

                # ---------------------------------------------
                # SELECT PORTFOLIO
                # ---------------------------------------------

                selected = select_with_buffer(
                    daily=daily,
                    top_n=top_n,
                    previous_holdings=previous_holdings,
                    buffer_size=buffer_size,
                )

                selected_tickers = set(
                    selected["ticker"]
                )

                # ---------------------------------------------
                # TURNOVER
                # ---------------------------------------------

                if not previous_holdings:

                    turnover = 1.0

                else:

                    # Number of stocks entering + leaving.
                    changed_positions = len(
                        selected_tickers
                        ^
                        previous_holdings
                    )

                    turnover = (
                        changed_positions
                        /
                        max(
                            len(selected_tickers),
                            1
                        )
                    )

                # ---------------------------------------------
                # GROSS PORTFOLIO RETURN
                # ---------------------------------------------

                portfolio_return = (
                    selected[
                        "next_return"
                    ].mean()
                )

                # ---------------------------------------------
                # TRANSACTION COST
                # ---------------------------------------------

                transaction_cost_amount = (
                    turnover
                    * transaction_cost
                )

                net_return = (
                    portfolio_return
                    - transaction_cost_amount
                )

                daily_returns.append(
                    net_return
                )

                turnover_values.append(
                    turnover
                )

                # ---------------------------------------------
                # SAVE DAILY RESULT
                # ---------------------------------------------

                daily_results.append(
                    {
                        "buffer_size":
                            buffer_size,

                        "transaction_cost":
                            transaction_cost,

                        "date":
                            current_date,

                        "regime":
                            regime,

                        "top_n":
                            top_n,

                        "turnover":
                            turnover,

                        "gross_return":
                            portfolio_return,

                        "transaction_cost_amount":
                            transaction_cost_amount,

                        "return":
                            net_return,

                        "holdings":
                            len(
                                selected_tickers
                            ),
                    }
                )

                # ---------------------------------------------
                # UPDATE HOLDINGS
                # ---------------------------------------------

                previous_holdings = (
                    selected_tickers
                )

            # =================================================
            # OVERALL METRICS
            # =================================================

            metrics = calculate_metrics(
                daily_returns
            )

            average_turnover = (
                np.mean(turnover_values)
                if turnover_values
                else np.nan
            )

            all_results.append(
                {
                    "buffer_size":
                        buffer_size,

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
                f"Return : "
                f"{metrics['total_return']:.2%}"
            )

            print(
                f"Sharpe : "
                f"{metrics['sharpe']:.3f}"
            )

            print(
                f"Max DD : "
                f"{metrics['max_drawdown']:.2%}"
            )

            print(
                f"Turnover : "
                f"{average_turnover:.4f}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    daily_df = pd.DataFrame(
        daily_results
    )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BEST BUFFER CONFIGURATION"
    )
    print("=" * 70)

    # Practical reference:
    # 0.10% transaction cost

    practical = results_df[
        results_df[
            "transaction_cost"
        ] == 0.001
    ].copy()

    if not practical.empty:

        best = practical.loc[
            practical[
                "sharpe"
            ].idxmax()
        ]

        print(
            f"\nBest buffer size: "
            f"{int(best['buffer_size'])}"
        )

        print(
            f"Transaction cost: "
            f"{best['transaction_cost']:.4f}"
        )

        print(
            f"Total return: "
            f"{best['total_return']:.2%}"
        )

        print(
            f"Annualized return: "
            f"{best['annualized_return']:.2%}"
        )

        print(
            f"Sharpe: "
            f"{best['sharpe']:.3f}"
        )

        print(
            f"Max drawdown: "
            f"{best['max_drawdown']:.2%}"
        )

        print(
            f"Average turnover: "
            f"{best['average_turnover']:.4f}"
        )

    # ========================================================
    # REGIME ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BUFFER PERFORMANCE BY REGIME"
    )
    print("=" * 70)

    regime_grouped = (
        daily_df
        .groupby(
            [
                "buffer_size",
                "transaction_cost",
                "regime",
            ]
        )
    )

    for (
        buffer_size,
        transaction_cost,
        regime,
    ), group in regime_grouped:

        metrics = calculate_metrics(
            group["return"]
        )

        regime_results.append(
            {
                "buffer_size":
                    buffer_size,

                "transaction_cost":
                    transaction_cost,

                "regime":
                    regime,

                "days":
                    len(group),

                **metrics,

                "average_turnover":
                    group[
                        "turnover"
                    ].mean(),
            }
        )

    regime_results_df = pd.DataFrame(
        regime_results
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    daily_df.to_csv(
        DAILY_OUTPUT_PATH,
        index=False
    )

    regime_results_df.to_csv(
        REGIME_OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BUFFER / TRANSACTION COST COMPARISON"
    )
    print("=" * 70)

    display_columns = [
        "buffer_size",
        "transaction_cost",
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "winning_days",
        "average_turnover",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # TURNOVER REDUCTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "TURNOVER REDUCTION ANALYSIS"
    )
    print("=" * 70)

    baseline = results_df[
        (
            results_df[
                "buffer_size"
            ] == 0
        )
        &
        (
            results_df[
                "transaction_cost"
            ] == 0.001
        )
    ]

    if not baseline.empty:

        baseline_turnover = (
            baseline[
                "average_turnover"
            ].iloc[0]
        )

        print(
            f"\nBaseline turnover "
            f"(buffer=0): "
            f"{baseline_turnover:.4f}"
        )

        for buffer_size in BUFFER_SIZES:

            row = results_df[
                (
                    results_df[
                        "buffer_size"
                    ] == buffer_size
                )
                &
                (
                    results_df[
                        "transaction_cost"
                    ] == 0.001
                )
            ]

            if row.empty:

                continue

            turnover = row[
                "average_turnover"
            ].iloc[0]

            if baseline_turnover != 0:

                reduction = (
                    1
                    -
                    turnover
                    /
                    baseline_turnover
                )

            else:

                reduction = np.nan

            print(
                f"Buffer {buffer_size:>2}: "
                f"turnover={turnover:.4f} | "
                f"reduction="
                f"{reduction:.2%}"
            )

    # ========================================================
    # BUFFER EFFECT CHECK
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BUFFER EFFECT VALIDATION"
    )
    print("=" * 70)

    turnover_check = (
        results_df[
            results_df[
                "transaction_cost"
            ] == 0.001
        ]
        [
            [
                "buffer_size",
                "average_turnover",
                "total_return",
                "sharpe",
            ]
        ]
        .sort_values(
            "buffer_size"
        )
    )

    print(
        turnover_check.to_string(
            index=False
        )
    )

    unique_turnovers = (
        turnover_check[
            "average_turnover"
        ]
        .round(10)
        .nunique()
    )

    if unique_turnovers == 1:

        print(
            "\nWARNING:"
        )

        print(
            "Buffer sizes still produce "
            "identical turnover."
        )

        print(
            "This indicates that the ranking "
            "buffer may not be useful for "
            "this particular dataset."
        )

    else:

        print(
            "\nSUCCESS:"
        )

        print(
            "Buffer size is affecting "
            "portfolio turnover."
        )

    # ========================================================
    # FILE LOCATIONS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nOverall results:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nDaily results:\n"
        f"{DAILY_OUTPUT_PATH}"
    )

    print(
        f"\nRegime results:\n"
        f"{REGIME_OUTPUT_PATH}"
    )

    print("\n")
    print("=" * 70)
    print(
        "TURNOVER CONTROL BACKTEST COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()