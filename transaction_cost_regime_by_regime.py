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

MODEL_PATH = os.path.join(
    "models",
    "transaction_cost_regime_model.joblib"
)

OUTPUT_PATH = os.path.join(
    "data",
    "processed",
    "transaction_cost_regime_by_regime_detailed.csv"
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.70

TRANSACTION_COSTS = [
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
]

TOP_N_BULL = 20
TOP_N_BEAR = 50
TOP_N_HIGH_VOL = 50
TOP_N_SIDEWAYS = 20


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

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    annualized_return = (
        equity.iloc[-1]
        ** (252 / len(returns))
        - 1
    )

    volatility = (
        returns.std(ddof=1)
        * np.sqrt(252)
    )

    if returns.std(ddof=1) != 0:

        sharpe = (
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(252)
        )

    else:

        sharpe = 0.0

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    winning_days = (
        (returns > 0).mean()
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
# REGIME MAPPING
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
# MARKET REGIME CALCULATION
# ============================================================

def calculate_market_regime(df):

    print("\nCalculating market returns...")

    market = (
        df.groupby("date")["return_1d"]
        .mean()
        .sort_index()
    )

    market_return = market.copy()

    market_volatility = (
        market
        .rolling(20)
        .std()
    )

    market_ma = (
        market
        .rolling(20)
        .mean()
    )

    regime_df = pd.DataFrame({
        "market_return": market_return,
        "market_volatility": market_volatility,
        "market_ma": market_ma,
    })

    volatility_threshold = (
        market_volatility
        .rolling(60)
        .median()
    )

    regimes = []

    for date, row in regime_df.iterrows():

        ret = row["market_return"]
        ma = row["market_ma"]
        vol = row["market_volatility"]
        vol_threshold = volatility_threshold.loc[date]

        if pd.isna(ma) or pd.isna(vol):

            regime = "SIDEWAYS"

        elif (
            ret > ma
            and ret > 0
        ):

            regime = "BULL"

        elif (
            ret < ma
            and ret < 0
        ):

            if (
                not pd.isna(vol_threshold)
                and vol > vol_threshold
            ):
                regime = "BEAR_HIGH_VOL"

            else:
                regime = "BEAR"

        else:

            regime = "SIDEWAYS"

        regimes.append(regime)

    regime_df["regime"] = regimes

    print("\nREGIME DISTRIBUTION")
    print("-" * 60)

    print(
        regime_df["regime"]
        .value_counts()
    )

    return regime_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRANSACTION COST REGIME-BY-REGIME ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        DATA_PATH,
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

    regime_df = calculate_market_regime(df)

    df = df.merge(
        regime_df[
            [
                "market_return",
                "market_volatility",
                "regime",
            ]
        ],
        left_on="date",
        right_index=True,
        how="left"
    )

    df = df.dropna(
        subset=[
            "regime",
            "next_return"
        ]
    )

    print(
        f"\nRows after regime merge: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    excluded = {
        "date",
        "ticker",

        # FUTURE INFORMATION
        "next_close",
        "next_return",
        "target",

        # REGIME INFORMATION
        "regime",
        "market_return",
        "market_volatility",
        "market_ma",
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

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    test[
        "predicted_return"
    ] = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # RUN EXPERIMENT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "RUNNING TRANSACTION COST "
        "ANALYSIS BY REGIME"
    )
    print("=" * 70)

    dates_test = sorted(
        test["date"].unique()
    )

    all_results = []

    for transaction_cost in TRANSACTION_COSTS:

        print("\n")
        print("-" * 70)

        print(
            f"TRANSACTION COST: "
            f"{transaction_cost:.4f}"
        )

        previous_holdings = set()

        daily_records = []

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

            current_holdings = set(
                selected["ticker"]
            )

            # ------------------------------------------------
            # TURNOVER
            # ------------------------------------------------

            if len(previous_holdings) == 0:

                turnover = 1.0

            else:

                turnover = (
                    len(
                        current_holdings
                        ^ previous_holdings
                    )
                    /
                    max(
                        len(current_holdings),
                        1
                    )
                )

            # ------------------------------------------------
            # RAW RETURN
            # ------------------------------------------------

            raw_return = selected[
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
                raw_return
                - cost
            )

            daily_records.append(
                {
                    "date": current_date,
                    "regime": regime,
                    "top_n": top_n,
                    "raw_return": raw_return,
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "net_return": net_return,
                }
            )

            previous_holdings = (
                current_holdings
            )

        daily_df = pd.DataFrame(
            daily_records
        )

        # ----------------------------------------------------
        # BY REGIME
        # ----------------------------------------------------

        for regime, group in daily_df.groupby(
            "regime"
        ):

            metrics = calculate_metrics(
                group["net_return"]
            )

            all_results.append(
                {
                    "transaction_cost":
                        transaction_cost,

                    "cost_percent":
                        transaction_cost * 100,

                    "regime":
                        regime,

                    "days":
                        len(group),

                    "top_n":
                        group["top_n"].mode()
                        .iloc[0],

                    "total_return":
                        metrics[
                            "total_return"
                        ],

                    "annualized_return":
                        metrics[
                            "annualized_return"
                        ],

                    "volatility":
                        metrics[
                            "volatility"
                        ],

                    "sharpe":
                        metrics[
                            "sharpe"
                        ],

                    "max_drawdown":
                        metrics[
                            "max_drawdown"
                        ],

                    "winning_days":
                        metrics[
                            "winning_days"
                        ],

                    "average_turnover":
                        group[
                            "turnover"
                        ].mean(),

                    "average_raw_return":
                        group[
                            "raw_return"
                        ].mean(),

                    "average_cost":
                        group[
                            "transaction_cost"
                        ].mean(),
                }
            )

            print(
                f"{regime:15s}"
                f" Days: {len(group):3d}"
                f" | Return: "
                f"{metrics['total_return'] * 100:8.2f}%"
                f" | Sharpe: "
                f"{metrics['sharpe']:7.3f}"
                f" | Max DD: "
                f"{metrics['max_drawdown'] * 100:8.2f}%"
            )

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "TRANSACTION COST × REGIME COMPARISON"
    )
    print("=" * 70)

    display_columns = [
        "transaction_cost",
        "regime",
        "days",
        "top_n",
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

    # --------------------------------------------------------
    # BREAK-EVEN ANALYSIS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "BREAK-EVEN TRANSACTION COST ANALYSIS"
    )
    print("=" * 70)

    for regime in sorted(
        results_df["regime"].unique()
    ):

        regime_data = results_df[
            results_df["regime"] == regime
        ].sort_values(
            "transaction_cost"
        )

        profitable = regime_data[
            regime_data[
                "total_return"
            ] > 0
        ]

        if not profitable.empty:

            highest_profitable = (
                profitable[
                    "transaction_cost"
                ].max()
            )

            print(
                f"{regime:15s} "
                f"profitable through "
                f"{highest_profitable * 100:.2f}% "
                f"transaction cost"
            )

        else:

            print(
                f"{regime:15s} "
                f"not profitable at tested costs"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        OUTPUT_PATH
    )

    print("\n")
    print("=" * 70)
    print(
        "TRANSACTION COST × REGIME "
        "ANALYSIS COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()