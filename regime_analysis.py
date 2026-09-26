import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


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
    "regime_analysis_results.csv"
)

TOP_N_VALUES = [5, 10, 20, 50]

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MARKET REGIME ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "ticker"]
).reset_index(drop=True)

print(f"Rows   : {len(df):,}")
print(f"Stocks : {df['ticker'].nunique()}")
print(f"Dates  : {df['date'].nunique()}")


# ============================================================
# CLEAN DATA
# ============================================================

excluded_columns = [
    "date",
    "ticker",
    "target",
    "next_close",
    "next_return"
]

feature_columns = [
    col
    for col in df.select_dtypes(include=[np.number]).columns
    if col not in excluded_columns
]

print(f"Features: {len(feature_columns)}")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=feature_columns + ["next_return"]
).reset_index(drop=True)


# ============================================================
# MARKET RETURN
# ============================================================

print("\nCalculating market returns...")

daily_market = (
    df.groupby("date")["next_return"]
    .mean()
    .reset_index()
)

daily_market = daily_market.sort_values(
    "date"
).reset_index(drop=True)

daily_market["market_cumulative"] = (
    1 + daily_market["next_return"]
).cumprod()


# ============================================================
# MARKET VOLATILITY
# ============================================================

daily_market["rolling_volatility"] = (
    daily_market["next_return"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)


# ============================================================
# MARKET TREND
# ============================================================

daily_market["market_ma20"] = (
    daily_market["market_cumulative"]
    .rolling(20)
    .mean()
)

daily_market["market_ma60"] = (
    daily_market["market_cumulative"]
    .rolling(60)
    .mean()
)


# ============================================================
# MARKET REGIME CLASSIFICATION
# ============================================================

def classify_regime(row):

    if pd.isna(row["rolling_volatility"]):
        return "UNKNOWN"

    market_return = row["next_return"]

    ma20 = row["market_ma20"]
    ma60 = row["market_ma60"]

    volatility = row["rolling_volatility"]

    # High volatility threshold
    high_volatility = volatility >= 0.30

    # Trend
    if ma20 > ma60:

        if high_volatility:
            return "BULL_HIGH_VOL"

        return "BULL"

    elif ma20 < ma60:

        if high_volatility:
            return "BEAR_HIGH_VOL"

        return "BEAR"

    else:

        if high_volatility:
            return "SIDEWAYS_HIGH_VOL"

        return "SIDEWAYS"


daily_market["regime"] = daily_market.apply(
    classify_regime,
    axis=1
)


# ============================================================
# REMOVE UNKNOWN
# ============================================================

daily_market = daily_market[
    daily_market["regime"] != "UNKNOWN"
].copy()


print("\nREGIME DISTRIBUTION")
print("-" * 60)

regime_counts = (
    daily_market["regime"]
    .value_counts()
)

print(regime_counts)


# ============================================================
# MERGE REGIME INTO STOCK DATA
# ============================================================

df = df.merge(
    daily_market[
        [
            "date",
            "regime"
        ]
    ],
    on="date",
    how="inner"
)

print(
    f"\nRows after regime merge: {len(df):,}"
)


# ============================================================
# TIME SERIES SPLIT
# ============================================================

dates = sorted(
    df["date"].unique()
)

split_index = int(
    len(dates) * 0.70
)

train_dates = dates[:split_index]
test_dates = dates[split_index:]

train = df[
    df["date"].isin(train_dates)
].copy()

test = df[
    df["date"].isin(test_dates)
].copy()

print("\nTIME SERIES SPLIT")
print("-" * 60)

print(
    f"Train : {train['date'].min().date()} → "
    f"{train['date'].max().date()}"
)

print(
    f"Test  : {test['date'].min().date()} → "
    f"{test['date'].max().date()}"
)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")


# ============================================================
# TRAIN RANKING MODEL
# ============================================================

print("\nTraining ranking model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    train[feature_columns],
    train["next_return"]
)

print("Model trained successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

test["predicted_return"] = model.predict(
    test[feature_columns]
)


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(
    returns
):

    returns = pd.Series(
        returns
    ).dropna()

    if len(returns) == 0:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "winning_days": np.nan,
            "average_return": np.nan
        }

    cumulative = (
        1 + returns
    ).cumprod()

    total_return = (
        cumulative.iloc[-1] - 1
    )

    annualized_return = (
        (1 + total_return)
        ** (252 / len(returns))
        - 1
    )

    volatility = (
        returns.std()
        * np.sqrt(252)
    )

    if returns.std() > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0.0

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    winning_days = (
        returns > 0
    ).mean()

    average_return = returns.mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_days": winning_days,
        "average_return": average_return
    }


# ============================================================
# RUN REGIME STRATEGIES
# ============================================================

results = []

print("\n")
print("=" * 70)
print("RUNNING REGIME STRATEGY ANALYSIS")
print("=" * 70)


for top_n in TOP_N_VALUES:

    print("\n")
    print("-" * 70)
    print(f"ML TOP {top_n}")
    print("-" * 70)

    daily_results = []

    for date, day in test.groupby("date"):

        day = day.sort_values(
            "predicted_return",
            ascending=False
        )

        if len(day) < top_n:
            continue

        selected = day.head(
            top_n
        )

        portfolio_return = selected[
            "next_return"
        ].mean()

        regime = day[
            "regime"
        ].iloc[0]

        daily_results.append({
            "date": date,
            "regime": regime,
            "return": portfolio_return
        })

    daily_results = pd.DataFrame(
        daily_results
    )

    # --------------------------------------------------------
    # EACH REGIME
    # --------------------------------------------------------

    for regime, regime_data in daily_results.groupby(
        "regime"
    ):

        metrics = calculate_metrics(
            regime_data["return"]
        )

        results.append({
            "strategy": "ML",
            "top_n": top_n,
            "regime": regime,
            "days": len(regime_data),
            **metrics
        })

        print(
            f"{regime:<20} "
            f"Days: {len(regime_data):>3} | "
            f"Return: {metrics['total_return'] * 100:>7.2f}% | "
            f"Sharpe: {metrics['sharpe']:>6.3f} | "
            f"Max DD: {metrics['max_drawdown'] * 100:>7.2f}%"
        )


# ============================================================
# EQUAL-WEIGHT BENCHMARK
# ============================================================

print("\n")
print("=" * 70)
print("EQUAL-WEIGHT BENCHMARK BY REGIME")
print("=" * 70)


benchmark_daily = []

for date, day in test.groupby("date"):

    market_return = day[
        "next_return"
    ].mean()

    regime = day[
        "regime"
    ].iloc[0]

    benchmark_daily.append({
        "date": date,
        "regime": regime,
        "return": market_return
    })


benchmark_daily = pd.DataFrame(
    benchmark_daily
)


for regime, regime_data in benchmark_daily.groupby(
    "regime"
):

    metrics = calculate_metrics(
        regime_data["return"]
    )

    results.append({
        "strategy": "Equal Weight",
        "top_n": 0,
        "regime": regime,
        "days": len(regime_data),
        **metrics
    })

    print(
        f"{regime:<20} "
        f"Days: {len(regime_data):>3} | "
        f"Return: {metrics['total_return'] * 100:>7.2f}% | "
        f"Sharpe: {metrics['sharpe']:>6.3f} | "
        f"Max DD: {metrics['max_drawdown'] * 100:>7.2f}%"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 70)
print("FINAL REGIME COMPARISON")
print("=" * 70)

display_columns = [
    "strategy",
    "top_n",
    "regime",
    "days",
    "total_return",
    "annualized_return",
    "volatility",
    "sharpe",
    "max_drawdown",
    "winning_days"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST REGIMES
# ============================================================

print("\n")
print("=" * 70)
print("BEST ML CONFIGURATION BY REGIME")
print("=" * 70)


ml_results = results_df[
    results_df["strategy"] == "ML"
].copy()


for regime in ml_results[
    "regime"
].unique():

    subset = ml_results[
        ml_results["regime"] == regime
    ]

    if len(subset) == 0:
        continue

    best = subset.loc[
        subset["sharpe"].idxmax()
    ]

    print(
        f"{regime:<20} "
        f"Best Top-{int(best['top_n'])} | "
        f"Return: {best['total_return'] * 100:.2f}% | "
        f"Sharpe: {best['sharpe']:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("MARKET REGIME ANALYSIS COMPLETE")
print("=" * 70)