import os
import joblib
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
    "transaction_cost_results.csv"
)

RANKING_MODEL_PATH = os.path.join(
    "models",
    "ranking_random_forest.joblib"
)

TOP_N_VALUES = [5, 10, 20, 50]

TRANSACTION_COSTS = [
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030
]

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRANSACTION COST BACKTEST")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

print(f"Rows   : {len(df):,}")
print(f"Stocks : {df['ticker'].nunique()}")
print(f"Dates  : {df['date'].nunique()}")


# ============================================================
# IDENTIFY FEATURES
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


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=feature_columns + ["next_return"]
).reset_index(drop=True)


# ============================================================
# TIME SERIES TEST SPLIT
# ============================================================

dates = sorted(df["date"].unique())

split_index = int(len(dates) * 0.70)

train_dates = dates[:split_index]
test_dates = dates[split_index:]

train = df[df["date"].isin(train_dates)].copy()
test = df[df["date"].isin(test_dates)].copy()

print("\nTIME SERIES SPLIT")
print("-" * 50)

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
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

test["predicted_return"] = model.predict(
    test[feature_columns]
)


# ============================================================
# PORTFOLIO BACKTEST FUNCTION
# ============================================================

def calculate_strategy(
    data,
    top_n,
    transaction_cost
):

    daily_returns = []

    previous_holdings = set()

    for date, day in data.groupby("date"):

        day = day.sort_values(
            "predicted_return",
            ascending=False
        )

        if len(day) < top_n:
            continue

        selected = day.head(top_n)

        current_holdings = set(
            selected["ticker"]
        )

        gross_return = selected[
            "next_return"
        ].mean()

        # ----------------------------------------------------
        # Turnover
        # ----------------------------------------------------

        if len(previous_holdings) == 0:

            turnover = 1.0

        else:

            added = current_holdings - previous_holdings
            removed = previous_holdings - current_holdings

            turnover = (
                len(added) + len(removed)
            ) / (2 * top_n)

        cost = turnover * transaction_cost

        net_return = gross_return - cost

        daily_returns.append({
            "date": date,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": cost,
            "net_return": net_return
        })

        previous_holdings = current_holdings

    result = pd.DataFrame(daily_returns)

    if len(result) == 0:
        return None

    returns = result["net_return"]

    total_return = (
        (1 + returns).prod() - 1
    )

    annualized_return = (
        (1 + total_return)
        ** (252 / len(result))
        - 1
    )

    volatility = (
        returns.std() * np.sqrt(252)
    )

    if volatility > 0:

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    cumulative = (
        1 + returns
    ).cumprod()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
    ) - 1

    max_drawdown = drawdown.min()

    winning_days = (
        returns > 0
    ).mean()

    average_trade = returns.mean()

    average_turnover = result[
        "turnover"
    ].mean()

    return {
        "top_n": top_n,
        "transaction_cost": transaction_cost,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_days": winning_days,
        "average_return": average_trade,
        "average_turnover": average_turnover
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

print("\n")
print("=" * 70)
print("RUNNING TRANSACTION COST EXPERIMENTS")
print("=" * 70)

for top_n in TOP_N_VALUES:

    print(f"\n{'-' * 60}")
    print(f"TOP {top_n}")
    print(f"{'-' * 60}")

    for cost in TRANSACTION_COSTS:

        result = calculate_strategy(
            test,
            top_n,
            cost
        )

        if result is None:
            continue

        results.append(result)

        print(
            f"Cost {cost * 100:.2f}% | "
            f"Return {result['total_return'] * 100:.2f}% | "
            f"Sharpe {result['sharpe']:.3f} | "
            f"Max DD {result['max_drawdown'] * 100:.2f}%"
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("TRANSACTION COST COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
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
    os.path.abspath(OUTPUT_PATH)
)

print("\n")
print("=" * 70)
print("TRANSACTION COST BACKTEST COMPLETE")
print("=" * 70)