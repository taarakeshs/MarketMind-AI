import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DAILY_RESULTS_PATH = os.path.join(
    "data",
    "processed",
    "turnover_control_daily.csv"
)

OVERALL_RESULTS_PATH = os.path.join(
    "data",
    "processed",
    "turnover_control_results.csv"
)

OUTPUT_PATH = os.path.join(
    "data",
    "processed",
    "turnover_robustness_validation.csv"
)

TRANSACTION_COST = 0.001

BUFFER_SIZES = [0, 5, 10, 15, 20]


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

    if len(returns) > 1:

        annualized_return = (
            cumulative.iloc[-1]
            ** (252 / len(returns))
        ) - 1

    else:

        annualized_return = np.nan

    volatility = (
        returns.std() * np.sqrt(252)
    )

    if (
        returns.std() > 0
        and not np.isnan(returns.std())
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
        cumulative / running_max - 1
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("TURNOVER CONTROL ROBUSTNESS VALIDATION")
print("=" * 70)

print("\nLoading daily results...")

if not os.path.exists(DAILY_RESULTS_PATH):

    print(
        f"\nERROR: File not found:\n"
        f"{DAILY_RESULTS_PATH}"
    )

    print(
        "\nRun turnover_control_backtest.py first."
    )

    raise SystemExit

daily_df = pd.read_csv(
    DAILY_RESULTS_PATH,
    parse_dates=["date"]
)

print(
    f"Rows loaded: {len(daily_df):,}"
)

print(
    f"Dates: {daily_df['date'].min().date()} "
    f"-> {daily_df['date'].max().date()}"
)


# ============================================================
# FILTER PRACTICAL TRANSACTION COST
# ============================================================

df = daily_df[
    daily_df["transaction_cost"]
    == TRANSACTION_COST
].copy()

print(
    f"\nTransaction cost selected: "
    f"{TRANSACTION_COST:.2%}"
)

print(
    f"Rows after filtering: "
    f"{len(df):,}"
)


# ============================================================
# OVERALL BUFFER COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL BUFFER COMPARISON")
print("=" * 70)

overall_results = []

for buffer_size in BUFFER_SIZES:

    subset = df[
        df["buffer_size"] == buffer_size
    ].copy()

    if subset.empty:
        continue

    metrics = calculate_metrics(
        subset["return"]
    )

    average_turnover = (
        subset["turnover"].mean()
    )

    overall_results.append(
        {
            "buffer_size": buffer_size,
            "total_return": metrics[
                "total_return"
            ],
            "annualized_return": metrics[
                "annualized_return"
            ],
            "volatility": metrics[
                "volatility"
            ],
            "sharpe": metrics[
                "sharpe"
            ],
            "max_drawdown": metrics[
                "max_drawdown"
            ],
            "winning_days": metrics[
                "winning_days"
            ],
            "average_turnover":
                average_turnover,
        }
    )


overall_df = pd.DataFrame(
    overall_results
)

print()

print(
    overall_df.to_string(
        index=False,
        formatters={
            "total_return":
                "{:.2%}".format,

            "annualized_return":
                "{:.2%}".format,

            "volatility":
                "{:.2%}".format,

            "sharpe":
                "{:.3f}".format,

            "max_drawdown":
                "{:.2%}".format,

            "winning_days":
                "{:.2%}".format,

            "average_turnover":
                "{:.4f}".format,
        }
    )
)


# ============================================================
# BASELINE
# ============================================================

baseline_row = overall_df[
    overall_df["buffer_size"] == 0
]

if baseline_row.empty:

    print(
        "\nERROR: Buffer 0 baseline not found."
    )

    raise SystemExit

baseline = baseline_row.iloc[0]

baseline_return = baseline[
    "total_return"
]

baseline_sharpe = baseline[
    "sharpe"
]

baseline_turnover = baseline[
    "average_turnover"
]

baseline_drawdown = baseline[
    "max_drawdown"
]


# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("IMPROVEMENT VS BASELINE")
print("=" * 70)

comparison_results = []

for _, row in overall_df.iterrows():

    buffer_size = int(
        row["buffer_size"]
    )

    turnover_reduction = (
        1
        -
        row["average_turnover"]
        /
        baseline_turnover
    )

    return_difference = (
        row["total_return"]
        -
        baseline_return
    )

    sharpe_difference = (
        row["sharpe"]
        -
        baseline_sharpe
    )

    drawdown_improvement = (
        abs(baseline_drawdown)
        -
        abs(row["max_drawdown"])
    )

    comparison_results.append(
        {
            "buffer_size":
                buffer_size,

            "turnover_reduction":
                turnover_reduction,

            "return_difference":
                return_difference,

            "sharpe_difference":
                sharpe_difference,

            "drawdown_improvement":
                drawdown_improvement,
        }
    )


comparison_df = pd.DataFrame(
    comparison_results
)

print()

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "turnover_reduction":
                "{:.2%}".format,

            "return_difference":
                "{:.2%}".format,

            "sharpe_difference":
                "{:.3f}".format,

            "drawdown_improvement":
                "{:.2%}".format,
        }
    )
)


# ============================================================
# TIME PERIOD ROBUSTNESS
# ============================================================

print("\n")
print("=" * 70)
print("TIME PERIOD ROBUSTNESS")
print("=" * 70)

dates = sorted(
    df["date"].unique()
)

if len(dates) >= 3:

    split_1 = dates[
        len(dates) // 3
    ]

    split_2 = dates[
        2 * len(dates) // 3
    ]

    periods = {
        "EARLY":
            df[
                df["date"] < split_1
            ],

        "MIDDLE":
            df[
                (
                    df["date"] >= split_1
                )
                &
                (
                    df["date"] < split_2
                )
            ],

        "LATE":
            df[
                df["date"] >= split_2
            ],
    }

else:

    periods = {
        "FULL":
            df
    }


period_results = []

for period_name, period_df in periods.items():

    for buffer_size in BUFFER_SIZES:

        subset = period_df[
            period_df["buffer_size"]
            == buffer_size
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(
            subset["return"]
        )

        period_results.append(
            {
                "period":
                    period_name,

                "buffer_size":
                    buffer_size,

                "return":
                    metrics[
                        "total_return"
                    ],

                "sharpe":
                    metrics[
                        "sharpe"
                    ],

                "max_drawdown":
                    metrics[
                        "max_drawdown"
                    ],

                "turnover":
                    subset[
                        "turnover"
                    ].mean(),
            }
        )


period_df = pd.DataFrame(
    period_results
)

print()

print(
    period_df.to_string(
        index=False,
        formatters={
            "return":
                "{:.2%}".format,

            "sharpe":
                "{:.3f}".format,

            "max_drawdown":
                "{:.2%}".format,

            "turnover":
                "{:.4f}".format,
        }
    )
)


# ============================================================
# REGIME ROBUSTNESS
# ============================================================

print("\n")
print("=" * 70)
print("REGIME ROBUSTNESS")
print("=" * 70)

regime_results = []

for regime in sorted(
    df["regime"].dropna().unique()
):

    regime_data = df[
        df["regime"] == regime
    ]

    for buffer_size in BUFFER_SIZES:

        subset = regime_data[
            regime_data[
                "buffer_size"
            ] == buffer_size
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(
            subset["return"]
        )

        regime_results.append(
            {
                "regime":
                    regime,

                "buffer_size":
                    buffer_size,

                "days":
                    len(subset),

                "return":
                    metrics[
                        "total_return"
                    ],

                "sharpe":
                    metrics[
                        "sharpe"
                    ],

                "max_drawdown":
                    metrics[
                        "max_drawdown"
                    ],

                "turnover":
                    subset[
                        "turnover"
                    ].mean(),
            }
        )


regime_df = pd.DataFrame(
    regime_results
)

print()

if not regime_df.empty:

    print(
        regime_df.to_string(
            index=False,
            formatters={
                "return":
                    "{:.2%}".format,

                "sharpe":
                    "{:.3f}".format,

                "max_drawdown":
                    "{:.2%}".format,

                "turnover":
                    "{:.4f}".format,
            }
        )
    )

else:

    print(
        "No regime data available."
    )


# ============================================================
# CONSISTENCY SCORE
# ============================================================

print("\n")
print("=" * 70)
print("BUFFER CONSISTENCY SCORE")
print("=" * 70)

consistency_results = []

for buffer_size in BUFFER_SIZES:

    subset = overall_df[
        overall_df["buffer_size"]
        == buffer_size
    ]

    if subset.empty:
        continue

    score = 0

    # Positive Sharpe
    if subset["sharpe"].iloc[0] > 1:
        score += 1

    # Positive return
    if subset["total_return"].iloc[0] > 0:
        score += 1

    # Lower turnover than baseline
    if (
        subset[
            "average_turnover"
        ].iloc[0]
        <
        baseline_turnover
    ):
        score += 1

    # Better drawdown than baseline
    if (
        abs(
            subset[
                "max_drawdown"
            ].iloc[0]
        )
        <
        abs(baseline_drawdown)
    ):
        score += 1

    # Better Sharpe than baseline
    if (
        subset[
            "sharpe"
        ].iloc[0]
        >
        baseline_sharpe
    ):
        score += 1

    consistency_results.append(
        {
            "buffer_size":
                buffer_size,

            "consistency_score":
                score,

            "maximum_score":
                5,
        }
    )


consistency_df = pd.DataFrame(
    consistency_results
)

print()

print(
    consistency_df.to_string(
        index=False
    )
)


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n")
print("=" * 70)
print("ROBUSTNESS CONCLUSION")
print("=" * 70)

best_sharpe_row = overall_df.loc[
    overall_df["sharpe"].idxmax()
]

best_return_row = overall_df.loc[
    overall_df["total_return"].idxmax()
]

lowest_turnover_row = overall_df.loc[
    overall_df["average_turnover"].idxmin()
]

print(
    f"\nHighest Sharpe buffer: "
    f"{int(best_sharpe_row['buffer_size'])}"
)

print(
    f"Sharpe: "
    f"{best_sharpe_row['sharpe']:.3f}"
)

print(
    f"\nHighest return buffer: "
    f"{int(best_return_row['buffer_size'])}"
)

print(
    f"Return: "
    f"{best_return_row['total_return']:.2%}"
)

print(
    f"\nLowest turnover buffer: "
    f"{int(lowest_turnover_row['buffer_size'])}"
)

print(
    f"Turnover: "
    f"{lowest_turnover_row['average_turnover']:.4f}"
)


# ============================================================
# SAVE
# ============================================================

output_df = overall_df.merge(
    comparison_df,
    on="buffer_size",
    how="left"
)

output_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("=" * 70)
print("OUTPUT SAVED")
print("=" * 70)

print(
    f"\n{OUTPUT_PATH}"
)

print("\n")
print("=" * 70)
print("ROBUSTNESS VALIDATION COMPLETE")
print("=" * 70)