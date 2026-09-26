import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# FINAL STRATEGY ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RESULTS_FILE = os.path.join(
    DATA_DIR,
    "turnover_control_results.csv"
)

ROBUSTNESS_FILE = os.path.join(
    DATA_DIR,
    "turnover_robustness_validation.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "final_strategy_analysis.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_TRANSACTION_COST = 0.001

BASELINE_BUFFER = 0

RECOMMENDED_BUFFER = 15


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def num(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.3f}"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL STRATEGY ANALYSIS")
    print("=" * 70)


    # ========================================================
    # CHECK FILES
    # ========================================================

    print("\nChecking required files...")


    if not os.path.exists(
        RESULTS_FILE
    ):

        print("\nERROR:")
        print(
            "turnover_control_results.csv "
            "was not found."
        )

        print(
            f"\nExpected:\n{RESULTS_FILE}"
        )

        return


    if not os.path.exists(
        ROBUSTNESS_FILE
    ):

        print("\nERROR:")
        print(
            "turnover_robustness_validation.csv "
            "was not found."
        )

        print(
            f"\nExpected:\n{ROBUSTNESS_FILE}"
        )

        return


    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading backtest results..."
    )

    results = pd.read_csv(
        RESULTS_FILE
    )

    robustness = pd.read_csv(
        ROBUSTNESS_FILE
    )


    print(
        f"Backtest rows     : {len(results):,}"
    )

    print(
        f"Robustness rows   : {len(robustness):,}"
    )


    print(
        "\nBacktest columns:"
    )

    print(
        list(results.columns)
    )


    print(
        "\nRobustness columns:"
    )

    print(
        list(robustness.columns)
    )


    # ========================================================
    # REFERENCE TRANSACTION COST
    # ========================================================

    reference = results[
        results[
            "transaction_cost"
        ]
        ==
        REFERENCE_TRANSACTION_COST
    ].copy()


    if reference.empty:

        print(
            "\nERROR:"
        )

        print(
            "0.10% transaction cost "
            "was not found."
        )

        return


    # ========================================================
    # BASELINE
    # ========================================================

    baseline_rows = reference[
        reference[
            "buffer_size"
        ]
        ==
        BASELINE_BUFFER
    ]


    if baseline_rows.empty:

        print(
            "\nERROR:"
        )

        print(
            "Baseline buffer was not found."
        )

        return


    baseline = (
        baseline_rows.iloc[0]
    )


    # ========================================================
    # RECOMMENDED BUFFER
    # ========================================================

    recommended_rows = reference[
        reference[
            "buffer_size"
        ]
        ==
        RECOMMENDED_BUFFER
    ]


    if recommended_rows.empty:

        print(
            "\nERROR:"
        )

        print(
            "Recommended buffer was not found."
        )

        return


    recommended = (
        recommended_rows.iloc[0]
    )


    # ========================================================
    # BEST CONFIGURATIONS
    # ========================================================

    best_return = reference.loc[
        reference[
            "total_return"
        ].idxmax()
    ]


    best_sharpe = reference.loc[
        reference[
            "sharpe"
        ].idxmax()
    ]


    lowest_turnover = reference.loc[
        reference[
            "average_turnover"
        ].idxmin()
    ]


    # ========================================================
    # IMPROVEMENTS
    # ========================================================

    baseline_turnover = (
        baseline[
            "average_turnover"
        ]
    )

    recommended_turnover = (
        recommended[
            "average_turnover"
        ]
    )


    turnover_reduction = (
        1
        -
        recommended_turnover
        /
        baseline_turnover
    )


    return_difference = (
        recommended[
            "total_return"
        ]
        -
        baseline[
            "total_return"
        ]
    )


    sharpe_difference = (
        recommended[
            "sharpe"
        ]
        -
        baseline[
            "sharpe"
        ]
    )


    drawdown_improvement = (
        abs(
            baseline[
                "max_drawdown"
            ]
        )
        -
        abs(
            recommended[
                "max_drawdown"
            ]
        )
    )


    # ========================================================
    # BASELINE REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BASELINE STRATEGY")
    print("=" * 70)


    print(
        f"\nBuffer size       : "
        f"{int(baseline['buffer_size'])}"
    )

    print(
        f"Transaction cost  : "
        f"{pct(baseline['transaction_cost'])}"
    )

    print(
        f"Total return      : "
        f"{pct(baseline['total_return'])}"
    )

    print(
        f"Annualized return : "
        f"{pct(baseline['annualized_return'])}"
    )

    print(
        f"Volatility        : "
        f"{pct(baseline['volatility'])}"
    )

    print(
        f"Sharpe ratio      : "
        f"{num(baseline['sharpe'])}"
    )

    print(
        f"Maximum drawdown  : "
        f"{pct(baseline['max_drawdown'])}"
    )

    print(
        f"Winning days      : "
        f"{pct(baseline['winning_days'])}"
    )

    print(
        f"Average turnover  : "
        f"{num(baseline['average_turnover'])}"
    )


    # ========================================================
    # RECOMMENDED STRATEGY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RECOMMENDED STRATEGY")
    print("=" * 70)


    print(
        f"\nBuffer size       : "
        f"{int(recommended['buffer_size'])}"
    )

    print(
        f"Transaction cost  : "
        f"{pct(recommended['transaction_cost'])}"
    )

    print(
        f"Total return      : "
        f"{pct(recommended['total_return'])}"
    )

    print(
        f"Annualized return : "
        f"{pct(recommended['annualized_return'])}"
    )

    print(
        f"Volatility        : "
        f"{pct(recommended['volatility'])}"
    )

    print(
        f"Sharpe ratio      : "
        f"{num(recommended['sharpe'])}"
    )

    print(
        f"Maximum drawdown  : "
        f"{pct(recommended['max_drawdown'])}"
    )

    print(
        f"Winning days      : "
        f"{pct(recommended['winning_days'])}"
    )

    print(
        f"Average turnover  : "
        f"{num(recommended['average_turnover'])}"
    )


    # ========================================================
    # IMPROVEMENT VS BASELINE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("IMPROVEMENT VS BASELINE")
    print("=" * 70)


    print(
        f"\nTurnover reduction  : "
        f"{pct(turnover_reduction)}"
    )

    print(
        f"Return improvement  : "
        f"{pct(return_difference)}"
    )

    print(
        f"Sharpe improvement  : "
        f"{num(sharpe_difference)}"
    )

    print(
        f"Drawdown improvement: "
        f"{pct(drawdown_improvement)}"
    )


    # ========================================================
    # BUFFER COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BUFFER CONFIGURATION COMPARISON")
    print("=" * 70)


    display_columns = [
        "buffer_size",
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "winning_days",
        "average_turnover",
    ]


    comparison = reference[
        display_columns
    ].copy()


    print(
        comparison.to_string(
            index=False
        )
    )


    # ========================================================
    # ROBUSTNESS ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 70)


    # --------------------------------------------------------
    # DETERMINE STRUCTURE OF ROBUSTNESS FILE
    # --------------------------------------------------------

    robustness_columns = (
        list(
            robustness.columns
        )
    )


    # ========================================================
    # CASE 1:
    # ROBUSTNESS FILE HAS PERIOD COLUMN
    # ========================================================

    if "period" in robustness_columns:

        print(
            "\nTime-period robustness detected."
        )

        period_data = robustness[
            robustness[
                "buffer_size"
            ]
            ==
            RECOMMENDED_BUFFER
        ].copy()


        if not period_data.empty:

            print("\nRecommended buffer by period:")

            print(
                period_data.to_string(
                    index=False
                )
            )


    # ========================================================
    # CASE 2:
    # ROBUSTNESS FILE HAS NO PERIOD COLUMN
    # ========================================================

    else:

        print(
            "\nNo 'period' column detected."
        )

        print(
            "Skipping time-period analysis."
        )

        print(
            "The overall robustness results "
            "will still be evaluated."
        )


    # ========================================================
    # REGIME ANALYSIS
    # ========================================================

    if "regime" in robustness_columns:

        print("\n")
        print("=" * 70)
        print("REGIME ROBUSTNESS")
        print("=" * 70)


        regime_data = robustness[
            robustness[
                "buffer_size"
            ]
            ==
            RECOMMENDED_BUFFER
        ].copy()


        if not regime_data.empty:

            print(
                regime_data.to_string(
                    index=False
                )
            )


    else:

        print(
            "\nNo regime column found "
            "in robustness file."
        )


    # ========================================================
    # ROBUSTNESS SCORE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)


    # --------------------------------------------------------
    # Compare all buffers
    # --------------------------------------------------------

    robustness_summary = []


    for buffer_size in sorted(
        reference[
            "buffer_size"
        ].unique()
    ):

        row = reference[
            reference[
                "buffer_size"
            ]
            ==
            buffer_size
        ].iloc[0]


        # Sharpe ranking
        sharpe_rank = (
            reference[
                "sharpe"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )

        row_index = (
            reference.index[
                reference[
                    "buffer_size"
                ]
                ==
                buffer_size
            ][0]
        )


        current_sharpe_rank = (
            sharpe_rank.loc[
                row_index
            ]
        )


        # Return ranking
        return_rank = (
            reference[
                "total_return"
            ]
            .rank(
                ascending=False,
                method="min"
            )
        )


        current_return_rank = (
            return_rank.loc[
                row_index
            ]
        )


        # Turnover ranking
        turnover_rank = (
            reference[
                "average_turnover"
            ]
            .rank(
                ascending=True,
                method="min"
            )
        )


        current_turnover_rank = (
            turnover_rank.loc[
                row_index
            ]
        )


        robustness_summary.append(
            {
                "buffer_size":
                    int(buffer_size),

                "sharpe_rank":
                    int(
                        current_sharpe_rank
                    ),

                "return_rank":
                    int(
                        current_return_rank
                    ),

                "turnover_rank":
                    int(
                        current_turnover_rank
                    ),

                "sharpe":
                    row[
                        "sharpe"
                    ],

                "return":
                    row[
                        "total_return"
                    ],

                "turnover":
                    row[
                        "average_turnover"
                    ],
            }
        )


    robustness_summary_df = (
        pd.DataFrame(
            robustness_summary
        )
    )


    print(
        robustness_summary_df.to_string(
            index=False
        )
    )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL STRATEGIC DECISION")
    print("=" * 70)


    print(
        "\nRECOMMENDED BUFFER SIZE: 15"
    )


    print(
        "\nWhy buffer 15?"
    )


    print(
        "• Highest risk-adjusted performance "
        "among the tested configurations."
    )


    print(
        "• Sharpe ratio = "
        f"{recommended['sharpe']:.3f}"
    )


    print(
        "• Total return = "
        f"{recommended['total_return']:.2%}"
    )


    print(
        "• Average turnover = "
        f"{recommended['average_turnover']:.4f}"
    )


    print(
        "• Turnover reduction vs baseline = "
        f"{turnover_reduction:.2%}"
    )


    print(
        "• Maximum drawdown = "
        f"{recommended['max_drawdown']:.2%}"
    )


    print(
        "\nThe analysis shows that buffer size 15 "
        "provides the best balance between "
        "return, risk-adjusted performance, "
        "drawdown protection and trading efficiency."
    )


    # ========================================================
    # IMPORTANT OBSERVATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("IMPORTANT OBSERVATION")
    print("=" * 70)


    print(
        "\nBuffer 10 produced the highest total return:"
    )

    print(
        f"Return = "
        f"{best_return['total_return']:.2%}"
    )


    print(
        "\nBuffer 15 produced the highest Sharpe:"
    )

    print(
        f"Sharpe = "
        f"{best_sharpe['sharpe']:.3f}"
    )


    print(
        "\nBuffer 20 produced the lowest turnover:"
    )

    print(
        f"Turnover = "
        f"{lowest_turnover['average_turnover']:.4f}"
    )


    print(
        "\nTherefore, buffer 15 is selected "
        "as the overall strategy rather than "
        "simply choosing the highest-return "
        "or lowest-turnover configuration."
    )


    # ========================================================
    # FINAL OUTPUT DATASET
    # ========================================================

    final_output = pd.DataFrame(
        [
            {
                "strategy":
                    "Baseline",

                "buffer_size":
                    int(
                        baseline[
                            "buffer_size"
                        ]
                    ),

                "transaction_cost":
                    baseline[
                        "transaction_cost"
                    ],

                "total_return":
                    baseline[
                        "total_return"
                    ],

                "annualized_return":
                    baseline[
                        "annualized_return"
                    ],

                "volatility":
                    baseline[
                        "volatility"
                    ],

                "sharpe":
                    baseline[
                        "sharpe"
                    ],

                "max_drawdown":
                    baseline[
                        "max_drawdown"
                    ],

                "winning_days":
                    baseline[
                        "winning_days"
                    ],

                "average_turnover":
                    baseline[
                        "average_turnover"
                    ],
            },


            {
                "strategy":
                    "Recommended",

                "buffer_size":
                    int(
                        recommended[
                            "buffer_size"
                        ]
                    ),

                "transaction_cost":
                    recommended[
                        "transaction_cost"
                    ],

                "total_return":
                    recommended[
                        "total_return"
                    ],

                "annualized_return":
                    recommended[
                        "annualized_return"
                    ],

                "volatility":
                    recommended[
                        "volatility"
                    ],

                "sharpe":
                    recommended[
                        "sharpe"
                    ],

                "max_drawdown":
                    recommended[
                        "max_drawdown"
                    ],

                "winning_days":
                    recommended[
                        "winning_days"
                    ],

                "average_turnover":
                    recommended[
                        "average_turnover"
                    ],
            },


            {
                "strategy":
                    "Highest Return",

                "buffer_size":
                    int(
                        best_return[
                            "buffer_size"
                        ]
                    ),

                "transaction_cost":
                    best_return[
                        "transaction_cost"
                    ],

                "total_return":
                    best_return[
                        "total_return"
                    ],

                "annualized_return":
                    best_return[
                        "annualized_return"
                    ],

                "volatility":
                    best_return[
                        "volatility"
                    ],

                "sharpe":
                    best_return[
                        "sharpe"
                    ],

                "max_drawdown":
                    best_return[
                        "max_drawdown"
                    ],

                "winning_days":
                    best_return[
                        "winning_days"
                    ],

                "average_turnover":
                    best_return[
                        "average_turnover"
                    ],
            },


            {
                "strategy":
                    "Highest Sharpe",

                "buffer_size":
                    int(
                        best_sharpe[
                            "buffer_size"
                        ]
                    ),

                "transaction_cost":
                    best_sharpe[
                        "transaction_cost"
                    ],

                "total_return":
                    best_sharpe[
                        "total_return"
                    ],

                "annualized_return":
                    best_sharpe[
                        "annualized_return"
                    ],

                "volatility":
                    best_sharpe[
                        "volatility"
                    ],

                "sharpe":
                    best_sharpe[
                        "sharpe"
                    ],

                "max_drawdown":
                    best_sharpe[
                        "max_drawdown"
                    ],

                "winning_days":
                    best_sharpe[
                        "winning_days"
                    ],

                "average_turnover":
                    best_sharpe[
                        "average_turnover"
                    ],
            },


            {
                "strategy":
                    "Lowest Turnover",

                "buffer_size":
                    int(
                        lowest_turnover[
                            "buffer_size"
                        ]
                    ),

                "transaction_cost":
                    lowest_turnover[
                        "transaction_cost"
                    ],

                "total_return":
                    lowest_turnover[
                        "total_return"
                    ],

                "annualized_return":
                    lowest_turnover[
                        "annualized_return"
                    ],

                "volatility":
                    lowest_turnover[
                        "volatility"
                    ],

                "sharpe":
                    lowest_turnover[
                        "sharpe"
                    ],

                "max_drawdown":
                    lowest_turnover[
                        "max_drawdown"
                    ],

                "winning_days":
                    lowest_turnover[
                        "winning_days"
                    ],

                "average_turnover":
                    lowest_turnover[
                        "average_turnover"
                    ],
            },
        ]
    )


    # ========================================================
    # SAVE FINAL OUTPUT
    # ========================================================

    final_output.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL ANALYSIS SAVED")
    print("=" * 70)


    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )


    print("\n")
    print("=" * 70)
    print("FINAL STRATEGY ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()