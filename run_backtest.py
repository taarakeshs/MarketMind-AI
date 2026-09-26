from pathlib import Path
import pandas as pd
import numpy as np

from src.backtest_engine import (
    backtest_all_stocks,
    simulate_equal_weight_portfolio,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_FILE = (
    REPORT_DIR / "predictions.parquet"
)


# ============================================================
# FIND PREDICTION COLUMNS
# ============================================================

def find_prediction_column(df):

    candidates = [
        "Pred_hist_gradient_boosting",
        "Pred_hist_gradient_boosting_regressor",
        "predicted_return",
        "Predicted_Return",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    # Automatic search
    prediction_columns = [
        c for c in df.columns
        if str(c).lower().startswith("pred_")
        and "dir" not in str(c).lower()
    ]

    if prediction_columns:
        return prediction_columns[0]

    raise RuntimeError(
        "Could not find regression prediction column."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STOCK MARKET INTELLIGENCE")
    print("BACKTEST + PORTFOLIO ANALYSIS")
    print("=" * 70)

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTION_FILE}\n\n"
            "Run classical ML training first."
        )

    print()
    print("Loading predictions...")

    df = pd.read_parquet(
        PREDICTION_FILE
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    rename = {}

    for column in df.columns:

        lower = str(column).lower()

        if lower == "ticker":
            rename[column] = "ticker"

        elif lower == "date":
            rename[column] = "date"

        elif lower == "close":
            rename[column] = "close"

    df = df.rename(
        columns=rename
    )

    prediction_column = (
        find_prediction_column(df)
    )

    print()
    print(
        "Prediction column:",
        prediction_column
    )

    df = df.rename(
        columns={
            prediction_column:
                "predicted_return"
        }
    )

    required = [
        "ticker",
        "date",
        "close",
        "predicted_return",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric cleaning
    # --------------------------------------------------------

    for column in [
        "close",
        "predicted_return",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required
    )

    print()
    print(
        "Rows after cleaning:",
        len(df)
    )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    stock_results, daily_results = (
        backtest_all_stocks(df)
    )

    # --------------------------------------------------------
    # SAVE STOCK RESULTS
    # --------------------------------------------------------

    stock_file = (
        REPORT_DIR
        / "backtest_stock_results.csv"
    )

    stock_results.to_csv(
        stock_file,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE DAILY RESULTS
    # --------------------------------------------------------

    daily_file = (
        REPORT_DIR
        / "backtest_daily_results.parquet"
    )

    daily_results.to_parquet(
        daily_file,
        index=False,
    )

    # --------------------------------------------------------
    # PORTFOLIO
    # --------------------------------------------------------

    portfolio_daily, portfolio_metrics = (
        simulate_equal_weight_portfolio(
            daily_results
        )
    )

    portfolio_file = (
        REPORT_DIR
        / "portfolio_daily.parquet"
    )

    portfolio_daily.to_parquet(
        portfolio_file,
        index=False,
    )

    # --------------------------------------------------------
    # PORTFOLIO METRICS
    # --------------------------------------------------------

    portfolio_metrics_df = pd.DataFrame(
        [portfolio_metrics]
    )

    metrics_file = (
        REPORT_DIR
        / "portfolio_metrics.csv"
    )

    portfolio_metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    # --------------------------------------------------------
    # TOP STOCKS
    # --------------------------------------------------------

    top_stocks = (
        stock_results
        .sort_values(
            "Sharpe_Ratio",
            ascending=False,
        )
        .head(20)
    )

    top_file = (
        REPORT_DIR
        / "top_20_stocks.csv"
    )

    top_stocks.to_csv(
        top_file,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print()
    print("PORTFOLIO RESULTS")
    print("-" * 70)

    for key, value in portfolio_metrics.items():

        if isinstance(value, float):

            if "Return" in key:
                print(
                    f"{key:25s}: "
                    f"{value:.2%}"
                )

            elif "Drawdown" in key:
                print(
                    f"{key:25s}: "
                    f"{value:.2%}"
                )

            else:
                print(
                    f"{key:25s}: "
                    f"{value:.4f}"
                )

        else:

            print(
                f"{key:25s}: "
                f"{value}"
            )

    print()
    print("Files generated:")
    print(
        stock_file
    )
    print(
        daily_file
    )
    print(
        portfolio_file
    )
    print(
        metrics_file
    )
    print(
        top_file
    )

    print()
    print("=" * 70)
    print("NEXT: STREAMLIT DASHBOARD")
    print("=" * 70)


if __name__ == "__main__":
    main()