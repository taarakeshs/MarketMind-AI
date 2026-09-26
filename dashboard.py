import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stock Market Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        background: #111827;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #263244;
        margin-bottom: 10px;
    }

    .metric-title {
        font-size: 14px;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: white;
    }

    .success-box {
        padding: 18px;
        border-radius: 12px;
        background: #0f2d22;
        border: 1px solid #1f6f50;
        margin-bottom: 20px;
    }

    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background: #302714;
        border: 1px solid #806b2e;
        margin-bottom: 20px;
    }

    h1 {
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def find_file(*names):

    for name in names:

        path = REPORT_DIR / name

        if path.exists():
            return path

    return None


def load_csv(*names):

    path = find_file(*names)

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_parquet(*names):

    path = find_file(*names)

    if path is None:
        return None

    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def money(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"₹{value:,.2f}"


def percentage(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def number(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_all_data():

    metrics = load_csv(
        "strategy_metrics.csv",
        "profit_strategy_metrics.csv",
        "profit_strategy_v2_metrics.csv",
    )

    strategy_comparison = load_csv(
        "strategy_comparison.csv"
    )

    best_strategy = load_csv(
        "best_strategy.csv"
    )

    predictions = load_parquet(
        "predictions.parquet"
    )

    if predictions is None:

        prediction_csv = find_file(
            "predictions.csv"
        )

        if prediction_csv:
            predictions = pd.read_csv(
                prediction_csv
            )

    portfolio_daily = load_parquet(
        "portfolio_daily.parquet"
    )

    if portfolio_daily is None:

        portfolio_daily = load_csv(
            "portfolio_daily.csv"
        )

    daily_strategy = load_csv(
        "profit_strategy_daily_results.csv",
        "profit_strategy_v2_daily.csv",
    )

    trades = load_csv(
        "profit_strategy_v2_trades.csv",
        "profit_strategy_trades.csv",
    )

    signals = load_csv(
        "profit_strategy_v2_signals.csv",
        "profit_strategy_signals.csv",
    )

    return {
        "metrics": metrics,
        "strategy_comparison": strategy_comparison,
        "best_strategy": best_strategy,
        "predictions": predictions,
        "portfolio_daily": portfolio_daily,
        "daily_strategy": daily_strategy,
        "trades": trades,
        "signals": signals,
    }


data = load_all_data()


# ============================================================
# HEADER
# ============================================================

st.title("📈 Stock Market Intelligence System")

st.markdown(
    """
    ### AI-Powered Market Prediction & Profitability Engine

    A multi-model machine learning system that combines:

    **53 engineered features → 5 ML models → ensemble predictions → 
    confidence filtering → strategy optimization → portfolio intelligence**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Dashboard",
        "Market Predictions",
        "Strategy Analysis",
        "Portfolio Performance",
        "Trading Signals",
        "Model Performance",
        "About System",
    ],
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header("Executive Dashboard")

    strategy_df = data["strategy_comparison"]

    if strategy_df is not None and not strategy_df.empty:

        best_return_row = strategy_df.loc[
            strategy_df["Total_Return"].idxmax()
        ]

        best_sharpe_row = strategy_df.loc[
            strategy_df["Sharpe_Ratio"].idxmax()
        ]

        lowest_dd_row = strategy_df.loc[
            strategy_df["Maximum_Drawdown"].idxmax()
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Best Strategy Return",
                percentage(
                    best_return_row["Total_Return"]
                ),
                best_return_row["strategy"],
            )

        with col2:

            st.metric(
                "Best Sharpe Ratio",
                number(
                    best_sharpe_row["Sharpe_Ratio"]
                ),
                best_sharpe_row["strategy"],
            )

        with col3:

            st.metric(
                "Lowest Drawdown",
                percentage(
                    lowest_dd_row["Maximum_Drawdown"]
                ),
                lowest_dd_row["strategy"],
            )

        with col4:

            st.metric(
                "Strategies Tested",
                len(strategy_df),
            )

        st.divider()

        st.subheader("Strategy Comparison")

        display_df = strategy_df.copy()

        if "Total_Return" in display_df.columns:

            display_df["Return"] = (
                display_df["Total_Return"]
                .map(percentage)
            )

        if "Maximum_Drawdown" in display_df.columns:

            display_df["Drawdown"] = (
                display_df["Maximum_Drawdown"]
                .map(percentage)
            )

        if "Sharpe_Ratio" in display_df.columns:

            display_df["Sharpe"] = (
                display_df["Sharpe_Ratio"]
                .round(2)
            )

        columns = [
            c for c in [
                "strategy",
                "Return",
                "Sharpe",
                "Drawdown",
                "Win_Rate",
                "Profit_Factor",
                "Total_Trades",
            ]
            if c in display_df.columns
        ]

        st.dataframe(
            display_df[columns],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Return vs Risk")

        fig = px.scatter(
            strategy_df,
            x="Maximum_Drawdown",
            y="Total_Return",
            size="Total_Trades",
            color="strategy",
            text="strategy",
            title="Strategy Risk / Return Profile",
        )

        fig.update_traces(
            textposition="top center"
        )

        fig.update_layout(
            xaxis_title="Maximum Drawdown",
            yaxis_title="Total Return",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.warning(
            "Strategy comparison data was not found."
        )


# ============================================================
# MARKET PREDICTIONS
# ============================================================

elif page == "Market Predictions":

    st.header("🤖 Market Predictions")

    predictions = data["predictions"]

    if predictions is None:

        st.error(
            "predictions.parquet was not found."
        )

    else:

        predictions = predictions.copy()

        predictions["date"] = pd.to_datetime(
            predictions["date"]
        )

        latest_date = predictions["date"].max()

        latest = predictions[
            predictions["date"] == latest_date
        ].copy()

        st.info(
            f"Latest prediction date: {latest_date.date()}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Stocks Analyzed",
                latest["ticker"].nunique(),
            )

        with col2:
            st.metric(
                "Prediction Records",
                len(latest),
            )

        with col3:

            if "pred_return_lightgbm" in latest.columns:

                avg_prediction = latest[
                    "pred_return_lightgbm"
                ].mean()

                st.metric(
                    "Average Expected Return",
                    percentage(avg_prediction),
                )

        st.divider()

        # ----------------------------------------------------
        # Model selector
        # ----------------------------------------------------

        model_columns = [
            c for c in predictions.columns
            if c.startswith("pred_return_")
        ]

        if model_columns:

            selected_model = st.selectbox(
                "Prediction Model",
                model_columns,
            )

            top_predictions = latest.sort_values(
                selected_model,
                ascending=False,
            ).head(20)

            st.subheader(
                "Top Predicted Opportunities"
            )

            display = top_predictions[
                [
                    "ticker",
                    "close",
                    selected_model,
                ]
            ].copy()

            display["Expected Return"] = (
                display[selected_model]
                .map(percentage)
            )

            display["Price"] = (
                display["close"]
                .map(lambda x: f"₹{x:,.2f}")
            )

            st.dataframe(
                display[
                    [
                        "ticker",
                        "Price",
                        "Expected Return",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            fig = px.bar(
                top_predictions,
                x="ticker",
                y=selected_model,
                title="Top Predicted Returns",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        direction_columns = [
            c for c in predictions.columns
            if c.startswith("pred_direction_")
        ]

        if direction_columns:

            st.subheader(
                "Directional Model Consensus"
            )

            direction_data = latest.copy()

            direction_data["up_votes"] = (
                direction_data[
                    direction_columns
                ]
                .sum(axis=1)
            )

            consensus = (
                direction_data
                .sort_values(
                    "up_votes",
                    ascending=False,
                )
                .head(20)
            )

            st.dataframe(
                consensus[
                    [
                        "ticker",
                        "up_votes",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# STRATEGY ANALYSIS
# ============================================================

elif page == "Strategy Analysis":

    st.header("🎯 Strategy Analysis")

    strategy_df = data["strategy_comparison"]

    if strategy_df is None:

        st.error(
            "strategy_comparison.csv not found."
        )

    else:

        st.subheader(
            "Optimized Trading Strategies"
        )

        st.dataframe(
            strategy_df,
            use_container_width=True,
            hide_index=True,
        )

        # Return chart

        st.subheader(
            "Strategy Returns"
        )

        fig = px.bar(
            strategy_df.sort_values(
                "Total_Return",
                ascending=False,
            ),
            x="strategy",
            y="Total_Return",
            text="Total_Return",
            title="Strategy Total Return",
        )

        fig.update_traces(
            texttemplate="%{text:.2%}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # Sharpe

        st.subheader(
            "Risk-Adjusted Performance"
        )

        fig2 = px.bar(
            strategy_df.sort_values(
                "Sharpe_Ratio",
                ascending=False,
            ),
            x="strategy",
            y="Sharpe_Ratio",
            text="Sharpe_Ratio",
            title="Sharpe Ratio Comparison",
        )

        fig2.update_traces(
            texttemplate="%{text:.2f}"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

        # Drawdown

        st.subheader(
            "Maximum Drawdown"
        )

        fig3 = px.bar(
            strategy_df.sort_values(
                "Maximum_Drawdown",
                ascending=False,
            ),
            x="strategy",
            y="Maximum_Drawdown",
            text="Maximum_Drawdown",
            title="Maximum Drawdown Comparison",
        )

        fig3.update_traces(
            texttemplate="%{text:.2%}"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True,
        )

        st.success(
            """
            Strategy selection is based on historical backtesting.
            The system does not guarantee future profitability.
            """
        )


# ============================================================
# PORTFOLIO PERFORMANCE
# ============================================================

elif page == "Portfolio Performance":

    st.header("💰 Portfolio Performance")

    portfolio = data["portfolio_daily"]

    if portfolio is None:

        st.warning(
            "Portfolio daily data was not found."
        )

    else:

        portfolio = portfolio.copy()

        if "date" in portfolio.columns:

            portfolio["date"] = pd.to_datetime(
                portfolio["date"]
            )

        capital_column = None

        for c in [
            "capital",
            "portfolio_value",
            "equity",
            "Final_Capital",
        ]:

            if c in portfolio.columns:

                capital_column = c
                break

        if capital_column:

            starting = float(
                portfolio[capital_column].iloc[0]
            )

            ending = float(
                portfolio[capital_column].iloc[-1]
            )

            total_return = (
                ending / starting - 1
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Starting Capital",
                    money(starting),
                )

            with col2:

                st.metric(
                    "Current Capital",
                    money(ending),
                )

            with col3:

                st.metric(
                    "Total Return",
                    percentage(total_return),
                )

            fig = px.line(
                portfolio,
                x="date",
                y=capital_column,
                title="Portfolio Equity Curve",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # Drawdown

        if "drawdown" in portfolio.columns:

            st.subheader(
                "Portfolio Drawdown"
            )

            fig2 = px.area(
                portfolio,
                x="date",
                y="drawdown",
                title="Drawdown",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )


# ============================================================
# TRADING SIGNALS
# ============================================================

elif page == "Trading Signals":

    st.header("🚦 Trading Signals")

    signals = data["signals"]

    if signals is None:

        st.warning(
            "Signal file not found."
        )

    else:

        signals = signals.copy()

        if "date" in signals.columns:

            signals["date"] = pd.to_datetime(
                signals["date"]
            )

        latest_date = signals["date"].max()

        latest = signals[
            signals["date"] == latest_date
        ].copy()

        st.info(
            f"Latest signal date: {latest_date.date()}"
        )

        if "signal" in latest.columns:

            buy_count = (
                latest["signal"]
                .eq("BUY")
                .sum()
            )

            sell_count = (
                latest["signal"]
                .eq("SELL")
                .sum()
            )

            hold_count = (
                latest["signal"]
                .eq("HOLD")
                .sum()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "BUY",
                buy_count,
            )

            c2.metric(
                "SELL",
                sell_count,
            )

            c3.metric(
                "HOLD",
                hold_count,
            )

        st.subheader(
            "Current Opportunities"
        )

        display_columns = [
            c for c in [
                "date",
                "ticker",
                "signal",
                "confidence",
                "ensemble_return",
                "opportunity_score",
                "position_weight",
                "up_votes",
            ]
            if c in latest.columns
        ]

        st.dataframe(
            latest.sort_values(
                "opportunity_score",
                ascending=False,
            )[display_columns],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🧠 Machine Learning Model Performance")

    regression = load_csv(
        "regression_model_comparison.csv"
    )

    classification = load_csv(
        "classification_model_comparison.csv"
    )

    if regression is not None:

        st.subheader(
            "Regression Models"
        )

        st.dataframe(
            regression,
            use_container_width=True,
            hide_index=True,
        )

        if "RMSE" in regression.columns:

            fig = px.bar(
                regression.sort_values("RMSE"),
                x="Model",
                y="RMSE",
                title="Regression RMSE",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if "R2" in regression.columns:

            fig2 = px.bar(
                regression,
                x="Model",
                y="R2",
                title="Regression R²",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )

    if classification is not None:

        st.subheader(
            "Classification Models"
        )

        st.dataframe(
            classification,
            use_container_width=True,
            hide_index=True,
        )

        if "F1" in classification.columns:

            fig3 = px.bar(
                classification.sort_values(
                    "F1",
                    ascending=False,
                ),
                x="Model",
                y="F1",
                title="Classification F1 Score",
            )

            st.plotly_chart(
                fig3,
                use_container_width=True,
            )


# ============================================================
# ABOUT SYSTEM
# ============================================================

elif page == "About System":

    st.header("ℹ️ About the System")

    st.markdown(
        """
        ## Stock Market Intelligence System

        This project combines machine learning, feature engineering,
        ensemble prediction and quantitative strategy optimization.

        ### Pipeline

        **1. Market Data**

        Historical market data is collected for a large stock universe.

        ↓

        **2. Feature Engineering**

        53 engineered market features are generated.

        ↓

        **3. Machine Learning**

        Multiple regression and classification models are trained.

        Models include:

        - Random Forest
        - Extra Trees
        - Histogram Gradient Boosting
        - XGBoost
        - LightGBM

        ↓

        **4. Ensemble Prediction**

        Model predictions are combined to estimate:

        - Expected return
        - Direction
        - Confidence
        - Model agreement

        ↓

        **5. Strategy Engine**

        The system filters opportunities using:

        - Expected return
        - Directional consensus
        - Confidence
        - Volatility
        - Position limits

        ↓

        **6. Portfolio Backtesting**

        Candidate strategies are tested using historical data.

        ↓

        **7. Dashboard**

        The final system presents:

        - Market predictions
        - Trading opportunities
        - Strategy comparison
        - Portfolio performance
        - Risk metrics
        - Model performance

        ---

        ### Important

        Historical backtesting is not a guarantee of future market
        performance. This dashboard is intended for research,
        experimentation and demonstration.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Stock Market Intelligence System • Machine Learning + Quantitative Strategy Engine"
)