
import os, sys
sys.path.append(os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from config import *
from src.features import create_features, FEATURES


def add_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compatibility shim: normalise column names and run the feature pipeline.

    app.py receives a DataFrame with Title-Case columns from yfinance
    (Close, High, Low, …).  src.features.create_features expects lower-case.
    This wrapper normalises both ways so the rest of app.py can work
    unchanged via the FEATURES list.
    """
    import pandas as _pd

    out = df.copy()

    # Map Title-Case yfinance columns → lower-case expected by feature engine
    col_map = {
        c: c.lower()
        for c in out.columns
        if c.lower() in ("open", "high", "low", "close", "volume",
                         "adj close", "adj_close")
    }
    # Also handle the Ticker column used for groupby
    out.columns = [col_map.get(c, c) for c in out.columns]

    result = create_features(out)

    return result

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

st.title("📈 Stock Market Intelligence & Prediction System")
st.caption("Indian + US equities | ML predictions | signals | backtesting | risk")

@st.cache_data(ttl=300)
def live_data(ticker):
    df = yf.download(
        ticker, period="5d", interval="1m",
        auto_adjust=False, progress=False, prepost=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()

@st.cache_data(ttl=900)
def historical(ticker):
    df = yf.download(
        ticker, period="2y", auto_adjust=False, progress=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()

st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Market", "Prediction", "Model Comparison", "Risk & Backtest"
])

with tab1:
    st.subheader(f"Latest data: {ticker}")
    live = live_data(ticker)
    if live.empty:
        st.warning("No live/near-live data returned by the data provider.")
    else:
        latest = live.iloc[-1]
        cols = st.columns(4)
        cols[0].metric("Latest", f"{float(latest['Close']):.2f}")
        cols[1].metric("Day High", f"{float(live['High'].max()):.2f}")
        cols[2].metric("Day Low", f"{float(live['Low'].min()):.2f}")
        cols[3].metric("Volume", f"{float(live['Volume'].sum()):,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=live["Datetime"] if "Datetime" in live else live.iloc[:,0],
            y=live["Close"], mode="lines", name="Price"
        ))
        fig.update_layout(height=450, title="Intraday Price")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Next-day prediction")
    hist = historical(ticker)

    if hist.empty:
        st.warning("Ticker not found or historical data unavailable.")
    else:
        hist["Ticker"] = ticker
        feat = add_features(hist)
        usable = feat.dropna(subset=FEATURES)

        if usable.empty:
            st.warning("Not enough historical observations.")
        else:
            latest = usable.iloc[-1]
            model_path = MODEL_DIR / "XGBoost_regressor.pkl"
            clf_path = MODEL_DIR / "XGBoost_classifier.pkl"

            if not model_path.exists():
                st.info("Train the models first with: py src/train_classical.py")
            else:
                model = joblib.load(model_path)
                pred_return = float(model.predict(
                    pd.DataFrame([latest[FEATURES].values], columns=FEATURES)
                )[0])

                predicted_price = float(latest["Close"]) * (1+pred_return)

                if pred_return > BUY_THRESHOLD:
                    signal = "BUY"
                elif pred_return < SELL_THRESHOLD:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                c = st.columns(3)
                c[0].metric("Current Close", f"{float(latest['Close']):.2f}")
                c[1].metric("Predicted Return", f"{pred_return*100:.2f}%")
                c[2].metric("Predicted Next Close", f"{predicted_price:.2f}")

                st.success(f"Signal: **{signal}**")

                if clf_path.exists():
                    clf = joblib.load(clf_path)
                    prob = clf.predict_proba(
                        pd.DataFrame([latest[FEATURES].values], columns=FEATURES)
                    )[0,1]
                    st.metric("UP probability", f"{prob*100:.1f}%")

with tab3:
    st.subheader("Model comparison")
    reg_file = REPORT_DIR / "regression_model_comparison.csv"
    clf_file = REPORT_DIR / "classification_model_comparison.csv"

    if reg_file.exists():
        st.dataframe(pd.read_csv(reg_file), use_container_width=True)
    else:
        st.info("Run classical training first.")

    if clf_file.exists():
        st.dataframe(pd.read_csv(clf_file), use_container_width=True)

with tab4:
    st.subheader("Backtest & risk")
    bt_file = REPORT_DIR / "backtest_by_stock.csv"

    if bt_file.exists():
        bt = pd.read_csv(bt_file)
        st.dataframe(bt, use_container_width=True)

        if len(bt):
            st.metric("Average Sharpe", f"{bt['Sharpe'].mean():.2f}")
            st.metric("Average Max Drawdown", f"{bt['Max_Drawdown'].mean()*100:.2f}%")
    else:
        st.info("Run classical training/backtesting first.")
