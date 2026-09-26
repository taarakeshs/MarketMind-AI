
# Stock Market Intelligence & Prediction System

A modular multi-market ML system for Indian + US equities.

## Included
- 500+ stock universe support
- Indian NSE + US stocks
- Historical and near-live Yahoo Finance data
- Technical feature engineering
- Next-day return / price prediction
- UP/DOWN classification
- Buy/Hold/Sell signal
- Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with TimeSeriesSplit
- Walk-forward validation
- LSTM and GRU time-series models
- Backtesting
- Portfolio simulation
- Sharpe ratio
- Maximum drawdown
- Confusion matrix
- Model comparison
- Streamlit dashboard
- Model/data caching
- Per-ticker evaluation

## Important
This is an educational research system, not financial advice. Yahoo Finance data is not guaranteed to be real-time. Live quotes can be delayed depending on exchange/data availability. Backtests do not guarantee future performance.

## Quick start

### 1. Create environment
Windows:
```bash
py -m venv .venv
.venv\Scripts\activate
```

### 2. Install
```bash
py -m pip install -r requirements.txt
```

### 3. Train classical models
```bash
py src/train_classical.py
```

### 4. Train deep learning models
```bash
py src/train_deep.py
```

### 5. Run dashboard
```bash
streamlit run app.py
```

The first training run downloads a large amount of data. Start with the defaults in `config.py`, verify the pipeline, then increase the universe.

## Architecture

Data -> Features -> Time split -> Walk-forward validation -> Model training -> Signals -> Backtest -> Risk -> Dashboard
