
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-01-01"
END_DATE = None

# 500+ target universe. The loaders use current S&P 500 constituents plus
# a maintained Indian large/mid-cap list. Expand INDIA_TICKERS as needed.
USE_SP500 = True
USE_INDIA = True

# Start at 50 while testing the pipeline. Set to None for the full universe.
MAX_TICKERS = 550

TEST_FRACTION = 0.20
WALK_FORWARD_SPLITS = 5

RANDOM_STATE = 42

# Classical models
RF_ESTIMATORS = 250

# Tuning
TUNE_MODELS = True
TUNING_CV_SPLITS = 3
TUNING_N_ITER = 12

# Deep learning
SEQUENCE_LENGTH = 30
DL_EPOCHS = 12
DL_BATCH_SIZE = 256
DL_PATIENCE = 3
DL_TICKER_LIMIT = None  # None = use all available tickers

# Trading
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.0
BUY_THRESHOLD = 0.003
SELL_THRESHOLD = -0.003

# Yahoo Finance is useful for research/prototyping, but not guaranteed real-time.
