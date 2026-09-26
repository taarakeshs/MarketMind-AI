from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)

BACKTEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "backtest_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "enhanced_signals.csv"
)


def main():

    print("=" * 70)
    print("MULTI-FACTOR SIGNAL ENGINE")
    print("=" * 70)

    print()
    print("Loading feature dataset...")

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
    )

    backtest = pd.read_csv(
        BACKTEST_FILE,
        parse_dates=["date"],
    )

    print(
        f"Rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # KEEP ONLY BACKTEST PERIOD
    # --------------------------------------------------------

    test_dates = set(
        backtest["date"]
    )

    df = df[
        df["date"].isin(
            test_dates
        )
    ].copy()

    # --------------------------------------------------------
    # MERGE MODEL PROBABILITY
    # --------------------------------------------------------

    probability_data = (
        backtest[
            [
                "date",
                "ticker",
                "probability_up",
            ]
        ]
    )

    df = df.merge(
        probability_data,
        on=[
            "date",
            "ticker",
        ],
        how="inner",
    )

    print(
        f"Signals generated: "
        f"{len(df):,}"
    )

    # ========================================================
    # FACTOR 1 — ML CONFIDENCE
    # ========================================================

    # Convert probability into
    # directional confidence.

    df["ml_confidence"] = (
        abs(
            df["probability_up"]
            - 0.50
        )
        * 2
    )

    # 0 = no confidence
    # 1 = maximum confidence

    # ========================================================
    # FACTOR 2 — MOMENTUM
    # ========================================================

    df["momentum_score"] = (
        (
            df["return_5d"] > 0
        ).astype(float)
        * 0.5
        +
        (
            df["return_20d"] > 0
        ).astype(float)
        * 0.5
    )

    # ========================================================
    # FACTOR 3 — TREND
    # ========================================================

    df["trend_score"] = (
        (
            df["price_to_sma_20"] > 1
        ).astype(float)
        * 0.4
        +
        (
            df["price_to_sma_50"] > 1
        ).astype(float)
        * 0.3
        +
        (
            df["price_to_sma_200"] > 1
        ).astype(float)
        * 0.3
    )

    # ========================================================
    # FACTOR 4 — RSI
    # ========================================================

    # Prefer moderate RSI rather than
    # extreme overbought conditions.

    df["rsi_score"] = np.where(
        df["rsi"].between(
            45,
            65,
        ),
        1.0,
        np.where(
            df["rsi"].between(
                35,
                75,
            ),
            0.5,
            0.0,
        ),
    )

    # ========================================================
    # FACTOR 5 — VOLATILITY
    # ========================================================

    # Lower volatility receives
    # higher risk score.

    volatility_rank = (
        df.groupby("date")
        ["volatility_20d"]
        .rank(
            pct=True,
            ascending=True,
        )
    )

    df["volatility_score"] = (
        1 - volatility_rank
    )

    # ========================================================
    # FACTOR 6 — VOLUME
    # ========================================================

    df["volume_score"] = np.where(
        df["volume_ratio"] >= 1.0,
        1.0,
        0.0,
    )

    # ========================================================
    # ML DIRECTION
    # ========================================================

    df["ml_direction"] = np.where(
        df["probability_up"] >= 0.50,
        1,
        -1,
    )

    # ========================================================
    # COMPOSITE SCORE
    # ========================================================

    df["composite_score"] = (
        df["ml_confidence"] * 0.35
        +
        df["momentum_score"] * 0.20
        +
        df["trend_score"] * 0.20
        +
        df["rsi_score"] * 0.10
        +
        df["volatility_score"] * 0.10
        +
        df["volume_score"] * 0.05
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    df["signal"] = 0

    # Strong bullish configuration

    bullish = (
        (df["ml_direction"] == 1)
        &
        (df["probability_up"] >= 0.55)
        &
        (df["composite_score"] >= 0.60)
    )

    # Strong bearish configuration

    bearish = (
        (df["ml_direction"] == -1)
        &
        (df["probability_up"] <= 0.45)
        &
        (df["composite_score"] >= 0.60)
    )

    df.loc[
        bullish,
        "signal"
    ] = 1

    df.loc[
        bearish,
        "signal"
    ] = -1

    # ========================================================
    # POSITION SIZE
    # ========================================================

    df["position_size"] = (
        df["ml_confidence"]
        * df["composite_score"]
    )

    # Cap exposure.

    df["position_size"] = (
        df["position_size"]
        .clip(
            0,
            1,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("SIGNAL SUMMARY")
    print("=" * 70)

    print()

    print(
        "BUY signals :",
        int(
            (
                df["signal"] == 1
            ).sum()
        ),
    )

    print(
        "HOLD signals:",
        int(
            (
                df["signal"] == 0
            ).sum()
        ),
    )

    print(
        "SELL signals:",
        int(
            (
                df["signal"] == -1
            ).sum()
        ),
    )

    print()

    print(
        "Average ML probability:",
        round(
            df[
                "probability_up"
            ].mean(),
            4,
        ),
    )

    print(
        "Average composite score:",
        round(
            df[
                "composite_score"
            ].mean(),
            4,
        ),
    )

    print(
        "Average position size:",
        round(
            df[
                "position_size"
            ].mean(),
            4,
        ),
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Enhanced signals saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("SIGNAL ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()