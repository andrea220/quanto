"""
Labeling utilities for supervised learning on financial time series.

Provides event-based labeling compatible with Purged K-Fold CV and sample
weight calculation (engine/validation.py, engine/weights.py — Blocco D/E).

Reference: López de Prado (2018), Advances in Financial Machine Learning, Ch. 3–4.

Classes
-------
FixedHorizonLabeler
    Labels each signal event with a fixed forward horizon [t0, t0+horizon].

Planned (Blocco D)
------------------
SignalExitLabeler
    Labels events using an exit condition (stop-loss / take-profit / max holding).
"""

import polars as pl
from typing import Optional


class FixedHorizonLabeler:
    """
    Labels signal events with a fixed-horizon outcome.

    For each bar where ``signal_col != 0`` (event at t0), computes:

    - ``t1_date``     : date of bar at t0 + horizon
    - ``y``           : forward return over [t0, t1] signed by signal direction
                        = (price[t1] / price[t0] - 1) * sign(signal)
    - ``holding_time``: actual number of bars from t0 to t1

    Rows where t1 falls beyond the series end have ``null`` y, t1_date and
    holding_time. Drop them before training::

        events = labeler.label(df).drop_nulls("y")

    The output DataFrame is the building block for:

    - **Sample weights** — bar concurrency in overlapping [t0, t1] intervals
    - **Purged K-Fold CV** — embargo windows based on [t0, t1] overlap
    - **Meta-labeling** — binary y as secondary classifier target

    Args:
        horizon    : Number of forward bars from t0 to t1 (must be > 0).
        signal_col : Column name containing the signal (0 = no event).
        price_col  : Price column for computing returns (default: ``"close"``).

    Example:
        >>> from engine.engine import Researcher
        >>> from engine.factor import MarketStructure
        >>> from engine.labeling import FixedHorizonLabeler
        >>>
        >>> researcher = Researcher([MarketStructure(1)], feed, start, end, "eod", ["AAPL"])
        >>> df = researcher.get_data()
        >>>
        >>> labeler = FixedHorizonLabeler(horizon=5, signal_col="ms_1_signal")
        >>> events = labeler.label(df).drop_nulls("y")
        >>> print(events)
        # shape: (N, 8) — t0_date, t1_date, ticker, signal, price_t0, price_t1, y, holding_time
    """

    def __init__(
        self,
        horizon: int,
        signal_col: str,
        price_col: str = "close",
    ) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        self.horizon = horizon
        self.signal_col = signal_col
        self.price_col = price_col

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute labels for all signal bars in df.

        Args:
            df: DataFrame with columns ``[date, ticker, <price_col>, <signal_col>]``.
                Typically the output of ``Researcher.get_data()``.
                If a ``time`` column is present it is included in the sort key
                (correct behaviour for intraday data).

        Returns:
            DataFrame with one row per signal event and columns:

            ============  =====================================================
            t0_date       bar date at signal generation
            t1_date       bar date at t0 + horizon (null if beyond series end)
            ticker        ticker symbol
            signal        original signal value at t0
            price_t0      price at t0
            price_t1      price at t1 (null if beyond series end)
            y             signed forward return (null if beyond series end)
            holding_time  horizon in bars (null if beyond series end)
            ============  =====================================================

        Raises:
            ValueError: if required columns are missing from df.
        """
        required = {"date", "ticker", self.price_col, self.signal_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        h = self.horizon
        sort_cols = ["ticker", "date"] + (["time"] if "time" in df.columns else [])

        events = (
            df
            .sort(sort_cols)
            .with_columns([
                pl.col(self.price_col).shift(-h).over("ticker").alias("_price_t1"),
                pl.col("date").shift(-h).over("ticker").alias("t1_date"),
            ])
            .filter(pl.col(self.signal_col) != 0)
            .with_columns([
                pl.when(pl.col("_price_t1").is_not_null())
                  .then(
                      (pl.col("_price_t1") / pl.col(self.price_col) - 1.0)
                      * pl.col(self.signal_col).sign().cast(pl.Float64)
                  )
                  .otherwise(pl.lit(None, dtype=pl.Float64))
                  .alias("y"),
                pl.when(pl.col("_price_t1").is_not_null())
                  .then(pl.lit(h, dtype=pl.Int32))
                  .otherwise(pl.lit(None, dtype=pl.Int32))
                  .alias("holding_time"),
            ])
            .rename({"date": "t0_date", self.price_col: "price_t0"})
            .rename({"_price_t1": "price_t1"})
            .select([
                "t0_date",
                "t1_date",
                "ticker",
                pl.col(self.signal_col).alias("signal"),
                "price_t0",
                "price_t1",
                "y",
                "holding_time",
            ])
        )
        return events
