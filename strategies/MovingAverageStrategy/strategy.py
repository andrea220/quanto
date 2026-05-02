import sys
import polars as pl
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
# Path relativo alla root del progetto (2 livelli sopra da strategies/MovingAverageStrategy/)
data_dir = Path("../../database")

from engine.datafeed import DataFeed, MarketData
from engine.engine import *
from engine.factor import *
from engine.plotting import *
from engine.reports import *

# ── Parametri ────────────────────────────────────────────────────────────────
universe = ['SX5E']
start_date = '2020-01-01'
end_date = '2026-12-31'

feed = DataFeed(prices_base=data_dir)


# ── Strategia ─────────────────────────────────────────────────────────────────
class MovingAverageStrategy(Strategy):

    def __init__(self) -> None:
        super().__init__()
        self.ticker = 'SX5E'
        self.factors = [
            MovingAverage(50)
        ]
        self.position_open = False

    def on_bar(self):
        if self.period <= 1:
            return
        close      = self.market_data.get(self.ticker, 0)['close'].item()
        prev_close = self.market_data.get(self.ticker, 1)['close'].item()
        ma_50      = self.market_data.get(self.ticker, 1)['ma_50'].item()
        prev_ma_50 = self.market_data.get(self.ticker, 2)['ma_50'].item()
        size = self.portfolio.cash / close

        # Open position
        if close > ma_50 and prev_close < prev_ma_50 and not self.position_open:
            self.buy(self.ticker, size)
            self.position_open = True

        # Close position
        if self.position_open:
            if close < ma_50 and prev_close > prev_ma_50:
                self.close_position(self.ticker)
                self.position_open = False


# ── Backtest ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    starting_balance = 100_000.

    exec_model = ExecutionModel(mode="on_close", slippage_bps=0.0, commission_bps=0.0)
    risk       = RiskManager(max_leverage=4)
    reporter   = ReportWriter(out_dir="backtest_reports/moving_average")

    strat = MovingAverageStrategy()
    strat.backtest_refactor(
        starting_balance,
        start_date, end_date,
        "eod",
        universe,
        feed,
        exec_model,
        risk,
        reporter,
        intraday_log=True,
    )

    analytics = StrategyAnalytics(strat)
    analytics.summary()
    analytics.export_excel("backtest_reports/MovingAverageStrategy.xlsx")