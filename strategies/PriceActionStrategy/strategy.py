import sys
import polars as pl
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))
# Path relativo alla root del progetto (2 livelli sopra da strategies/PriceActionStrategy/)
data_dir = Path("../../database")

from engine.datafeed import DataFeed, MarketData
from engine.engine import *
from engine.factor import *
from engine.plotting import *
from engine.reports import *

# ── Parametri ────────────────────────────────────────────────────────────────
universe   = ['SX5E']
start_date = '2020-01-01'
end_date   = '2026-12-31'

feed = DataFeed(prices_base=data_dir)


# ── Strategia ─────────────────────────────────────────────────────────────────
class PriceActionStrategy(Strategy):
    """
    Strategia basata sui cambi di struttura di mercato (MSS – Market Structure Shift).

    Regole:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  LONG  : segnale MSS bull (+2) → close > ultimo swing high             │
    │          (il mercato era in trend ribassista e rompe al rialzo)         │
    │  EXIT  : segnale MSS bear (-2) O BOS down (-1)                         │
    │                                                                          │
    │  SHORT : segnale MSS bear (-2) → close < ultimo swing low              │
    │          (il mercato era in trend rialzista e rompe al ribasso)         │
    │  EXIT  : segnale MSS bull (+2) O BOS up  (+1)                          │
    └──────────────────────────────────────────────────────────────────────────┘

    Filtro FVG: entra solo se il livello di rottura è vicino a un FVG attivo
    (opzionale, abilitato con use_fvg_filter=True).

    Un solo trade aperto alla volta; size = tutto il capitale disponibile.
    """

    def __init__(self, swing_window: int = 2, use_fvg_filter: bool = False) -> None:
        super().__init__()
        self.ticker          = 'SX5E'
        self.swing_window    = swing_window
        self.use_fvg_filter  = use_fvg_filter
        self.position_open   = False   # True=long, False=flat, None=short
        self.position_dir    = 0       # +1=long, -1=short, 0=flat

        self.factors = [
            MarketStructure(swing_window=swing_window),
            FairValueGap(atr_period=50, threshold=0.5),
        ]

    def on_bar(self):
        if self.period <= 2:
            return

        ms_signal_col = f"ms_{self.swing_window}_signal"
        fvg_bull_col  = f"fvg_50_bull"
        fvg_bear_col  = f"fvg_50_bear"

        # Usa bar confermata (lag=1 per evitare lookahead sul close corrente)
        bar      = self.market_data.get(self.ticker, 1)
        bar_prev = self.market_data.get(self.ticker, 2)

        signal      = bar[ms_signal_col].item()
        prev_signal = bar_prev[ms_signal_col].item()
        close       = bar['close'].item()

        # Filtro FVG: segnale deve coincidere con un FVG attivo sulla stessa bar
        fvg_bull = bar[fvg_bull_col].item() if self.use_fvg_filter else 1.0
        fvg_bear = bar[fvg_bear_col].item() if self.use_fvg_filter else -1.0

        size = self.portfolio.cash / close if close > 0 else 0

        # ── Gestione posizione corrente ───────────────────────────────────────
        if self.position_dir == 1:           # long aperto
            # Chiudi su MSS bear o BOS down
            if signal <= -1:
                self.close_position(self.ticker)
                self.position_dir = 0

        elif self.position_dir == -1:        # short aperto
            # Chiudi su MSS bull o BOS up
            if signal >= 1:
                self.close_position(self.ticker)
                self.position_dir = 0

        # ── Nuovi segnali di ingresso ─────────────────────────────────────────
        if self.position_dir == 0:
            # MSS Bull: cambio struttura rialzista
            if signal == 2 and (not self.use_fvg_filter or fvg_bull > 0):
                if size > 0:
                    self.buy(self.ticker, size)
                    self.position_dir = 1

            # MSS Bear: cambio struttura ribassista
            elif signal == -2 and (not self.use_fvg_filter or fvg_bear < 0):
                if size > 0:
                    self.sell(self.ticker, size)
                    self.position_dir = -1


# ── Backtest ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    starting_balance = 100_000.

    exec_model = ExecutionModel(mode="on_close", slippage_bps=0.0, commission_bps=0.0)
    risk       = RiskManager(max_leverage=4)
    reporter   = ReportWriter(out_dir="backtest_reports/price_action")

    strat = PriceActionStrategy(swing_window=2, use_fvg_filter=False)
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
    analytics.export_excel("backtest_reports/PriceActionStrategy.xlsx")
