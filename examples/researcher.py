"""
Esempio completo di Researcher.

Mostra:
  - Costruzione con fattori (MovingAverage, Return)
  - get_data()     carica dati + calcola fattori (risultato cached)
  - get_data_eod() barre EOD raw senza fattori (contesto intraday)
  - plot()         grafico interattivo Plotly con tutti i fattori

Uso:
    python examples/researcher.py
    python examples/researcher.py --ticker AAPL --start 2024-01-01
"""

import argparse
import sys
import io
from datetime import date
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.datafeed import DataFeed
from engine.engine import Researcher
from engine.factor import MovingAverage, Return

PRICES_BASE = str(Path(__file__).parent.parent / "database")
SEP = "-" * 60


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main():
    parser = argparse.ArgumentParser(description="Esempio Researcher")
    parser.add_argument("--ticker", default="AAPL", help="Simbolo (default: AAPL)")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--freq", default="eod",
                        choices=["eod", "1m", "5m", "1h", "4h"])
    parser.add_argument("--no-plot", action="store_true",
                        help="Salta il grafico (utile in ambienti senza display)")
    args = parser.parse_args()

    feed = DataFeed(prices_base=PRICES_BASE)

    # Definizione fattori
    factors = [
        MovingAverage(20),
        MovingAverage(50),
        Return(1),
    ]

    researcher = Researcher(
        factors=factors,
        feed=feed,
        start_date=args.start,
        end_date=args.end,
        frequency=args.freq,
        tickers=[args.ticker],
    )

    # ------------------------------------------------------------------
    # 1. get_data() — prima chiamata: carica + calcola + salva in cache
    # ------------------------------------------------------------------
    section("1. get_data() — prima chiamata (load + compute + cache)")
    import time
    t0 = time.perf_counter()
    df = researcher.get_data()
    elapsed_first = time.perf_counter() - t0

    if df is None:
        print(f"Nessun dato per '{args.ticker}' ({args.start} - {args.end}, {args.freq}).")
        sys.exit(1)

    print(f"Righe (post-warmup) : {df.height}")
    print(f"Colonne             : {df.columns}")
    print(f"Tempo               : {elapsed_first * 1000:.1f} ms")
    print(f"\nUltime 5 righe:")
    print(df.select(["date", "close", "ma_20", "ma_50", "ret_1"]).tail(5))

    # ------------------------------------------------------------------
    # 2. get_data() — seconda chiamata: ritorna dalla cache
    # ------------------------------------------------------------------
    section("2. get_data() — seconda chiamata (da cache)")
    t0 = time.perf_counter()
    df2 = researcher.get_data()
    elapsed_cached = time.perf_counter() - t0

    print(f"Stessa istanza  : {df is df2}")
    print(f"Tempo           : {elapsed_cached * 1000:.2f} ms  "
          f"(speedup: {elapsed_first / max(elapsed_cached, 1e-9):.0f}x)")

    # ------------------------------------------------------------------
    # 3. get_data_eod() — barre EOD raw (senza fattori, non cached)
    # ------------------------------------------------------------------
    section("3. get_data_eod() — EOD raw per contesto intraday")
    df_eod = researcher.get_data_eod()
    if df_eod is not None:
        print(f"Righe EOD : {df_eod.height}")
        print(f"Colonne   : {df_eod.columns}")
        print(df_eod.select(["date", "open", "high", "low", "close"]).tail(3))
    else:
        print("Nessun dato EOD disponibile.")

    # ------------------------------------------------------------------
    # 4. plot() — grafico interattivo
    # ------------------------------------------------------------------
    section("4. plot() — grafico interattivo")
    if args.no_plot:
        print("(skipped — rimuovi --no-plot per visualizzare il grafico)")
    else:
        fig = researcher.plot(
            ticker=args.ticker,
            chart_type="candlestick",
            theme="plotly_dark",
        )
        fig.show()
        print("Grafico aperto nel browser.")


if __name__ == "__main__":
    main()
