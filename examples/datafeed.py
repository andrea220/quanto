"""
Esempio completo di DataFeed e MarketData.

Mostra:
  - get_market_data()      carica dati in memoria come DataFrame
  - scan_prices()          lazy scan (senza caricare tutto in RAM)
  - get_data_excel()       export su file Excel
  - MarketData             accesso bar-by-bar (prezzi, OHLC, slice storici)

Uso:
    python examples/datafeed_last5.py AAPL
    python examples/datafeed_last5.py SPY --start 2024-01-01 --end 2025-12-31
    python examples/datafeed_last5.py AAPL --freq 1m
"""

import argparse
import sys
import io
from datetime import date
from pathlib import Path

# Forza UTF-8 su stdout (necessario su Windows con codepage legacy)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.datafeed import DataFeed, MarketData

PRICES_BASE = str(Path(__file__).parent.parent / "database")

SEP = "-" * 60


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main():
    parser = argparse.ArgumentParser(description="Esempio DataFeed + MarketData")
    parser.add_argument("ticker", help="Simbolo (es. AAPL, SPY)")
    parser.add_argument("--start", default="2020-01-01", help="Data inizio YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", default=str(date.today()), help="Data fine YYYY-MM-DD (default: oggi)")
    parser.add_argument(
        "--freq",
        default="eod",
        choices=["eod", "1m", "5m", "1h", "4h"],
        help="Frequenza dei dati (default: eod)",
    )
    parser.add_argument("--excel", action="store_true", help="Esporta i dati su file Excel")
    args = parser.parse_args()

    feed = DataFeed(prices_base=PRICES_BASE)

    # ------------------------------------------------------------------
    # 1. get_market_data — carica tutto in un DataFrame Polars
    # ------------------------------------------------------------------
    section("1. get_market_data() — ultime 5 righe")
    df = feed.get_market_data(args.start, args.end, args.freq, [args.ticker])

    if df.is_empty():
        print(f"Nessun dato trovato per '{args.ticker}' ({args.start} - {args.end}, {args.freq}).")
        sys.exit(1)

    print(f"Righe totali caricate: {df.height}")
    print(df.tail(5))

    # ------------------------------------------------------------------
    # 2. scan_prices — LazyFrame (utile per grandi dataset)
    # ------------------------------------------------------------------
    section("2. scan_prices() — LazyFrame, conta righe senza caricare tutto")
    lf = feed.scan_prices(args.start, args.end, args.freq, [args.ticker])
    count = lf.select("date").collect(engine="streaming").height
    print(f"Righe disponibili via LazyFrame: {count}")

    # ------------------------------------------------------------------
    # 3. get_data_excel — export Excel (solo se --excel e' passato)
    # ------------------------------------------------------------------
    section("3. get_data_excel() — export su Excel")
    if args.excel:
        path = feed.get_data_excel(args.start, args.end, args.freq, [args.ticker])
        print(f"File Excel creato: {path}")
    else:
        print("(skipped — passa --excel per abilitare l'export)")

    # ------------------------------------------------------------------
    # 4. MarketData — accesso bar-by-bar
    # ------------------------------------------------------------------
    section("4. MarketData — accesso bar-by-bar")
    md = MarketData.from_datafeed(feed, args.start, args.end, args.freq, [args.ticker])

    print(f"Simboli caricati : {md.symbols()}")
    print(f"Data corrente    : {md.current_date()}")
    print(f"Ticker presente  : {md.has_symbol(args.ticker)}")

    # Prezzo corrente (bar piu' recente)
    close_now = md.price(args.ticker, "close")
    print(f"\nClose corrente (T)   : {close_now}")

    # Prezzo 1 e 2 barre fa
    close_t1 = md.price(args.ticker, "close", 1)
    close_t2 = md.price(args.ticker, "close", 2)
    print(f"Close T-1            : {close_t1}")
    print(f"Close T-2            : {close_t2}")

    # OHLC corrente
    ohlc = md.ohlc(args.ticker)
    print(f"\nOHLC corrente        : {ohlc}")

    # Volume corrente
    vol = md.volume(args.ticker)
    print(f"Volume corrente      : {vol}")

    # Ultimi 5 close come Series
    last5_close = md.price(args.ticker, "close", slice(None, 5))
    print(f"\nUltimi 5 close (Series):\n{last5_close}")

    # Ultimi 5 bar come DataFrame (date + OHLCV)
    last5_df = md.get(args.ticker, slice(None, 5))
    print(f"\nUltimi 5 bar (DataFrame):\n{last5_df}")

    # Bar corrente per tutti i ticker
    current_bar = md.get_current_bar()
    print(f"\nBar corrente (tutti i ticker):\n{current_bar}")


if __name__ == "__main__":
    main()
