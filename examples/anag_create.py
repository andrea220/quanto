"""
Esempio: creare una nuova entry nell'anagrafica.

Uso:
    python examples/anag_create.py MSFT
    python examples/anag_create.py MSFT --anag config/anag.json

Dopo aver creato l'entry, compila manualmente bbg_ticker (e gli altri
campi che ti servono) in config/anag.json, poi lancia download.bat.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datafetch.ticker_manager import TickerManager


def main():
    parser = argparse.ArgumentParser(description="Crea una entry vuota in anag.json")
    parser.add_argument("ticker", help="Simbolo del ticker (es. MSFT, SX5E)")
    parser.add_argument(
        "--anag",
        default="config/anag.json",
        help="Path al file anag.json (default: config/anag.json)",
    )
    args = parser.parse_args()

    manager = TickerManager(universe_path=Path(args.anag))

    try:
        manager.create_anag(args.ticker)
        print(f"Entry creata per '{args.ticker.upper()}' in {args.anag}")
        print("Compila bbg_ticker (e gli altri campi) a mano nel JSON, poi lancia download.bat")
    except ValueError as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
