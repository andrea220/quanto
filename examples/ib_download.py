"""
Scarica dati IB per un ticker.

Uso:
    python examples/ib_download.py SPY
    python examples/ib_download.py SPY --port 7496
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datafetch.ib.workflow import download_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# ib_insync logga posizioni e portfolio alla connessione — non ci servono
logging.getLogger("ib_insync").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Scarica dati IB per un ticker")
    parser.add_argument("ticker", help="Simbolo (es. SPY, AAPL)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=1, dest="client_id")
    args = parser.parse_args()

    try:
        download_ticker(args.ticker, host=args.host, port=args.port, client_id=args.client_id)
        print(f"Download completato per '{args.ticker.upper()}'")
    except ValueError as e:
        print(f"Errore anagrafica: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
