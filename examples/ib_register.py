"""
Registra un ticker in anag.json popolando i campi da IB (operazione one-time).

Uso:
    python examples/ib_register.py MSFT
    python examples/ib_register.py MSFT --port 7496
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datafetch.ib.workflow import register_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("ib_insync").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Registra un ticker in anag.json da IB")
    parser.add_argument("ticker", help="Simbolo (es. MSFT, AAPL)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=1, dest="client_id")
    args = parser.parse_args()

    try:
        register_ticker(args.ticker, host=args.host, port=args.port, client_id=args.client_id)
        print(f"'{args.ticker.upper()}' registrato in anag.json")
    except RuntimeError as e:
        print(f"Errore IB: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
