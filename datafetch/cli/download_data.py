#!/usr/bin/env python3
"""
Script semplice per scaricare dati storici per un ticker.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datafetch.ticker_manager import TickerManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_ib(ticker: str, ticker_info: dict):
    """Scarica dati da Interactive Brokers."""
    from datafetch.ib.download_data import get_or_update_ticker
    from datafetch.ib.connection import IBConnection
    
    ib_ticker = ticker_info.get('ib_ticker', ticker)
    exchange = ticker_info.get('exchange')
    currency = ticker_info.get('currency')
    
    logger.info(f"Scaricando dati per {ticker} da IB (ticker: {ib_ticker})...")
    
    connection = IBConnection()
    try:
        get_or_update_ticker(
            ticker=ib_ticker,
            connection=connection,
            exchange=exchange,
            currency=currency
        )
        logger.info(f"Dati scaricati con successo per {ticker}")
        return True
    except Exception as e:
        logger.error(f"Errore durante il download: {e}")
        return False
    finally:
        connection.disconnect()


def download_bbg(ticker: str, ticker_info: dict):
    """Scarica dati da Bloomberg."""
    from datafetch.bbg.download_data import get_or_update_ticker
    from datafetch.bbg.connection import BloombergConnection
    
    bbg_ticker = ticker_info.get('bbg_ticker')
    if not bbg_ticker:
        logger.error(f"Bloomberg ticker non trovato per {ticker} nell'anagrafica")
        logger.info(f"Aggiungi il bbg_ticker con: python -m datafetch.cli.add_ticker {ticker} --source bbg --bbg-ticker 'TICKER US Equity'")
        return False
    
    logger.info(f"Scaricando dati per {ticker} da Bloomberg (ticker: {bbg_ticker})...")
    
    connection = BloombergConnection()
    try:
        get_or_update_ticker(
            ticker=ticker,
            bbg_ticker=bbg_ticker,
            connection=connection
        )
        logger.info(f"Dati scaricati con successo per {ticker}")
        return True
    except Exception as e:
        logger.error(f"Errore durante il download: {e}")
        return False
    finally:
        connection.disconnect()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Scarica dati storici per un ticker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scarica dati da IB
  python -m datafetch.cli.download_data SPY --source ib

  # Scarica dati da Bloomberg
  python -m datafetch.cli.download_data SPY --source bbg
        """
    )
    
    parser.add_argument(
        'ticker',
        help='Ticker symbol'
    )
    
    parser.add_argument(
        '--source',
        choices=['ib', 'bbg'],
        required=True,
        help='Source provider (ib o bbg)'
    )
    
    parser.add_argument(
        '--anag-path',
        dest='anag_path',
        type=Path,
        help='Path to anagrafica JSON file (default: config/anag.json)'
    )
    
    args = parser.parse_args()
    
    # Initialize ticker manager
    manager = TickerManager(universe_path=args.anag_path)
    
    # Get ticker info from anagrafica
    ticker_info = manager.get_ticker(args.ticker)
    
    if not ticker_info:
        logger.error(f"Ticker {args.ticker} non trovato nell'anagrafica")
        logger.info(f"Usa 'python -m datafetch.cli.add_ticker {args.ticker} --source {args.source}' per aggiungerlo")
        sys.exit(1)
    
    # Download based on source
    success = False
    if args.source == 'ib':
        success = download_ib(args.ticker, ticker_info)
    elif args.source == 'bbg':
        success = download_bbg(args.ticker, ticker_info)
    
    if success:
        logger.info(f"Download completato per {args.ticker}")
        sys.exit(0)
    else:
        logger.error(f"Download fallito per {args.ticker}")
        sys.exit(1)


if __name__ == '__main__':
    main()
