#!/usr/bin/env python3
"""
CLI script to refresh data for a single ticker from a selected provider.
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


def refresh_ib_ticker(ticker: str, ticker_info: dict, eod_duration: str = '20 Y', intraday_duration: str = '2 Y'):
    """
    Refresh ticker data from Interactive Brokers.
    
    Args:
        ticker: Ticker symbol
        ticker_info: Ticker information from universe
        eod_duration: Duration for EOD data download
        intraday_duration: Duration for intraday data download
    """
    from datafetch.ib.download_data import get_or_update_ticker
    from datafetch.ib.connection import IBConnection
    
    ib_ticker = ticker_info.get('ib_ticker', ticker)
    exchange = ticker_info.get('exchange')
    currency = ticker_info.get('currency')
    
    logger.info(f"Refreshing {ticker} from IB (ticker: {ib_ticker})...")
    
    try:
        connection = IBConnection()
        eod_path, intraday_path = get_or_update_ticker(
            ticker=ib_ticker,
            eod_duration=eod_duration,
            intraday_duration=intraday_duration,
            connection=connection,
            exchange=exchange,
            currency=currency
        )
        logger.info(f"Successfully refreshed {ticker} from IB")
        return True
    except Exception as e:
        logger.error(f"Failed to refresh {ticker} from IB: {e}")
        return False
    finally:
        try:
            connection.disconnect()
        except:
            pass


def refresh_bbg_ticker(ticker: str, ticker_info: dict):
    """
    Refresh ticker data from Bloomberg.
    
    Args:
        ticker: Ticker symbol
        ticker_info: Ticker information from universe
    """
    # TODO: Implement Bloomberg data fetching
    logger.warning(f"Bloomberg provider not yet implemented for {ticker}")
    logger.info(f"Would refresh {ticker} from Bloomberg (ticker: {ticker_info.get('bbg_ticker', 'N/A')})")
    return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Refresh data for a single ticker from a selected provider',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh ticker from IB
  python -m datafetch.cli.refresh_ticker AAPL --provider ib

  # Refresh ticker from Bloomberg
  python -m datafetch.cli.refresh_ticker AAPL --provider bbg
        """
    )
    
    parser.add_argument(
        'ticker',
        help='Ticker symbol to refresh'
    )
    
    parser.add_argument(
        '--provider',
        choices=['ib', 'bbg'],
        required=True,
        help='Data provider to use (ib or bbg)'
    )
    
    parser.add_argument(
        '--universe-path',
        dest='universe_path',
        type=Path,
        help='Path to ticker universe JSON file (default: config/anag.json)'
    )
    
    parser.add_argument(
        '--eod-duration',
        dest='eod_duration',
        default='20 Y',
        help='Duration for EOD data download (default: 20 Y - maximum available)'
    )
    
    parser.add_argument(
        '--intraday-duration',
        dest='intraday_duration',
        default='30 D',
        help='Duration for intraday data download (default: 30 D - IB limit for 1-min bars)'
    )
    
    args = parser.parse_args()
    
    # Initialize ticker manager
    manager = TickerManager(universe_path=args.universe_path)
    
    # Get ticker info from universe
    ticker_info = manager.get_ticker(args.ticker)
    
    if not ticker_info:
        logger.warning(f"Ticker {args.ticker} not found in universe")
        logger.info("Creating minimal ticker info...")
        ticker_info = {
            'ib_ticker': args.ticker,
            'bbg_ticker': None
        }
    
    # Refresh based on provider
    success = False
    if args.provider == 'ib':
        success = refresh_ib_ticker(
            args.ticker, 
            ticker_info,
            eod_duration=args.eod_duration,
            intraday_duration=args.intraday_duration
        )
    elif args.provider == 'bbg':
        success = refresh_bbg_ticker(args.ticker, ticker_info)
    
    if success:
        logger.info(f"Successfully refreshed {args.ticker} from {args.provider.upper()}")
        sys.exit(0)
    else:
        logger.error(f"Failed to refresh {args.ticker} from {args.provider.upper()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
