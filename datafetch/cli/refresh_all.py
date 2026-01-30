#!/usr/bin/env python3
"""
CLI script to refresh all tickers in the universe from a selected provider.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datafetch.ticker_manager import TickerManager
from datafetch.ib.download_data import list_available_tickers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def refresh_ib_ticker(ticker: str, ticker_info: dict):
    """
    Refresh ticker data from Interactive Brokers.
    
    Args:
        ticker: Ticker symbol
        ticker_info: Ticker information from universe
        
    Returns:
        bool: True if successful, False otherwise
    """
    from datafetch.ib.download_data import get_or_update_ticker
    from datafetch.ib.connection import IBConnection
    
    ib_ticker = ticker_info.get('ib_ticker', ticker)
    exchange = ticker_info.get('exchange')
    currency = ticker_info.get('currency')
    
    try:
        connection = IBConnection()
        eod_path, intraday_path = get_or_update_ticker(
            ticker=ib_ticker,
            connection=connection,
            exchange=exchange,
            currency=currency
        )
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
        
    Returns:
        bool: True if successful, False otherwise
    """
    # TODO: Implement Bloomberg data fetching
    logger.warning(f"Bloomberg provider not yet implemented for {ticker}")
    return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Refresh all tickers in the universe from a selected provider',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh all tickers from IB
  python -m datafetch.cli.refresh_all --provider ib

  # Refresh all tickers from Bloomberg
  python -m datafetch.cli.refresh_all --provider bbg

  # Refresh only tickers that don't exist in database
  python -m datafetch.cli.refresh_all --provider ib --new-only
        """
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
        '--new-only',
        dest='new_only',
        action='store_true',
        help='Only refresh tickers that don\'t exist in database'
    )
    
    parser.add_argument(
        '--skip-existing',
        dest='skip_existing',
        action='store_true',
        help='Skip tickers that already exist in database (opposite of --new-only)'
    )
    
    args = parser.parse_args()
    
    # Initialize ticker manager
    manager = TickerManager(universe_path=args.universe_path)
    
    # Get all tickers from universe
    tickers = manager.list_tickers()
    
    if not tickers:
        logger.warning("No tickers found in universe")
        sys.exit(0)
    
    logger.info(f"Found {len(tickers)} tickers in universe")
    
    # Filter tickers based on options
    tickers_to_process = []
    for ticker in tickers:
        ticker_info = manager.get_ticker(ticker)
        
        if args.provider == 'ib':
            # Check if IB ticker exists in database
            ib_ticker = ticker_info.get('ib_ticker', ticker)
            # Import here to avoid circular imports
            from datafetch.ib.download_data import _ticker_exists
            exists_in_db = _ticker_exists(ib_ticker)
        elif args.provider == 'bbg':
            # For Bloomberg, assume not implemented yet
            exists_in_db = False
        else:
            exists_in_db = False
        
        if args.new_only:
            # Only process if doesn't exist
            if not exists_in_db:
                tickers_to_process.append((ticker, ticker_info))
        elif args.skip_existing:
            # Skip if exists
            if not exists_in_db:
                tickers_to_process.append((ticker, ticker_info))
        else:
            # Process all
            tickers_to_process.append((ticker, ticker_info))
    
    if not tickers_to_process:
        logger.info("No tickers to process")
        sys.exit(0)
    
    logger.info(f"Processing {len(tickers_to_process)} tickers...")
    
    # Process each ticker
    success_count = 0
    fail_count = 0
    
    for ticker, ticker_info in tickers_to_process:
        logger.info(f"Processing {ticker}...")
        
        if args.provider == 'ib':
            success = refresh_ib_ticker(ticker, ticker_info)
        elif args.provider == 'bbg':
            success = refresh_bbg_ticker(ticker, ticker_info)
        else:
            logger.error(f"Unknown provider: {args.provider}")
            success = False
        
        if success:
            success_count += 1
            logger.info(f"✓ Successfully refreshed {ticker}")
        else:
            fail_count += 1
            logger.error(f"✗ Failed to refresh {ticker}")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Refresh complete: {success_count} succeeded, {fail_count} failed")
    logger.info("=" * 60)
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
