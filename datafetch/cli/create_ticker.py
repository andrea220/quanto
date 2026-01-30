#!/usr/bin/env python3
"""
CLI script to add or update tickers in the ticker universe.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datafetch.ticker_manager import TickerManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_ticker_info_from_ib(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch ticker information from Interactive Brokers.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        dict with ticker information or None if failed
    """
    try:
        from datafetch.ib.connection import IBConnection
        from ib_insync import Stock
        from config.ib_config import get_config
        
        config = get_config()
        connection = IBConnection()
        
        try:
            ib = connection.get_ib()
            
            # Create contract
            contract = Stock(ticker, config['default_exchange'], config['default_currency'])
            
            # Qualify contract to get details
            qualified = ib.qualifyContracts(contract)
            
            if not qualified:
                logger.warning(f"Could not qualify contract for {ticker}")
                return None
            
            contract = qualified[0]
            
            # Get contract details
            details_list = ib.reqContractDetails(contract)
            
            info = {
                'ib_ticker': ticker,
                'exchange': contract.exchange or config['default_exchange'],
                'currency': contract.currency or config['default_currency'],
                'name': None,
                'asset_type': contract.secType or 'STK'
            }
            
            # Try to get name from contract details
            if details_list:
                detail = details_list[0]
                if hasattr(detail, 'longName') and detail.longName:
                    info['name'] = detail.longName
                elif hasattr(detail, 'marketName') and detail.marketName:
                    info['name'] = detail.marketName
            
            return info
            
        finally:
            connection.disconnect()
            
    except Exception as e:
        logger.warning(f"Failed to fetch info from IB for {ticker}: {e}")
        return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Add or update a ticker in the ticker universe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a new ticker - automatically fetches info from IB
  python -m datafetch.cli.create_ticker SPY

  # Add a ticker with Bloomberg mapping (still fetches IB info)
  python -m datafetch.cli.create_ticker SPY --bbg "SPY US Equity"

  # Add a ticker without fetching from IB (manual entry)
  python -m datafetch.cli.create_ticker SPY --no-fetch --ib SPY --exchange ARCA

  # Update an existing ticker
  python -m datafetch.cli.create_ticker SPY --bbg "SPY US Equity" --update
        """
    )
    
    parser.add_argument(
        'ticker',
        help='Primary ticker symbol (key in universe)'
    )
    
    parser.add_argument(
        '--ib',
        dest='ib_ticker',
        help='Interactive Brokers ticker symbol (if not provided, uses ticker argument)'
    )
    
    parser.add_argument(
        '--bbg',
        dest='bbg_ticker',
        help='Bloomberg ticker symbol (e.g., "AAPL US Equity")'
    )
    
    parser.add_argument(
        '--name',
        help='Company/Asset name'
    )
    
    parser.add_argument(
        '--exchange',
        help='Exchange code (e.g., NASDAQ, NYSE)'
    )
    
    parser.add_argument(
        '--currency',
        help='Currency code (e.g., USD, EUR)'
    )
    
    parser.add_argument(
        '--asset-type',
        dest='asset_type',
        help='Asset type (e.g., EQUITY, FUTURE, OPTION)'
    )
    
    parser.add_argument(
        '--universe-path',
        dest='universe_path',
        type=Path,
        help='Path to ticker universe JSON file (default: config/anag.json)'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing ticker instead of adding new one'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate ticker format before adding/updating'
    )
    
    parser.add_argument(
        '--no-fetch',
        dest='no_fetch',
        action='store_true',
        help='Do not fetch ticker info from IB automatically'
    )
    
    args = parser.parse_args()
    
    # Initialize ticker manager
    manager = TickerManager(universe_path=args.universe_path)
    
    # Determine IB ticker (use argument if provided, otherwise use ticker)
    ib_ticker = args.ib_ticker or args.ticker
    
    # Fetch ticker info from IB if not disabled and not updating
    ib_info = None
    if not args.no_fetch and not args.update:
        logger.info(f"Fetching ticker information from IB for {ib_ticker}...")
        ib_info = fetch_ticker_info_from_ib(ib_ticker)
        if ib_info:
            logger.info(f"Found info: {ib_info}")
        else:
            logger.warning("Could not fetch info from IB, using provided/default values")
    
    # Use IB info to fill in missing parameters
    if ib_info:
        if not args.exchange:
            args.exchange = ib_info.get('exchange')
        if not args.currency:
            args.currency = ib_info.get('currency')
        if not args.name:
            args.name = ib_info.get('name')
        if not args.asset_type:
            args.asset_type = ib_info.get('asset_type')
    
    # Validate ticker format if requested
    if args.validate:
        if ib_ticker:
            if not manager.validate_ticker(ib_ticker, provider='ib'):
                logger.error(f"Invalid IB ticker format: {ib_ticker}")
                sys.exit(1)
        
        if args.bbg_ticker:
            if not manager.validate_ticker(args.bbg_ticker, provider='bbg'):
                logger.error(f"Invalid Bloomberg ticker format: {args.bbg_ticker}")
                sys.exit(1)
    
    try:
        if args.update:
            # Update existing ticker
            updates = {}
            if args.ib_ticker:
                updates['ib_ticker'] = args.ib_ticker
            if args.bbg_ticker:
                updates['bbg_ticker'] = args.bbg_ticker
            if args.name:
                updates['name'] = args.name
            if args.exchange:
                updates['exchange'] = args.exchange
            if args.currency:
                updates['currency'] = args.currency
            if args.asset_type:
                updates['asset_type'] = args.asset_type
            
            if not updates:
                logger.error("No fields to update. Specify at least one field to update.")
                sys.exit(1)
            
            manager.update_ticker(args.ticker, **updates)
            logger.info(f"Successfully updated ticker {args.ticker}")
        else:
            # Add new ticker - ib_ticker is required (use ticker if not provided)
            # Always include all fields, even if empty
            manager.add_ticker(
                ticker=args.ticker,
                ib_ticker=ib_ticker,
                bbg_ticker=args.bbg_ticker or None,  # Explicit None if not provided
                name=args.name or None,
                exchange=args.exchange or None,
                currency=args.currency or None,
                asset_type=args.asset_type or None
            )
            logger.info(f"Successfully added ticker {args.ticker}")
    
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
