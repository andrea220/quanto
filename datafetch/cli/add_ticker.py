#!/usr/bin/env python3
"""
Script semplice per aggiungere ticker all'anagrafica.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datafetch.ticker_manager import TickerManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Aggiungi ticker all\'anagrafica',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggiungi ticker con source IB
  python -m datafetch.cli.add_ticker SPY --source ib

  # Aggiungi ticker con source IB e Bloomberg
  python -m datafetch.cli.add_ticker SPY --source ib --bbg-ticker "SPY US Equity"
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
        '--ib-ticker',
        dest='ib_ticker',
        help='IB ticker (se diverso da ticker)'
    )
    
    parser.add_argument(
        '--bbg-ticker',
        dest='bbg_ticker',
        help='Bloomberg ticker (es. "AAPL US Equity")'
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
    
    # Determine IB ticker
    ib_ticker = args.ib_ticker or args.ticker
    
    try:
        # Check if ticker exists
        existing = manager.get_ticker(args.ticker)
        
        if existing:
            # Update existing ticker
            updates = {}
            if args.source == 'ib':
                updates['ib_ticker'] = ib_ticker
            if args.bbg_ticker:
                updates['bbg_ticker'] = args.bbg_ticker
            
            manager.update_ticker(args.ticker, **updates)
            print(f"Ticker {args.ticker} aggiornato nell'anagrafica")
        else:
            # Add new ticker
            manager.add_ticker(
                ticker=args.ticker,
                ib_ticker=ib_ticker if args.source == 'ib' else None,
                bbg_ticker=args.bbg_ticker if args.source == 'bbg' else None,
                name=None,
                exchange=None,
                currency=None,
                asset_type=None
            )
            print(f"Ticker {args.ticker} aggiunto all'anagrafica")
    
    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
