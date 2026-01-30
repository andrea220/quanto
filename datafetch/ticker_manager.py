"""
Ticker universe manager for managing ticker mappings between IB and Bloomberg.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading

# Setup logging
logger = logging.getLogger(__name__)

# Default path to ticker universe JSON
DEFAULT_UNIVERSE_PATH = Path(__file__).parent.parent / "config" / "anag.json"


class TickerManager:
    """
    Manager for ticker universe JSON file.
    
    Provides thread-safe operations for reading and writing ticker mappings.
    """
    
    def __init__(self, universe_path: Optional[Path] = None):
        """
        Initialize the TickerManager.
        
        Args:
            universe_path: Path to ticker universe JSON file. 
                          Defaults to config/anag.json
        """
        self.universe_path = Path(universe_path) if universe_path else DEFAULT_UNIVERSE_PATH
        self._lock = threading.Lock()
        
        # Ensure parent directory exists
        self.universe_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not self.universe_path.exists():
            self._initialize_universe()
    
    def _initialize_universe(self):
        """Initialize empty universe file."""
        with self._lock:
            with open(self.universe_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            logger.info(f"Initialized ticker universe file: {self.universe_path}")
    
    def load_universe(self) -> Dict[str, Any]:
        """
        Load ticker universe from JSON file.
        
        Returns:
            dict: Ticker universe dictionary
        """
        with self._lock:
            try:
                with open(self.universe_path, 'r', encoding='utf-8') as f:
                    universe = json.load(f)
                # Remove metadata keys (starting with _)
                return {k: v for k, v in universe.items() if not k.startswith('_')}
            except FileNotFoundError:
                logger.warning(f"Universe file not found: {self.universe_path}")
                return {}
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing universe JSON: {e}")
                raise ValueError(f"Invalid JSON in universe file: {e}")
    
    def save_universe(self, universe: Dict[str, Any]) -> None:
        """
        Save ticker universe to JSON file.
        
        Args:
            universe: Ticker universe dictionary to save
        """
        with self._lock:
            # Preserve metadata keys if they exist
            existing_universe = {}
            if self.universe_path.exists():
                try:
                    with open(self.universe_path, 'r', encoding='utf-8') as f:
                        existing_universe = json.load(f)
                except:
                    pass
            
            # Merge metadata back
            metadata = {k: v for k, v in existing_universe.items() if k.startswith('_')}
            merged_universe = {**metadata, **universe}
            
            # Write to temporary file first, then rename (atomic write)
            temp_path = self.universe_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(merged_universe, f, indent=2, ensure_ascii=False)
            
            temp_path.replace(self.universe_path)
            logger.debug(f"Saved ticker universe to {self.universe_path}")
    
    def add_ticker(
        self,
        ticker: str,
        ib_ticker: Optional[str] = None,
        bbg_ticker: Optional[str] = None,
        name: Optional[str] = None,
        exchange: Optional[str] = None,
        currency: Optional[str] = None,
        asset_type: Optional[str] = None
    ) -> None:
        """
        Add a new ticker to the universe.
        
        Args:
            ticker: Primary ticker symbol (key in universe)
            ib_ticker: Interactive Brokers ticker symbol (optional)
            bbg_ticker: Bloomberg ticker symbol (optional)
            name: Company/Asset name (optional)
            exchange: Exchange code (optional)
            currency: Currency code (optional)
            asset_type: Asset type (optional)
        """
        ticker = ticker.upper().strip()
        ib_ticker = ib_ticker.upper().strip() if ib_ticker else None
        
        universe = self.load_universe()
        
        if ticker in universe:
            logger.warning(f"Ticker {ticker} already exists. Use update_ticker() to modify.")
            raise ValueError(f"Ticker {ticker} already exists in universe")
        
        # Always include all standard fields, even if None/empty
        ticker_data = {
            "ib_ticker": ib_ticker,
            "bbg_ticker": bbg_ticker.strip() if bbg_ticker else None,
            "name": name.strip() if name else None,
            "exchange": exchange.strip().upper() if exchange else None,
            "currency": currency.strip().upper() if currency else None,
            "asset_type": asset_type.strip().upper() if asset_type else None,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
        
        universe[ticker] = ticker_data
        self.save_universe(universe)
        logger.info(f"Added ticker {ticker} to universe")
    
    def update_ticker(
        self,
        ticker: str,
        **updates
    ) -> None:
        """
        Update an existing ticker in the universe.
        
        Args:
            ticker: Ticker symbol to update
            **updates: Fields to update (ib_ticker, bbg_ticker, name, etc.)
        """
        ticker = ticker.upper().strip()
        
        universe = self.load_universe()
        
        if ticker not in universe:
            logger.warning(f"Ticker {ticker} not found. Use add_ticker() to add new ticker.")
            raise ValueError(f"Ticker {ticker} not found in universe")
        
        # Ensure all standard fields exist (initialize with None if missing)
        standard_fields = ['ib_ticker', 'bbg_ticker', 'name', 'exchange', 'currency', 'asset_type']
        for field in standard_fields:
            if field not in universe[ticker]:
                universe[ticker][field] = None
        
        # Update fields
        for key, value in updates.items():
            if value is not None:
                if key in ['ib_ticker', 'bbg_ticker', 'exchange', 'currency', 'asset_type']:
                    universe[ticker][key] = str(value).strip().upper()
                elif key == 'name':
                    universe[ticker][key] = str(value).strip()
                else:
                    universe[ticker][key] = value
            elif key in standard_fields:
                # Explicitly set to None if provided as None
                universe[ticker][key] = None
        
        # Update last_updated timestamp
        universe[ticker]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        self.save_universe(universe)
        logger.info(f"Updated ticker {ticker} in universe")
    
    def get_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker information from universe.
        
        Args:
            ticker: Ticker symbol to retrieve
            
        Returns:
            dict: Ticker information, or None if not found
        """
        ticker = ticker.upper().strip()
        universe = self.load_universe()
        return universe.get(ticker)
    
    def list_tickers(self) -> List[str]:
        """
        List all tickers in the universe.
        
        Returns:
            list: List of ticker symbols
        """
        universe = self.load_universe()
        return sorted(universe.keys())
    
    def remove_ticker(self, ticker: str) -> None:
        """
        Remove a ticker from the universe.
        
        Args:
            ticker: Ticker symbol to remove
        """
        ticker = ticker.upper().strip()
        
        universe = self.load_universe()
        
        if ticker not in universe:
            logger.warning(f"Ticker {ticker} not found in universe")
            raise ValueError(f"Ticker {ticker} not found in universe")
        
        del universe[ticker]
        self.save_universe(universe)
        logger.info(f"Removed ticker {ticker} from universe")
    
    def validate_ticker(self, ticker: str, provider: str = "ib") -> bool:
        """
        Validate ticker format for a specific provider.
        
        Args:
            ticker: Ticker symbol to validate
            provider: Provider name ('ib' or 'bbg')
            
        Returns:
            bool: True if ticker format is valid
        """
        ticker = ticker.strip()
        
        if provider.lower() == "ib":
            # IB tickers are typically alphanumeric, no spaces
            return bool(ticker and ticker.replace('_', '').replace('-', '').isalnum())
        elif provider.lower() == "bbg":
            # Bloomberg tickers typically have format: SYMBOL EXCHANGE TYPE
            # e.g., "AAPL US Equity", "ES1 Index"
            return bool(ticker and len(ticker.split()) >= 2)
        else:
            logger.warning(f"Unknown provider: {provider}")
            return False
