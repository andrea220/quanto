"""
Historical data fetcher for OHLCV data from Interactive Brokers.
"""

import logging
from typing import Optional, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
from ib_insync import Stock, IB, util

from datafetch.ib.connection import IBConnection
from datafetch.ib.utils import (
    validate_symbol,
    validate_timeframe,
    validate_duration,
    generate_filename,
    save_to_parquet
)
from config.ib_config import get_config


# Setup logging
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetch historical OHLCV data from Interactive Brokers.
    
    This class provides methods to retrieve historical bar data for stocks
    with configurable timeframes and durations.
    
    Example:
        >>> fetcher = HistoricalDataFetcher()
        >>> df = fetcher.fetch('AAPL', timeframe='1 min', duration='1 D')
        >>> print(df.head())
    """
    
    def __init__(
        self,
        connection: Optional[IBConnection] = None,
        exchange: Optional[str] = None,
        currency: Optional[str] = None
    ):
        """
        Initialize the historical data fetcher.
        
        Args:
            connection: Optional IBConnection instance. If None, creates a new one.
            exchange: Default exchange for stocks. Defaults to config value.
            currency: Default currency. Defaults to config value.
        """
        self.config = get_config()
        self.connection = connection or IBConnection()
        self.exchange = exchange or self.config['default_exchange']
        self.currency = currency or self.config['default_currency']
        
        # Rate limiting
        self._request_count = 0
        self._request_times = []
        
        logger.info(f"HistoricalDataFetcher initialized - Exchange: {self.exchange}, Currency: {self.currency}")
    
    def _check_rate_limit(self):
        """
        Check and enforce rate limiting for historical data requests.
        
        IB limits historical data requests to approximately 60 requests per 10 minutes.
        
        Raises:
            RuntimeError: If rate limit is exceeded
        """
        now = datetime.now()
        interval = self.config['rate_limit_interval']
        max_requests = self.config['max_requests_per_interval']
        
        # Remove old requests outside the interval
        self._request_times = [
            t for t in self._request_times
            if (now - t).total_seconds() < interval
        ]
        
        if len(self._request_times) >= max_requests:
            wait_time = interval - (now - self._request_times[0]).total_seconds()
            logger.warning(
                f"Rate limit reached ({max_requests} requests per {interval}s). "
                f"Wait {wait_time:.0f} seconds."
            )
            raise RuntimeError(
                f"Rate limit exceeded. Please wait {wait_time:.0f} seconds before making more requests."
            )
        
        self._request_times.append(now)
    
    def fetch(
        self,
        symbol: str,
        timeframe: str = '1 min',
        duration: str = '1 D',
        end_datetime: Union[str, datetime, None] = '',
        what_to_show: str = 'TRADES',
        use_rth: bool = True,
        exchange: Optional[str] = None,
        currency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            timeframe: Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
            duration: Duration of historical data (e.g., '1 D', '5 D', '1 W', '1 M')
            end_datetime: End date/time for historical data. Empty string means now.
            what_to_show: Type of data ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            use_rth: If True, only regular trading hours data
            exchange: Optional exchange override
            currency: Optional currency override
            
        Returns:
            pd.DataFrame: OHLCV data with DateTimeIndex
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If rate limit is exceeded or fetch fails
        """
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        duration = validate_duration(duration)
        
        # Check rate limit
        self._check_rate_limit()
        
        # Use provided or default exchange/currency
        exchange = exchange or self.exchange
        currency = currency or self.currency
        
        logger.info(
            f"Fetching historical data: {symbol} | "
            f"Timeframe: {timeframe} | Duration: {duration} | "
            f"Exchange: {exchange} | Currency: {currency}"
        )
        
        try:
            # Get IB connection
            ib = self.connection.get_ib()
            
            # Create stock contract
            contract = Stock(symbol, exchange, currency)
            
            # Qualify contract (resolve to specific contract)
            qualified_contracts = ib.qualifyContracts(contract)
            
            if not qualified_contracts:
                raise ValueError(f"Could not qualify contract for symbol: {symbol}")
            
            contract = qualified_contracts[0]
            logger.info(f"Qualified contract: {contract}")
            
            # Request historical data
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,  # Format dates as strings
                keepUpToDate=False
            )
            
            if not bars:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = util.df(bars)
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.index.name = 'datetime'
            
            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch historical data: {e}")
    
    def fetch_and_save(
        self,
        symbol: str,
        timeframe: str = '1 min',
        duration: str = '1 D',
        directory: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> tuple[pd.DataFrame, Path]:
        """
        Fetch historical data and save to Parquet file.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
            duration: Duration of historical data (e.g., '1 D', '5 D', '1 W')
            directory: Optional directory to save file. Defaults to 'data/'
            filename: Optional custom filename. If None, auto-generates.
            **kwargs: Additional arguments passed to fetch()
            
        Returns:
            tuple: (DataFrame, Path to saved file)
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If fetch or save fails
        """
        # Fetch data
        df = self.fetch(symbol, timeframe, duration, **kwargs)
        
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return df, None
        
        # Generate filename if not provided
        if filename is None:
            filename = generate_filename(symbol, 'historical', timeframe)
        
        # Save to Parquet
        filepath = save_to_parquet(df, filename, directory)
        
        logger.info(f"Historical data saved: {filepath}")
        
        return df, filepath
    
    def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: str = '1 min',
        duration: str = '1 D',
        **kwargs
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            timeframe: Bar size
            duration: Duration of historical data
            **kwargs: Additional arguments passed to fetch()
            
        Returns:
            dict: Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}...")
                df = self.fetch(symbol, timeframe, duration, **kwargs)
                results[symbol] = df
                
                # Small delay between requests to be respectful
                util.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def fetch_multiple_and_save(
        self,
        symbols: list[str],
        timeframe: str = '1 min',
        duration: str = '1 D',
        directory: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> dict[str, tuple[pd.DataFrame, Path]]:
        """
        Fetch and save historical data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            timeframe: Bar size
            duration: Duration of historical data
            directory: Optional directory to save files
            **kwargs: Additional arguments passed to fetch()
            
        Returns:
            dict: Dictionary mapping symbols to (DataFrame, filepath) tuples
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching and saving data for {symbol}...")
                df, filepath = self.fetch_and_save(
                    symbol, timeframe, duration, directory, **kwargs
                )
                results[symbol] = (df, filepath)
                
                # Small delay between requests
                util.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to fetch and save data for {symbol}: {e}")
                results[symbol] = (pd.DataFrame(), None)
        
        return results

