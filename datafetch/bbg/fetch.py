"""
Historical data fetcher for OHLCV data from Bloomberg using xbbg.
"""

import logging
from typing import Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from datafetch.bbg.connection import BloombergConnection

# Setup logging
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetch historical OHLCV data from Bloomberg using xbbg.
    
    This class provides methods to retrieve historical bar data for stocks
    with configurable timeframes and date ranges.
    
    Example:
        >>> fetcher = HistoricalDataFetcher()
        >>> df = fetcher.fetch('AAPL US Equity', start_date='2024-01-01', end_date='2024-12-31')
        >>> print(df.head())
    """
    
    def __init__(
        self,
        connection: Optional[BloombergConnection] = None
    ):
        """
        Initialize the historical data fetcher.
        
        Args:
            connection: Optional BloombergConnection instance. If None, creates a new one.
        """
        self.connection = connection or BloombergConnection()
        self.connection.connect()
        
        logger.info("HistoricalDataFetcher initialized for Bloomberg")
    
    def fetch(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = '1 day',
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a Bloomberg ticker.
        
        Args:
            symbol: Bloomberg ticker symbol (e.g., 'AAPL US Equity')
            start_date: Start date for historical data (string 'YYYY-MM-DD' or datetime)
            end_date: End date for historical data (string 'YYYY-MM-DD' or datetime). Defaults to today.
            timeframe: Bar size ('1 day' for EOD, '1 min' for intraday)
            **kwargs: Additional arguments (not used currently)
            
        Returns:
            pd.DataFrame: OHLCV data with DateTimeIndex
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If fetch fails
        """
        from xbbg import blp
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Set default end_date to today if not provided
        if end_date is None:
            end_date = datetime.now().date()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
        
        # Set default start_date if not provided (1 year back)
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
        
        # Convert dates to strings for xbbg
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        logger.info(
            f"Fetching Bloomberg historical data: {symbol} | "
            f"Timeframe: {timeframe} | Start: {start_date} | End: {end_date}"
        )
        
        try:
            # Determine if EOD or intraday
            if timeframe == '1 day' or 'day' in timeframe.lower():
                # EOD data using bdh (bar data historical)
                df = blp.bdh(
                    tickers=symbol,
                    flds=['open', 'high', 'low', 'close', 'volume'],
                    start_date=start_str,
                    end_date=end_str
                )
                
                # xbbg returns MultiIndex DataFrame with ticker as column level
                # Flatten the structure
                if isinstance(df.columns, pd.MultiIndex):
                    # Get the first ticker's data
                    ticker_cols = df.columns.get_level_values(0).unique()
                    if len(ticker_cols) > 0:
                        ticker_name = ticker_cols[0]
                        df = df[ticker_name].copy()
                        # Rename columns to lowercase
                        df.columns = [col.lower() for col in df.columns]
                else:
                    # Already flat, just rename to lowercase
                    df.columns = [col.lower() for col in df.columns]
                
                # Map Bloomberg field names to standard names
                # Bloomberg uses different field names for indices vs equities
                column_mapping = {
                    'px_last': 'close',  # Bloomberg uses PX_LAST for indices
                    'last_price': 'close',
                    'px_open': 'open',
                    'px_high': 'high',
                    'px_low': 'low',
                    'volume': 'volume',
                    'px_volume': 'volume',
                }
                
                # Rename columns if they match Bloomberg field names
                df.rename(columns=column_mapping, inplace=True)
                
            else:
                # Intraday data using bdib (bar data intraday)
                # For intraday, we need to fetch day by day
                df_list = []
                current_date = start_date
                
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y-%m-%d')
                    try:
                        df_day = blp.bdib(
                            ticker=symbol,
                            dt=date_str,
                            session='allday'  # or 'day' for regular trading hours
                        )
                        
                        if not df_day.empty:
                            # Handle MultiIndex columns (tuples) or regular columns
                            if isinstance(df_day.columns, pd.MultiIndex):
                                # Extract second level (column names) from MultiIndex
                                # First level is ticker, second level is field name
                                df_day.columns = df_day.columns.get_level_values(1)
                                # Rename to lowercase
                                df_day.columns = [col.lower() for col in df_day.columns]
                            else:
                                # Rename columns to lowercase
                                df_day.columns = [str(col).lower() if not isinstance(col, str) else col.lower() 
                                                  for col in df_day.columns]
                            df_list.append(df_day)
                        
                        current_date += timedelta(days=1)
                    except Exception as e:
                        logger.warning(f"Failed to fetch intraday data for {symbol} on {date_str}: {e}")
                        current_date += timedelta(days=1)
                        continue
                
                if df_list:
                    df = pd.concat(df_list, axis=0)
                    df = df.sort_index()
                else:
                    df = pd.DataFrame()
            
            if df.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            
            # Ensure we have the right columns (open, high, low, close, volume)
            # Check what columns we actually have
            actual_cols = [col.lower() for col in df.columns]
            
            # Map any remaining Bloomberg-specific column names
            column_mapping = {
                'px_last': 'close',
                'last_price': 'close',
                'px_open': 'open',
                'px_high': 'high',
                'px_low': 'low',
                'volume': 'volume',
                'px_volume': 'volume',
            }
            
            # Rename columns if they match Bloomberg field names
            rename_dict = {}
            for old_col in df.columns:
                old_lower = old_col.lower()
                if old_lower in column_mapping:
                    rename_dict[old_col] = column_mapping[old_lower]
            
            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                # Add missing columns with NaN
                for col in missing_cols:
                    df[col] = pd.NA
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            
            df.index.name = 'datetime'
            
            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch Bloomberg historical data: {e}")
    
    def fetch_and_save(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = '1 day',
        directory: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> tuple[pd.DataFrame, Path]:
        """
        Fetch historical data and save to Parquet file.
        
        Args:
            symbol: Bloomberg ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Bar size ('1 day' or '1 min')
            directory: Optional directory to save file
            filename: Optional custom filename
            **kwargs: Additional arguments passed to fetch()
            
        Returns:
            tuple: (DataFrame, Path to saved file)
        """
        # Fetch data
        df = self.fetch(symbol, start_date, end_date, timeframe, **kwargs)
        
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return df, None
        
        # Save logic would go here (similar to IB version)
        # For now, just return the DataFrame
        logger.info(f"Historical data fetched for {symbol} (not saved)")
        
        return df, None
