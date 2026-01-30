"""
Download and update ticker data module for Bloomberg.

This module provides functions to download and update EOD and intraday data
for stock tickers from Bloomberg, storing them in an organized directory structure.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from datafetch.bbg.connection import BloombergConnection
from datafetch.bbg.fetch import HistoricalDataFetcher
from datafetch.ib.utils import validate_symbol, load_from_parquet

# Import normalization function from IB module (same structure)
from datafetch.ib.download_data import _normalize_dataframe, _export_to_annual_files, _get_last_saved_date, _ticker_exists, _get_ticker_directory

# Setup logging
logger = logging.getLogger(__name__)

# Data directory (same as IB)
DATA_DIR = Path("database")
DATA_DIR.mkdir(exist_ok=True)


def download_data(
    ticker: str,
    bbg_ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    connection: Optional[BloombergConnection] = None
) -> Tuple[Path, Path]:
    """
    Download complete historical data for a new ticker from Bloomberg.
    
    This function downloads:
    1. EOD (End of Day) data with daily bars
    2. Intraday data with 1-minute bars
    
    Data is saved in the structure: database/TICKER/
    
    Args:
        ticker: Base ticker symbol (e.g., 'AAPL')
        bbg_ticker: Bloomberg ticker symbol (e.g., 'AAPL US Equity')
        start_date: Start date for historical data (YYYY-MM-DD). Defaults to 20 years ago.
        end_date: End date for historical data (YYYY-MM-DD). Defaults to today.
        connection: Optional BloombergConnection instance
        
    Returns:
        tuple: (eod_filepath, intraday_filepath) - both point to ticker_dir
        
    Raises:
        ValueError: If ticker already exists (use update_ticker instead)
        RuntimeError: If download fails
        
    Example:
        >>> eod_path, intraday_path = download_data('AAPL', 'AAPL US Equity')
        >>> print(f"EOD data saved to: {eod_path}")
        >>> print(f"Intraday data saved to: {intraday_path}")
    """
    ticker = validate_symbol(ticker)
    
    # Check if ticker already exists
    if _ticker_exists(ticker):
        raise ValueError(
            f"Ticker {ticker} already exists. Use update_ticker() to update existing data."
        )
    
    logger.info(f"Starting download for new ticker: {ticker} (Bloomberg: {bbg_ticker})")
    
    # Create ticker directory
    ticker_dir = _get_ticker_directory(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {ticker_dir}")
    
    # Initialize connection and fetcher
    conn = connection or BloombergConnection()
    fetcher = HistoricalDataFetcher(connection=conn)
    
    try:
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().date()
        else:
            end_date = pd.to_datetime(end_date).date()
        
        if start_date is None:
            # Default: 20 years of EOD data
            start_date = end_date - timedelta(days=20*365)
        else:
            start_date = pd.to_datetime(start_date).date()
        
        # Download EOD data
        logger.info(f"Downloading EOD data for {ticker} (from {start_date} to {end_date})")
        df_eod_raw = fetcher.fetch(
            symbol=bbg_ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe='1 day'
        )
        
        if not df_eod_raw.empty:
            # Normalize to standard format (same as IB)
            df_eod = _normalize_dataframe(df_eod_raw, ticker, 'eod')
            # Export to annual files
            _export_to_annual_files(df_eod, ticker_dir)
            logger.info(f"Saved {len(df_eod)} EOD bars to annual files")
        else:
            logger.warning(f"No EOD data retrieved for {ticker}")
        
        # Download intraday data (1 minute) - last 6 months
        logger.info(f"Downloading 1-minute intraday data for {ticker} (last 6 months)")
        
        intraday_start = end_date - timedelta(days=180)  # 6 months
        df_intraday_raw = fetcher.fetch(
            symbol=bbg_ticker,
            start_date=intraday_start,
            end_date=end_date,
            timeframe='1 min'
        )
        
        if not df_intraday_raw.empty:
            # Normalize to standard format (same as IB)
            df_intraday = _normalize_dataframe(df_intraday_raw, ticker, 'intraday')
            # Export to annual files
            _export_to_annual_files(df_intraday, ticker_dir)
            logger.info(f"Saved {len(df_intraday)} intraday bars to annual files")
        else:
            logger.warning(f"No intraday data retrieved for {ticker}")
        
        logger.info(f"Successfully downloaded data for {ticker}")
        # Return ticker_dir as both paths (for compatibility, but files are now annual)
        return ticker_dir, ticker_dir
        
    except Exception as e:
        logger.error(f"Failed to download data for {ticker}: {e}")
        raise RuntimeError(f"Failed to download Bloomberg data for {ticker}: {e}")


def update_ticker(
    ticker: str,
    bbg_ticker: str,
    connection: Optional[BloombergConnection] = None
) -> Tuple[Path, Path]:
    """
    Update existing ticker data with latest available data from Bloomberg.
    
    This function:
    1. Downloads new EOD data since last saved date
    2. Downloads new intraday data (last 30 days)
    3. Merges with existing data and saves to annual files
    
    Args:
        ticker: Stock ticker symbol
        bbg_ticker: Bloomberg ticker symbol
        connection: Optional BloombergConnection instance
        
    Returns:
        tuple: (eod_filepath, intraday_filepath) - both point to ticker_dir
        
    Raises:
        ValueError: If ticker doesn't exist
        RuntimeError: If update fails
    """
    ticker = validate_symbol(ticker)
    
    # Check if ticker exists
    if not _ticker_exists(ticker):
        raise ValueError(
            f"Ticker {ticker} does not exist. Use download_data() to download initial data."
        )
    
    logger.info(f"Updating data for existing ticker: {ticker} (Bloomberg: {bbg_ticker})")
    
    ticker_dir = _get_ticker_directory(ticker)
    conn = connection or BloombergConnection()
    fetcher = HistoricalDataFetcher(connection=conn)
    
    try:
        # Get last saved date
        last_date = _get_last_saved_date(ticker_dir)
        end_date = datetime.now().date()
        
        if last_date:
            # Update from last saved date
            start_date = last_date
            logger.info(f"Updating EOD data from {last_date} to {end_date}")
        else:
            # No existing data, download full history
            start_date = end_date - timedelta(days=20*365)
            logger.info(f"No existing data found, downloading full EOD history")
        
        # Update EOD data
        if start_date < end_date:
            df_eod_raw = fetcher.fetch(
                symbol=bbg_ticker,
                start_date=start_date,
                end_date=end_date,
                timeframe='1 day'
            )
            if not df_eod_raw.empty:
                df_eod = _normalize_dataframe(df_eod_raw, ticker, 'eod')
                _export_to_annual_files(df_eod, ticker_dir)
        
        # Update intraday data (last 30 days)
        intraday_start = end_date - timedelta(days=30)
        df_intraday_raw = fetcher.fetch(
            symbol=bbg_ticker,
            start_date=intraday_start,
            end_date=end_date,
            timeframe='1 min'
        )
        if not df_intraday_raw.empty:
            df_intraday = _normalize_dataframe(df_intraday_raw, ticker, 'intraday')
            _export_to_annual_files(df_intraday, ticker_dir)
        
        logger.info(f"Successfully updated data for {ticker}")
        # Return ticker_dir as both paths (for compatibility, but files are now annual)
        return ticker_dir, ticker_dir
        
    except Exception as e:
        logger.error(f"Failed to update data for {ticker}: {e}")
        raise RuntimeError(f"Failed to update Bloomberg data for {ticker}: {e}")


def get_or_update_ticker(
    ticker: str,
    bbg_ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    connection: Optional[BloombergConnection] = None
) -> Tuple[Path, Path]:
    """
    Smart function that downloads or updates ticker data based on existence.
    
    This function automatically:
    - Calls download_data() if ticker doesn't exist
    - Calls update_ticker() if ticker already exists
    
    Args:
        ticker: Base ticker symbol (e.g., 'AAPL')
        bbg_ticker: Bloomberg ticker symbol (e.g., 'AAPL US Equity')
        start_date: Start date for initial download (default: 20 years ago)
        end_date: End date for initial download (default: today)
        connection: Optional BloombergConnection instance
        
    Returns:
        tuple: (eod_filepath, intraday_filepath)
        
    Example:
        >>> eod_path, intraday_path = get_or_update_ticker('AAPL', 'AAPL US Equity')
        >>> # Automatically downloads if new, updates if exists
    """
    ticker = validate_symbol(ticker)
    
    if _ticker_exists(ticker):
        logger.info(f"Ticker {ticker} exists, updating...")
        return update_ticker(ticker, bbg_ticker, connection)
    else:
        logger.info(f"Ticker {ticker} does not exist, downloading...")
        return download_data(ticker, bbg_ticker, start_date, end_date, connection)
