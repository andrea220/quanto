"""
Download and update ticker data module.

This module provides functions to download and update EOD and intraday data
for stock tickers, storing them in an organized directory structure.
"""

import logging
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from ib_insync import Stock

from datafetch.ib.connection import IBConnection
from datafetch.ib.fetch import HistoricalDataFetcher
from datafetch.ib.utils import validate_symbol, load_from_parquet

# Setup logging
logger = logging.getLogger(__name__)

# Data directory - sempre nella root del progetto
DATA_DIR = Path(__file__).parent.parent.parent / "database"
DATA_DIR.mkdir(exist_ok=True)


def _get_ticker_directory(ticker: str) -> Path:
    """
    Get the directory path for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Path: Directory path for the ticker
    """
    ticker = validate_symbol(ticker)
    ticker_dir = DATA_DIR / ticker
    return ticker_dir


def _ticker_exists(ticker: str) -> bool:
    """
    Check if a ticker directory and data files already exist.
    
    Checks for annual parquet files in the format {ticker}/{YYYY}.parquet
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        bool: True if ticker data exists, False otherwise
    """
    ticker_dir = _get_ticker_directory(ticker)
    
    if not ticker_dir.exists():
        return False
    
    # Check if any annual parquet files exist
    parquet_files = [f for f in os.listdir(ticker_dir) if f.endswith(".parquet")]
    if not parquet_files:
        return False
    
    # Check if any file is a year file (YYYY.parquet)
    for f in parquet_files:
        base = os.path.splitext(f)[0]
        if base.isdigit():
            return True
    
    return False


def _get_last_saved_date(ticker_dir: Path) -> Optional[datetime.date]:
    """
    Get the last saved date for a ticker from annual parquet files.
    
    Args:
        ticker_dir: Directory containing ticker data files
        
    Returns:
        Last date present in the data, or None if no data exists
    """
    if not ticker_dir.exists() or not ticker_dir.is_dir():
        return None
    
    files = [f for f in os.listdir(ticker_dir) if f.endswith(".parquet")]
    years = []
    for f in files:
        base = os.path.splitext(f)[0]
        if base.isdigit():
            years.append(int(base))
    
    if not years:
        return None
    
    y_max = max(years)
    path = ticker_dir / f"{y_max}.parquet"
    if not path.exists():
        return None
    
    try:
        df = pd.read_parquet(path, columns=["date"])
    except Exception:
        return None
    
    if df.empty or "date" not in df.columns:
        return None
    
    s = pd.to_datetime(df["date"]).dt.date
    return s.max() if not s.empty else None


def _normalize_dataframe(df: pd.DataFrame, ticker: str, data_type: str) -> pd.DataFrame:
    """
    Normalize raw IB DataFrame into standard table format.
    
    Converts DataFrame with datetime index to format with separate date/time columns:
    ['date', 'time', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'insertion_time', 'type']
    
    Args:
        df: Raw DataFrame from IB with datetime index and OHLCV columns
        ticker: Ticker symbol
        data_type: Type of data ('eod' or 'intraday')
        
    Returns:
        Normalized DataFrame with standard columns
    """
    if df.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'date', 'time', 'ticker', 'open', 'high', 'low', 'close',
            'volume', 'insertion_time', 'type'
        ])
    
    # Save index name before reset
    index_name = df.index.name
    
    # Reset index to get datetime as column
    # The index name might be 'datetime' or 'date' depending on how IB returns data
    df = df.reset_index()
    
    # Find datetime column (could be 'date', 'datetime', or index name)
    datetime_col = None
    
    # Check common column names first
    for col in ['date', 'datetime']:
        if col in df.columns:
            datetime_col = col
            break
    
    # If not found, check if index name is now a column
    if datetime_col is None and index_name:
        if index_name in df.columns:
            datetime_col = index_name
    
    # If still not found, try to find any datetime-like column
    if datetime_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                break
    
    if datetime_col is None:
        raise ValueError("Could not find datetime column in DataFrame")
    
    # Convert to UTC datetime, then to Europe/Rome timezone
    df['caldt'] = pd.to_datetime(df[datetime_col], utc=True)
    if df['caldt'].dt.tz is None:
        df['caldt'] = df['caldt'].dt.tz_localize('UTC')
    df['caldt'] = df['caldt'].dt.tz_convert('Europe/Rome')
    
    # Extract date and time
    df['date'] = df['caldt'].dt.date
    df['time'] = df['caldt'].dt.time
    
    # Add ticker and type
    df['ticker'] = ticker
    df['type'] = data_type
    
    # Add insertion time
    now = datetime.now().replace(second=0, microsecond=0)
    df['insertion_time'] = now
    
    # Select and rename columns
    # Map IB columns to standard columns
    column_map = {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    }
    
    # Build final DataFrame with required columns
    result_cols = ['date', 'time', 'ticker', 'open', 'high', 'low', 'close',
                   'volume', 'insertion_time', 'type']
    
    result_df = pd.DataFrame()
    result_df['date'] = df['date']
    result_df['time'] = df['time']
    result_df['ticker'] = df['ticker']
    
    # Map OHLCV columns
    for std_col, ib_col in column_map.items():
        if ib_col in df.columns:
            result_df[std_col] = pd.to_numeric(df[ib_col], errors='coerce')
        else:
            result_df[std_col] = pd.NA
    
    result_df['insertion_time'] = df['insertion_time']
    result_df['type'] = df['type']
    
    return result_df[result_cols]


def _export_to_annual_files(df: pd.DataFrame, ticker_dir: Path) -> None:
    """
    Export normalized DataFrame to annual Parquet files.
    
    Saves data in the format {ticker_dir}/{YYYY}.parquet with incremental merging
    and deduplication. Similar to the export function in oldcode.py.
    
    Args:
        df: Normalized DataFrame with columns: date, time, ticker, open, high, low, close, volume, insertion_time, type
        ticker_dir: Directory where to save annual files
    """
    if df.empty:
        return
    
    df_orig = df.copy()
    cols = df_orig.columns.tolist()
    
    # Create timestamp from date and time for grouping
    ts = pd.to_datetime(df_orig["date"].astype(str) + " " + df_orig["time"].astype(str), utc=True)
    df_orig["timestamp"] = ts
    df_orig["date_only"] = ts.dt.date
    df_orig["year"] = ts.dt.year
    
    # Get last saved date to filter new data
    last_date = _get_last_saved_date(ticker_dir)
    if last_date:
        df_orig = df_orig[df_orig["date_only"] >= last_date]
    
    if df_orig.empty:
        return
    
    # Group by ticker (should be single ticker, but keep for compatibility)
    for ticker, grp in df_orig.groupby("ticker"):
        ticker_dir_path = ticker_dir
        
        # Group by year
        for y, sub in grp.groupby("year"):
            os.makedirs(ticker_dir_path, exist_ok=True)
            path = ticker_dir_path / f"{int(y)}.parquet"
            
            df_new = sub[cols].copy()
            
            if path.exists():
                df_old = pd.read_parquet(path)
                
                # Ensure columns match
                if set(cols) != set(df_old.columns):
                    df_old = df_old.reindex(columns=cols, fill_value=pd.NA)
                
                # Remove last date if updating same year (overwrite protection)
                if last_date and (int(y) == last_date.year):
                    d_last = pd.to_datetime(last_date).date()
                    old_dates = pd.to_datetime(df_old["date"]).dt.date
                    df_old = df_old.loc[old_dates != d_last]
                
                df_cat = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_cat = df_new
            
            # Deduplicate and sort
            df_to_save = (
                df_cat.drop_duplicates(subset=["date", "time", "type"], keep="last")
                      .sort_values(["date", "time"], kind="stable")
            )
            
            # Save with zstd compression
            df_to_save.to_parquet(path, index=False, compression="zstd")


def _save_contract_details(
    ticker: str,
    ticker_dir: Path,
    connection: Optional[IBConnection] = None,
    exchange: Optional[str] = None,
    currency: Optional[str] = None
) -> Optional[Path]:
    """
    Get and save contract details for a ticker to JSON file.
    
    Args:
        ticker: Stock ticker symbol
        ticker_dir: Directory where to save the file
        connection: Optional IBConnection instance
        exchange: Optional exchange override
        currency: Optional currency override
        
    Returns:
        Path to saved JSON file, or None if failed
    """
    from config.ib_config import get_config
    
    config = get_config()
    conn = connection or IBConnection()
    exchange = exchange or config['default_exchange']
    currency = currency or config['default_currency']
    
    contract_details_file = ticker_dir / f"{ticker}_contract_details.json"
    
    try:
        ib = conn.get_ib()
        
        # Create contract
        contract = Stock(ticker, exchange, currency)
        
        # Qualify contract first
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            logger.warning(f"Could not qualify contract for {ticker}, skipping contract details")
            return None
        
        # Use the qualified contract
        contract = qualified[0]
        
        # Request contract details
        details_list = ib.reqContractDetails(contract)
        
        if not details_list:
            logger.warning(f"No contract details returned for {ticker}")
            return None
        
        # Convert contract details to dictionary
        all_details = []
        for detail in details_list:
            contract_obj = detail.contract
            
            # Convert contract to dict
            contract_dict = {
                'conId': contract_obj.conId,
                'symbol': contract_obj.symbol,
                'secType': contract_obj.secType,
                'exchange': contract_obj.exchange,
                'primaryExchange': contract_obj.primaryExchange or None,
                'currency': contract_obj.currency,
                'localSymbol': contract_obj.localSymbol,
                'tradingClass': contract_obj.tradingClass or None,
                'multiplier': contract_obj.multiplier or None,
            }
            
            # Convert contract details to dict using getattr for safe attribute access
            # Some attributes may not exist depending on contract type
            details_dict = {
                'contract': contract_dict,
                'marketName': getattr(detail, 'marketName', None) or None,
                'minTick': getattr(detail, 'minTick', None) or None,
                'orderTypes': getattr(detail, 'orderTypes', None) or None,
                'validExchanges': getattr(detail, 'validExchanges', None) or None,
                'priceMagnifier': getattr(detail, 'priceMagnifier', None) or None,
                'underConId': getattr(detail, 'underConId', None) or None,
                'longName': getattr(detail, 'longName', None) or None,
                'contractMonth': getattr(detail, 'contractMonth', None) or None,
                'industry': getattr(detail, 'industry', None) or None,
                'category': getattr(detail, 'category', None) or None,
                'subcategory': getattr(detail, 'subcategory', None) or None,
                'timeZoneId': getattr(detail, 'timeZoneId', None) or None,
                'tradingHours': getattr(detail, 'tradingHours', None) or None,
                'liquidHours': getattr(detail, 'liquidHours', None) or None,
                'evRule': getattr(detail, 'evRule', None) or None,
                'evMultiplier': getattr(detail, 'evMultiplier', None) or None,
                'mdSizeMultiplier': getattr(detail, 'mdSizeMultiplier', None) or None,
                'aggGroup': getattr(detail, 'aggGroup', None) or None,
                'underSymbol': getattr(detail, 'underSymbol', None) or None,
                'underSecType': getattr(detail, 'underSecType', None) or None,
                'marketRuleIds': getattr(detail, 'marketRuleIds', None) or None,
                'realExpirationDate': getattr(detail, 'realExpirationDate', None) or None,
                'lastTradeDate': getattr(detail, 'lastTradeDate', None) or None,
                'stockType': getattr(detail, 'stockType', None) or None,
                'minSize': getattr(detail, 'minSize', None) or None,
                'sizeIncrement': getattr(detail, 'sizeIncrement', None) or None,
                'suggestedSizeIncrement': getattr(detail, 'suggestedSizeIncrement', None) or None,
            }
            
            # Remove None values for cleaner JSON
            details_dict = {k: v for k, v in details_dict.items() if v is not None}
            all_details.append(details_dict)
        
        # Add metadata
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'exchange_used': exchange,
            'currency_used': currency,
            'contracts_found': len(all_details),
            'contract_details': all_details
        }
        
        # Save to JSON
        with open(contract_details_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved contract details to {contract_details_file}")
        return contract_details_file
        
    except Exception as e:
        logger.warning(f"Failed to save contract details for {ticker}: {e}")
        return None


def download_data(
    ticker: str,
    eod_duration: str = '20 Y',  # Maximum available historical data (IB limit ~20 years)
    intraday_duration: str = '30 D',  # Maximum intraday data (IB limit ~30 days for 1-min bars)
    connection: Optional[IBConnection] = None,
    exchange: Optional[str] = None,
    currency: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Download complete historical data for a new ticker.
    
    This function downloads:
    1. EOD (End of Day) data with daily bars
    2. Intraday data with 1-minute bars
    
    Data is saved in the structure: database/TICKER/
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        eod_duration: Duration for EOD data (default: '20 Y' - maximum available ~20 years)
        intraday_duration: Duration for intraday data (default: '30 D' - IB limit for 1-min bars)
        connection: Optional IBConnection instance
        exchange: Optional exchange override
        currency: Optional currency override
        
    Returns:
        tuple: (eod_filepath, intraday_filepath)
        
    Raises:
        ValueError: If ticker already exists (use update_ticker instead)
        RuntimeError: If download fails
        
    Example:
        >>> eod_path, intraday_path = download_data('AAPL')
        >>> print(f"EOD data saved to: {eod_path}")
        >>> print(f"Intraday data saved to: {intraday_path}")
    """
    ticker = validate_symbol(ticker)
    
    # Check if ticker already exists
    if _ticker_exists(ticker):
        raise ValueError(
            f"Ticker {ticker} already exists. Use update_ticker() to update existing data."
        )
    
    logger.info(f"Starting download for new ticker: {ticker}")
    
    # Create ticker directory
    ticker_dir = _get_ticker_directory(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {ticker_dir}")
    
    # Initialize connection and fetcher
    conn = connection or IBConnection()
    fetcher = HistoricalDataFetcher(connection=conn, exchange=exchange, currency=currency)
    
    try:
        # Download EOD data
        logger.info(f"Downloading EOD data for {ticker} (duration: {eod_duration})")
        df_eod_raw = fetcher.fetch(
            symbol=ticker,
            timeframe='1 day',
            duration=eod_duration,
            use_rth=True
        )
        
        if not df_eod_raw.empty:
            # Normalize to standard format
            df_eod = _normalize_dataframe(df_eod_raw, ticker, 'eod')
            # Export to annual files
            _export_to_annual_files(df_eod, ticker_dir)
            logger.info(f"Saved {len(df_eod)} EOD bars to annual files")
        else:
            logger.warning(f"No EOD data retrieved for {ticker}")
        
        # Download intraday data (1 minute) - split into chunks to avoid timeouts
        logger.info(f"Downloading 1-minute intraday data for {ticker} (duration: {intraday_duration})")
        
        # For intraday data, IB limits are ~30 days per request for 1-min bars
        # Use single request (default is 30 D which is the IB limit)
        df_intraday_raw = fetcher.fetch(
            symbol=ticker,
            timeframe='1 min',
            duration=intraday_duration,
            use_rth=True
        )
        
        if not df_intraday_raw.empty:
            # Normalize to standard format
            df_intraday = _normalize_dataframe(df_intraday_raw, ticker, 'intraday')
            # Export to annual files
            _export_to_annual_files(df_intraday, ticker_dir)
            logger.info(f"Saved {len(df_intraday)} intraday bars to annual files")
        else:
            logger.warning(f"No intraday data retrieved for {ticker}")
        
        # Save contract details
        logger.info(f"Fetching contract details for {ticker}")
        contract_details_file = _save_contract_details(
            ticker=ticker,
            ticker_dir=ticker_dir,
            connection=conn,
            exchange=exchange,
            currency=currency
        )
        if contract_details_file:
            logger.info(f"Contract details saved to {contract_details_file}")
        
        logger.info(f"Successfully downloaded data for {ticker}")
        # Return ticker_dir as both paths (for compatibility, but files are now annual)
        return ticker_dir, ticker_dir
        
    except Exception as e:
        logger.error(f"Failed to download data for {ticker}: {e}")
        # Clean up directory if download failed
        if ticker_dir.exists() and not any(ticker_dir.iterdir()):
            ticker_dir.rmdir()
            logger.info(f"Removed empty directory: {ticker_dir}")
        raise RuntimeError(f"Failed to download data for {ticker}: {e}")


def update_ticker(
    ticker: str,
    connection: Optional[IBConnection] = None,
    exchange: Optional[str] = None,
    currency: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Update existing ticker data with the latest available data.
    
    This function:
    1. Loads existing data files
    2. Determines the last available date
    3. Downloads only new data since the last date
    4. Appends new data and saves the updated files
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        connection: Optional IBConnection instance
        exchange: Optional exchange override
        currency: Optional currency override
        
    Returns:
        tuple: (eod_filepath, intraday_filepath)
        
    Raises:
        ValueError: If ticker doesn't exist (use download_data instead)
        RuntimeError: If update fails
        
    Example:
        >>> eod_path, intraday_path = update_ticker('AAPL')
        >>> print(f"Updated EOD data: {eod_path}")
        >>> print(f"Updated intraday data: {intraday_path}")
    """
    ticker = validate_symbol(ticker)
    
    # Check if ticker exists
    if not _ticker_exists(ticker):
        raise ValueError(
            f"Ticker {ticker} does not exist. Use download_data() to download initial data."
        )
    
    logger.info(f"Starting update for ticker: {ticker}")
    
    ticker_dir = _get_ticker_directory(ticker)
    
    # Initialize connection and fetcher
    conn = connection or IBConnection()
    fetcher = HistoricalDataFetcher(connection=conn, exchange=exchange, currency=currency)
    
    try:
        # Get last saved date from annual files
        last_date = _get_last_saved_date(ticker_dir)
        
        if last_date:
            # Calculate days since last update
            now = datetime.now().date()
            days_since = (now - last_date).days
            
            if days_since > 0:
                # Update EOD data
                logger.info(f"Updating EOD data for {ticker} (last date: {last_date})")
                duration_eod = f"{min(days_since + 5, 30)} D"  # Add buffer, max 30 days
                df_eod_raw = fetcher.fetch(
                    symbol=ticker,
                    timeframe='1 day',
                    duration=duration_eod,
                    use_rth=True
                )
                
                if not df_eod_raw.empty:
                    # Normalize to standard format
                    df_eod = _normalize_dataframe(df_eod_raw, ticker, 'eod')
                    # Export to annual files (will filter and merge automatically)
                    _export_to_annual_files(df_eod, ticker_dir)
                    logger.info(f"Updated EOD data: added {len(df_eod)} bars")
                else:
                    logger.info("No new EOD data retrieved")
                
                # Update intraday data (max 30 days for 1-min bars due to IB limits)
                logger.info(f"Updating 1-minute intraday data for {ticker}")
                duration_intraday = f"{min(days_since + 2, 30)} D"  # Add buffer, max 30 days
                df_intraday_raw = fetcher.fetch(
                    symbol=ticker,
                    timeframe='1 min',
                    duration=duration_intraday,
                    use_rth=True
                )
                
                if not df_intraday_raw.empty:
                    # Normalize to standard format
                    df_intraday = _normalize_dataframe(df_intraday_raw, ticker, 'intraday')
                    # Export to annual files (will filter and merge automatically)
                    _export_to_annual_files(df_intraday, ticker_dir)
                    logger.info(f"Updated intraday data: added {len(df_intraday)} bars")
                else:
                    logger.info("No new intraday data retrieved")
            else:
                logger.info("Data is already up to date")
        else:
            logger.warning("Could not determine last saved date, downloading fresh data")
            # If we can't determine last date, download recent data
            df_eod_raw = fetcher.fetch(
                symbol=ticker,
                timeframe='1 day',
                duration='30 D',
                use_rth=True
            )
            if not df_eod_raw.empty:
                df_eod = _normalize_dataframe(df_eod_raw, ticker, 'eod')
                _export_to_annual_files(df_eod, ticker_dir)
            
            df_intraday_raw = fetcher.fetch(
                symbol=ticker,
                timeframe='1 min',
                duration='30 D',
                use_rth=True
            )
            if not df_intraday_raw.empty:
                df_intraday = _normalize_dataframe(df_intraday_raw, ticker, 'intraday')
                _export_to_annual_files(df_intraday, ticker_dir)
        
        # Save contract details if file doesn't exist
        contract_details_file = ticker_dir / f"{ticker}_contract_details.json"
        if not contract_details_file.exists():
            logger.info(f"Contract details file not found, fetching for {ticker}")
            contract_details_file = _save_contract_details(
                ticker=ticker,
                ticker_dir=ticker_dir,
                connection=conn,
                exchange=exchange,
                currency=currency
            )
            if contract_details_file:
                logger.info(f"Contract details saved to {contract_details_file}")
        else:
            logger.debug(f"Contract details file already exists: {contract_details_file}")
        
        logger.info(f"Successfully updated data for {ticker}")
        # Return ticker_dir as both paths (for compatibility, but files are now annual)
        return ticker_dir, ticker_dir
        
    except Exception as e:
        logger.error(f"Failed to update data for {ticker}: {e}")
        raise RuntimeError(f"Failed to update data for {ticker}: {e}")


def get_or_update_ticker(
    ticker: str,
    eod_duration: str = '20 Y',  # Maximum available historical data
    intraday_duration: str = '30 D',  # Maximum intraday data (IB limit for 1-min bars)
    connection: Optional[IBConnection] = None,
    exchange: Optional[str] = None,
    currency: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Smart function that downloads or updates ticker data based on existence.
    
    This function automatically:
    - Calls download_data() if ticker doesn't exist
    - Calls update_ticker() if ticker already exists
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        eod_duration: Duration for initial EOD download (default: '20 Y' - maximum available)
        intraday_duration: Duration for initial intraday download (default: '30 D' - IB limit for 1-min bars)
        connection: Optional IBConnection instance
        exchange: Optional exchange override
        currency: Optional currency override
        
    Returns:
        tuple: (eod_filepath, intraday_filepath)
        
    Example:
        >>> eod_path, intraday_path = get_or_update_ticker('AAPL')
        >>> # Automatically downloads if new, updates if exists
    """
    ticker = validate_symbol(ticker)
    
    if _ticker_exists(ticker):
        logger.info(f"Ticker {ticker} exists, updating...")
        return update_ticker(ticker, connection, exchange, currency)
    else:
        logger.info(f"Ticker {ticker} does not exist, downloading...")
        return download_data(
            ticker, eod_duration, intraday_duration,
            connection, exchange, currency
        )


def list_available_tickers() -> list[str]:
    """
    List all tickers that have been downloaded.
    
    Returns:
        list: List of ticker symbols available in the data directory
        
    Example:
        >>> tickers = list_available_tickers()
        >>> print(f"Available tickers: {', '.join(tickers)}")
    """
    if not DATA_DIR.exists():
        return []
    
    tickers = [
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    return sorted(tickers)


def get_ticker_info(ticker: str) -> dict:
    """
    Get information about a ticker's data files.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        dict: Dictionary with ticker information including file paths,
              sizes, and date ranges
              
    Example:
        >>> info = get_ticker_info('AAPL')
        >>> print(f"EOD bars: {info['eod_bars']}")
        >>> print(f"Intraday bars: {info['intraday_bars']}")
    """
    ticker = validate_symbol(ticker)
    ticker_dir = _get_ticker_directory(ticker)
    
    info = {
        'ticker': ticker,
        'exists': _ticker_exists(ticker),
        'directory': str(ticker_dir),
        'annual_files': [],
        'contract_details_file': None,
        'eod_bars': 0,
        'intraday_bars': 0,
        'eod_date_range': None,
        'intraday_date_range': None,
        'has_contract_details': False
    }
    
    if not info['exists']:
        return info
    
    # Load all annual files
    annual_files = []
    if ticker_dir.exists():
        files = [f for f in os.listdir(ticker_dir) if f.endswith(".parquet")]
        for f in files:
            base = os.path.splitext(f)[0]
            if base.isdigit():
                annual_files.append(ticker_dir / f)
    
    info['annual_files'] = [str(f) for f in annual_files]
    
    # Load and aggregate EOD and intraday data from annual files
    try:
        if annual_files:
            # Load all annual files
            dfs = []
            for f in annual_files:
                try:
                    df = pd.read_parquet(f)
                    if 'type' in df.columns:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading annual file {f}: {e}")
            
            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                
                # Filter EOD data
                if 'type' in df_all.columns:
                    df_eod = df_all[df_all['type'] == 'eod'].copy()
                    if not df_eod.empty:
                        info['eod_bars'] = len(df_eod)
                        if 'date' in df_eod.columns:
                            dates = pd.to_datetime(df_eod['date'])
                            info['eod_date_range'] = (
                                str(dates.min().date()),
                                str(dates.max().date())
                            )
                    
                    # Filter intraday data
                    df_intraday = df_all[df_all['type'] == 'intraday'].copy()
                    if not df_intraday.empty:
                        info['intraday_bars'] = len(df_intraday)
                        if 'date' in df_intraday.columns:
                            dates = pd.to_datetime(df_intraday['date'])
                            info['intraday_date_range'] = (
                                str(dates.min().date()),
                                str(dates.max().date())
                            )
    except Exception as e:
        logger.error(f"Error reading annual files: {e}")
    
    # Contract details file info
    contract_details_file = ticker_dir / f"{ticker}_contract_details.json"
    if contract_details_file.exists():
        info['contract_details_file'] = str(contract_details_file)
        info['has_contract_details'] = True
        try:
            # Read JSON to get basic info
            with open(contract_details_file, 'r', encoding='utf-8') as f:
                contract_data = json.load(f)
                info['contract_details_timestamp'] = contract_data.get('timestamp')
                info['contract_details_count'] = contract_data.get('contracts_found', 0)
        except Exception as e:
            logger.error(f"Error reading contract details file: {e}")
    
    return info


def load_contract_details(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Load contract details from JSON file for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        dict: Contract details dictionary, or None if file doesn't exist
        
    Example:
        >>> details = load_contract_details('AAPL')
        >>> if details:
        ...     print(f"Found {details['contracts_found']} contract(s)")
        ...     print(f"Exchange: {details['contract_details'][0]['contract']['exchange']}")
    """
    ticker = validate_symbol(ticker)
    ticker_dir = _get_ticker_directory(ticker)
    contract_details_file = ticker_dir / f"{ticker}_contract_details.json"
    
    if not contract_details_file.exists():
        logger.warning(f"Contract details file not found: {contract_details_file}")
        return None
    
    try:
        with open(contract_details_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading contract details: {e}")
        return None

