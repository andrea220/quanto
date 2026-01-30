"""
Utility functions for data storage, validation, and logging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd


# Setup logging
logger = logging.getLogger(__name__)


# Data directory - sempre nella root del progetto
DATA_DIR = Path(__file__).parent.parent.parent / "database"
DATA_DIR.mkdir(exist_ok=True)


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_symbol(symbol: str) -> str:
    """
    Validate and clean a stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        str: Cleaned and validated symbol
        
    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    # Clean and uppercase
    symbol = symbol.strip().upper()
    
    if not symbol.isalnum():
        raise ValueError(f"Symbol contains invalid characters: {symbol}")
    
    return symbol


def validate_timeframe(timeframe: str) -> str:
    """
    Validate timeframe/bar size string.
    
    Args:
        timeframe: Timeframe string (e.g., '1 min', '5 mins', '1 hour', '1 day')
        
    Returns:
        str: Validated timeframe
        
    Raises:
        ValueError: If timeframe is invalid
    """
    valid_timeframes = [
        '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
        '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
        '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
        '1 day', '1 week', '1 month'
    ]
    
    if timeframe not in valid_timeframes:
        raise ValueError(
            f"Invalid timeframe: {timeframe}. Must be one of: {', '.join(valid_timeframes)}"
        )
    
    return timeframe


def validate_duration(duration: str) -> str:
    """
    Validate duration string for historical data requests.
    
    Args:
        duration: Duration string (e.g., '1 D', '5 D', '1 W', '1 M', '1 Y')
        
    Returns:
        str: Validated duration
        
    Raises:
        ValueError: If duration is invalid
    """
    import re
    
    # Pattern: number followed by space and unit (S, D, W, M, Y)
    pattern = r'^\d+\s[SDWMY]$'
    
    if not re.match(pattern, duration):
        raise ValueError(
            f"Invalid duration format: {duration}. "
            "Expected format: '<number> <unit>' where unit is S (seconds), "
            "D (days), W (weeks), M (months), or Y (years). Example: '5 D'"
        )
    
    return duration


def generate_filename(
    symbol: str,
    data_type: str,
    timeframe: Optional[str] = None,
    extension: str = 'parquet'
) -> str:
    """
    Generate a standardized filename for data storage.
    
    Args:
        symbol: Stock symbol
        data_type: Type of data ('historical' or 'realtime')
        timeframe: Optional timeframe for historical data
        extension: File extension (default: 'parquet')
        
    Returns:
        str: Generated filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    symbol_clean = symbol.upper().replace(' ', '_')
    
    if timeframe:
        timeframe_clean = timeframe.replace(' ', '_')
        filename = f"{symbol_clean}_{data_type}_{timeframe_clean}_{timestamp}.{extension}"
    else:
        filename = f"{symbol_clean}_{data_type}_{timestamp}.{extension}"
    
    return filename


def save_to_parquet(
    df: pd.DataFrame,
    filename: str,
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Save a DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        filename: Filename (without path)
        directory: Optional directory path. Defaults to DATA_DIR
        
    Returns:
        Path: Full path to saved file
        
    Raises:
        ValueError: If DataFrame is empty
        IOError: If save fails
    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame")
    
    # Use default data directory if not specified
    if directory is None:
        directory = DATA_DIR
    else:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    
    filepath = directory / filename
    
    try:
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.info(f"Saved {len(df)} rows to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save to Parquet: {e}")
        raise IOError(f"Failed to save data to {filepath}: {e}")


def load_from_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from Parquet format.
    
    Args:
        filepath: Path to Parquet file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_parquet(filepath, engine='pyarrow')
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load from Parquet: {e}")
        raise IOError(f"Failed to load data from {filepath}: {e}")


def save_to_csv(
    df: pd.DataFrame,
    filename: str,
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Save a DataFrame to CSV format.
    
    Args:
        df: DataFrame to save
        filename: Filename (without path)
        directory: Optional directory path. Defaults to DATA_DIR
        
    Returns:
        Path: Full path to saved file
        
    Raises:
        ValueError: If DataFrame is empty
        IOError: If save fails
    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame")
    
    if directory is None:
        directory = DATA_DIR
    else:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    
    filepath = directory / filename
    
    try:
        df.to_csv(filepath, index=True)
        logger.info(f"Saved {len(df)} rows to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save to CSV: {e}")
        raise IOError(f"Failed to save data to {filepath}: {e}")

