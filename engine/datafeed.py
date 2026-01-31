import os
import polars as pl
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, List, Union, TYPE_CHECKING
from datetime import datetime, date, time
from pathlib import Path

class DataFeed:
    """
    I/O loader for loading price data from parquet files.
    
    This class is responsible for file system I/O operations, scanning and loading
    price data from a directory structure where price files are organized by ticker
    and year. It handles the low-level data loading operations and returns Polars
    DataFrames/LazyFrames for consumption by higher-level components.
    
    The expected directory structure is:
        prices_base/
            <ticker>/
                <year>.parquet
    
    The parquet files should contain columns: date, time, ticker, open, high,
    low, close, volume, and type (where type is either "eod" or "intraday").
    
    This class is purely an I/O layer and does not contain business logic.
    It is used by MarketData and other components to load raw data from storage.
    
    Attributes:
        prices_base: Base directory path containing price data files.
            Should contain subdirectories for each ticker, with parquet files
            named by year (e.g., "2020.parquet").
    
    Example:
        >>> feed = DataFeed(prices_base="/path/to/prices")
        >>> lf = feed.scan_prices("2020-01-01", "2023-12-31", "eod", ["SPY US Equity", "QQQ US Equity"])
        >>> df = feed.get_market_data("2020-01-01", "2023-12-31", "eod", ["SPY US Equity"])
    """

    def __init__(
            self,
            prices_base: str,
        ):
        """
        Initialize the DataFeed I/O loader with a base directory for price files.
        
        Args:
            prices_base: Base directory path containing price data files.
                The directory should be organized with subdirectories for each
                ticker, containing parquet files named by year.
        """
        self.prices_base = prices_base

    def _price_paths(self, ticker: str, start: str, end: str) -> List[str]:
        """
        Get price file paths for a ticker within a date range.
        
        This method scans the price directory for the given ticker and returns
        all parquet files that fall within the specified date range. Files are
        expected to be named by year (e.g., "2020.parquet").
        
        Args:
            ticker: Ticker symbol (e.g., "SPY US Equity").
                Used to locate the ticker's subdirectory in prices_base.
            start: Start date in YYYY-MM-DD format.
                The method will include files for years from this date onwards.
            end: End date in YYYY-MM-DD format.
                The method will include files for years up to this date.
            
        Returns:
            List of existing parquet file paths for the ticker within the
            date range. Only files that actually exist on disk are included.
            Returns an empty list if no files are found.
        """
        y0, y1 = int(start[:4]), int(end[:4])
        paths = []
        for y in range(y0, y1 + 1):
            p = os.path.join(self.prices_base, ticker, f"{y}.parquet")
            if os.path.exists(p):
                paths.append(p)
        return paths

    def scan_prices(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["1m", "5m", "1h", "4h", "eod"],
        tickers: List[str]
    ) -> pl.LazyFrame:
        """
        Create a lazy frame for price data.
        
        This method scans parquet files for the specified tickers and date range,
        filters by the configured frequency (eod or intraday: 1m, 5m, 1h, 4h), and returns a
        Polars LazyFrame. The data is not loaded into memory until explicitly
        collected, making this efficient for large datasets.
        
        For intraday data, the method loads 1-minute bars from the parquet files
        and then resamples them to the requested frequency (5m, 1h, 4h) if needed.
        The resampling uses standard OHLC aggregation:
            - Open: First open price in the period
            - High: Maximum high price in the period
            - Low: Minimum low price in the period
            - Close: Last close price in the period
            - Volume: Sum of volume in the period
        
        The returned LazyFrame contains the following columns:
            - date: Date of the price bar
            - time: Time of the price bar (for intraday data)
            - ticker: Ticker symbol
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            - type: Bar type ("eod" or "intraday")
        
        Args:
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            frequency: "eod" for end-of-day or "1m", "5m", "1h", "4h" for intraday data
            tickers: List of ticker symbols to load.
                Each ticker should correspond to a subdirectory in prices_base.
            
        Returns:
            Polars LazyFrame with price data for the specified tickers and
            date range, resampled to the requested frequency. If no files are found,
            returns an empty LazyFrame with the expected schema.
        """
        paths: List[str] = []
        for t in tickers:
            paths += self._price_paths(t, start_date, end_date)
        if not paths:
            # <<< schema minimo atteso dal resto della pipeline
            schema = {
                "date": pl.Date,
                "time": pl.Time,
                "ticker": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "insertion_time": pl.Datetime(time_unit="us", time_zone=None),
                "type": pl.Utf8,
            }
            return pl.DataFrame(schema=schema).lazy()  # LazyFrame con colonne note (vuoto)

        lf = pl.scan_parquet(paths).filter(
            (pl.col("date").cast(pl.Date) >= pl.lit(start_date).cast(pl.Date)) &
            (pl.col("date").cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))
        )
        lf = lf.filter(pl.col("type") == ("eod" if frequency == "eod" else "intraday"))
        
        # For intraday data, resample to the requested frequency if needed
        if frequency != "eod" and frequency != "1m":
            # Combine date and time into a datetime column for resampling
            lf = lf.with_columns(
                pl.datetime(
                    pl.col("date").dt.year(),
                    pl.col("date").dt.month(),
                    pl.col("date").dt.day(),
                    pl.col("time").dt.hour(),
                    pl.col("time").dt.minute(),
                    pl.col("time").dt.second()
                ).alias("datetime")
            )
            
            # Group by ticker and resample to the requested frequency
            lf = lf.group_by_dynamic(
                "datetime",
                every=frequency,
                by="ticker",
                closed="left"
            ).agg([
                pl.first("date").alias("date"),
                pl.first("time").alias("time"),
                pl.first("open").alias("open"),
                pl.max("high").alias("high"),
                pl.min("low").alias("low"),
                pl.last("close").alias("close"),
                pl.sum("volume").alias("volume"),
                pl.first("insertion_time").alias("insertion_time"),
                pl.first("type").alias("type"),
            ]).drop("datetime").select([
                "date",
                "time",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "insertion_time",
                "type",
            ])
        
        return lf

    def get_market_data(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["1m", "5m", "1h", "4h", "eod"],
        tickers: List[str]
    ) -> pl.DataFrame:
        """
        Load market data from parquet files and return as a Polars DataFrame.
        
        This method performs the I/O operation of loading price data from disk,
        applying filters based on configuration, and returning the collected data.
        The data is loaded into memory as a DataFrame (unlike scan_prices which
        returns a LazyFrame).
        
        Args:
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            frequency: "eod" for end-of-day or "1m", "5m", "1h", "4h" for intraday data
            tickers: List of ticker symbols to load
            
        Returns:
            Polars DataFrame with market data for the specified tickers and
            configuration. Returns an empty DataFrame if no data is found.
        """
        market_feed = self.scan_prices(start_date, end_date, frequency, tickers)
        return market_feed.collect(engine='streaming')

    def get_data_excel(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["1m", "5m", "1h", "4h", "eod"],
        tickers: List[str],
        output_path: Optional[str] = None
    ) -> str:
        """
        Load market data and export to Excel file.
        
        This method loads market data using get_market_data and exports it to an Excel file.
        If no output path is specified, the file is saved in the current working directory
        with an auto-generated filename based on tickers, date range, and frequency.
        
        Args:
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            frequency: "eod" for end-of-day or "1m", "5m", "1h", "4h" for intraday data
            tickers: List of ticker symbols to load
            output_path: Optional path for the output Excel file. If None, saves in current
                directory with auto-generated filename (e.g., "SX5E_2025-01-01_2025-12-31_eod.xlsx")
            
        Returns:
            Path to the created Excel file
            
        Examples:
            >>> feed = DataFeed(prices_base="/path/to/prices")
            >>> # Save to current directory with auto-generated name
            >>> path = feed.get_data_excel("2025-01-01", "2025-12-31", "eod", ["SX5E"])
            >>> # Save to specific path
            >>> path = feed.get_data_excel("2025-01-01", "2025-12-31", "eod", ["SX5E"], 
            ...                            output_path="/path/to/output.xlsx")
        """
        # Get market data
        df = self.get_market_data(start_date, end_date, frequency, tickers)
        
        # Generate output path if not provided
        if output_path is None:
            # Create filename from tickers, dates, and frequency
            ticker_str = "_".join(tickers[:3])  # Use first 3 tickers for filename
            if len(tickers) > 3:
                ticker_str += f"_and_{len(tickers)-3}_more"
            # Sanitize ticker string for filename (remove spaces and special chars)
            ticker_str = ticker_str.replace(" ", "_").replace("/", "_")
            filename = f"{ticker_str}_{start_date}_{end_date}_{frequency}.xlsx"
            output_path = str(Path.cwd() / filename)
        else:
            # Ensure output_path is a string
            output_path = str(output_path)
            # Ensure .xlsx extension
            if not output_path.endswith('.xlsx'):
                output_path += '.xlsx'
        
        # Write to Excel
        df.write_excel(output_path)
        
        return output_path


@dataclass
class MarketData:
    """Container for market prices with historical access.
    
    This class provides a powerful interface for accessing current and historical
    prices. It supports time-based offsets (T, T-1, T-2, etc.) and slicing for
    retrieving ranges of historical data.
    
    Attributes:
        data: Polars DataFrame with historical price data.
            Required columns: date, time, ticker, open, high, low, close, volume
        current_timestamp: Tuple (date, time) representing the current bar.
            If None, uses the last row in the DataFrame as current.
        _ticker_cache: Internal cache for filtered ticker data (performance optimization)
    
    Examples:
        >>> # Get current price
        >>> market_data.price("SPY", "close")
        450.25
        
        >>> # Get price 1 bar ago (T-1)
        >>> market_data.price("SPY", "close", 1)
        449.80
        
        >>> # Get last 100 prices (using slice notation :100)
        >>> market_data.price("SPY", "close", slice(None, 100))
        <polars.Series>
        
        >>> # Get prices from T-10 to T-1
        >>> market_data.price("SPY", "close", slice(1, 11))
        <polars.Series>
    """
    data: pl.DataFrame
    current_timestamp: Optional[tuple] = None
    _ticker_cache: Dict[str, pl.DataFrame] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Validate and prepare data after initialization."""
        required_cols = ["date", "time", "ticker", "open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Set current_timestamp if not provided
        if self.current_timestamp is None and self.data.height > 0:
            last_row = self.data.tail(1)
            self.current_timestamp = (
                last_row["date"][0],
                last_row["time"][0]
            )
        
        # Ensure data is sorted
        if self.data.height > 0:
            self.data = self.data.sort(["date", "time", "ticker"])
    
    def get_ticker_data(self, ticker: str) -> Optional[pl.DataFrame]:
        """Get filtered and sorted data for a ticker (with caching).
        
        Args:
            ticker: Symbol to filter for
            
        Returns:
            Filtered DataFrame sorted by date/time, or None if ticker not found
        """
        if ticker not in self._ticker_cache:
            ticker_data = self.data.filter(pl.col("ticker") == ticker)
            if ticker_data.height == 0:
                return None
            self._ticker_cache[ticker] = ticker_data.sort(["date", "time"])
        
        return self._ticker_cache[ticker]
    
    def _get_current_index(self, ticker_data: pl.DataFrame) -> Optional[int]:
        """Get the index of the current timestamp in ticker data.
        
        Args:
            ticker_data: Filtered DataFrame for a specific ticker
            
        Returns:
            Index of current bar, or None if not found
        """
        if self.current_timestamp is None:
            return ticker_data.height - 1
        
        current_date, current_time = self.current_timestamp
        mask = (pl.col("date") == current_date) & (pl.col("time") == current_time)
        matches = ticker_data.filter(mask)
        
        if matches.height == 0:
            return None
        
        # Find index in sorted ticker_data
        indices = ticker_data.with_row_index("_idx").filter(mask)["_idx"]
        return int(indices[0])
    
    def price(self, ticker: str, price_type: str = "close", 
              offset: Optional[Union[int, slice]] = None) -> Union[float, pl.Series, None]:
        """
        Get price(s) for a ticker at specified time offset.
        
        Args:
            ticker: Symbol to get price for
            price_type: Price type ("open", "high", "low", "close", "volume")
            offset: Time offset or slice:
                - None or 0: Current price (at current_timestamp)
                - Positive int (1, 2, ...): Price N bars ago (T-N)
                - Negative int (-1, -2, ...): Price N bars ahead (T+N, if available)
                - slice(None, N) or :N: Last N prices up to current (inclusive)
                - slice(M, N): Prices from T-N+1 to T-M (M < N, both relative to current)
                - slice(-N, None): Last N prices (alternative syntax)
        
        Returns:
            - float: Single price if offset is int or None
            - pl.Series: Series of prices if offset is slice
            - None: If ticker not found, price_type invalid, or offset out of range
        
        Examples:
            >>> # Current price
            >>> market_data.price("SPY", "close")
            450.25
            
            >>> # Price 1 bar ago
            >>> market_data.price("SPY", "close", 1)
            449.80
            
            >>> # Last 100 prices (using slice notation :100)
            >>> market_data.price("SPY", "close", slice(None, 100))
            <polars.Series with up to 100 values>
            
            >>> # Prices from T-10 to T-1
            >>> market_data.price("SPY", "close", slice(1, 11))
            <polars.Series with 10 values>
        """
        # Validate price_type
        if price_type not in ["open", "high", "low", "close", "volume"]:
            return None
        
        # Get ticker data
        ticker_data = self.get_ticker_data(ticker)
        if ticker_data is None:
            return None
        
        # Get current index
        current_idx = self._get_current_index(ticker_data)
        if current_idx is None:
            return None
        
        # Handle different offset types
        if offset is None or offset == 0:
            # Current price
            return float(ticker_data[price_type][current_idx])
        
        elif isinstance(offset, int):
            # Integer offset: T-1, T-2, T+1, etc.
            target_idx = current_idx - offset
            if target_idx < 0 or target_idx >= ticker_data.height:
                return None
            return float(ticker_data[price_type][target_idx])
        
        elif isinstance(offset, slice):
            # Slice: :100, -100:, 1:11, etc.
            start = offset.start
            stop = offset.stop
            step = offset.step if offset.step is not None else 1
            
            # Handle slice(None, N) - last N prices (most common case)
            if start is None and stop is not None:
                # :N means last N prices up to and including current
                start_idx = max(0, current_idx - stop + 1)
                stop_idx = current_idx + 1
                if start_idx >= stop_idx:
                    return None
                prices = ticker_data[price_type][start_idx:stop_idx:step]
                return prices
            
            # Handle slice(-N, None) - last N prices (alternative)
            elif start is not None and start < 0 and stop is None:
                start_idx = max(0, current_idx + start + 1)
                stop_idx = current_idx + 1
                if start_idx >= stop_idx:
                    return None
                prices = ticker_data[price_type][start_idx:stop_idx:step]
                return prices
            
            # Handle slice(M, N) - range from T-N+1 to T-M
            elif start is not None and stop is not None:
                # Both are relative to current_idx
                # slice(1, 11) means from T-10 to T-0 (indices current_idx-10 to current_idx)
                if start < 0 or stop < 0:
                    # Negative indices relative to current
                    start_idx = current_idx + start + 1 if start < 0 else current_idx - start + 1
                    stop_idx = current_idx + stop + 1 if stop < 0 else current_idx - stop + 1
                else:
                    # Positive indices: slice(1, 11) means T-10 to T-0
                    start_idx = current_idx - stop + 1
                    stop_idx = current_idx - start + 1
                
                start_idx = max(0, start_idx)
                stop_idx = min(current_idx + 1, stop_idx)
                
                if start_idx >= stop_idx:
                    return None
                
                prices = ticker_data[price_type][start_idx:stop_idx:step]
                return prices
            
            # Handle slice(None, None) - all prices up to current
            elif start is None and stop is None:
                prices = ticker_data[price_type][:current_idx + 1:step]
                return prices
        
        return None
    
    def ohlc(self, ticker: str, offset: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Get OHLC (Open, High, Low, Close) for a ticker at specified offset.
        
        Args:
            ticker: Symbol to get OHLC for
            offset: Time offset (None/0 for current, 1 for T-1, etc.)
        
        Returns:
            Dictionary with keys "open", "high", "low", "close", or None if not found
        """
        if offset is None:
            offset = 0
        
        o = self.price(ticker, "open", offset)
        h = self.price(ticker, "high", offset)
        l = self.price(ticker, "low", offset)
        c = self.price(ticker, "close", offset)
        
        if any(p is None for p in [o, h, l, c]):
            return None
        
        return {"open": o, "high": h, "low": l, "close": c}
    
    def volume(self, ticker: str, offset: Optional[int] = None) -> Optional[float]:
        """Get volume for a ticker at specified offset."""
        return self.price(ticker, "volume", offset)
    
    def get(self, ticker: str, offset: Optional[Union[int, slice]] = None) -> Optional[Union[pl.DataFrame, pl.Series]]:
        """
        Get DataFrame row(s) for a ticker at specified time offset.
        
        Args:
            ticker: Symbol to get data for
            offset: Time offset or slice:
                - None or 0: Current row (at current_timestamp)
                - Positive int (1, 2, ...): Row N bars ago (T-N)
                - Negative int (-1, -2, ...): Row N bars ahead (T+N, if available)
                - slice(None, N) or :N: Last N rows up to current (inclusive)
                - slice(M, N): Rows from T-N+1 to T-M (M < N, both relative to current)
        
        Returns:
            - pl.DataFrame: Single row as DataFrame if offset is int or None
            - pl.DataFrame: Multiple rows as DataFrame if offset is slice
            - None: If ticker not found or offset out of range
        
        Examples:
            >>> # Current row
            >>> market_data.get("SPY")
            <polars.DataFrame with 1 row>
            
            >>> # Row 10 bars ago (T-10)
            >>> market_data.get("SPY", 10)
            <polars.DataFrame with 1 row>
            
            >>> # Last 10 rows
            >>> market_data.get("SPY", slice(None, 10))
            <polars.DataFrame with up to 10 rows>
        """
        # Get ticker data
        ticker_data = self.get_ticker_data(ticker)
        if ticker_data is None:
            return None
        
        # Get current index
        current_idx = self._get_current_index(ticker_data)
        if current_idx is None:
            return None
        
        # Handle different offset types
        if offset is None or offset == 0:
            # Current row
            return ticker_data.slice(current_idx, 1)
        
        elif isinstance(offset, int):
            # Integer offset: T-1, T-2, T+1, etc.
            target_idx = current_idx - offset
            if target_idx < 0 or target_idx >= ticker_data.height:
                return None
            return ticker_data.slice(target_idx, 1)
        
        elif isinstance(offset, slice):
            # Slice: :100, -100:, 1:11, etc.
            start = offset.start
            stop = offset.stop
            step = offset.step if offset.step is not None else 1
            
            # Handle slice(None, N) - last N rows (most common case)
            if start is None and stop is not None:
                # :N means last N rows up to and including current
                start_idx = max(0, current_idx - stop + 1)
                stop_idx = current_idx + 1
                if start_idx >= stop_idx:
                    return None
                return ticker_data.slice(start_idx, stop_idx - start_idx)
            
            # Handle slice(-N, None) - last N rows (alternative)
            elif start is not None and start < 0 and stop is None:
                start_idx = max(0, current_idx + start + 1)
                stop_idx = current_idx + 1
                if start_idx >= stop_idx:
                    return None
                return ticker_data.slice(start_idx, stop_idx - start_idx)
            
            # Handle slice(M, N) - range from T-N+1 to T-M
            elif start is not None and stop is not None:
                # Both are relative to current_idx
                # slice(1, 11) means from T-10 to T-0 (indices current_idx-10 to current_idx)
                if start < 0 or stop < 0:
                    # Negative indices relative to current
                    start_idx = current_idx + start + 1 if start < 0 else current_idx - start + 1
                    stop_idx = current_idx + stop + 1 if stop < 0 else current_idx - stop + 1
                else:
                    # Positive indices: slice(1, 11) means T-10 to T-0
                    start_idx = current_idx - stop + 1
                    stop_idx = current_idx - start + 1
                
                start_idx = max(0, start_idx)
                stop_idx = min(current_idx + 1, stop_idx)
                
                if start_idx >= stop_idx:
                    return None
                
                return ticker_data.slice(start_idx, stop_idx - start_idx)
            
            # Handle slice(None, None) - all rows up to current
            elif start is None and stop is None:
                return ticker_data.slice(0, current_idx + 1)
        
        return None
    
    def get_current_bar(self) -> pl.DataFrame:
        """Get DataFrame with only the current bar data (all tickers at current_timestamp).
        
        This method filters the data to return only rows matching the current_timestamp,
        which represents all tickers at the current bar. This is useful for execution
        models that need price data for all tickers at the current time.
        
        Returns:
            DataFrame filtered to current_timestamp only, containing all tickers.
            If current_timestamp is None, returns last row per ticker.
        
        Examples:
            >>> # Get current bar for all tickers
            >>> current_bar = market_data.get_current_bar()
            >>> # Contains: date, time, ticker, open, high, low, close, volume
        """
        if self.current_timestamp is None:
            # If no timestamp specified, return last row per ticker
            return self.data.sort(['date', 'time']).group_by('ticker', maintain_order=True).tail(1)
        
        current_date, current_time = self.current_timestamp
        return self.data.filter(
            (pl.col("date") == current_date) & 
            (pl.col("time") == current_time)
        )
    
    def get_bar(self, index: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Get DataFrame with bar data for all tickers at specified time offset.
        
        This method is similar to get() but works on all tickers without requiring
        a ticker parameter. It returns all tickers at a specific bar offset from
        the current timestamp.
        
        Args:
            index: Time offset:
                - None or 0: Current bar (at current_timestamp)
                - Positive int (1, 2, ...): Bar N bars ago (T-N)
                - Negative int (-1, -2, ...): Bar N bars ahead (T+N, if available)
        
        Returns:
            DataFrame with all tickers at the specified bar, or None if offset out of range
        
        Examples:
            >>> # Current bar for all tickers
            >>> market_data.get_bar()
            <polars.DataFrame with all tickers at current timestamp>
            
            >>> # Bar 10 bars ago (T-10)
            >>> market_data.get_bar(10)
            <polars.DataFrame with all tickers at T-10>
        """
        # Handle offset
        if index is None:
            return self.get_current_bar()
               
        # Get unique timestamps sorted chronologically
        unique_timestamps = self.data.select(["date", "time"]).unique().sort(["date", "time"])
        
        if unique_timestamps.height == 0:
            return None
        
        # Find current timestamp index
        if self.current_timestamp is None:
            # If no timestamp specified, use last timestamp
            current_idx = unique_timestamps.height - 1
        else:
            current_date, current_time = self.current_timestamp
            mask = (pl.col("date") == current_date) & (pl.col("time") == current_time)
            matches = unique_timestamps.filter(mask)
            if matches.height == 0:
                return None
            # Find index in sorted unique_timestamps
            indices = unique_timestamps.with_row_index("_idx").filter(mask)["_idx"]
            current_idx = int(indices[0])
        
        # Calculate target index
        target_idx = current_idx - index
        if target_idx < 0 or target_idx >= unique_timestamps.height:
            return None
        
        # Get target timestamp
        target_row = unique_timestamps.slice(target_idx, 1)
        target_date = target_row["date"][0]
        target_time = target_row["time"][0]
        
        # Return all tickers at this timestamp
        return self.data.filter(
            (pl.col("date") == target_date) & 
            (pl.col("time") == target_time)
        )
    
    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol exists in market data."""
        return symbol in self.data["ticker"].unique().to_list()
    
    def symbols(self) -> List[str]:
        """Get list of all symbols in market data."""
        return self.data["ticker"].unique().to_list()
    
    def current_date(self) -> Optional[date]:
        """Get current date."""
        if self.current_timestamp:
            return self.current_timestamp[0]
        return None
    
    def current_time(self) -> Optional[time]:
        """Get current time."""
        if self.current_timestamp:
            return self.current_timestamp[1]
        return None
    
    def clear_cache(self):
        """Clear internal ticker cache (useful if data is updated)."""
        self._ticker_cache.clear()
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, 
                      current_timestamp: Optional[tuple] = None) -> "MarketData":
        """
        Create MarketData from historical DataFrame.
        
        Args:
            df: DataFrame with columns: date, time, ticker, open, high, low, close, volume
            current_timestamp: Tuple (date, time) for current bar. If None, uses last row.
        
        Returns:
            MarketData instance
        """
        return cls(data=df, current_timestamp=current_timestamp)
    
    @classmethod
    def from_datafeed(
        cls,
        feed: "DataFeed",
        start_date: str,
        end_date: str,
        frequency: Literal["1m", "5m", "1h", "4h", "eod"],
        tickers: List[str],
        current_timestamp: Optional[tuple] = None
    ) -> "MarketData":
        """
        Create MarketData from DataFeed I/O loader.
        
        Args:
            feed: DataFeed instance for I/O operations
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            frequency: "eod" for end-of-day or "1m", "5m", "1h", "4h" for intraday data
            tickers: List of ticker symbols to load
            current_timestamp: Tuple (date, time) for current bar. If None, uses last row.
        
        Returns:
            MarketData instance with historical data loaded from DataFeed
        """
        df = feed.get_market_data(start_date, end_date, frequency, tickers)
        return cls.from_dataframe(df, current_timestamp=current_timestamp)

