import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import blpapi

from datafetch.ticker_manager import *

# Setup logging
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / "database"


class BloombergDataDownloader:
    """
    Classe unificata per gestire il download completo di dati Bloomberg:
    fetch, normalizzazione e salvataggio.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the Bloomberg data downloader.
        
        Args:
            data_dir: Optional custom data directory. Defaults to DATA_DIR.
        """
        self.data_dir = data_dir or DATA_DIR
        self.ticker_manager = TickerManager()
    
    # ============================================================================
    # Metodi per fetching dati da Bloomberg
    # ============================================================================
    
    def _fetch_daily(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """
        Fetch daily (EOD) historical data from Bloomberg using blpapi.
        
        Args:
            symbol: Bloomberg ticker symbol (e.g., 'SX5E Index')
            start_date: Start date (datetime.date or datetime.datetime)
            end_date: End date (datetime.date or datetime.datetime)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                         indexed by datetime
        """
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost('localhost')
        sessionOptions.setServerPort(8194)
        session = blpapi.Session(sessionOptions)
        
        if not session.start():
            raise RuntimeError("Failed to start Bloomberg session")
        
        try:
            if not session.openService("//blp/refdata"):
                raise RuntimeError("Failed to open //blp/refdata service")
            
            refDataService = session.getService("//blp/refdata")
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Append security (securities is an array in HistoricalDataRequest)
            request.append("securities", symbol)
            
            # Convert dates to strings in YYYYMMDD format
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            request.set("startDate", start_str)
            request.set("endDate", end_str)
            
            # Set fields to retrieve
            request.append("fields", "OPEN")
            request.append("fields", "HIGH")
            request.append("fields", "LOW")
            request.append("fields", "PX_LAST")  # Close price
            request.append("fields", "VOLUME")
            
            # Set periodicity to daily
            request.set("periodicityAdjustment", "ACTUAL")
            request.set("periodicitySelection", "DAILY")
            
            session.sendRequest(request)
            
            bars = []
            while True:
                event = session.nextEvent(500)
                if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    for msg in event:
                        if msg.messageType() == "HistoricalDataResponse":
                            securityData = msg.getElement("securityData")
                            fieldData = securityData.getElement("fieldData")
                            
                            for i in range(fieldData.numValues()):
                                bar = fieldData.getValue(i)
                                bar_date = bar.getElement("date").getValue()
                                
                                # Extract field values, handling missing data
                                bar_dict = {
                                    'datetime': pd.to_datetime(bar_date)
                                }
                                
                                # Get field values if they exist
                                if bar.hasElement("OPEN"):
                                    bar_dict['open'] = bar.getElement("OPEN").getValue()
                                if bar.hasElement("HIGH"):
                                    bar_dict['high'] = bar.getElement("HIGH").getValue()
                                if bar.hasElement("LOW"):
                                    bar_dict['low'] = bar.getElement("LOW").getValue()
                                if bar.hasElement("PX_LAST"):
                                    bar_dict['close'] = bar.getElement("PX_LAST").getValue()
                                if bar.hasElement("VOLUME"):
                                    bar_dict['volume'] = bar.getElement("VOLUME").getValue()
                                
                                bars.append(bar_dict)
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if bars:
                df = pd.DataFrame(bars)
                df.set_index('datetime', inplace=True)
                return df.sort_index()
            return pd.DataFrame()
        finally:
            session.stop()
    
    def _fetch_intraday(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """
        Fetch intraday historical data from Bloomberg using blpapi.
        
        Args:
            symbol: Bloomberg ticker symbol (e.g., 'SX5E Index')
            start_date: Start date (datetime.date or datetime.datetime)
            end_date: End date (datetime.date or datetime.datetime)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                         indexed by datetime
        """
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost('localhost')
        sessionOptions.setServerPort(8194)
        session = blpapi.Session(sessionOptions)
        
        if not session.start():
            raise RuntimeError("Failed to start Bloomberg session")
        
        try:
            if not session.openService("//blp/refdata"):
                raise RuntimeError("Failed to open //blp/refdata service")
            
            refDataService = session.getService("//blp/refdata")
            request = refDataService.createRequest("IntradayBarRequest")
            
            request.set("security", symbol)
            request.set("eventType", "TRADE")
            request.set("interval", 1)
            
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            
            request.set("startDateTime", start_dt)
            request.set("endDateTime", end_dt)
            
            session.sendRequest(request)
            
            bars = []
            while True:
                event = session.nextEvent(500)
                if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    for msg in event:
                        if msg.messageType() == "IntradayBarResponse":
                            barData = msg.getElement("barData")
                            barTickData = barData.getElement("barTickData")
                            for i in range(barTickData.numValues()):
                                bar = barTickData.getValue(i)
                                bar_time = bar.getElement("time").getValue()
                                bars.append({
                                    'datetime': pd.to_datetime(bar_time),
                                    'open': bar.getElement("open").getValue(),
                                    'high': bar.getElement("high").getValue(),
                                    'low': bar.getElement("low").getValue(),
                                    'close': bar.getElement("close").getValue(),
                                    'volume': bar.getElement("volume").getValue()
                                })
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if bars:
                df = pd.DataFrame(bars)
                df.set_index('datetime', inplace=True)
                return df.sort_index()
            return pd.DataFrame()
        finally:
            session.stop()
    
    def fetch(
        self,
        symbol: str,
        start_date,
        end_date,
        timeframe: str = '1 day'
    ) -> pd.DataFrame:
        """
        Fetch historical data from Bloomberg.
        
        Args:
            symbol: Bloomberg ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe string ('1 day', '1 min', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data indexed by datetime
        """
        logger.info(
            f"Fetching Bloomberg historical data: {symbol} | "
            f"Timeframe: {timeframe} | Start: {start_date} | End: {end_date}"
        )
        
        try:
            # Determine if EOD or intraday
            if timeframe == '1 day' or 'day' in timeframe.lower():
                # EOD data using blpapi HistoricalDataRequest
                df = self._fetch_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            else:
                # Intraday data using blpapi IntradayBarRequest
                intraday_start = end_date - timedelta(days=365)
                df = self._fetch_intraday(symbol=symbol,
                                        start_date=intraday_start,
                                        end_date=end_date)
            
            if df.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch Bloomberg historical data: {e}")
    
    # ============================================================================
    # Metodi helper per gestione directory e file
    # ============================================================================
    
    def _get_ticker_directory(self, ticker: str) -> Path:
        """
        Get the directory path for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Path: Directory path for the ticker
        """
        return self.data_dir / ticker
    
    def _ticker_exists(self, ticker: str) -> bool:
        """
        Check if a ticker directory and data files already exist.
        
        Checks for annual parquet files in the format {ticker}/{YYYY}.parquet
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            bool: True if ticker data exists, False otherwise
        """
        ticker_dir = self._get_ticker_directory(ticker)
        
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
    
    # ============================================================================
    # Metodi per normalizzazione e export
    # ============================================================================
    
    def _normalize_dataframe(self, df: pd.DataFrame, ticker: str, data_type: str) -> pd.DataFrame:
        """
        Normalize raw DataFrame into standard table format.
        
        Converts DataFrame with datetime index to format with separate date/time columns:
        ['date', 'time', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'insertion_time', 'type']
        
        Args:
            df: Raw DataFrame with datetime index and OHLCV columns
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
        
        # Build final DataFrame with required columns
        result_cols = ['date', 'time', 'ticker', 'open', 'high', 'low', 'close',
                       'volume', 'insertion_time', 'type']
        
        result_df = pd.DataFrame()
        result_df['date'] = df['date']
        result_df['time'] = df['time']
        result_df['ticker'] = df['ticker']
        
        # Map OHLCV columns
        column_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }
        
        for std_col, bbg_col in column_map.items():
            if bbg_col in df.columns:
                result_df[std_col] = pd.to_numeric(df[bbg_col], errors='coerce')
            else:
                result_df[std_col] = pd.NA
        
        result_df['insertion_time'] = df['insertion_time']
        result_df['type'] = df['type']
        
        return result_df[result_cols]
    
    def _get_last_saved_date(self, ticker_dir: Path, data_type: Optional[str] = None) -> Optional[datetime.date]:
        """
        Get the last saved date for a ticker from annual parquet files.
        
        Args:
            ticker_dir: Directory containing ticker data files
            data_type: Optional filter by data type ('eod' or 'intraday'). 
                      If None, returns the last date across all types.
            
        Returns:
            Last date present in the data for the specified type, or None if no data exists
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
            # Read date and type columns if filtering by type
            columns = ["date", "type"] if data_type else ["date"]
            df = pd.read_parquet(path, columns=columns)
        except Exception:
            return None
        
        if df.empty or "date" not in df.columns:
            return None
        
        # Filter by data type if specified
        if data_type and "type" in df.columns:
            df = df[df["type"] == data_type]
            if df.empty:
                return None
        
        s = pd.to_datetime(df["date"]).dt.date
        return s.max() if not s.empty else None
    
    def _export_to_annual_files(self, df: pd.DataFrame, ticker_dir: Path) -> None:
        """
        Export normalized DataFrame to annual Parquet files.
        
        Saves data in the format {ticker_dir}/{YYYY}.parquet with incremental merging
        and deduplication. Filters by data type to avoid conflicts between EOD and intraday data.
        
        Args:
            df: Normalized DataFrame with columns: date, time, ticker, open, high, low, close, volume, insertion_time, type
            ticker_dir: Directory where to save annual files
        """
        if df.empty:
            return
        
        df_orig = df.copy()
        cols = df_orig.columns.tolist()
        
        # Get the data type from the DataFrame (should be consistent within a single call)
        data_types = df_orig["type"].unique() if "type" in df_orig.columns else [None]
        data_type = data_types[0] if len(data_types) == 1 else None
        
        # Create timestamp from date and time for grouping
        ts = pd.to_datetime(df_orig["date"].astype(str) + " " + df_orig["time"].astype(str), utc=True)
        df_orig["timestamp"] = ts
        df_orig["date_only"] = ts.dt.date
        df_orig["year"] = ts.dt.year
        
        # Get last saved date filtered by data type to avoid conflicts
        last_date = self._get_last_saved_date(ticker_dir, data_type=data_type)
        if last_date:
            # Only filter if we have a last date for this specific data type
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
                    
                    # Remove records from old data that have same (date, time, type) as new data
                    # This prevents duplicates when updating the same type of data
                    if data_type and "type" in df_old.columns and "type" in df_new.columns:
                        # Create a set of (date, time, type) tuples from new data for fast lookup
                        new_keys = set(zip(
                            pd.to_datetime(df_new["date"]).dt.date,
                            df_new["time"],
                            df_new["type"]
                        ))
                        
                        # Filter out old records that match new records
                        old_keys = zip(
                            pd.to_datetime(df_old["date"]).dt.date,
                            df_old["time"],
                            df_old["type"]
                        )
                        mask = [key not in new_keys for key in old_keys]
                        df_old = df_old[mask]
                    
                    # Also remove last date if updating same year and same type (overwrite protection)
                    if last_date and data_type and (int(y) == last_date.year):
                        if "type" in df_old.columns:
                            d_last = pd.to_datetime(last_date).date()
                            old_dates = pd.to_datetime(df_old["date"]).dt.date
                            old_types = df_old["type"]
                            # Only remove if it's the same type
                            mask = (old_dates == d_last) & (old_types == data_type)
                            df_old = df_old.loc[~mask]
                    
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
    
    # ============================================================================
    # Metodo principale per download completo
    # ============================================================================
    
    def download_data(self, ticker: str) -> None:
        """
        Download and save data for a ticker.
        
        Downloads both EOD (20 years) and intraday (last 6 months) data,
        normalizes and saves to annual parquet files.
        
        Args:
            ticker: Ticker symbol to download
        """
        self.ticker_manager.validate_ticker(ticker)
        
        if self._ticker_exists(ticker):
            raise ValueError(
                f"Ticker {ticker} already exists. Use update_ticker() to update existing data."
            )
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=20*365)
        
        anag = self.ticker_manager.get_anag(ticker)
        if anag is None:
            raise ValueError(
                f"Ticker '{ticker}' non trovato nella configurazione. "
                f"Assicurati che il ticker sia presente in anag.json. "
                f"Nota: usa solo il simbolo (es. 'SX5E'), non il Bloomberg ticker completo (es. 'SX5E Index')."
            )
        
        bbg_ticker = anag['bbg_ticker']
        logger.info(f"Starting download for new ticker: {ticker} (Bloomberg: {bbg_ticker})")
        
        # Create ticker directory
        ticker_dir = self._get_ticker_directory(ticker)
        ticker_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {ticker_dir}")
        
        # Download EOD data
        logger.info(f"Downloading EOD data for {ticker} (from {start_date} to {end_date})")
        df_eod_raw = self.fetch(
            symbol=bbg_ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe='day'
        )
        
        if not df_eod_raw.empty:
            df_eod = self._normalize_dataframe(df_eod_raw, ticker, 'eod')
            self._export_to_annual_files(df_eod, ticker_dir)
            logger.info(f"Saved {len(df_eod)} EOD bars to annual files")
        else:
            logger.warning(f"No EOD data retrieved for {ticker}")
        
        # Download intraday data (1 minute) 
        logger.info(f"Downloading 1-minute intraday data for {ticker}")
        intraday_start = end_date - timedelta(days=365)  
        df_intraday_raw = self.fetch(
            symbol=bbg_ticker,
            start_date=intraday_start,
            end_date=end_date,
            timeframe='1 min'
        )
        
        if not df_intraday_raw.empty:
            df_intraday = self._normalize_dataframe(df_intraday_raw, ticker, 'intraday')
            self._export_to_annual_files(df_intraday, ticker_dir)
            logger.info(f"Saved {len(df_intraday)} intraday bars to annual files")
        else:
            logger.warning(f"No intraday data retrieved for {ticker}")
        
        logger.info(f"Successfully downloaded data for {ticker}")
    
    def update_ticker(self, ticker: str) -> None:
        """
        Update existing ticker data by downloading only new data.
        
        Downloads EOD data from the last saved date and intraday data from 3 days before
        the last EOD date, normalizes and saves to annual parquet files.
        
        Args:
            ticker: Ticker symbol to update
        """
        self.ticker_manager.validate_ticker(ticker)
        
        # Check if ticker exists
        if not self._ticker_exists(ticker):
            raise ValueError(
                f"Ticker {ticker} does not exist. Use download_data() to download initial data."
            )
        
        anag = self.ticker_manager.get_anag(ticker)
        if anag is None:
            raise ValueError(
                f"Ticker '{ticker}' non trovato nella configurazione. "
                f"Assicurati che il ticker sia presente in anag.json. "
                f"Nota: usa solo il simbolo (es. 'SX5E'), non il Bloomberg ticker completo (es. 'SX5E Index')."
            )
        
        bbg_ticker = anag['bbg_ticker']
        logger.info(f"Updating data for existing ticker: {ticker} (Bloomberg: {bbg_ticker})")
        
        ticker_dir = self._get_ticker_directory(ticker)
        end_date = datetime.now().date()
        
        # Get last saved date for EOD data
        last_date_eod = self._get_last_saved_date(ticker_dir, data_type='eod')
        
        if last_date_eod:
            # Update from last saved date
            start_date = last_date_eod
            logger.info(f"Updating EOD data from {last_date_eod} to {end_date}")
        else:
            # No existing data, download full history
            start_date = end_date - timedelta(days=20*365)
            logger.info(f"No existing EOD data found, downloading full EOD history")
        
        # Update EOD data
        if start_date < end_date:
            df_eod_raw = self.fetch(
                symbol=bbg_ticker,
                start_date=start_date,
                end_date=end_date,
                timeframe='day'
            )
            
            if not df_eod_raw.empty:
                df_eod = self._normalize_dataframe(df_eod_raw, ticker, 'eod')
                self._export_to_annual_files(df_eod, ticker_dir)
                logger.info(f"Saved {len(df_eod)} EOD bars to annual files")
            else:
                logger.warning(f"No new EOD data retrieved for {ticker}")
        
        # Update intraday data (from 3 days before last EOD date)
        if last_date_eod:
            intraday_start = last_date_eod - timedelta(days=3)
            logger.info(f"Updating 1-minute intraday data for {ticker} (from {intraday_start} to {end_date})")
        else:
            # If no EOD data exists, use start_date - 3 days as fallback
            intraday_start = start_date - timedelta(days=3)
            logger.info(f"Updating 1-minute intraday data for {ticker} (from {intraday_start} to {end_date})")
        
        df_intraday_raw = self.fetch(
            symbol=bbg_ticker,
            start_date=intraday_start,
            end_date=end_date,
            timeframe='1 min'
        )
        
        if not df_intraday_raw.empty:
            df_intraday = self._normalize_dataframe(df_intraday_raw, ticker, 'intraday')
            self._export_to_annual_files(df_intraday, ticker_dir)
            logger.info(f"Saved {len(df_intraday)} intraday bars to annual files")
        else:
            logger.warning(f"No intraday data retrieved for {ticker}")
        
        logger.info(f"Successfully updated data for {ticker}")


def download_or_update(ticker):
    """
    Download or update ticker data.
    
    Args:
        ticker: Ticker symbol (e.g., 'SX5E')
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker non può essere vuoto")
    
    ticker = ticker.strip()
    print(f"Elaborazione ticker: {ticker}")
    
    downloader = BloombergDataDownloader()
    if downloader._ticker_exists(ticker):
        downloader.update_ticker(ticker)
    else:
        downloader.download_data(ticker)
