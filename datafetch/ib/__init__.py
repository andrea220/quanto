"""
quanto-ib: Interactive Brokers Data Retrieval System

A Python library for retrieving historical OHLCV data, real-time bid/ask
data, news, volatility data, and trading from Interactive Brokers using ib_insync.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from datafetch.ib.connection import IBConnection
from datafetch.ib.fetch import HistoricalDataFetcher
from datafetch.ib.download_data import (
    download_data,
    update_ticker,
    get_or_update_ticker,
    list_available_tickers,
    get_ticker_info,
    load_contract_details
)

# Optional imports for modules that may not exist yet
try:
    from datafetch.ib.realtime_data import RealtimeDataStreamer
except ImportError:
    RealtimeDataStreamer = None

try:
    from datafetch.ib.news import NewsFetcher
except ImportError:
    NewsFetcher = None

try:
    from datafetch.ib.volatility import VolatilityFetcher
except ImportError:
    VolatilityFetcher = None

try:
    from datafetch.ib.fundamental_data import (
        download_fundamental_data,
        update_fundamental_data,
        get_or_update_fundamental_data,
        load_fundamental_data,
        get_fundamental_info,
        FUNDAMENTAL_REPORT_TYPES
    )
except ImportError:
    download_fundamental_data = None
    update_fundamental_data = None
    get_or_update_fundamental_data = None
    load_fundamental_data = None
    get_fundamental_info = None
    FUNDAMENTAL_REPORT_TYPES = None

try:
    from datafetch.ib.data_manager import DataManager
except ImportError:
    DataManager = None

__all__ = [
    'IBConnection',
    'HistoricalDataFetcher',
    'RealtimeDataStreamer',
    'NewsFetcher',
    'VolatilityFetcher',
    'DataManager',
    'download_data',
    'update_ticker',
    'get_or_update_ticker',
    'list_available_tickers',
    'get_ticker_info',
    'load_contract_details',
    'download_fundamental_data',
    'update_fundamental_data',
    'get_or_update_fundamental_data',
    'load_fundamental_data',
    'get_fundamental_info',
    'FUNDAMENTAL_REPORT_TYPES',
]

