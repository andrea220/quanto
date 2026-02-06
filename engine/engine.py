from engine.factor import Factor
from engine.datafeed import DataFeed

import polars as pl
from typing import List, Optional

class Researcher:
    """Research engine for computing factors on market data.
    
    Loads all data into memory and computes factors in a single pass.
    Data is cached after first load for fast subsequent access.
    """
    
    def __init__(self,
                 factors: list[Factor],
                 feed: DataFeed,
                 start_date: str,
                 end_date: str,
                 frequency: str,
                 tickers: List[str]):
        """Initialize the researcher.
        
        Parameters:
        -----------
        factors : list[Factor]
            List of factors to compute on the data
        feed : DataFeed
            Data feed to load market data from
        start_date : str
            Start date for data loading (format: YYYY-MM-DD)
        end_date : str
            End date for data loading (format: YYYY-MM-DD)
        frequency : str
            Data frequency (e.g., '1d', '1h', '1m')
        tickers : List[str]
            List of ticker symbols to load
        """
        self.factors = factors
        self.feed = feed
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.tickers = tickers
        self.max_warmup = max([f.warmup() for f in factors], default=0)
        self._data_cache: Optional[pl.DataFrame] = None

    def get_data(self) -> Optional[pl.DataFrame]:
        """Load all data with computed factors.
        
        Data is cached after first load for fast subsequent access.
        Call this method multiple times without performance penalty.
        
        Returns:
        --------
        pl.DataFrame or None
            DataFrame with computed factors, or None if no data available.
            Warmup rows (where factors are null) are automatically removed.
        """
        # Return cached data if available
        if self._data_cache is not None:
            return self._data_cache
        
        # Load all data at once
        df = self.feed.scan_prices(
            self.start_date, 
            self.end_date, 
            self.frequency, 
            self.tickers
        ).collect(engine="streaming")
        
        if df.height == 0:
            return None
        
        # Compute all factors
        for factor in self.factors:
            df = factor.compute(df)
        
        # Remove warmup rows (rows where any factor is null)
        if self.factors:
            # Get all column names from all factors
            factor_columns = []
            for f in self.factors:
                factor_columns.extend(f.get_column_names())
            
            valid_mask = pl.all_horizontal([
                pl.col(col).is_not_null() for col in factor_columns
            ])
            valid_indices = df.with_row_index().filter(valid_mask)
            
            if valid_indices.height > 0:
                first_valid_idx = valid_indices["index"].min()
                df = df.slice(first_valid_idx, df.height - first_valid_idx)
        
        # Cache the result
        self._data_cache = df
        return df
    
    def plot(
        self,
        ticker: Optional[str] = None,
        plot_factors: Optional[List] = None,
        title: Optional[str] = None,
        height: int = 600,
        width: Optional[int] = None,
        chart_type: str = 'line',
        theme: str = 'plotly_white'
    ):
        """
        Create an interactive plot with price data and factors.
        
        This is a convenience method that uses the Plotter class internally.
        All factors from the researcher are automatically plotted unless
        plot_factors is specified.
        
        Parameters:
        -----------
        ticker : str, optional
            Specific ticker to plot. If None, uses first ticker from tickers list.
        plot_factors : List[Factor], optional
            List of specific factors to plot. If None, plots all factors.
        title : str, optional
            Chart title. Defaults to "{ticker} - Price and Indicators"
        height : int, default=600
            Height of the plot in pixels
        width : int, optional
            Width of the plot in pixels. If None, uses full width
        chart_type : str, default='line'
            Type of price chart: 'line' for line chart, 'candlestick' for OHLC candlestick chart
        theme : str, default='plotly_white'
            Plotly template theme
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object that can be displayed with fig.show()
        
        Examples:
        ---------
        >>> researcher = Researcher(factors, feed, start_date, end_date, frequency, tickers)
        >>> fig = researcher.plot()
        >>> fig.show()
        >>> 
        >>> # Plot specific ticker with candlestick
        >>> fig = researcher.plot(ticker='SX5E', chart_type='candlestick')
        >>> fig.show()
        """
        from engine.plotting import Plotter
        
        plotter = Plotter(researcher=self)
        return plotter.plot(
            ticker=ticker,
            plot_factors=plot_factors,
            title=title,
            height=height,
            width=width,
            chart_type=chart_type,
            theme=theme
        )
