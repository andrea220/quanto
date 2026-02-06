"""
Plotting module for financial data visualization.

This module provides tools for creating interactive plots of price data and technical indicators.
Supports Polars DataFrames with OHLCV data and custom factors/indicators.
"""

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .factor import Factor
    from .engine import Researcher


class Plotter:
    """
    A class for creating financial charts with price data and indicators.
    
    Can be initialized with either a Researcher object (recommended) or a DataFrame.
    
    Parameters:
    -----------
    researcher : Researcher, optional
        Researcher object containing factors and data. If provided, automatically
        uses factors and their plot configurations.
    df : pl.DataFrame, optional
        Polars DataFrame containing financial data with columns:
        - date or timestamp: datetime column
        - ticker: security identifier
        - close: closing prices
        - open, high, low, volume (optional, for future candlestick support)
        Required if researcher is not provided.
    ticker : str, optional
        Specific ticker to plot. If None and df contains multiple tickers,
        will plot the first ticker found. Ignored if researcher is provided.
    """
    
    def __init__(
        self, 
        researcher: Optional['Researcher'] = None,
        df: Optional[pl.DataFrame] = None, 
        ticker: Optional[str] = None
    ):
        if researcher is not None:
            # Initialize from Researcher
            self.researcher = researcher
            self.factors = researcher.factors
            self.df = researcher.get_data()
            if self.df is None:
                raise ValueError("Researcher returned no data")
            
            # Store default ticker (first from researcher's tickers list)
            self.default_ticker = ticker if ticker is not None else (researcher.tickers[0] if researcher.tickers else None)
            if self.default_ticker and self.default_ticker not in researcher.tickers:
                raise ValueError(f"Ticker '{self.default_ticker}' not in researcher's tickers list")
        elif df is not None:
            # Initialize from DataFrame (backward compatibility)
            self.researcher = None
            self.factors = []
            self.df = df
            self.default_ticker = ticker
        else:
            raise ValueError("Either 'researcher' or 'df' must be provided")
        
        # Identify the time column (either 'date' or 'timestamp')
        self.time_col = None
        for col in ['date', 'timestamp', 'time']:
            if col in self.df.columns:
                self.time_col = col
                break
        
        if self.time_col is None:
            raise ValueError("DataFrame must contain a 'date' or 'timestamp' column")
        
        # Check required columns
        if 'close' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
    
    def plot(
        self, 
        ticker: Optional[str] = None,
        plot_factors: Optional[List[Union[str, 'Factor']]] = None,
        title: Optional[str] = None,
        height: int = 600,
        width: Optional[int] = None,
        chart_type: str = 'line',
        theme: str = 'plotly_white'
    ) -> go.Figure:
        """
        Create an interactive plot with close price and optional indicators.
        
        Parameters:
        -----------
        ticker : str, optional
            Specific ticker to plot. If None, uses the default ticker from initialization
            or the first ticker available in the data.
        plot_factors : List[Union[str, Factor]], optional
            List of factor names (strings) or Factor objects to plot.
            If None and Plotter was initialized with Researcher, automatically
            plots all factors from the researcher.
            Factor objects automatically configure panel placement and styling.
            Factors with panel='price' plot with price data (top panel).
            Factors with panel='indicator' plot in separate subplot below.
        title : str, optional
            Chart title. Defaults to "{ticker} - Price and Indicators"
        height : int, default=600
            Height of the plot in pixels
        width : int, optional
            Width of the plot in pixels. If None, uses full width
        chart_type : str, default='line'
            Type of price chart: 'line' for line chart of close prices,
            'candlestick' for OHLC candlestick chart
        theme : str, default='plotly_white'
            Plotly template theme
        
        Returns:
        --------
        go.Figure
            Plotly figure object that can be displayed with fig.show()
        """
        # Determine which ticker to use
        plot_ticker = ticker if ticker is not None else self.default_ticker
        
        # If still no ticker, try to get from data
        if plot_ticker is None:
            if 'ticker' in self.df.columns:
                available_tickers = self.df.select(pl.col('ticker')).unique().to_series().to_list()
                if available_tickers:
                    plot_ticker = available_tickers[0]
                else:
                    plot_ticker = "Unknown"
            else:
                plot_ticker = "Unknown"
        
        # Validate ticker if researcher is used
        if self.researcher is not None and plot_ticker not in self.researcher.tickers:
            raise ValueError(f"Ticker '{plot_ticker}' not in researcher's tickers list: {self.researcher.tickers}")
        
        # Filter data by ticker
        if 'ticker' in self.df.columns:
            df_filtered = self.df.filter(pl.col('ticker') == plot_ticker)
        else:
            df_filtered = self.df
        
        # Sort by time
        df_filtered = df_filtered.sort(self.time_col)
        
        # Auto-use factors from researcher if not specified
        if plot_factors is None:
            if self.researcher is not None:
                plot_factors = self.factors
            else:
                plot_factors = []
        
        # Validate chart type
        if chart_type not in ['line', 'candlestick']:
            raise ValueError("chart_type must be 'line' or 'candlestick'")
        
        # Check for OHLC columns if candlestick requested
        if chart_type == 'candlestick':
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(
                    f"Candlestick chart requires OHLC columns. Missing: {missing_cols}"
                )
        
        # Parse factors and organize by panel
        price_panel_factors = []
        indicator_panel_factors = []
        
        for factor in plot_factors:
            factor_info = self._parse_factor(factor)
            
            # Validate all columns exist in dataframe
            for col_name in factor_info['column_names']:
                if col_name not in df_filtered.columns:
                    raise ValueError(f"Factor column '{col_name}' not found in DataFrame")
            
            # Organize by panel
            if factor_info['panel'] == 'price':
                price_panel_factors.append(factor_info)
            else:  # 'indicator'
                indicator_panel_factors.append(factor_info)
        
        # Determine subplot configuration
        # Each indicator factor gets its own panel
        n_indicator_panels = len(indicator_panel_factors)
        has_indicator_panel = n_indicator_panels > 0
        n_rows = 1 + n_indicator_panels  # 1 for price + indicators
        
        # Set default title
        if title is None:
            chart_name = "Candlestick" if chart_type == 'candlestick' else "Price"
            title = f"{plot_ticker} - {chart_name} and Indicators"
        
        # Create subplots with appropriate configuration
        if has_indicator_panel:
            # Calculate row heights: Price: 50%, Indicators: 50% (split equally)
            price_height = 0.5
            indicator_total = 0.5
            indicator_height = indicator_total / n_indicator_panels if n_indicator_panels > 0 else 0
            
            row_heights = [price_height]
            row_heights.extend([indicator_height] * n_indicator_panels)
            
            # Build specs: price panel has secondary y-axis, indicator panels don't
            specs = [[{"secondary_y": True}]]  # Price panel
            specs.extend([[{"secondary_y": False}]] * n_indicator_panels)  # Indicator panels
            
            # Build subplot titles
            subplot_titles = [title]
            for factor_info in indicator_panel_factors:
                # Use first column name or base name for title
                title_name = factor_info['column_names'][0]
                # Remove suffix like '_upper', '_lower', '_middle' for cleaner title
                if '_' in title_name:
                    base_name = '_'.join(title_name.split('_')[:-1])
                    if base_name:
                        title_name = base_name
                subplot_titles.append(title_name)
            
            fig = make_subplots(
                rows=n_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=row_heights,
                specs=specs,
                subplot_titles=subplot_titles
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=[title]
            )
        
        # Get time data
        time_data = df_filtered.select(pl.col(self.time_col)).to_series()
        
        # Plot price (line or candlestick)
        if chart_type == 'line':
            close_data = df_filtered.select(pl.col('close')).to_series()
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=close_data,
                    name='Close',
                    line=dict(color='#2E86AB', width=2),
                    mode='lines'
                ),
                row=1, col=1,
                secondary_y=False
            )
        else:  # candlestick
            open_data = df_filtered.select(pl.col('open')).to_series()
            high_data = df_filtered.select(pl.col('high')).to_series()
            low_data = df_filtered.select(pl.col('low')).to_series()
            close_data = df_filtered.select(pl.col('close')).to_series()
            
            fig.add_trace(
                go.Candlestick(
                    x=time_data,
                    open=open_data,
                    high=high_data,
                    low=low_data,
                    close=close_data,
                    name='OHLC',
                    increasing_line_color='#26A69A',  # Green for up candles
                    decreasing_line_color='#EF5350',  # Red for down candles
                    xaxis='x',
                    yaxis='y'
                ),
                row=1, col=1,
                secondary_y=False
            )
            
            # Explicitly disable rangeslider for candlestick charts
            fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        
        # Plot factors in price panel
        for factor_info in price_panel_factors:
            self._add_factor_traces(fig, time_data, factor_info, row=1, df_filtered=df_filtered)
        
        # Plot each indicator factor in its own panel (starting from row 2)
        for idx, factor_info in enumerate(indicator_panel_factors):
            self._add_factor_traces(fig, time_data, factor_info, row=2 + idx, df_filtered=df_filtered)
        
        # Update layout
        fig.update_layout(
            template=theme,
            height=height,
            width=width,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update x-axes
        # Disable rangeslider on all rows, especially important for candlestick charts
        for row in range(1, n_rows + 1):
            fig.update_xaxes(
                rangeslider=dict(visible=False),
                type='date',
                row=row, col=1
            )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
        
        # Update y-axis labels for each indicator panel
        for idx, factor_info in enumerate(indicator_panel_factors):
            # Use first column name for y-axis label
            y_label = factor_info['column_names'][0]
            # Clean up label (remove suffixes for cleaner display)
            if '_' in y_label:
                base_name = '_'.join(y_label.split('_')[:-1])
                if base_name:
                    y_label = base_name
            fig.update_yaxes(title_text=y_label, row=2 + idx, col=1)
        
        return fig
    
    def _parse_factor(self, factor: Union[str, 'Factor']) -> dict:
        """
        Parse a factor (string or Factor object) into plotting configuration.
        
        For Factor objects, extracts plot configuration and handles multi-column factors
        (like DonchianChannels which has upper, lower, middle columns).
        
        Returns:
        --------
        dict with keys: column_names (list), panel, color, line_style, line_width, opacity, show_in_legend
        """
        if isinstance(factor, str):
            # String factor name - use defaults
            return {
                'column_names': [factor],  # Single column
                'panel': 'price',  # Default to price panel for string names
                'color': None,  # Will be auto-assigned
                'line_style': 'solid',
                'line_width': 1.5,
                'opacity': 0.8,
                'show_in_legend': True
            }
        else:
            # Factor object - extract configuration
            plot_cfg = factor.plot_config
            column_names = factor.get_column_names()
            
            return {
                'column_names': column_names,  # Can be multiple columns
                'panel': plot_cfg.panel,
                'color': plot_cfg.color,
                'line_style': plot_cfg.line_style,
                'line_width': plot_cfg.line_width,
                'opacity': plot_cfg.opacity,
                'show_in_legend': plot_cfg.show_in_legend
            }
    
    def _add_factor_traces(
        self, 
        fig: go.Figure, 
        time_data: pl.Series, 
        factor_info: dict, 
        row: int,
        df_filtered: pl.DataFrame
    ):
        """
        Add factor trace(s) to the figure. Handles both single-column and multi-column factors.
        
        Parameters:
        -----------
        fig : go.Figure
            The plotly figure to add trace to
        time_data : pl.Series
            Time series data for x-axis
        factor_info : dict
            Factor configuration dictionary from _parse_factor
        row : int
            Subplot row number
        df_filtered : pl.DataFrame
            Filtered DataFrame for the specific ticker
        """
        column_names = factor_info['column_names']
        base_color = factor_info['color']
        
        # Map line style names to plotly dash values
        dash_map = {
            'solid': 'solid',
            'dash': 'dash',
            'dot': 'dot',
            'dashdot': 'dashdot'
        }
        
        # Color palette for multi-column factors (e.g., DonchianChannels)
        multi_colors = {
            'upper': '#26A69A',  # Green/teal for upper
            'lower': '#EF5350',  # Red for lower
            'middle': '#FFA726'  # Orange for middle
        }
        
        # For each column in the factor
        for col_name in column_names:
            # Get factor data from filtered dataframe
            factor_data = df_filtered.select(pl.col(col_name)).to_series()
            
            # Determine color
            color = base_color
            if color is None:
                # For multi-column factors, use specific colors
                if len(column_names) > 1:
                    # Extract suffix (upper, lower, middle)
                    if '_upper' in col_name:
                        color = multi_colors['upper']
                    elif '_lower' in col_name:
                        color = multi_colors['lower']
                    elif '_middle' in col_name:
                        color = multi_colors['middle']
                    else:
                        # Fallback: use hash for consistent assignment
                        color_idx = hash(col_name) % 8
                        colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                                 '#BC4B51', '#5E548E', '#E07A5F', '#3D5A80']
                        color = colors[color_idx]
                else:
                    # Single column: use hash for consistent color assignment
                    color_idx = hash(col_name) % 8
                    colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                             '#BC4B51', '#5E548E', '#E07A5F', '#3D5A80']
                    color = colors[color_idx]
            
            # Create trace
            trace = go.Scatter(
                x=time_data,
                y=factor_data,
                name=col_name,
                line=dict(
                    color=color,
                    width=factor_info['line_width'],
                    dash=dash_map.get(factor_info['line_style'], 'solid')
                ),
                mode='lines',
                opacity=factor_info['opacity'],
                showlegend=factor_info['show_in_legend']
            )
            
            # Add trace to appropriate subplot
            if row == 1:
                # Price panel - use primary y-axis
                fig.add_trace(trace, row=row, col=1, secondary_y=False)
            else:
                # Indicator panel - no secondary y-axis
                fig.add_trace(trace, row=row, col=1)
    
    def _should_use_secondary_axis(self, price_data: pl.Series, indicator_data: pl.Series) -> bool:
        """
        Determine if an indicator should be plotted on secondary y-axis.
        Uses the ratio of ranges to decide.
        """
        # Skip if indicator has null values
        if indicator_data.null_count() == len(indicator_data):
            return False
        
        # Calculate ranges (excluding nulls)
        price_range = price_data.max() - price_data.min()
        indicator_range = indicator_data.max() - indicator_data.min()
        
        if price_range == 0 or indicator_range == 0:
            return False
        
        # Calculate scale difference
        ratio = max(price_range, indicator_range) / min(price_range, indicator_range)
        
        # Use secondary axis if scales differ by more than 10x
        return ratio > 10
    
    def plot_candlestick(
        self,
        ticker: Optional[str] = None,
        plot_factors: Optional[List[Union[str, 'Factor']]] = None,
        title: Optional[str] = None,
        height: int = 600,
        width: Optional[int] = None,
        theme: str = 'plotly_white'
    ) -> go.Figure:
        """
        Convenience method to create candlestick chart.
        Wrapper around plot() with chart_type='candlestick'.
        
        Parameters:
        -----------
        ticker : str, optional
            Specific ticker to plot
        plot_factors : List[Union[str, Factor]], optional
            List of indicators to plot
        title : str, optional
            Chart title
        height : int, default=600
            Height of the plot
        width : int, optional
            Width of the plot
        theme : str, default='plotly_white'
            Plotly theme
        
        Returns:
        --------
        go.Figure
            Plotly figure with candlestick chart
        """
        return self.plot(
            ticker=ticker,
            plot_factors=plot_factors,
            title=title,
            height=height,
            width=width,
            chart_type='candlestick',
            theme=theme
        )


def plot_timeseries(
    researcher: Optional['Researcher'] = None,
    df: Optional[pl.DataFrame] = None,
    ticker: Optional[str] = None,
    plot_factors: Optional[List[Union[str, 'Factor']]] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: Optional[int] = None,
    chart_type: str = 'line',
    theme: str = 'plotly_white'
) -> go.Figure:
    """
    Convenience function to quickly plot financial time series data.
    
    Can be used with either a Researcher object (recommended) or a DataFrame.
    
    Parameters:
    -----------
    researcher : Researcher, optional
        Researcher object containing factors and data. If provided, automatically
        uses factors and their plot configurations.
    df : pl.DataFrame, optional
        Polars DataFrame with financial data. Required if researcher is not provided.
    ticker : str, optional
        Specific ticker to plot. Ignored if researcher is provided.
    plot_factors : List[Union[str, Factor]], optional
        List of indicator column names (strings) or Factor objects to plot.
        If None and researcher is provided, plots all factors automatically.
        Factor objects automatically configure panel placement and styling.
    title : str, optional
        Chart title
    height : int, default=600
        Height in pixels
    width : int, optional
        Width in pixels
    chart_type : str, default='line'
        Type of price chart: 'line' for line chart, 'candlestick' for OHLC candles
    theme : str, default='plotly_white'
        Plotly theme
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    
    Examples:
    ---------
    >>> from engine.plotting import plot_timeseries
    >>> from engine.engine import Researcher
    >>> 
    >>> # Using Researcher (recommended)
    >>> researcher = Researcher(factors, feed, start_date, end_date, frequency, tickers)
    >>> fig = plot_timeseries(researcher=researcher)
    >>> fig.show()
    >>> 
    >>> # Using DataFrame (backward compatibility)
    >>> fig = plot_timeseries(df=df, ticker='SX5E', 
    ...                       plot_factors=['ma_20', 'ma_50'])
    >>> fig.show()
    >>> 
    >>> # Candlestick chart with Researcher
    >>> fig = plot_timeseries(researcher=researcher, chart_type='candlestick')
    >>> fig.show()
    """
    plotter = Plotter(researcher=researcher, df=df, ticker=ticker)
    return plotter.plot(
        ticker=ticker,
        plot_factors=plot_factors,
        title=title,
        height=height,
        width=width,
        chart_type=chart_type,
        theme=theme
    )


def plot_portfolio_balance(
    daily_equity,
    starting_balance: float,
    benchmark_equity=None,
    title: str = "Strategy Value Over Time",
    height: int = 600,
    width: Optional[int] = None,
    theme: str = 'plotly_white',
    show_starting_line: bool = True
) -> go.Figure:
    """
    Plot portfolio equity curve over time with optional benchmark comparison.
    
    This function creates an interactive plot showing the evolution of portfolio value
    over time. It supports comparison with a benchmark strategy and handles both
    empty portfolios (no trades) and active portfolios.
    
    Parameters:
    -----------
    daily_equity : pd.DataFrame or dict-like
        DataFrame with columns 'ref_date' and 'daily_equity', or a dict-like object
        with these attributes. If empty or None, shows starting balance line.
    starting_balance : float
        Initial portfolio value
    benchmark_equity : pd.DataFrame or dict-like, optional
        Benchmark equity data with same structure as daily_equity.
        If provided, will be plotted as comparison line.
    title : str, default="Strategy Value Over Time"
        Chart title
    height : int, default=600
        Chart height in pixels
    width : int, optional
        Chart width in pixels. If None, uses full width
    theme : str, default='plotly_white'
        Plotly template theme ('plotly', 'plotly_white', 'plotly_dark', etc.)
    show_starting_line : bool, default=True
        Whether to show horizontal line at starting balance
    
    Returns:
    --------
    go.Figure
        Interactive Plotly figure object. Call .show() to display.
    
    Examples:
    ---------
    >>> from finresearch.plotting import plot_portfolio_balance
    >>> # Simple equity curve
    >>> fig = plot_portfolio_balance(
    ...     analytics.daily_equity,
    ...     starting_balance=10000
    ... )
    >>> fig.show()
    >>> 
    >>> # With benchmark comparison
    >>> fig = plot_portfolio_balance(
    ...     strategy_analytics.daily_equity,
    ...     starting_balance=10000,
    ...     benchmark_equity=buyhold_analytics.daily_equity,
    ...     title="Strategy vs Buy & Hold"
    ... )
    >>> fig.show()
    """
    import pandas as pd
    
    fig = go.Figure()
    
    # Check if daily_equity is empty
    is_empty = (
        daily_equity is None or 
        (hasattr(daily_equity, 'empty') and daily_equity.empty) or
        (hasattr(daily_equity, '__len__') and len(daily_equity) == 0)
    )
    
    if is_empty:
        # No trades - just show starting balance
        fig.add_hline(
            y=starting_balance, 
            line_dash="dash", 
            line_color="#666666",
            line_width=2,
            annotation_text=f"Starting Balance: ${starting_balance:,.2f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title={
                'text': title + " (No Trades)",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template=theme,
            height=height,
            width=width,
            hovermode='x unified',
            showlegend=True,
            yaxis=dict(
                tickformat='$,.0f',
                gridcolor='rgba(128,128,128,0.2)'
            ),
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.2)'
            )
        )
        return fig
    
    # Convert dates to datetime
    dates = pd.to_datetime(daily_equity['ref_date'])
    equity_values = daily_equity['daily_equity']
    
    # Calculate statistics for annotations
    total_return = ((equity_values.iloc[-1] - starting_balance) / starting_balance * 100) if len(equity_values) > 0 else 0
    max_value = equity_values.max()
    min_value = equity_values.min()
    
    # Plot main strategy equity
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_values,
        name="Strategy",
        line=dict(color='#2E86AB', width=3),
        mode='lines',
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                      '<b>Value:</b> $%{y:,.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Plot benchmark if provided
    if benchmark_equity is not None:
        is_benchmark_empty = (
            hasattr(benchmark_equity, 'empty') and benchmark_equity.empty or
            hasattr(benchmark_equity, '__len__') and len(benchmark_equity) == 0
        )
        
        if not is_benchmark_empty:
            benchmark_dates = pd.to_datetime(benchmark_equity['ref_date'])
            benchmark_values = benchmark_equity['daily_equity']
            
            fig.add_trace(go.Scatter(
                x=benchmark_dates,
                y=benchmark_values,
                name="Benchmark",
                line=dict(color='#26A69A', width=2.5, dash='dot'),
                mode='lines',
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                              '<b>Value:</b> $%{y:,.2f}<br>' +
                              '<extra></extra>'
            ))
    
    # Add starting balance reference line
    if show_starting_line:
        fig.add_hline(
            y=starting_balance,
            line_dash="dash",
            line_color="#666666",
            line_width=1.5,
            opacity=0.6,
            annotation_text=f"Start: ${starting_balance:,.0f}",
            annotation_position="left",
            annotation=dict(font_size=10, font_color="#666666")
        )
    
    # Update layout with professional styling
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Total Return: {total_return:+.2f}%</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template=theme,
        height=height,
        width=width,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        yaxis=dict(
            tickformat='$,.0f',
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.3)',
            zerolinewidth=1
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            type='date'
        ),
        plot_bgcolor='white',
        margin=dict(t=100, b=60, l=80, r=40)
    )
    
    return fig
