
"""Performance reporting and analysis module.

This module provides functionality for collecting, analyzing, and exporting
backtest performance data including trades, equity curves, positions, and
drawdown analysis.

Key Classes:
- ReportWriter: Collects and exports backtest performance data
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import polars as pl
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from io import StringIO
import plotly.graph_objects as go
from .plotting import plot_portfolio_balance 

class StrategyAnalytics:
    def __init__(self, strategy):
        self.strategy = strategy
        self.trades_df = strategy.positions_summary.copy()
        
        # Handle empty positions_summary DataFrame
        if self.trades_df.empty:
            # Create empty daily_equity with just the starting balance
            self.daily_equity = pd.DataFrame({
                'ref_date': [],
                'daily_equity': []
            })
        else:
            self.daily_equity = self.strategy.positions_summary.groupby('ref_date')['global_pnl'].sum().reset_index() 
            self.daily_equity = self.daily_equity.rename(columns={'global_pnl': 'daily_equity'})
            self.daily_equity['daily_equity'] = self.daily_equity['daily_equity'] + self.strategy.starting_balance


    def plot_balance(self, benchmark=None):
        """
        Plot portfolio equity curve over time using Plotly interactive charts.
        
        Parameters:
        -----------
        benchmark : StrategyAnalytics or str, optional
            Benchmark strategy analytics object to compare against, or a ticker string.
            If StrategyAnalytics, benchmark equity will be plotted alongside strategy equity.
            If str (ticker), will calculate buy-and-hold equity for that ticker from strategy history.
        
        Returns:
        --------
        go.Figure
            Interactive Plotly figure. Automatically displays in Jupyter notebooks.
        
        Examples:
        ---------
        >>> # Simple equity curve
        >>> analytics.plot_balance()
        >>> 
        >>> # With benchmark comparison (StrategyAnalytics)
        >>> buyhold_analytics = StrategyAnalytics(buyhold_strategy)
        >>> analytics.plot_balance(benchmark=buyhold_analytics)
        >>> 
        >>> # With benchmark ticker (buy-and-hold)
        >>> analytics.plot_balance(benchmark='SPY US Equity')
        """
        # Prepare benchmark equity if provided
        benchmark_equity = None
        if benchmark is not None:
            if isinstance(benchmark, str):
                # benchmark is a ticker string - calculate buy-and-hold equity
                if self.strategy.history is None or self.strategy.history.height == 0:
                    raise ValueError(f"Cannot calculate benchmark: strategy.history is empty")
                
                # Filter history for the ticker
                df_ticker = self.strategy.history.filter(pl.col('ticker') == benchmark)
                
                if df_ticker.height == 0:
                    raise ValueError(f"Ticker '{benchmark}' not found in strategy history")
                
                # Group by date and get the last close price of each day
                # Sort by date and time to ensure we get the last price of the day
                df_ticker = df_ticker.sort(['date', 'time'])
                daily_prices = df_ticker.group_by('date').agg(
                    pl.col('close').last().alias('close')
                ).sort('date')
                
                if daily_prices.height == 0:
                    raise ValueError(f"No daily prices found for ticker '{benchmark}'")
                
                # Convert to pandas for easier manipulation
                daily_prices_pd = daily_prices.to_pandas()
                
                # Calculate equity curve: starting_balance * (close_price / initial_close_price)
                initial_price = daily_prices_pd['close'].iloc[0]
                if initial_price == 0:
                    raise ValueError(f"Initial price for ticker '{benchmark}' is zero")
                
                daily_prices_pd['daily_equity'] = (
                    self.strategy.starting_balance * 
                    (daily_prices_pd['close'] / initial_price)
                )
                
                # Create benchmark_equity DataFrame with same structure as daily_equity
                benchmark_equity = pd.DataFrame({
                    'ref_date': daily_prices_pd['date'],
                    'daily_equity': daily_prices_pd['daily_equity']
                })
            else:
                # benchmark is a StrategyAnalytics object
                benchmark_equity = benchmark.daily_equity
        
        # Call the plotting function
        fig = plot_portfolio_balance(
            daily_equity=self.daily_equity,
            starting_balance=self.strategy.starting_balance,
            benchmark_equity=benchmark_equity,
            title="Strategy Value Over Time",
            height=600
        )
        
        # Show the figure (works in Jupyter/IPython)
        # fig.show()
        
        return fig
    
    def plot_price_with_signals(self, 
                                ticker: str,
                                plot_factors: Optional[List] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                chart_type: str = 'line',
                                height: int = 600):
        """
        Plot price chart with trading signals overlaid.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to plot
        plot_factors : List, optional
            List of factor names or Factor objects to overlay
        start_date : str, optional
            Start date filter (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date filter (format: 'YYYY-MM-DD')
        chart_type : str, default='line'
            Chart type: 'line' or 'candlestick'
        height : int, default=600
            Chart height in pixels
        
        Returns:
        --------
        go.Figure
            Interactive Plotly figure with price and signals
        
        Examples:
        ---------
        >>> analytics = StrategyAnalytics(strategy)
        >>> fig = analytics.plot_price_with_signals('SPY US Equity')
        >>> fig.show()
        """
        from .plotting import plot_timeseries
        
        # Get price data from strategy history
        df = self.strategy.history
        
        # Filter by ticker
        df_ticker = df.filter(pl.col('ticker') == ticker)
        
        # Filter by date range if provided
        if start_date is not None:
            df_ticker = df_ticker.filter(pl.col('date') >= pl.lit(start_date).cast(pl.Date))
        if end_date is not None:
            df_ticker = df_ticker.filter(pl.col('date') <= pl.lit(end_date).cast(pl.Date))
        
        if df_ticker.height == 0:
            print(f"No data found for {ticker}")
            return go.Figure()
        
        # Create base price chart using plot_timeseries
        fig = plot_timeseries(
            df=df_ticker,
            ticker=ticker,
            plot_factors=plot_factors,
            title=f"{ticker} - Price with Signals",
            height=height,
            chart_type=chart_type
        )
        
        # Convert signals to DataFrame for easier processing
        signals_df = pd.DataFrame(self.strategy.signals)
        
        if not signals_df.empty:
            # Filter signals for this ticker
            ticker_signals = signals_df[signals_df['ticker'] == ticker].copy()
            
            if not ticker_signals.empty:
                # Convert datetime string to datetime
                ticker_signals['datetime'] = pd.to_datetime(ticker_signals['datetime'])
                
                # Filter by date range if provided
                if start_date is not None:
                    ticker_signals = ticker_signals[ticker_signals['datetime'] >= pd.to_datetime(start_date)]
                if end_date is not None:
                    ticker_signals = ticker_signals[ticker_signals['datetime'] <= pd.to_datetime(end_date)]
                
                # Merge with price data to get the price at signal time
                df_ticker_pd = df_ticker.to_pandas()
                df_ticker_pd['datetime'] = pd.to_datetime(df_ticker_pd['date'].astype(str) + ' ' + df_ticker_pd['time'].astype(str))
                
                # Merge signals with prices
                signals_with_price = ticker_signals.merge(
                    df_ticker_pd[['datetime', 'close']], 
                    on='datetime', 
                    how='left'
                )
                
                # Separate signals by type
                long_signals = signals_with_price[signals_with_price['signal'] == 1]
                short_signals = signals_with_price[signals_with_price['signal'] == -1]
                close_signals = signals_with_price[signals_with_price['signal'] == 0]
                
                # Add LONG signals (buy markers)
                if not long_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=long_signals['datetime'],
                        y=long_signals['close'],
                        mode='markers',
                        name='LONG',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        showlegend=True
                    ))
                
                # Add SHORT signals (sell markers)
                if not short_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=short_signals['datetime'],
                        y=short_signals['close'],
                        mode='markers',
                        name='SHORT',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        showlegend=True
                    ))
                
                # Add CLOSE signals (exit markers)
                if not close_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=close_signals['datetime'],
                        y=close_signals['close'],
                        mode='markers',
                        name='CLOSE',
                        marker=dict(
                            symbol='circle',
                            size=10,
                            color='gray',
                            line=dict(width=2, color='darkgray')
                        ),
                        showlegend=True
                    ))
        
        return fig

    def trade_log(self):
        """
        Genera un DataFrame con due righe per ogni trade: una per l'apertura e una per la chiusura.

        :return: DataFrame con trade log dettagliato.
        """
        if self.trades_df.empty:
            return pd.DataFrame()
            
        trade_log = []

        for trade_id in self.trades_df['trade_id'].unique():
            
            entry_row = self.trades_df[(self.trades_df['trade_id'] == trade_id) & (self.trades_df['is_alive'] == True)].iloc[0]
            exit_row = self.trades_df[(self.trades_df['trade_id'] == trade_id) & (self.trades_df['is_alive'] == False)].iloc[0]

            trade_log.append(entry_row.to_dict())
            trade_log.append(exit_row.to_dict())
        return pd.DataFrame(trade_log)
    
    def plot_price(self, ticker,
               plot_signals=True,
               factor=None,
               start_date=None,
               end_date=None):
        # Serie dei prezzi di chiusura
        price_series = self.backtester.prices_to_df('close')[ticker]
        price_series.index = pd.to_datetime(price_series.index)

        # Filtro per date se richiesto
        if start_date is not None or end_date is not None:
            sd = pd.to_datetime(start_date) if start_date is not None else price_series.index.min()
            ed = pd.to_datetime(end_date)   if end_date   is not None else price_series.index.max()
            price_series = price_series.loc[sd:ed]

        plt.figure(figsize=(12, 6))
        plt.plot(price_series.index, price_series.values,
                label=ticker, color="black")

        # Fattore aggiuntivo (se richiesto)
        if factor is not None:
            factor_series = self.backtester.factors_history[factor][ticker]
            factor_series.index = pd.to_datetime(factor_series.index)
            if start_date is not None or end_date is not None:
                factor_series = factor_series.loc[sd:ed]
            plt.plot(factor_series.index, factor_series.values,
                    label=factor, color="blue")

        if plot_signals:
            # recupero df dei segnali: indice a date, colonna 'signal' con -1,0,+1
            signal_df = self.signals(ticker)
            signal_df.index = pd.to_datetime(signal_df.index)

            # filtro segnali per date
            if start_date is not None or end_date is not None:
                signal_df = signal_df.loc[sd:ed]

            # riallineo i segnali ai prezzi (inner join)
            aligned = signal_df.join(price_series.rename("price"), how="inner")

            # separo i 3 casi
            buys   = aligned[ aligned['signal'] ==  1 ]
            closes = aligned[ aligned['signal'] ==  0 ]
            sells  = aligned[ aligned['signal'] == -1 ]

            # plotto i marker
            plt.scatter(buys.index,   buys['price'],   marker='^', s=100,
                        label="Buy",  c='green', edgecolors='green')
            plt.scatter(sells.index,  sells['price'],  marker='v', s=100,
                        label="Sell", c='red',  edgecolors='red')
            plt.scatter(closes.index, closes['price'], marker='o', s=100,
                        label="Close", c='gray', edgecolors='gray')

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(f"{ticker} price with signals")
        plt.legend()
        plt.grid(True)
        plt.show()

    def signals(self, symbol: str) -> pd.DataFrame:
        """
        Restituisce un DataFrame dei segnali per il simbolo specificato:
        1  = apertura long (buy)
        0  = chiusura di posizione (close)
        -1  = apertura short (sell)
        """
        # 1) filtro per symbol
        df = self.trade_log()
        if df.empty:
            return pd.DataFrame(columns=['signal'])
            
        df_sym = df[df['symbol'] == symbol].copy()
        if df_sym.empty:
            return pd.DataFrame(columns=['signal'])
        
        # 2) calcolo del segnale:
        #    quantity == 0 => close (0)
        #    quantity > 0  => apertura: segnale = side (1 per buy, -1 per sell)
        df_sym['signal'] = np.where(
            df_sym['quantity'] == 0,
            0,
            df_sym['side'].clip(-1, 1)  # garantisce -1 o +1
        )
        
        # 3) imposto ref_date come indice e ordino
        df_signals = df_sym.set_index('ref_date').sort_index()[['signal']]
        return df_signals

    def expected_return(self):
            """ Calcola il rendimento medio giornaliero. """
            if self.daily_equity.empty:
                return 0.0
            self.daily_equity[self.daily_equity == 0] = np.nan  
            returns = self.daily_equity['daily_equity'].pct_change().dropna()
            return returns.mean() if not returns.empty else 0.0

    def std_deviation(self):
        """ Calcola la std dev dei rendimento medi giornalieri. """
        if self.daily_equity.empty:
            return 0.0
        self.daily_equity[self.daily_equity == 0] = np.nan  
        returns = self.daily_equity['daily_equity'].pct_change().dropna()
        return returns.std() if not returns.empty else 0.0
    
    def sharpe_ratio(self):
        """ Calcola lo Sharpe Ratio del portafoglio con riskfree=0. """
        if self.daily_equity.empty:
            return np.nan
        self.daily_equity[self.daily_equity == 0] = np.nan  
        returns = self.daily_equity['daily_equity'].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return np.nan
        return (returns.mean()) / returns.std()

    def win_rate(self):
        """ Calcola la percentuale di trade vincenti. """
        closed_trades = self.trades_df[self.trades_df['is_alive'] == False]
        wins = (closed_trades['closed_pnl'] > 0).sum()
        total = len(closed_trades)
        return wins / total if total > 0 else 0

    def average_pnl(self):
        """ Calcola il profitto medio per trade. """
        if self.trades_df.empty:
            return 0.0
        closed_trades = self.trades_df[self.trades_df['is_alive'] == False]
        return closed_trades['closed_pnl'].mean() if not closed_trades.empty else 0.0

    def average_win(self):
        """ Calcola il P&L medio dei trade vincenti. """
        if self.trades_df.empty:
            return 0.0
        closed_trades = self.trades_df[self.trades_df['is_alive'] == False]
        wins = closed_trades[closed_trades['closed_pnl'] > 0]
        # return (wins['global_pnl']/1e6/wins['initial_spot'] ).mean()
        return wins['closed_pnl'].mean() if not wins.empty else 0.0

    def average_loss(self):
        """ Calcola il P&L medio dei trade perdenti. """
        if self.trades_df.empty:
            return 0.0
        closed_trades = self.trades_df[self.trades_df['is_alive'] == False]
        losses = closed_trades[closed_trades['closed_pnl'] < 0]
        return losses['closed_pnl'].mean() if not losses.empty else 0.0
        # return (losses['global_pnl']/1e6/losses['initial_spot'] ).mean()

    def max_profit(self):
        """ Trova il trade più redditizio. """
        if self.trades_df.empty:
            return 0.0
        return self.trades_df['closed_pnl'].max()

    def max_loss(self):
        """ Trova la peggior perdita. """
        if self.trades_df.empty:
            return 0.0
        return self.trades_df['closed_pnl'].min()

    def profit_factor(self):
        """ Calcola il profit factor (profitto totale / perdita totale). """
        if self.trades_df.empty:
            return 1.0
        closed_trades = self.trades_df[self.trades_df['is_alive'] == False]
        if closed_trades.empty:
            return 1.0
        total_wins = closed_trades[closed_trades['closed_pnl'] > 0]['closed_pnl'].sum()
        total_losses = abs(closed_trades[closed_trades['closed_pnl'] < 0]['closed_pnl'].sum())
        return total_wins / total_losses if total_losses > 0 else 1

    def number_of_trades(self):
        if self.trades_df.empty:
            return 0
        closed_trades = self.trades_df[self.trades_df['is_alive'] == False]
        return len(closed_trades['trade_id'].unique()) if not closed_trades.empty else 0
    
    def max_drawdown(self):
        """
        Calcola il massimo drawdown del portafoglio basato su global_pnl.
        
        :return: Valore massimo del drawdown.
        """
        if self.daily_equity.empty:
            return 0.0
        pnl_series = self.daily_equity['daily_equity']

        # Calcola il massimo storico fino a quel punto
        running_max = pnl_series.cummax()
        # Evitiamo la divisione per zero sostituendo gli zeri con NaN temporaneamente
        running_max[running_max == 0] = np.nan  
        drawdown = (pnl_series - running_max) / running_max  
        return drawdown.min() if not drawdown.empty else 0.0
    
    def summary(self):
        """ Restituisce un riassunto delle metriche di performance con 6 decimali formattati come stringhe. """
        def format_value(value):
            """Format numeric values, handling NaN and inf cases."""
            if pd.isna(value) or np.isinf(value):
                return "N/A"
            return f"{value:.6f}"
        
        return {
            "Expected Return": format_value(self.expected_return()),
            "Std Deviation": format_value(self.std_deviation()),
            "Sharpe Ratio": format_value(self.sharpe_ratio()),
            "N. Trade": self.number_of_trades(),  # Numero intero
            "Win Rate": format_value(self.win_rate()),
            "Average P&L": format_value(self.average_pnl()),
            "Average Win": format_value(self.average_win()),
            "Average Loss": format_value(self.average_loss()),
            "Max Profit": format_value(self.max_profit()),
            "Max Loss": format_value(self.max_loss()),
            "Profit Factor": format_value(self.profit_factor()),
            "Max Drawdown": format_value(self.max_drawdown()),
        }
    
@dataclass
class ReportWriter:
    """
    Collects and exports backtest performance data with integrated logging.
    
    The ReportWriter accumulates performance data during backtesting including
    trade fills, equity curve evolution, and position snapshots. It also provides
    comprehensive logging capabilities for strategy execution.
    
    Attributes:
        out_dir: Output directory for reports (default: "backtest_reports")
        trades: List of trade records
        equity_curve: List of equity snapshots over time
        positions_snap: List of position snapshots
    """
    out_dir: str = "backtest_reports"
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    positions_snap: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize logging after dataclass initialization."""
        self.logger = None
        self._log_stream = None
        self._log_bars = False
        self._strategy_name = "Strategy"
    
    def setup_logger(self, strategy_name: str):
        """Setup logger for the strategy.
        
        Args:
            strategy_name: Name of the strategy class for logger naming
        """
        self._strategy_name = strategy_name
        self.logger = logging.getLogger(f"Strategy.{strategy_name}")
        self.logger.setLevel(logging.INFO)
        
        # Evita duplicazione handler
        if self.logger.handlers:
            return
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def enable_file_logging(self, filepath: str = None):
        """Abilita logging su file.
        
        Args:
            filepath: Path del file di log (default: logs/{StrategyName}_backtest.log)
            
        Returns:
            Path del file di log creato
        """
        if self.logger is None:
            raise RuntimeError("Logger not initialized. Call setup_logger() first.")
        
        if filepath is None:
            filepath = f"logs/{self._strategy_name}_backtest.log"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        file_handler = logging.FileHandler(filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        self.logger.info(f"File logging enabled: {filepath}")
        return filepath
    
    def enable_memory_logging(self):
        """Cattura i log in memoria per accesso programmatico."""
        if self.logger is None:
            raise RuntimeError("Logger not initialized. Call setup_logger() first.")
        
        self._log_stream = StringIO()
        memory_handler = logging.StreamHandler(self._log_stream)
        memory_handler.setLevel(logging.DEBUG)
        memory_format = logging.Formatter(
            '%(asctime)s [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        memory_handler.setFormatter(memory_format)
        self.logger.addHandler(memory_handler)
    
    def get_logs(self) -> str:
        """Restituisce tutti i log catturati in memoria.
        
        Returns:
            String contenente tutti i log catturati
        """
        if self._log_stream is None:
            return "Memory logging not enabled. Call enable_memory_logging() before backtest."
        return self._log_stream.getvalue()
    
    def print_logs(self):
        """Stampa tutti i log catturati."""
        print(self.get_logs())
    
    def save_logs_to_file(self, filepath: str = None):
        """Salva i log catturati in un file.
        
        Args:
            filepath: Path del file (default: logs/{StrategyName}_summary.log)
            
        Returns:
            Path del file creato
        """
        if filepath is None:
            filepath = f"logs/{self._strategy_name}_summary.log"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.get_logs())
        
        print(f"Logs saved to: {filepath}")
        return filepath
    
    def set_log_level(self, level: int):
        """Imposta il livello di verbosità del logger.
        
        Args:
            level: Livello di logging (logging.DEBUG, INFO, WARNING, ERROR)
        """
        if self.logger is None:
            raise RuntimeError("Logger not initialized. Call setup_logger() first.")
        
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def enable_bar_logging(self, enabled: bool = True):
        """Abilita/disabilita logging di ogni singolo bar.
        
        Args:
            enabled: True per abilitare, False per disabilitare
        """
        self._log_bars = enabled
    
    def get_log_summary(self) -> dict:
        """Estrae statistiche dai log catturati.
        
        Returns:
            Dizionario con statistiche sui log
        """
        if self._log_stream is None:
            return {"error": "Memory logging not enabled"}
        
        logs = self.get_logs()
        lines = logs.split('\n')
        
        return {
            "total_lines": len(lines),
            "orders_submitted": len([l for l in lines if 'ORDER SUBMITTED' in l]),
            "orders_filled": len([l for l in lines if 'ORDER FILLED' in l]),
            "warnings": len([l for l in lines if 'WARNING' in l]),
            "errors": len([l for l in lines if 'ERROR' in l]),
        }

    def log_fill(self, ts: str, ticker: str, qty: float, price: float, commission_bps: float = 0.0):
        """Log a trade fill for reporting.
        
        Args:
            ts: Timestamp of the fill
            ticker: Symbol that was traded
            qty: Quantity traded (positive for buys, negative for sells)
            price: Execution price
            commission_bps: Commission rate in basis points
        """
        notional = qty * price
        commission = abs(notional) * (commission_bps / 10_000.0) if commission_bps else 0.0
        self.trades.append({
            "ts": ts, "ticker": ticker, "qty": qty, "price": price,
            "notional": notional, "commission": commission,
        })

    def log_equity(self, date_str: str, ts: str, equity: float, cash: float):
        """Log equity and cash levels.
        
        Args:
            date_str: Date string
            ts: Timestamp
            equity: Total portfolio equity
            cash: Cash balance
        """
        self.equity_curve.append({"date": date_str, "ts": ts, "equity": equity, "cash": cash})

    def log_positions(self, date_str: str, positions: Dict[str, float]):
        """Log position snapshot.
        
        Args:
            date_str: Date string
            positions: Dictionary mapping tickers to quantities
        """
        # salva come records {date, ticker, qty}
        for t, q in positions.items():
            self.positions_snap.append({"date": date_str, "ticker": t, "qty": q})

    def _to_df(self, rows: List[Dict]) -> pl.DataFrame:
        """Convert list of dictionaries to Polars DataFrame.
        
        Args:
            rows: List of dictionary records
            
        Returns:
            Polars DataFrame
        """
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def frames(self) -> Dict[str, pl.DataFrame]:
        """Get all collected data as DataFrames.
        
        Returns:
            Dictionary containing 'trades', 'equity', and 'positions' DataFrames
        """
        return {
            "trades": self._to_df(self.trades),
            "equity": self._to_df(self.equity_curve),
            "positions": self._to_df(self.positions_snap),
        }

    def save(self, fmt: str = "parquet") -> Dict[str, str]:
        """Save all collected data to disk.
        
        Exports trades, equity curve, positions, and calculates drawdown.
        Creates the output directory if it doesn't exist.
        
        Args:
            fmt: Output format - "parquet" or "csv"
            
        Returns:
            Dictionary mapping data type names to file paths
        """
        os.makedirs(self.out_dir, exist_ok=True)
        dfs = self.frames()
        paths = {}
        for name, df in dfs.items():
            if df.height == 0:
                continue
            path = os.path.join(self.out_dir, f"{name}.{fmt}")
            if fmt == "csv":
                df.write_csv(path)
            else:
                df.write_parquet(path)
            paths[name] = path
        # salva anche drawdown come csv rapido se equity presente
        if "equity" in dfs and dfs["equity"].height > 0:
            eq = dfs["equity"].with_columns(pl.col("date").cast(pl.Date)).sort("date")
            if "equity" in eq.columns:
                runmax = eq["equity"].cum_max()
                dd = (eq["equity"] / runmax - 1.0).alias("drawdown")
                dd_df = pl.DataFrame({"date": eq["date"], "drawdown": dd})
                dd_path = os.path.join(self.out_dir, f"drawdown.csv")
                dd_df.write_csv(dd_path)
                paths["drawdown"] = dd_path
        return paths
