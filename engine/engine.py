from engine.factor import Factor
from engine.datafeed import DataFeed, MarketData

import polars as pl
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from .reports import ReportWriter
from .execution import ExecutionModel, PositionType, OrderType, Fill, Order
from .risk import RiskManager
from .portfolio import Portfolio, Position, AssetType, PositionType
import pandas as pd
from datetime import datetime


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

class Strategy(ABC):
    """Refactored Strategy class using MarketData instead of DataFrames.
    
    This class provides the same interface as Strategy but uses the new
    data-agnostic MarketData, Position, and Portfolio classes.
    """
    def __init__(self):
        self.feed = None
        self.exec = None
        self.sizer = None
        self.risk = None
        self.clock = None
        self.reporter = None
        self.portfolio = None
        self.last_date = None 
        self.last_time = None
        self.factors: list[Factor] = []
        self.signals = []  # List of all signals: [{'ticker': str, 'datetime': str, 'signal': int}, ...]
        self.market_data: Optional[MarketData] = None  # MarketData instead of current_bar/history
        self.current_bar: Optional[pl.DataFrame] = None  # Keep for ExecutionModel compatibility
        self.history: Optional[pl.DataFrame] = None  # Keep for history tracking
    
    # Logging methods - delegate to ReportWriter (same as Strategy)
    def _ensure_reporter(self):
        """Assicura che il reporter sia inizializzato."""
        if self.reporter is None:
            self.reporter = ReportWriter()
            self.reporter.setup_logger(self.__class__.__name__)
    
    def enable_file_logging(self, filepath: str = None):
        """Abilita logging su file. Chiamalo prima di backtest() se vuoi i log su file."""
        self._ensure_reporter()
        return self.reporter.enable_file_logging(filepath)
    
    def enable_memory_logging(self):
        """Cattura i log anche in memoria per accesso programmatico."""
        self._ensure_reporter()
        self.reporter.enable_memory_logging()
    
    def get_logs(self) -> str:
        """Restituisce tutti i log catturati in memoria."""
        if self.reporter is None:
            return "Reporter not initialized."
        return self.reporter.get_logs()
    
    def print_logs(self):
        """Stampa tutti i log catturati."""
        if self.reporter:
            self.reporter.print_logs()
    
    def save_logs_to_file(self, filepath: str = None):
        """Salva i log catturati in un file."""
        if self.reporter is None:
            raise RuntimeError("Reporter not initialized.")
        return self.reporter.save_logs_to_file(filepath)
    
    def set_log_level(self, level: int):
        """Imposta il livello di verbosità (logging.DEBUG, INFO, WARNING, ERROR)."""
        if self.reporter:
            self.reporter.set_log_level(level)
    
    def enable_bar_logging(self, enabled: bool = True):
        """Abilita/disabilita logging di ogni singolo bar (molto verbose)."""
        if self.reporter:
            self.reporter.enable_bar_logging(enabled)
    
    def get_log_summary(self) -> dict:
        """Estrae statistiche dai log."""
        if self.reporter is None:
            return {"error": "Reporter not initialized"}
        return self.reporter.get_log_summary()

    def _get_last_date(self, start_date: str, end_date: str, frequency: str, tickers: List[str]):
        lf = self.feed.scan_prices(start_date, end_date, frequency, tickers)
        y = end_date[:4]
        df = lf.filter(
                (pl.col("date").cast(pl.Date) >= pl.lit(f"{y}-01-01").cast(pl.Date)) 
            ).collect(engine="streaming")
        self.last_date, self.last_time = df.select(["date", "time"]).tail(1).row(0)

    def on_start(self,
                 start_date: str,
                 end_date: str,
                 frequency: str,
                 tickers: List[str],
                 starting_balance: float,
                 feed: DataFeed, 
                 exec_model: Optional[ExecutionModel] = None,
                 risk: Optional[RiskManager] = None,
                 reporter: Optional[ReportWriter] = None):
        
        self.starting_balance = starting_balance
        self.balance = pl.DataFrame(schema={"date": pl.Date, "time": pl.Time, "balance": pl.Float64})
        self.feed = feed
        exec_model_temp = exec_model or ExecutionModel(mode="on_close")
        self.exec = exec_model_temp
        self.risk = risk or RiskManager()
        
        # Use existing reporter if already created by enable_*_logging(), otherwise create new
        if reporter is not None:
            self.reporter = reporter
        elif self.reporter is None:
            self.reporter = ReportWriter()
        
        # Setup logger nel reporter se non già fatto
        if self.reporter.logger is None:
            self.reporter.setup_logger(self.__class__.__name__)
        
        # Log automatico inizio backtest
        self.reporter.logger.info("=" * 60)
        self.reporter.logger.info(f"BACKTEST STARTED")
        self.reporter.logger.info("=" * 60)
        self.reporter.logger.info(f"Period: {start_date} → {end_date}")
        self.reporter.logger.info(f"Universe: {', '.join(tickers)}")
        self.reporter.logger.info(f"Starting Balance: ${starting_balance:,.2f}")
        self.reporter.logger.info(f"Execution Mode: {exec_model_temp.mode}")
        if exec_model:
            self.reporter.logger.info(f"Slippage: {exec_model.slippage_bps} bps | Commission: {exec_model.commission_bps} bps")
        self.reporter.logger.info("-" * 60)

        self.history = pl.DataFrame()
        self.current_bar = None
        self.market_data = None
        self.portfolio = Portfolio(cash=starting_balance)
        self.positions_summary = pd.DataFrame()
        self.period_pnl = None

        self._get_last_date(start_date, end_date, frequency, tickers)
    
    def on_end(self):
        # Log automatico fine backtest
        # Close all positions and track signals
        self.close_all_positions()
        
        if self.market_data:
            summary = pd.DataFrame(self.portfolio.get_positions_summary(self.market_data))
            self.positions_summary = pd.concat([self.positions_summary, summary], axis = 0)
        
            if summary.empty:
                pnl = 0
            else:
                pnl = summary['global_pnl'].sum()
        else:
            pnl = 0
        
        total_balance = self.starting_balance + pnl
        new_balance_row = pl.DataFrame({
            "date": [self.current_date],
            "time": [self.current_time],
            "balance": [total_balance]
        })
        self.balance = pl.concat([self.balance, new_balance_row])
        
        # Log risultati finali
        total_pnl = total_balance - self.starting_balance
        pnl_pct = (total_pnl / self.starting_balance) * 100
        
        self.reporter.logger.info("-" * 60)
        self.reporter.logger.info(f"BACKTEST COMPLETED on {self.current_date}")
        self.reporter.logger.info("-" * 60)
        self.reporter.logger.info(f"Final Balance: ${total_balance:,.2f}")
        self.reporter.logger.info(f"Total P&L: ${total_pnl:,.2f} ({pnl_pct:+.2f}%)")
        total_trades = len(self.positions_summary['entry_time'].unique()) if not self.positions_summary.empty else 0
        self.reporter.logger.info(f"Total Trades: {total_trades}")
        self.reporter.logger.info("=" * 60)
    
    def submit_order(self, ticker: str, side: PositionType, qty: float, order_type: OrderType = OrderType.MKT, limit_price: Optional[float] = None):
        """Submit an order to the execution model - con logging automatico."""
        # Log automatico ordine
        self.reporter.logger.info(
            f"ORDER SUBMITTED → {ticker} | {side.name} {qty:.0f} shares | "
            f"Type: {order_type.value.upper()}"
            + (f" @ ${limit_price:.2f}" if limit_price else "")
        )
        
        order = Order(
            ticker=ticker,
            side=side,
            qty=qty,
            type=order_type,
            limit_price=limit_price
        )
        self.exec.queue_orders([order])
    
    def _close_opposite_positions(self, ticker: str, target_side: PositionType, quantity_to_close: float, 
                                 asset_type: AssetType = AssetType.EQUITY, exit_costs: float = 0.0) -> float:
        """Close positions with opposite side for a ticker.
        
        Closes the maximum available opposite positions and returns the remaining quantity
        that needs to be opened as a new trade.
        
        Args:
            ticker: Symbol to close opposite positions for
            target_side: The side we want to open (LONG or SHORT)
            quantity_to_close: Quantity of opposite positions to close
            asset_type: Asset type (default: EQUITY)
            exit_costs: Exit costs for closing positions (total, will be distributed proportionally)
        
        Returns:
            Remaining quantity that needs to be opened as a new trade
            (quantity_to_close - actual_quantity_closed)
        """
        if not self.market_data or not self.portfolio:
            return quantity_to_close
        
        if quantity_to_close <= 0:
            return quantity_to_close
        
        # Get all positions for this ticker and asset_type using Portfolio method
        positions = self.portfolio.get_positions(symbol=ticker, position_type=asset_type)
        
        # Filter to only open positions with opposite side
        opposite_side = PositionType.SHORT if target_side == PositionType.LONG else PositionType.LONG
        opposite_positions = [pos for pos in positions if pos.is_open and pos.position_type == opposite_side]
        
        if not opposite_positions:
            # No opposite positions, return full quantity to open
            return quantity_to_close
        
        # Calculate total quantity of opposite positions available
        total_opposite_quantity = sum(pos.quantity for pos in opposite_positions)
        
        # Check if there are any positions with the same side (non-opposite)
        same_side_positions = [pos for pos in positions if pos.is_open and pos.position_type == target_side]
        
        # Close the maximum available (can't close more than available)
        actual_quantity_to_close = min(quantity_to_close, total_opposite_quantity)
        
        if actual_quantity_to_close > 0:
            # If there are no same-side positions, we can use Portfolio.close_positions directly
            # Otherwise, we need to close manually to ensure only opposite positions are closed
            if not same_side_positions:
                # Use Portfolio.close_positions to close the maximum available quantity
                self.portfolio.close_positions(
                    ticker=ticker,
                    position_type=asset_type,
                    market_data=self.market_data,
                    exit_costs=exit_costs,
                    quantity=actual_quantity_to_close
                )
            else:
                # Need to close manually because there are both opposite and same-side positions
                # Use the same logic as Portfolio.close_positions but filter for opposite side
                cost_per_unit = exit_costs / actual_quantity_to_close if actual_quantity_to_close > 0 else 0.0
                
                remaining_quantity = actual_quantity_to_close
                
                for position in opposite_positions:
                    if remaining_quantity <= 0:
                        break
                    
                    price = self.market_data.price(position.symbol)
                    if price is None:
                        continue
                    
                    # Determine how much to close from this position
                    quantity_from_position = min(remaining_quantity, position.quantity)
                    
                    # Calculate proportional costs for this position
                    position_exit_costs = cost_per_unit * quantity_from_position
                    
                    # Calculate proceeds before closing
                    proceeds = price * position.position_type.value * quantity_from_position
                    
                    # Close position using Position.close method with explicit quantity
                    position.close(price, quantity=quantity_from_position, exit_costs=position_exit_costs)
                    
                    # Update portfolio cash using Portfolio's cash attribute
                    self.portfolio.cash += proceeds
                    
                    # Update remaining quantity to close
                    remaining_quantity -= quantity_from_position
            
            self.reporter.logger.info(
                f"CLOSED OPPOSITE POSITIONS → {ticker} | {opposite_side.name} {actual_quantity_to_close:.0f} shares"
            )
        
        # Return remaining quantity to open as new trade
        remaining_to_open = quantity_to_close - actual_quantity_to_close
        return remaining_to_open
    
    def buy(self, ticker: str, qty: float, asset_type: AssetType = AssetType.EQUITY, 
            order_type: OrderType = OrderType.MKT, limit_price: Optional[float] = None, 
            exit_costs: float = 0.0):
        """Buy (open LONG position) for a ticker.
        
        Before opening the LONG position, closes any existing SHORT positions for the same ticker.
        Closes the maximum available SHORT positions, then opens remaining quantity as new LONG trade.
        
        Args:
            ticker: Symbol to buy
            qty: Quantity to buy
            asset_type: Asset type (default: EQUITY)
            order_type: Order type (default: MKT)
            limit_price: Limit price for limit orders
            exit_costs: Exit costs for closing opposite positions
        """
        if not self.market_data:
            return
        
        # Close any existing SHORT positions first (up to the quantity being bought)
        # Returns remaining quantity that needs to be opened as new trade
        remaining_qty = self._close_opposite_positions(ticker, PositionType.LONG, qty, asset_type, exit_costs)
        
        # If there's remaining quantity, open it as a new LONG trade
        if remaining_qty > 0:
            self.submit_order(ticker, PositionType.LONG, remaining_qty, order_type, limit_price)
    
    def sell(self, ticker: str, qty: float, asset_type: AssetType = AssetType.EQUITY,
             order_type: OrderType = OrderType.MKT, limit_price: Optional[float] = None,
             exit_costs: float = 0.0):
        """Sell (open SHORT position) for a ticker.
        
        Before opening the SHORT position, closes any existing LONG positions for the same ticker.
        Closes the maximum available LONG positions, then opens remaining quantity as new SHORT trade.
        
        Args:
            ticker: Symbol to sell
            qty: Quantity to sell
            asset_type: Asset type (default: EQUITY)
            order_type: Order type (default: MKT)
            limit_price: Limit price for limit orders
            exit_costs: Exit costs for closing opposite positions
        """
        if not self.market_data:
            return
        
        # Close any existing LONG positions first (up to the quantity being sold)
        # Returns remaining quantity that needs to be opened as new trade
        remaining_qty = self._close_opposite_positions(ticker, PositionType.SHORT, qty, asset_type, exit_costs)
        
        # If there's remaining quantity, open it as a new SHORT trade
        if remaining_qty > 0:
            self.submit_order(ticker, PositionType.SHORT, remaining_qty, order_type, limit_price)
    
    def submit_orders(self, orders: List[Order]):
        """Submit multiple orders to the execution model - con logging automatico."""
        if not orders:
            return
        
        self.reporter.logger.info(f"SUBMITTING {len(orders)} ORDERS:")
        for order in orders:
            self.reporter.logger.info(
                f"  → {order.ticker} | {order.side.name} {order.qty:.0f} shares"
            )
        self.exec.queue_orders(orders)
    
    def close_position(self, ticker: str, asset_type: AssetType = AssetType.EQUITY):
        """Close all positions for a ticker and track signal.
        
        Closes all open positions for the specified ticker and asset type.
        Uses Portfolio.close_positions to close all positions completely.
        
        Args:
            ticker: Symbol to close positions for
            asset_type: Asset type to close (default: EQUITY)
        """
        # Close all positions for this ticker using Portfolio method
        self.portfolio.close_positions(ticker, asset_type, self.market_data)
        # Track signal as closed (0)
        self._update_signal(ticker, 0)
        # Log
        self.reporter.logger.info(f"POSITION CLOSED → {ticker}")
    
    def close_all_positions(self):
        """Close all open positions and track signals."""
        if not self.market_data:
            return
        
        # Get all open tickers before closing
        open_tickers = list(set([pos.symbol for pos in self.portfolio.positions if pos.is_open]))
        
        # Close all
        self.portfolio.close_all_positions(self.market_data)
        
        # Track signals for all closed positions
        for ticker in open_tickers:
            self._update_signal(ticker, 0)
        
        # Log
        if open_tickers:
            self.reporter.logger.info(f"ALL POSITIONS CLOSED → {', '.join(open_tickers)}")
    
    def get_bar(self, ticker: str) -> Optional[pl.DataFrame]:
        """Get the current bar data for a specific ticker using MarketData."""
        if not self.market_data:
            return None
        return self.market_data.get(ticker)
    
    def get_data(self, ticker: str, index: int = 0) -> Optional[pl.DataFrame]:
        """Get historical bar data for a specific ticker at a given time index using MarketData."""
        if not self.market_data:
            return None
        return self.market_data.get(ticker, index)
    
    def _update_signal(self, ticker: str, signal: int):
        """Update signal tracking for a ticker."""
        datetime_str = f"{self.current_date} {self.current_time}"
        self.signals.append({
            'ticker': ticker,
            'datetime': datetime_str,
            'signal': signal
        })
        
        # Log signal change
        signal_desc = {1: "LONG", -1: "SHORT", 0: "CLOSE/FLAT"}
        self.reporter.logger.debug(
            f"SIGNAL UPDATE → {ticker}: {signal_desc.get(signal, signal)} @ {datetime_str}"
        )
    
    def _create_price_map(self, market_data: MarketData) -> Dict[str, Dict[str, float]]:
        """Create price map from MarketData for execution model.
        
        Args:
            market_data: MarketData instance with current prices
            
        Returns:
            Dictionary with structure {ticker: {"open": price, "close": price, ...}}
        """
        price_map = {}
        for ticker in market_data.symbols():
            ohlc = market_data.ohlc(ticker)
            if ohlc:
                price_map[ticker] = ohlc
        return price_map
    
    def _process_fills(self, fills: List[Fill]):
        """Process fills and update portfolio with new positions - con logging automatico.
        
        Args:
            fills: List of Fill objects from execution
        """
        if not fills:
            return
        
        for fill in fills:
            # Log automatico fill
            notional = fill.qty * fill.price
            self.reporter.logger.info(
                f"ORDER FILLED → {fill.ticker} | {fill.side.name} {fill.qty:.0f} @ ${fill.price:.2f} | "
                f"Notional: ${notional:,.2f}"
            )
            
            # Calculate entry costs
            entry_costs = notional * (fill.commission_bps / 10_000.0)
            
            # Create Position (new data-agnostic interface)
            position = Position(
                symbol=fill.ticker,
                asset_type=AssetType.EQUITY,
                quantity=fill.qty,
                entry_price=fill.price,
                position_type=fill.side,
                entry_time=datetime.combine(self.current_date, self.current_time),
                entry_costs=entry_costs
            )
            
            self.portfolio.add_position(position)
            
            # Update signals tracking
            signal = 1 if fill.side == PositionType.LONG else -1
            self._update_signal(fill.ticker, signal)
            
            # Log stato portafoglio dopo fill
            self.reporter.logger.debug(
                f"Portfolio: {len(self.portfolio.positions)} positions | "
                f"Cash: ${self.portfolio.cash:,.2f}"
            )
    
    def execute_pending_orders(self):
        """Execute any pending orders using current MarketData - con logging automatico."""
        if not self.exec._pending or not self.market_data:
            return
        
        n_pending = len(self.exec._pending)
        self.reporter.logger.debug(f"Executing {n_pending} pending orders...")
        
        # Use current MarketData directly - get_current_bar() provides current bar data
        # ExecutionModel needs price_map format, which we create from MarketData
        price_map = self._create_price_map(self.market_data)
        fills = self.exec.execute(price_map)
        self._process_fills(fills)
    
    @abstractmethod
    def on_bar(self):
        """Called for each bar in the backtest. Override this method to implement strategy logic.
        
        You can access:
        - self.market_data: MarketData instance with historical prices
        - self.portfolio: Portfolio instance with current positions
        - self.current_date, self.current_time: Current bar timestamp
        """
        return
    
    def backtest(self,
                 starting_balance: float,
                 start_date: str,
                 end_date: str,
                 frequency: str,
                 tickers: List[str],
                 feed: DataFeed, 
                 exec_model: Optional[ExecutionModel] = None,
                 risk: Optional[RiskManager] = None,
                 reporter: Optional[ReportWriter] = None):
        
        researcher = Researcher(self.factors, feed, start_date, end_date, frequency, tickers)
        self.on_start(start_date, end_date, frequency, tickers, starting_balance, feed, exec_model, risk, reporter)
    
        # Use Clock for temporal iteration - zero overhead, clean separation
        data = researcher.get_data()
        if data is None:
            return
        clock = Clock(data, self.last_date, self.last_time)
        bar_count = 0
        self.period = 0 
        
        for market_data in clock:
            bar_count += 1
            self.market_data = market_data
            self.current_date, self.current_time = market_data.current_timestamp
            self.current_bar = market_data.get_current_bar()
            self.history = market_data.data  # Full history DataFrame
            
            # Log automatico bar (opzionale, se abilitato)
            if self.reporter._log_bars:
                self.reporter.logger.debug(
                    f"Bar #{bar_count}: {self.current_date} {self.current_time} | "
                    f"Positions: {len(self.portfolio.positions)} | "
                    f"Cash: ${self.portfolio.cash:,.2f}"
                )

            # Check if this is the last bar
            if self.current_date == self.last_date and self.current_time == self.last_time:
                self.on_end()
                break

            # Execute pending orders from previous bar (for "next_open" mode)
            if self.exec.mode == "next_open":
                self.execute_pending_orders()
            
            # Let strategy generate new orders
            self.on_bar()
            
            # Execute orders immediately if in "on_close" mode
            if self.exec.mode == "on_close":
                self.execute_pending_orders()
            
            # Update portfolio summary and balance
            if self.market_data:
                summary = pd.DataFrame(self.portfolio.get_positions_summary(self.market_data))
                self.positions_summary = pd.concat([self.positions_summary, summary], axis = 0) 
                if summary.empty:
                    pnl = 0
                else:
                    pnl = summary['global_pnl'].sum()
            else:
                pnl = 0
            
            # Update balance DataFrame with current date, time and balance
            total_balance = self.starting_balance + pnl 
            new_balance_row = pl.DataFrame({
                "date": [self.current_date],
                "time": [self.current_time],
                "balance": [total_balance]
            })
            self.balance = pl.concat([self.balance, new_balance_row])
            self.period += 1


        
class Clock:
    """Clock for iterating through bars in temporal order."""
    
    def __init__(self, data: pl.DataFrame, last_date, last_time):
        """Initialize Clock with data and end conditions.
        
        Args:
            data: DataFrame with computed factors
            last_date: Last date to process
            last_time: Last time to process
        """
        self.data = data
        self.last_date = last_date
        self.last_time = last_time
    
    def __iter__(self):
        """Iterate through bars - yields MarketData directly."""
        if self.data.height == 0:
            return
        
        # Group by date and time
        grouped = self.data.sort(['date', 'time']).group_by(
            ['date', 'time'], maintain_order=True
        )
        
        for group_key, group_data in grouped:
            current_date, current_time = group_key
            
            # Filter history up to current timestamp
            history = self.data.filter(
                (pl.col("date") <= current_date) |
                ((pl.col("date") == current_date) & (pl.col("time") <= current_time))
            )
            
            # Create MarketData directly
            market_data = MarketData.from_dataframe(
                history,
                current_timestamp=(current_date, current_time)
            )
            
            yield market_data
            
            # Check end condition after yielding (allows processing last bar)
            if current_date == self.last_date and current_time == self.last_time:
                return

