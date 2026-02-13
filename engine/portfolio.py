from enum import Enum
from datetime import datetime, date, time
import random
from typing import Dict, Optional, Literal, List, Union
from .datafeed import MarketData
import numpy as np

class PositionType(Enum):
    LONG = 1
    SHORT = -1

class AssetType(Enum):
    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"


class TradeId:
    _existing_ids = set()

    @classmethod
    def generate(cls) -> int:
        """Genera un numero intero univoco di 5 cifre per identificare una posizione."""
        while True:
            trade_id = random.randint(10000, 99999)  # Genera un numero di 5 cifre
            if trade_id not in cls._existing_ids:
                cls._existing_ids.add(trade_id)
                return trade_id
            

class Position:
    def __init__(
        self,
        symbol: str,
        asset_type: AssetType,
        quantity: float,
        entry_price: float,
        position_type: PositionType,
        entry_time: datetime,
        entry_costs: float = 0.0,
    ):
        self.symbol = symbol
        self.asset_type = asset_type
        self.quantity = quantity
        self.entry_price = entry_price
        self.position_type = position_type
        self.entry_time = entry_time

        # --- State ---
        self.is_open = True
        self.trade_id = TradeId.generate()

        # --- Costs ---
        self.entry_costs = entry_costs
        self.exit_costs = 0.0

        # --- Accounting ---
        # Entry costs are paid upfront and immediately affect realized PnL
        self.realized_pnl = -entry_costs

    def unrealized_pnl(self, price: float) -> float:
        """
        Unrealized PnL from price movements only (no costs).
        """
        if not self.is_open:
            return 0.0

        return (price - self.entry_price) * self.position_type.value * self.quantity

    def total_pnl(self, price: float) -> float:
        """
        Economic PnL = realized (already net of costs) + unrealized.
        """
        return self.realized_pnl + self.unrealized_pnl(price)

    def close(
        self,
        price: float,
        quantity: Optional[float] = None,
        exit_costs: float = 0.0,
    ) -> float:
        """
        Close position fully or partially at given execution price.

        Returns realized PnL from price movements only (before costs).
        """
        if not self.is_open:
            return 0.0

        if quantity is None:
            quantity = self.quantity

        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if quantity > self.quantity:
            raise ValueError("Cannot close more than current quantity")

        realized = (price - self.entry_price) * self.position_type.value * quantity

        # Update accounting
        self.realized_pnl += realized - exit_costs
        self.exit_costs += exit_costs
        self.quantity -= quantity

        if self.quantity == 0:
            self.is_open = False


    def notional(self) -> float:
        """
        Absolute exposure at entry.
        """
        return abs(self.entry_price * self.quantity)

class Portfolio:
    def __init__(self, cash: float):
        self.cash: float = cash
        self.positions: list[Position] = []  # Lista delle posizioni attive

    def add_position(self, position: Position):
        """Aggiunge una posizione al portafoglio."""
        self.positions.append(position)
        # Deduct notional value and entry costs from cash
        notional = position.notional()
        self.cash -= (notional + position.entry_costs)

    def calculate_total_value(self, market_data: MarketData) -> float:
        """
        Calcola il valore totale del portafoglio sommando il valore di tutte le posizioni.
        
        Args:
            market_data: MarketData instance with current prices
            
        Returns:
            Total portfolio value (cash + positions value)
        """
        positions_value = 0.0
        for position in self.open_positions():
            price = market_data.price(position.symbol)
            if price is None:
                continue
            # Current value = price * quantity * position_type (positive for LONG, negative for SHORT)
            positions_value += price * position.quantity * position.position_type.value
        return self.cash + positions_value
    
    def get_closed_pnl(self) -> float:
        """Calcola il profit and loss totale del portafoglio (realizzato)."""
        return sum(position.realized_pnl for position in self.positions)
    
    def get_ctv(self, market_data: MarketData) -> float:
        """
        Calcola il Current Trade Value (valore corrente delle posizioni aperte).
        
        Args:
            market_data: MarketData instance with current prices
            
        Returns:
            Total current trade value
        """
        total_ctv = 0.0
        for position in self.open_positions():
            price = market_data.price(position.symbol)
            if price is None:
                continue
            total_ctv += price * position.quantity
        return total_ctv
    
    def calculate_open_pnl(self, market_data: MarketData) -> float:
        """
        Calcola il profit and loss totale del portafoglio (non realizzato).
        
        Args:
            market_data: MarketData instance with current prices
            
        Returns:
            Total unrealized PnL
        """
        total_unrealized = 0.0
        for position in self.open_positions():
            price = market_data.price(position.symbol)
            if price is None:
                continue
            total_unrealized += position.unrealized_pnl(price)
        return total_unrealized
    
    def global_pnl(self, market_data: MarketData) -> float:
        """
        Calcola il profit and loss totale del portafoglio (realizzato + non realizzato).
        
        Args:
            market_data: MarketData instance with current prices
            
        Returns:
            Total PnL (realized + unrealized)
        """
        open_pnl = self.calculate_open_pnl(market_data)
        closed_pnl = self.get_closed_pnl()
        return open_pnl + closed_pnl      

    def get_positions_summary(self, market_data: MarketData, ref_date: str = None, ref_time: str = None):
        """
        Restituisce un riepilogo delle posizioni nel portafoglio.
        
        Args:
            market_data: MarketData instance with current prices
            ref_date: Reference date (optional, extracted from market_data.timestamp if not provided)
            ref_time: Reference time (optional)
            
        Returns:
            List of dictionaries with position summary information
        """
        # Use timestamp from market_data if ref_date not provided
        if ref_date is None and market_data.current_timestamp is not None:
            ref_date = str(market_data.current_timestamp[0])
            if ref_time is None:
                ref_time = str(market_data.current_timestamp[1])
        
        summary = []
        for position in self.positions:
            if position.is_open:
                price = market_data.price(position.symbol, "close")
                if price is not None:
                    open_pnl = position.unrealized_pnl(price)
                    current_value = price * position.quantity
                    total_pnl = position.total_pnl(price)
                else:
                    price = None
                    open_pnl = 0.0
                    current_value = 0.0
                    total_pnl = position.realized_pnl
            else:
                price = None
                open_pnl = 0.0
                current_value = 0.0
                total_pnl = position.realized_pnl
            
            summary.append({
                "ref_date": ref_date,
                "ref_time": ref_time,
                "entry_time": str(position.entry_time),
                "symbol": position.symbol,
                "trade_id": position.trade_id,
                "type": position.asset_type.value,
                "side": position.position_type.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "is_alive": position.is_open,
                "current_value": current_value,
                "open_pnl": open_pnl,
                "closed_pnl": position.realized_pnl,
                "commissions": position.entry_costs + position.exit_costs,
                "global_pnl": total_pnl
            })
        return summary
    
    def get_positions(self, symbol: str, position_type: AssetType) -> list[Position]:
        """
        Restituisce tutte le posizioni associate a un determinato simbolo e tipo di posizione.

        Args:
            symbol: Simbolo dell'asset da cercare
            position_type: Tipo di asset (EQUITY, FUTURE, OPTION)
            
        Returns:
            Lista di posizioni con il simbolo e il tipo specificato
        """
        filtered_positions = []
        for position in self.positions:
            if position.symbol == symbol and position.asset_type == position_type:
                filtered_positions.append(position)
        return filtered_positions
    
    def open_positions(self) -> list[Position]:
        """
        Restituisce solo le posizioni aperte dal portafoglio.
        
        Returns:
            Lista di posizioni aperte da self.positions
        """
        return [pos for pos in self.positions if pos.is_open]
    
    def get_total_quantity(self, symbol: str, position_type: AssetType, open_only: bool = False) -> float:
        """
        Restituisce la quantità totale delle posizioni per un simbolo e tipo di asset.
        
        Args:
            symbol: Symbol to get quantity for
            position_type: Asset type to filter by
            open_only: If True, only count open positions (default: False)
            
        Returns:
            Total quantity for the symbol and asset type
        """
        positions = self.get_positions(symbol=symbol, position_type=position_type)
        if open_only:
            positions = [pos for pos in positions if pos.is_open]
        return np.sum([pos.quantity for pos in positions])
    
    def get_equity_quantity(self, symbol: str) -> float:
        """ 
        Restituisce la quantità totale delle posizioni per una tupla symbol-assettype.
        
        Args:
            symbol: Symbol to get quantity for
            
        Returns:
            Total quantity for the symbol
        """
        return self.get_total_quantity(symbol=symbol, position_type=AssetType.EQUITY)
        
    def close_all_positions(self, market_data: MarketData, exit_costs: float = 0.0):
        """
        Chiude tutte le posizioni aperte nel portafoglio.
        
        Args:
            market_data: MarketData instance with current prices
            exit_costs: Exit costs per position (default: 0.0)
        """
        for position in self.open_positions():
            price = market_data.price(position.symbol)
            if price is None:
                continue
            # Calculate proceeds before closing (quantity will change)
            proceeds = price * position.position_type.value * position.quantity
            # Close position
            position.close(price, exit_costs=exit_costs)
            # Add proceeds to cash
            self.cash += proceeds
                
    def close_positions(self, ticker: str, position_type: AssetType, market_data: MarketData, 
                       exit_costs: float = 0.0, quantity: Optional[float] = None):
        """
        Chiude le posizioni aperte per un determinato simbolo e tipo di asset.
        Può chiudere completamente tutte le posizioni o parzialmente fino a una quantità specificata.
        
        Args:
            ticker: Symbol to close positions for
            position_type: Asset type to close
            market_data: MarketData instance with current prices
            exit_costs: Total exit costs for the entire operation (default: 0.0).
                       If quantity is specified, costs are distributed proportionally.
            quantity: Total quantity to close. If None, closes all positions completely.
                      If specified, closes positions until this total quantity is reached.
        
        Examples:
            >>> # Close all positions for SPY with total costs of 10
            >>> portfolio.close_positions("SPY", AssetType.EQUITY, market_data, exit_costs=10)
            
            >>> # Close 55 units total with total costs of 10 (costs distributed proportionally)
            >>> portfolio.close_positions("SPY", AssetType.EQUITY, market_data, quantity=55, exit_costs=10)
        """
        positions = self.get_positions(symbol=ticker, position_type=position_type)
        
        # Filter to only open positions
        open_positions = [pos for pos in positions if pos.is_open]
        
        if not open_positions:
            return
        
        # Calculate total quantity available using helper method
        total_quantity_available = self.get_total_quantity(symbol=ticker, position_type=position_type, open_only=True)
        
        # Determine actual quantity to close
        if quantity is None:
            quantity_to_close_total = total_quantity_available
        else:
            # Check if requested quantity exceeds available quantity
            if quantity > total_quantity_available:
                raise ValueError(
                    f"Cannot close {quantity} units for {ticker} ({position_type.value}). "
                    f"Only {total_quantity_available} units available."
                )
            quantity_to_close_total = quantity
        
        # Calculate proportional cost per unit closed
        if total_quantity_available > 0:
            cost_per_unit = exit_costs / total_quantity_available
        else:
            cost_per_unit = 0.0
        
        # Close positions until we reach the target quantity
        remaining_quantity = quantity_to_close_total
        
        for position in open_positions:
            if remaining_quantity <= 0:
                break
            
            price = market_data.price(position.symbol)
            if price is None:
                continue
            
            # Determine how much to close from this position
            quantity_to_close = min(remaining_quantity, position.quantity)
            
            # Calculate proportional costs for this position
            position_exit_costs = cost_per_unit * quantity_to_close
            
            # Calculate proceeds before closing
            proceeds = price * position.position_type.value * quantity_to_close
            
            # Close position (fully or partially)
            position.close(price, quantity=quantity_to_close, exit_costs=position_exit_costs)
            
            # Add proceeds to cash
            self.cash += proceeds
            
            # Update remaining quantity to close
            remaining_quantity -= quantity_to_close

