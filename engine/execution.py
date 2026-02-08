
"""Order execution module.

This module handles the execution of trading orders with configurable
execution timing, slippage, and commission costs. It supports different
execution modes and realistic transaction cost modeling.

Key Classes:
- ExecutionModel: Handles order queuing and execution with cost modeling
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, List, Union, TYPE_CHECKING
from enum import Enum


class PositionType(Enum):
    LONG = 1
    SHORT = -1

@dataclass
class Fill:
    """Represents an executed trade.
    
    Attributes:
        ticker: Symbol that was traded
        qty: Quantity traded (positive for buys, negative for sells)
        price: Execution price
    """
    ticker: str
    qty: float
    price: float
    side: PositionType
    slippage_bps: float
    commission_bps: float

class OrderType(Enum):
    MKT = "mkt"
    LMT = "lmt"

@dataclass
class Order:
    """Represents a trading order.
    
    Attributes:
        ticker: Symbol to trade
        side: Order side - "BUY" or "SELL"
        qty: Quantity to trade (always positive)
        type: Order type - "MKT" for market or "LMT" for limit
        limit_price: Limit price for limit orders (None for market orders)
    """
    ticker: str
    side: PositionType
    qty: float
    type: OrderType = OrderType.MKT
    limit_price: Optional[float] = None



class ExecutionModel:
    """
    Executes orders based on the chosen policy.
    
    The ExecutionModel handles order execution with configurable timing,
    slippage, and commission costs. It supports two execution modes:
    - "next_open": Execute orders at the next bar's opening price
    - "on_close": Execute orders at the current bar's closing price
    
    Attributes:
        mode: Execution timing - "next_open" or "on_close"
        slippage_bps: Slippage cost in basis points
        commission_bps: Commission cost in basis points
    """
    def __init__(self, mode: str = "next_open", slippage_bps: float = 0.0, commission_bps: float = 0.0):
        """Initialize the execution model.
        
        Args:
            mode: Execution timing - "next_open" or "on_close"
            slippage_bps: Slippage cost in basis points (default: 0.0)
            commission_bps: Commission cost in basis points (default: 0.0)
            
        Raises:
            AssertionError: If mode is not "next_open" or "on_close"
        """
        assert mode in ("next_open", "on_close")
        self.mode = mode
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self._pending: List[Order] = []

    def queue_orders(self, orders: List[Order]) -> None:
        """Add orders to the execution queue.
        
        Args:
            orders: List of Order objects to queue for execution
        """
        if not orders: return
        self._pending.extend(orders)

    def _apply_slippage(self, price: float, side: PositionType) -> float:
        """Apply slippage to execution price.
        
        Slippage is applied as a percentage of the base price, with
        the direction depending on whether it's a buy or sell order.
        
        Args:
            price: Base execution price
            qty: Quantity being traded (positive for buys, negative for sells)
            
        Returns:
            Adjusted price after applying slippage
        """
        if self.slippage_bps == 0: return price
        bump = price * (self.slippage_bps / 10_000.0)
        return price + (bump * side.value)

    def execute(self, price_map: Dict[str, Dict[str, float]]) -> List[Fill]:
        """Execute queued orders and return fills.
        
        Processes all queued orders using the current price data and
        the configured execution mode. Orders are removed from the queue
        after execution.
        
        Args:
            price_map: Nested dictionary with structure {ticker: {"open": price, "close": price}}
            
        Returns:
            List of Fill objects representing executed trades
        """
        fills: List[Fill] = []
        if self.mode == "next_open":
            to_exec = self._pending
            self._pending = []
            for o in to_exec:
                px = price_map.get(o.ticker, {}).get("open")
                if px is None: continue
                qty = o.qty #if o.side == "BUY" else -o.qty
                px = self._apply_slippage(float(px), o.side)
                fills.append(Fill(ticker=o.ticker, qty=qty, price=px, side=o.side, slippage_bps=self.slippage_bps, commission_bps=self.commission_bps))
            return fills
        else:
            to_exec = self._pending
            self._pending = []
            for o in to_exec:
                px = price_map.get(o.ticker, {}).get("close")
                if px is None: continue
                qty = o.qty #if o.side == "BUY" else -o.qty
                px = self._apply_slippage(float(px), o.side)
                fills.append(Fill(ticker=o.ticker, qty=qty, price=px, side=o.side, slippage_bps=self.slippage_bps, commission_bps=self.commission_bps))
            return fills
