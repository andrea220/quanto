
"""Risk management and position sizing module.

This module provides risk management and position sizing functionality for
backtesting. It includes order sizing based on target weights and risk
constraints to prevent excessive leverage and concentration.

Key Classes:
- OrderSizer: Converts target portfolio weights to specific order quantities
- RiskManager: Applies risk limits and constraints to orders
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RiskManager:
    max_leverage: int = 4
   