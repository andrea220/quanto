from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import polars as pl


class PlotConfig:
    """Configuration for how a factor should be plotted."""
    
    def __init__(
        self, 
        panel: str = 'price',
        color: Optional[str] = None,
        line_style: str = 'solid',
        line_width: float = 1.5,
        opacity: float = 0.8,
        show_in_legend: bool = True,
        y_axis: str = 'left'
    ):
        """
        Parameters:
        -----------
        panel : str, default='price'
            Which panel to plot in: 'price' (with price data) or 'indicator' (separate subplot)
        color : str, optional
            Line color (hex or named color). If None, auto-assigned
        line_style : str, default='solid'
            Line style: 'solid', 'dash', 'dot', 'dashdot'
        line_width : float, default=1.5
            Width of the line
        opacity : float, default=0.8
            Line opacity (0-1)
        show_in_legend : bool, default=True
            Whether to show in legend
        y_axis : str, default='left'
            Which y-axis: 'left' or 'right'
        """
        self.panel = panel
        self.color = color
        self.line_style = line_style
        self.line_width = line_width
        self.opacity = opacity
        self.show_in_legend = show_in_legend
        self.y_axis = y_axis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        return {
            'panel': self.panel,
            'color': self.color,
            'line_style': self.line_style,
            'line_width': self.line_width,
            'opacity': self.opacity,
            'show_in_legend': self.show_in_legend,
            'y_axis': self.y_axis
        }


class Factor(ABC):
    def __init__(self, plot_config: Optional[PlotConfig] = None):
        self.name = self._build_name()
        self._plot_config = plot_config
    
    @abstractmethod
    def _build_name(self) -> str:
        """Build the factor name dynamically based on class-specific static name and parameters."""
        pass
    
    @abstractmethod
    def compute(self, market_data):
        return
    
    @abstractmethod
    def warmup(self):
        return
    
    def get_column_names(self) -> List[str]:
        """Return list of column names this factor creates.
        
        By default, returns [self.name]. Override for factors that create multiple columns.
        """
        return [self.name]
    
    @property
    def plot_config(self) -> PlotConfig:
        """Return the plotting configuration for this factor."""
        if self._plot_config is None:
            # Return default config if not set
            return PlotConfig()
        return self._plot_config

class MovingAverage(Factor):
    _static_name = "ma"
    
    def __init__(self, window, field='close', type: str = 'simple', plot_config: Optional[PlotConfig] = None):
        self.window = window
        self.field = field
        # store type while avoiding shadowing builtins in attributes
        self.ma_type = type.lower()
        if self.ma_type not in {"simple", "exponential"}:
            raise ValueError("type must be 'simple' or 'exponential'")
        
        # Create default plot config for Moving Averages (plot with price)
        if plot_config is None:
            plot_config = PlotConfig(
                panel='price',  # Plot on the same panel as price
                line_style='solid',
                line_width=1.5,
                opacity=0.8
            )
        super().__init__(plot_config=plot_config)
    
    def _build_name(self) -> str:
        """Build name like 'ma_20' or 'ma_20_ema'."""
        name_parts = [self._static_name, str(self.window)]
        if self.ma_type == "exponential":
            name_parts.append("ema")
        return "_".join(name_parts)

    def compute(self, market_data):
        if self.ma_type == "simple":
            expr = pl.col(self.field).rolling_mean(window_size=self.window)
        else:  # exponential
            # Use span-based EMA to be consistent with common definitions
            expr = pl.col(self.field).ewm_mean(span=self.window, adjust=False)
        return market_data.with_columns(
            expr.over("ticker").alias(self.name)
        )
    
    def warmup(self):
        return self.window
    
class Return(Factor):
    _static_name = "ret"
    
    def __init__(self, window: int = 1, field: str = "close", type: str = "simple", plot_config: Optional[PlotConfig] = None):
        self.window = window
        self.field = field
        self.ret_type = type.lower()
        if self.ret_type not in {"simple", "log"}:
            raise ValueError("type must be 'simple' or 'log'")
        
        # Create default plot config for Returns (separate indicator panel)
        if plot_config is None:
            plot_config = PlotConfig(
                panel='indicator',  # Plot in separate indicator panel
                line_style='solid',
                line_width=1.2,
                opacity=0.8
            )
        super().__init__(plot_config=plot_config)
    
    def _build_name(self) -> str:
        """Build name like 'ret_1' or 'ret_1_log'."""
        name_parts = [self._static_name, str(self.window)]
        if self.ret_type == "log":
            name_parts.append("log")
        return "_".join(name_parts)

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        if self.ret_type == "simple":
            expr = (pl.col(self.field) / pl.col(self.field).shift(self.window)) - 1
        else:  # log return
            expr = pl.col(self.field).log() - pl.col(self.field).log().shift(self.window)
        return market_data.with_columns(
            expr.over("ticker").alias(self.name)
        )

    def warmup(self) -> int:
        return self.window

class Momentum(Factor):
    _static_name = "mom"
    
    """
    Momentum factor that calculates returns over a lookback window,
    optionally skipping the most recent period.
    
    Parameters:
    -----------
    window : int
        Lookback period in bars
    skip : int, default=0
        Number of recent bars to skip (useful to avoid short-term reversals)
        For example, skip=1 calculates return from t-window to t-1 (excluding today)
    field : str, default='close'
        Price field to use for momentum calculation
    type : str, default='simple'
        Type of return: 'simple' for (P_t/P_{t-n} - 1) or 'log' for log returns
    normalize : bool, default=False
        If True, normalize momentum by its rolling standard deviation
    normalize_window : int, optional
        Window for normalization. If None, uses the same as 'window'
    """
    def __init__(self, 
                 window: int, 
                 skip: int = 0,
                 field: str = "close", 
                 type: str = "simple",
                 normalize: bool = False,
                 normalize_window: int = None,
                 plot_config: Optional[PlotConfig] = None):
        self.window = window
        self.skip = skip
        self.field = field
        self.mom_type = type.lower()
        self.normalize = normalize
        self.normalize_window = normalize_window if normalize_window is not None else window
        
        if self.mom_type not in {"simple", "log"}:
            raise ValueError("type must be 'simple' or 'log'")
        if self.skip < 0:
            raise ValueError("skip must be non-negative")
        if self.skip >= self.window:
            raise ValueError("skip must be less than window")
        
        # Create default plot config for Momentum (separate indicator panel)
        if plot_config is None:
            plot_config = PlotConfig(
                panel='indicator',  # Plot in separate indicator panel
                line_style='solid',
                line_width=1.5,
                opacity=0.85
            )
        super().__init__(plot_config=plot_config)
    
    def _build_name(self) -> str:
        """Build name like 'mom_20' or 'mom_20_skip1_log_norm'."""
        name_parts = [self._static_name, str(self.window)]
        if self.skip > 0:
            name_parts.append(f"skip{self.skip}")
        if self.mom_type == "log":
            name_parts.append("log")
        if self.normalize:
            name_parts.append("norm")
        return "_".join(name_parts)

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        # Calculate lookback shift considering skip period
        lookback_shift = self.window + self.skip
        recent_shift = self.skip if self.skip > 0 else None
        
        if self.mom_type == "simple":
            if recent_shift:
                # Return from t-lookback_shift to t-skip
                expr = (pl.col(self.field).shift(recent_shift) / 
                       pl.col(self.field).shift(lookback_shift)) - 1
            else:
                # Standard return from t-window to t
                expr = (pl.col(self.field) / 
                       pl.col(self.field).shift(lookback_shift)) - 1
        else:  # log return
            if recent_shift:
                expr = (pl.col(self.field).shift(recent_shift).log() - 
                       pl.col(self.field).shift(lookback_shift).log())
            else:
                expr = (pl.col(self.field).log() - 
                       pl.col(self.field).shift(lookback_shift).log())
        
        # First compute the momentum values
        result = market_data.with_columns(
            expr.over("ticker").alias(self.name)
        )
        
        # Apply normalization if requested (as a second step to avoid nested window ops)
        if self.normalize:
            result = result.with_columns(
                (pl.col(self.name) / 
                 pl.col(self.name).rolling_std(window_size=self.normalize_window)
                ).over("ticker").alias(self.name)
            )
        
        return result

    def warmup(self) -> int:
        # Total warmup needed: lookback + skip + normalization window (if applicable)
        base_warmup = self.window + self.skip
        if self.normalize:
            # Need additional bars for rolling std calculation
            return base_warmup + self.normalize_window
        return base_warmup

class DonchianChannels(Factor):
    _static_name = "dc"
    
    """
    Donchian Channels indicator that tracks the highest high and lowest low
    over a lookback period, creating upper and lower bands.
    
    Parameters:
    -----------
    window : int
        Lookback period in bars for the channels
    field_high : str, default='high'
        Price field to use for upper band (highest high)
    field_low : str, default='low'
        Price field to use for lower band (lowest low)
    include_middle : bool, default=True
        If True, also calculate the middle band (average of upper and lower)
    plot_config : PlotConfig, optional
        Custom plotting configuration
    """
    def __init__(self, 
                 window: int,
                 field_high: str = "high",
                 field_low: str = "low",
                 include_middle: bool = True,
                 plot_config: Optional[PlotConfig] = None):
        self.window = window
        self.field_high = field_high
        self.field_low = field_low
        self.include_middle = include_middle
        
        # Create default plot config for Donchian Channels (plot with price)
        if plot_config is None:
            plot_config = PlotConfig(
                panel='price',  # Plot on the same panel as price
                line_style='dash',
                line_width=1.2,
                opacity=0.7
            )
        super().__init__(plot_config=plot_config)
        
        # Generate column names based on the base name
        self.upper_name = f"{self.name}_upper"
        self.lower_name = f"{self.name}_lower"
        self.middle_name = f"{self.name}_middle"
    
    def _build_name(self) -> str:
        """Build name like 'dc_20'."""
        return f"{self._static_name}_{self.window}"
        
    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        # Calculate upper band (highest high over window)
        upper_expr = pl.col(self.field_high).rolling_max(window_size=self.window)
        
        # Calculate lower band (lowest low over window)
        lower_expr = pl.col(self.field_low).rolling_min(window_size=self.window)
        
        # Add upper and lower bands
        result = market_data.with_columns([
            upper_expr.over("ticker").alias(self.upper_name),
            lower_expr.over("ticker").alias(self.lower_name)
        ])
        
        # Optionally add middle band
        if self.include_middle:
            middle_expr = (pl.col(self.upper_name) + pl.col(self.lower_name)) / 2
            result = result.with_columns(
                middle_expr.alias(self.middle_name)
            )
        
        return result
    
    def get_column_names(self) -> List[str]:
        """Return list of column names: upper, lower, and optionally middle."""
        columns = [self.upper_name, self.lower_name]
        if self.include_middle:
            columns.append(self.middle_name)
        return columns
    
    def warmup(self) -> int:
        return self.window


class ATR(Factor):
    _static_name = "atr"
    
    """
    Average True Range (ATR) indicator that measures market volatility.
    
    The True Range is the maximum of:
    1. High - Low (current period range)
    2. |High - Previous Close| (gap up)
    3. |Low - Previous Close| (gap down)
    
    ATR is the simple moving average of True Range over the specified period.
    
    Parameters:
    -----------
    period : int
        Number of periods for the moving average of True Range
    field_high : str, default='high'
        Price field to use for high prices
    field_low : str, default='low'
        Price field to use for low prices
    field_close : str, default='close'
        Price field to use for close prices
    plot_config : PlotConfig, optional
        Custom plotting configuration
    """
    def __init__(self, 
                 period: int,
                 field_high: str = "high",
                 field_low: str = "low",
                 field_close: str = "close",
                 plot_config: Optional[PlotConfig] = None):
        self.period = period
        self.field_high = field_high
        self.field_low = field_low
        self.field_close = field_close
        
        if period < 1:
            raise ValueError("period must be >= 1")
        
        # Create default plot config for ATR (separate indicator panel)
        if plot_config is None:
            plot_config = PlotConfig(
                panel='indicator',  # Plot in separate indicator panel
                line_style='solid',
                line_width=1.5,
                opacity=0.8
            )
        super().__init__(plot_config=plot_config)
    
    def _build_name(self) -> str:
        """Build name like 'atr_14'."""
        return f"{self._static_name}_{self.period}"
    
    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        # Calculate True Range components
        # 1. High - Low
        tr1 = pl.col(self.field_high) - pl.col(self.field_low)
        
        # 2. |High - Previous Close|
        tr2 = (pl.col(self.field_high) - pl.col(self.field_close).shift(1)).abs()
        
        # 3. |Low - Previous Close|
        tr3 = (pl.col(self.field_low) - pl.col(self.field_close).shift(1)).abs()
        
        # True Range is the maximum of the three
        true_range = pl.max_horizontal([tr1, tr2, tr3])
        
        # ATR is the simple moving average of True Range
        atr_expr = true_range.rolling_mean(window_size=self.period)
        
        return market_data.with_columns(
            atr_expr.over("ticker").alias(self.name)
        )
    
    def warmup(self) -> int:
        # Need period bars for ATR calculation, plus 1 for previous close
        return self.period + 1
