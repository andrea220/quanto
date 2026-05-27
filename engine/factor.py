from abc import ABC, abstractmethod
from dataclasses import dataclass, replace as _dc_replace
from typing import Dict, Any, Optional, List
import polars as pl
import numpy as np


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
        y_axis: str = 'left',
        plot_type: str = 'lines',
        marker_symbol: str = 'circle',
        marker_size: int = 8,
        vline_positive_color: str = '#26A69A',
        vline_negative_color: str = '#EF5350',
        vline_min_abs: float = 0.0,
        vline_width: float = 1.5,
        vline_opacity: float = 0.45,
        fill_color: Optional[str] = None,
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
        plot_type : str, default='lines'
            Plotly mode: 'lines', 'markers', 'lines+markers', or 'vlines'
        marker_symbol : str, default='circle'
            Plotly marker symbol (e.g. 'circle', 'triangle-up', 'triangle-down', 'diamond')
        marker_size : int, default=8
            Marker size in pixels
        vline_positive_color : str, default='#26A69A'
            Color for vertical lines where value > 0 (used when plot_type='vlines')
        vline_negative_color : str, default='#EF5350'
            Color for vertical lines where value < 0 (used when plot_type='vlines')
        vline_min_abs : float, default=0.0
            Only draw vlines where abs(value) >= this threshold (e.g. 1.5 for MSS-only)
        vline_width : float, default=1.5
            Width of vertical lines in pixels
        vline_opacity : float, default=0.45
            Opacity of vertical lines (0-1)
        """
        self.panel = panel
        self.color = color
        self.line_style = line_style
        self.line_width = line_width
        self.opacity = opacity
        self.show_in_legend = show_in_legend
        self.y_axis = y_axis
        self.plot_type = plot_type
        self.marker_symbol = marker_symbol
        self.marker_size = marker_size
        self.vline_positive_color = vline_positive_color
        self.vline_negative_color = vline_negative_color
        self.vline_min_abs = vline_min_abs
        self.vline_width = vline_width
        self.vline_opacity = vline_opacity
        self.fill_color = fill_color   # used by plot_type='zone'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        return {
            'panel': self.panel,
            'color': self.color,
            'line_style': self.line_style,
            'line_width': self.line_width,
            'opacity': self.opacity,
            'show_in_legend': self.show_in_legend,
            'y_axis': self.y_axis,
            'plot_type': self.plot_type,
            'marker_symbol': self.marker_symbol,
            'marker_size': self.marker_size,
            'vline_positive_color': self.vline_positive_color,
            'vline_negative_color': self.vline_negative_color,
            'vline_min_abs': self.vline_min_abs,
            'vline_width': self.vline_width,
            'vline_opacity': self.vline_opacity,
            'fill_color': self.fill_color,
        }


@dataclass
class RenderInstruction:
    """
    Unità atomica di rendering: descrive come disegnare un gruppo di colonne.

    Prodotta da ``Factor.get_render_instructions()``, consumata da ``Plotter``.
    Immutabile; usare ``.with_overrides(**kwargs)`` per creare una copia modificata.

    Attributi
    ---------
    column_names    : lista di colonne coinvolte (1 per linea/marker, 2 per zona)
    plot_type       : 'lines' | 'markers' | 'zone' | 'vlines'
    panel           : 'price' | 'indicator'
    label           : etichetta in legenda (None → usa column_names[0])
    color           : colore bordi / linea / marker
    fill_color      : colore riempimento (solo zone)
    line_width      : spessore linea
    line_style      : 'solid' | 'dash' | 'dot' | 'dashdot'
    opacity         : opacità 0-1
    show_in_legend  : mostra in legenda
    marker_symbol   : simbolo marker Plotly
    marker_size     : dimensione marker px
    vline_*         : parametri per linee verticali (plot_type='vlines')
    """
    column_names:          List[str]
    plot_type:             str                    # 'lines' | 'markers' | 'zone' | 'vlines'
    panel:                 str             = 'price'
    label:                 Optional[str]   = None
    # Line / fill
    color:                 Optional[str]   = None
    fill_color:            Optional[str]   = None
    line_width:            float           = 1.5
    line_style:            str             = 'solid'
    opacity:               float           = 0.8
    show_in_legend:        bool            = True
    # Markers
    marker_symbol:         str             = 'circle'
    marker_size:           int             = 8
    # Vertical lines
    vline_positive_color:  str             = '#26A69A'
    vline_negative_color:  str             = '#EF5350'
    vline_min_abs:         float           = 0.0
    vline_width:           float           = 1.5
    vline_opacity:         float           = 0.45

    def with_overrides(self, **kwargs) -> 'RenderInstruction':
        """Restituisce una copia con i campi specificati sovrascritti."""
        return _dc_replace(self, **kwargs)


class Factor(ABC):
    def __init__(self, plot_config: Optional[PlotConfig] = None):
        self.name = self._build_name()
        self._plot_config = plot_config
    
    @abstractmethod
    def _build_name(self) -> str:
        """Build the factor name dynamically based on class-specific static name and parameters."""
        pass
    
    @abstractmethod
    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        """Compute the factor and return a new DataFrame with added columns."""
        pass
    
    @abstractmethod
    def warmup(self):
        return
    
    def t1_offset(self) -> Optional[int]:
        """Return the natural event horizon for this factor in bars.

        Used by the labeling framework (FixedHorizonLabeler / SignalExitLabeler)
        to pair a signal at t0 with its event end t1. When a factor has a natural
        holding horizon (e.g. a trend-following factor with a fixed exit rule),
        override this method to declare it.

        Returns:
            Number of bars from signal generation (t0) to event end (t1),
            or None if the factor does not define a natural event horizon.
            None is the default — the caller must supply the horizon explicitly.
        """
        return None

    def get_column_names(self) -> List[str]:
        """Return ALL column names produced by this factor.
        
        Used by the Researcher to detect warmup rows (null check).
        By default, returns [self.name]. Override for multi-column factors.
        """
        return [self.name]

    def get_plot_column_names(self) -> List[str]:
        """Return column names to plot.

        Defaults to get_column_names(). Override to exclude metadata columns
        (e.g. binary flags, signal integers) that should not appear on the chart.
        """
        return self.get_column_names()

    def get_render_instructions(self) -> List[RenderInstruction]:
        """
        Restituisce le istruzioni di rendering per questo factor.

        Default: una ``RenderInstruction`` per colonna di ``get_plot_column_names()``,
        usando i valori di ``PlotConfig``.

        Sovrascrivere nelle sottoclassi per:
        - zone accoppiate (top + btm)
        - colonne con stili diversi nella stessa serie (es. swing_high vs swing_low)
        - rendering multi-pannello
        """
        cfg = self.plot_config
        return [
            RenderInstruction(
                column_names=[col],
                plot_type=cfg.plot_type,
                panel=cfg.panel,
                color=cfg.color,
                fill_color=cfg.fill_color,
                line_width=cfg.line_width,
                line_style=cfg.line_style,
                opacity=cfg.opacity,
                show_in_legend=cfg.show_in_legend,
                marker_symbol=cfg.marker_symbol,
                marker_size=cfg.marker_size,
                vline_positive_color=cfg.vline_positive_color,
                vline_negative_color=cfg.vline_negative_color,
                vline_min_abs=cfg.vline_min_abs,
                vline_width=cfg.vline_width,
                vline_opacity=cfg.vline_opacity,
            )
            for col in self.get_plot_column_names()
        ]

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

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
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


# ══════════════════════════════════════════════════════════════════════════════
# ICT / Price Action factors (from PineScript "Pure Price Action ICT Tools")
# ══════════════════════════════════════════════════════════════════════════════

class SwingHighLow(Factor):
    """
    Pivot High / Pivot Low detection (N barre per lato).

    Corrisponde ai livelli Short/Intermediate/Long Term del PineScript:
      window=1 → Short Term  (3-bar pivot)
      window=2 → Intermediate Term (5-bar pivot)
      window=3 → Long Term (7-bar pivot)

    ⚠️  Usa shift(-window) per confermare il lato destro del pivot:
        corretta per ricerca/visualizzazione.
        In backtest, leggi il segnale con market_data.get(ticker, window).

    Colonne prodotte
    ----------------
    swing_{window}_high : prezzo del pivot high (NaN altrove)
    swing_{window}_low  : prezzo del pivot low  (NaN altrove)
    """
    _static_name = "swing"

    def __init__(
        self,
        window: int = 1,
        field_high: str = "high",
        field_low: str = "low",
        plot_config: Optional[PlotConfig] = None,
    ):
        self.window     = window
        self.field_high = field_high
        self.field_low  = field_low
        if plot_config is None:
            plot_config = PlotConfig(
                panel="price",
                opacity=0.9,
                plot_type='markers',
                marker_symbol='triangle-up',  # overridden per-column in plotter
                marker_size=10,
            )
        super().__init__(plot_config=plot_config)
        self.high_name = f"{self.name}_high"
        self.low_name  = f"{self.name}_low"

    def _build_name(self) -> str:
        return f"{self._static_name}_{self.window}"

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        n = self.window
        h = pl.col(self.field_high)
        l = pl.col(self.field_low)

        is_ph = pl.lit(True)
        is_pl = pl.lit(True)
        for i in range(1, n + 1):
            is_ph = is_ph & (h > h.shift(i))
            is_pl = is_pl & (l < l.shift(i))
        # Lato destro: >= come il PineScript (lookahead per conferma)
        for i in range(1, n + 1):
            is_ph = is_ph & (h >= h.shift(-i))
            is_pl = is_pl & (l <= l.shift(-i))

        swing_high = pl.when(is_ph).then(h).otherwise(pl.lit(None))
        swing_low  = pl.when(is_pl).then(l).otherwise(pl.lit(None))

        return market_data.with_columns([
            swing_high.over("ticker").alias(self.high_name),
            swing_low.over("ticker").alias(self.low_name),
        ])

    def get_column_names(self) -> List[str]:
        return [self.high_name, self.low_name]

    def warmup(self) -> int:
        return self.window * 2 + 1


class FairValueGap(Factor):
    """
    Fair Value Gap (FVG) / Liquidity Void.
    Replica del blocco "Liquidity Voids" del PineScript.

    Pattern a 3 candele con gap di prezzo tra candle[t] e candle[t-2]:
      Bullish FVG : low[t] > high[t-2]  AND  close[t-1] > high[t-2]
                    AND  (low[t] - high[t-2]) > ATR * threshold
      Bearish FVG : high[t] < low[t-2]  AND  close[t-1] < low[t-2]
                    AND  (low[t-2] - high[t]) > ATR * threshold

    Colonne prodotte
    ----------------
    fvg_{p}_bull :  1.0 dove Bullish FVG, 0.0 altrove
    fvg_{p}_bear : -1.0 dove Bearish FVG, 0.0 altrove
    fvg_{p}_top  : bordo superiore del gap (NaN se nessun FVG)
    fvg_{p}_btm  : bordo inferiore del gap (NaN se nessun FVG)
    """
    _static_name = "fvg"

    def __init__(
        self,
        atr_period: int = 200,
        threshold: float = 0.75,
        field_high: str = "high",
        field_low: str = "low",
        field_close: str = "close",
        plot_config: Optional[PlotConfig] = None,
    ):
        self.atr_period  = atr_period
        self.threshold   = threshold
        self.field_high  = field_high
        self.field_low   = field_low
        self.field_close = field_close
        if plot_config is None:
            plot_config = PlotConfig(panel="price", line_style="dash", opacity=0.6)
        super().__init__(plot_config=plot_config)
        self.bull_name = f"{self.name}_bull"
        self.bear_name = f"{self.name}_bear"
        self.top_name  = f"{self.name}_top"
        self.btm_name  = f"{self.name}_btm"

    def _build_name(self) -> str:
        return f"{self._static_name}_{self.atr_period}"

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        h = pl.col(self.field_high)
        l = pl.col(self.field_low)
        c = pl.col(self.field_close)

        tr  = pl.max_horizontal([h - l,
                                  (h - c.shift(1)).abs(),
                                  (l - c.shift(1)).abs()])
        atr     = tr.rolling_mean(window_size=self.atr_period)
        min_gap = atr * self.threshold

        bull_gap = l - h.shift(2)
        is_bull  = (bull_gap > min_gap) & (l > h.shift(2)) & (c.shift(1) > h.shift(2))

        bear_gap = l.shift(2) - h
        is_bear  = (bear_gap > min_gap) & (h < l.shift(2)) & (c.shift(1) < l.shift(2))

        bull_sig = pl.when(is_bull).then(pl.lit(1.0)).otherwise(pl.lit(0.0))
        bear_sig = pl.when(is_bear).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))

        gap_top = (pl.when(is_bull).then(l)
                     .when(is_bear).then(l.shift(2))
                     .otherwise(pl.lit(None)))
        gap_btm = (pl.when(is_bull).then(h.shift(2))
                     .when(is_bear).then(h)
                     .otherwise(pl.lit(None)))

        return market_data.with_columns([
            bull_sig.over("ticker").alias(self.bull_name),
            bear_sig.over("ticker").alias(self.bear_name),
            gap_top.over("ticker").alias(self.top_name),
            gap_btm.over("ticker").alias(self.btm_name),
        ])

    def get_column_names(self) -> List[str]:
        return [self.bull_name, self.bear_name, self.top_name, self.btm_name]

    def warmup(self) -> int:
        return self.atr_period + 2


class OrderBlock(Factor):
    """
    Order Block (semplificato).

    Bullish OB: si forma quando il close rompe al rialzo il massimo delle
    ultime ``window`` barre. Il corpo (open/close) della candela immediatamente
    precedente al breakout definisce il livello dell'OB.
    Bearish OB: analogo al ribasso.

    Imposta use_body=False per usare high/low ("Use Candle Body = Disabled").

    Colonne prodotte
    ----------------
    ob_{n}_bull_top : corpo superiore dell'OB rialzista (NaN se nessun OB)
    ob_{n}_bull_btm : corpo inferiore dell'OB rialzista
    ob_{n}_bear_top : corpo superiore dell'OB ribassista
    ob_{n}_bear_btm : corpo inferiore dell'OB ribassista
    """
    _static_name = "ob"

    def __init__(
        self,
        window: int = 10,
        use_body: bool = True,
        field_high: str = "high",
        field_low: str = "low",
        field_open: str = "open",
        field_close: str = "close",
        plot_config: Optional[PlotConfig] = None,
    ):
        self.window      = window
        self.use_body    = use_body
        self.field_high  = field_high
        self.field_low   = field_low
        self.field_open  = field_open
        self.field_close = field_close
        if plot_config is None:
            plot_config = PlotConfig(panel="price", line_style="dash", opacity=0.7)
        super().__init__(plot_config=plot_config)
        self.bull_top_name = f"{self.name}_bull_top"
        self.bull_btm_name = f"{self.name}_bull_btm"
        self.bear_top_name = f"{self.name}_bear_top"
        self.bear_btm_name = f"{self.name}_bear_btm"

    def _build_name(self) -> str:
        return f"{self._static_name}_{self.window}"

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        n = self.window
        h = pl.col(self.field_high)
        l = pl.col(self.field_low)
        o = pl.col(self.field_open)
        c = pl.col(self.field_close)

        prev_high_max = h.shift(1).rolling_max(window_size=n)
        prev_low_min  = l.shift(1).rolling_min(window_size=n)

        bull_breakout = c > prev_high_max
        bear_breakout = c < prev_low_min

        if self.use_body:
            ob_top = pl.max_horizontal([o.shift(1), c.shift(1)])
            ob_btm = pl.min_horizontal([o.shift(1), c.shift(1)])
        else:
            ob_top = h.shift(1)
            ob_btm = l.shift(1)

        return market_data.with_columns([
            pl.when(bull_breakout).then(ob_top).otherwise(pl.lit(None))
              .over("ticker").alias(self.bull_top_name),
            pl.when(bull_breakout).then(ob_btm).otherwise(pl.lit(None))
              .over("ticker").alias(self.bull_btm_name),
            pl.when(bear_breakout).then(ob_top).otherwise(pl.lit(None))
              .over("ticker").alias(self.bear_top_name),
            pl.when(bear_breakout).then(ob_btm).otherwise(pl.lit(None))
              .over("ticker").alias(self.bear_btm_name),
        ])

    def get_column_names(self) -> List[str]:
        return [self.bull_top_name, self.bull_btm_name,
                self.bear_top_name, self.bear_btm_name]

    def warmup(self) -> int:
        return self.window + 1


class BuySellLiquidity(Factor):
    """
    Buyside & Sellside Liquidity.
    Replica del blocco omonimo del PineScript.

    Costruisce un ZigZag di swing high/low e identifica cluster di livelli
    entro un margine proporzionale all'ATR. Quando almeno ``min_touches``
    swing si raggruppano nella stessa area, viene segnalata una zona di
    liquidità (buyside = cluster di highs, sellside = cluster di lows).

    Implementazione: Python loop per ticker (non vectorizzata).

    Parametri
    ---------
    swing_window  : barre per lato nella rilevazione dei pivot
    atr_period    : periodo ATR per il margine (ta.atr(10) nel PineScript)
    margin_divisor: margin = ATR / margin_divisor  (default 6.9 come PineScript)
    min_touches   : swing minimi nello stesso cluster (PineScript usa count > 2)

    Colonne prodotte
    ----------------
    liq_{sw}_{p}_bull       :  1.0 dove zona buyside rilevata, 0 altrove
    liq_{sw}_{p}_bear       : -1.0 dove zona sellside rilevata, 0 altrove
    liq_{sw}_{p}_bull_level : prezzo medio della zona buyside (NaN altrove)
    liq_{sw}_{p}_bear_level : prezzo medio della zona sellside (NaN altrove)
    """
    _static_name = "liq"

    def __init__(
        self,
        swing_window: int = 1,
        atr_period: int = 10,
        margin_divisor: float = 6.9,
        min_touches: int = 3,
        plot_config: Optional[PlotConfig] = None,
    ):
        self.swing_window   = swing_window
        self.atr_period     = atr_period
        self.margin_divisor = margin_divisor
        self.min_touches    = min_touches
        if plot_config is None:
            plot_config = PlotConfig(panel="price", line_style="dash", opacity=0.6)
        super().__init__(plot_config=plot_config)
        self.bull_name       = f"{self.name}_bull"
        self.bear_name       = f"{self.name}_bear"
        self.bull_level_name = f"{self.name}_bull_level"
        self.bear_level_name = f"{self.name}_bear_level"

    def _build_name(self) -> str:
        return f"{self._static_name}_{self.swing_window}_{self.atr_period}"

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int) -> np.ndarray:
        n   = len(highs)
        trs = np.full(n, np.nan)
        for i in range(1, n):
            trs[i] = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i - 1]),
                         abs(lows[i]  - closes[i - 1]))
        atrs = np.full(n, np.nan)
        for i in range(period - 1, n):
            atrs[i] = np.nanmean(trs[i - period + 1 : i + 1])
        return atrs

    @staticmethod
    def _detect_pivots(highs: np.ndarray, lows: np.ndarray,
                       w: int):
        n  = len(highs)
        sh = np.full(n, np.nan)
        sl = np.full(n, np.nan)
        for i in range(w, n - w):
            if all(highs[i] > highs[i - j] for j in range(1, w + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, w + 1)):
                sh[i] = highs[i]
            if all(lows[i] < lows[i - j] for j in range(1, w + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, w + 1)):
                sl[i] = lows[i]
        return sh, sl

    def _process_ticker(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray):
        n      = len(highs)
        w      = self.swing_window
        MAX_ZZ = 50

        atrs                    = self._rolling_atr(highs, lows, closes, self.atr_period)
        swing_highs, swing_lows = self._detect_pivots(highs, lows, w)

        # ZigZag: buffer newest-first, max MAX_ZZ entries
        zz_dirs   = np.zeros(MAX_ZZ, dtype=int)
        zz_prices = np.full(MAX_ZZ, np.nan)

        bull_sig   = np.zeros(n)
        bear_sig   = np.zeros(n)
        bull_level = np.full(n, np.nan)
        bear_level = np.full(n, np.nan)

        for i in range(n):
            atr    = atrs[i] if not np.isnan(atrs[i]) else 0.0
            margin = atr / self.margin_divisor if self.margin_divisor > 0 and atr > 0 else 0.0

            # ── Swing High → Buyside Liquidity ───────────────────────────
            sh = swing_highs[i]
            if not np.isnan(sh):
                if zz_dirs[0] < 1:                       # new up leg
                    zz_dirs[1:]   = zz_dirs[:-1]
                    zz_prices[1:] = zz_prices[:-1]
                    zz_dirs[0]    = 1
                    zz_prices[0]  = sh
                elif zz_dirs[0] == 1 and sh > zz_prices[0]:  # update if higher
                    zz_prices[0] = sh

                if margin > 0:
                    count, min_p, max_p = 0, np.inf, -np.inf
                    for j in range(MAX_ZZ):
                        if zz_dirs[j] == 1 and not np.isnan(zz_prices[j]):
                            if abs(zz_prices[j] - sh) < margin:
                                count += 1
                                min_p  = min(min_p, zz_prices[j])
                                max_p  = max(max_p, zz_prices[j])
                    if count >= self.min_touches:
                        bull_sig[i]   = 1.0
                        bull_level[i] = (min_p + max_p) / 2.0

            # ── Swing Low → Sellside Liquidity ───────────────────────────
            sl = swing_lows[i]
            if not np.isnan(sl):
                if zz_dirs[0] > -1:                      # new down leg
                    zz_dirs[1:]   = zz_dirs[:-1]
                    zz_prices[1:] = zz_prices[:-1]
                    zz_dirs[0]    = -1
                    zz_prices[0]  = sl
                elif zz_dirs[0] == -1 and sl < zz_prices[0]:
                    zz_prices[0] = sl

                if margin > 0:
                    count, min_p, max_p = 0, np.inf, -np.inf
                    for j in range(MAX_ZZ):
                        if zz_dirs[j] == -1 and not np.isnan(zz_prices[j]):
                            if abs(zz_prices[j] - sl) < margin:
                                count += 1
                                min_p  = min(min_p, zz_prices[j])
                                max_p  = max(max_p, zz_prices[j])
                    if count >= self.min_touches:
                        bear_sig[i]   = -1.0
                        bear_level[i] = (min_p + max_p) / 2.0

        return bull_sig, bear_sig, bull_level, bear_level

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        df      = market_data.with_row_index("__row_idx__")
        n_total = len(df)

        bull_sig_all   = np.zeros(n_total)
        bear_sig_all   = np.zeros(n_total)
        bull_level_all = np.full(n_total, np.nan)
        bear_level_all = np.full(n_total, np.nan)

        for ticker in df["ticker"].unique().to_list():
            df_t = df.filter(pl.col("ticker") == ticker).sort("date", "time")
            idx  = df_t["__row_idx__"].to_numpy()

            highs  = np.array(df_t["high"].to_list(),  dtype=float)
            lows   = np.array(df_t["low"].to_list(),   dtype=float)
            closes = np.array(df_t["close"].to_list(), dtype=float)

            bs, be, bl, bel = self._process_ticker(highs, lows, closes)

            bull_sig_all[idx]   = bs
            bear_sig_all[idx]   = be
            bull_level_all[idx] = bl
            bear_level_all[idx] = bel

        return df.drop("__row_idx__").with_columns([
            pl.Series(self.bull_name,       bull_sig_all.tolist()),
            pl.Series(self.bear_name,       bear_sig_all.tolist()),
            pl.Series(self.bull_level_name, bull_level_all.tolist()),
            pl.Series(self.bear_level_name, bear_level_all.tolist()),
        ])

    def get_column_names(self) -> List[str]:
        return [self.bull_name, self.bear_name,
                self.bull_level_name, self.bear_level_name]

    def warmup(self) -> int:
        return self.atr_period + self.swing_window * 2 + 1


class MarketStructure(Factor):
    """
    Market Structure — Break of Structure (BOS) e Market Structure Shift (MSS).
    Replica del blocco "Market Structures" del PineScript.

    Logica:
    - Quando il close supera l'ultimo swing high confermato:
        * trend era già rialzista → BOS up  (segnale = +1)
        * trend era ribassista   → MSS bull (segnale = +2)
    - Quando il close scende sotto l'ultimo swing low confermato:
        * trend era già ribassista → BOS down (segnale = -1)
        * trend era rialzista      → MSS bear (segnale = -2)

    Implementazione: Python loop per ticker (non vectorizzata).

    ⚠️  Il rilevamento dei pivot usa lookahead sul lato destro (highs[i + j])
        per confermare la formazione dello swing, identico a SwingHighLow.
        Questo è corretto per ricerca/visualizzazione.
        In backtest, leggi il segnale con market_data.get(ticker, swing_window).

    Colonne prodotte
    ----------------
    ms_{w}_signal :  0=niente | +1=BOS↑ | +2=MSS↑ | -1=BOS↓ | -2=MSS↓
    ms_{w}_level  : prezzo dello swing attraversato (NaN se nessun evento)
    ms_{w}_trend  : trend corrente  -1=bear | 0=undefined | +1=bull
    """
    _static_name = "ms"

    def __init__(
        self,
        swing_window: int = 1,
        field_high: str = "high",
        field_low: str = "low",
        field_close: str = "close",
        include_swings: bool = True,
        plot_config: Optional[PlotConfig] = None,
    ):
        self.swing_window   = swing_window
        self.field_high     = field_high
        self.field_low      = field_low
        self.field_close    = field_close
        self.include_swings = include_swings
        if plot_config is None:
            plot_config = PlotConfig(
                panel="price",
                opacity=0.9,
                plot_type='markers',        # default for swing_high/swing_low columns
                marker_symbol='triangle-up',  # overridden per-column in plotter
                marker_size=10,
                # vlines config used when signal column is rendered
                vline_positive_color='#26A69A',
                vline_negative_color='#EF5350',
                vline_min_abs=1.5,          # show only MSS (|signal| >= 2)
                vline_width=1.5,
                vline_opacity=0.45,
            )
        super().__init__(plot_config=plot_config)
        self.signal_name     = f"{self.name}_signal"
        self.level_name      = f"{self.name}_level"
        self.trend_name      = f"{self.name}_trend"
        self.swing_high_name = f"{self.name}_swing_high"
        self.swing_low_name  = f"{self.name}_swing_low"

    def _build_name(self) -> str:
        return f"{self._static_name}_{self.swing_window}"

    def _process_ticker(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray):
        n = len(highs)
        w = self.swing_window

        # Pivot detection (con lookahead sul lato destro)
        swing_highs = np.full(n, np.nan)
        swing_lows  = np.full(n, np.nan)
        for i in range(w, n - w):
            if all(highs[i] > highs[i - j] for j in range(1, w + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, w + 1)):
                swing_highs[i] = highs[i]
            if all(lows[i] < lows[i - j] for j in range(1, w + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, w + 1)):
                swing_lows[i] = lows[i]

        ms_signal = np.zeros(n)
        ms_level  = np.full(n, np.nan)
        ms_trend  = np.zeros(n)

        trend     = 0       # 0=undefined, 1=bullish, -1=bearish
        last_sh   = np.nan
        last_sl   = np.nan
        sh_broken = False
        sl_broken = False

        for i in range(n):
            # Aggiorna livelli di swing non appena vengono rilevati
            if not np.isnan(swing_highs[i]):
                last_sh   = swing_highs[i]
                sh_broken = False
            if not np.isnan(swing_lows[i]):
                last_sl   = swing_lows[i]
                sl_broken = False

            # Rottura rialzista
            if not np.isnan(last_sh) and not sh_broken and closes[i] > last_sh:
                sh_broken    = True
                ms_level[i]  = last_sh
                ms_signal[i] = 1.0 if trend == 1 else 2.0
                trend         = 1

            # Rottura ribassista
            if not np.isnan(last_sl) and not sl_broken and closes[i] < last_sl:
                sl_broken    = True
                ms_level[i]  = last_sl
                ms_signal[i] = -1.0 if trend == -1 else -2.0
                trend         = -1

            ms_trend[i] = float(trend)

        return ms_signal, ms_level, ms_trend, swing_highs, swing_lows

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        df      = market_data.with_row_index("__row_idx__")
        n_total = len(df)

        signal_all     = np.zeros(n_total)
        level_all      = np.full(n_total, np.nan)
        trend_all      = np.zeros(n_total)
        swing_high_all = np.full(n_total, np.nan)
        swing_low_all  = np.full(n_total, np.nan)

        for ticker in df["ticker"].unique().to_list():
            df_t = df.filter(pl.col("ticker") == ticker).sort("date", "time")
            idx  = df_t["__row_idx__"].to_numpy()

            highs  = np.array(df_t["high"].to_list(),  dtype=float)
            lows   = np.array(df_t["low"].to_list(),   dtype=float)
            closes = np.array(df_t["close"].to_list(), dtype=float)

            sig, lvl, trd, sh, sl = self._process_ticker(highs, lows, closes)

            signal_all[idx]     = sig
            level_all[idx]      = lvl
            trend_all[idx]      = trd
            swing_high_all[idx] = sh
            swing_low_all[idx]  = sl

        new_cols = [
            pl.Series(self.signal_name, signal_all.tolist()),
            pl.Series(self.level_name,  level_all.tolist()),
            pl.Series(self.trend_name,  trend_all.tolist()),
        ]
        if self.include_swings:
            new_cols += [
                pl.Series(self.swing_high_name, swing_high_all.tolist()),
                pl.Series(self.swing_low_name,  swing_low_all.tolist()),
            ]

        return df.drop("__row_idx__").with_columns(new_cols)

    def get_column_names(self) -> List[str]:
        cols = [self.signal_name, self.level_name, self.trend_name]
        if self.include_swings:
            cols += [self.swing_high_name, self.swing_low_name]
        return cols

    def get_render_instructions(self) -> List[RenderInstruction]:
        """
        Istruzioni di rendering esplicite per MarketStructure:
        - swing_high : markers triangle-up, colore teal
        - swing_low  : markers triangle-down, colore rosso
        - signal     : vlines (solo MSS per default, |signal| >= 2)
        level / trend non vengono plottati di default.
        """
        cfg = self.plot_config
        instrs: List[RenderInstruction] = []
        if self.include_swings:
            instrs.append(RenderInstruction(
                column_names=[self.swing_high_name],
                plot_type='markers',
                panel='price',
                color='#26A69A',
                marker_symbol='triangle-up',
                marker_size=cfg.marker_size,
                opacity=cfg.opacity,
                show_in_legend=True,
                label=self.swing_high_name,
            ))
            instrs.append(RenderInstruction(
                column_names=[self.swing_low_name],
                plot_type='markers',
                panel='price',
                color='#EF5350',
                marker_symbol='triangle-down',
                marker_size=cfg.marker_size,
                opacity=cfg.opacity,
                show_in_legend=True,
                label=self.swing_low_name,
            ))
        instrs.append(RenderInstruction(
            column_names=[self.signal_name],
            plot_type='vlines',
            panel='price',
            vline_positive_color=cfg.vline_positive_color,
            vline_negative_color=cfg.vline_negative_color,
            vline_min_abs=cfg.vline_min_abs,
            vline_width=cfg.vline_width,
            vline_opacity=cfg.vline_opacity,
            show_in_legend=True,
            label=self.signal_name,
        ))
        return instrs

    def warmup(self) -> int:
        return self.swing_window * 2 + 1


class PrevDayHighLow(Factor):
    """
    Previous Day High/Low zones (HoD / LoD).

    Zone di base (inizio giornata)
    ------------------------------
    Per ogni barra del giorno D, le zone partono dall'ultimo giorno di
    riferimento *non-inside* che precede D.  La zona è la WICK della candela
    specifica che ha generato il massimo/minimo di quel giorno:

      HoD zone : [day_high  …  max(open,close) della candela col high più alto]
      LoD zone : [min(open,close) della candela col low più basso  …  day_low]

    Inside-day rule
    ---------------
    Se il range (high-low) del giorno corrente è interamente contenuto nel
    range del giorno di riferimento, il riferimento rimane invariato (carry-
    forward).  Solo quando un giorno espande il range diventa il nuovo
    riferimento per i giorni successivi.

    Aggiornamento intraday
    ----------------------
    Se durante il giorno corrente il prezzo supera la zona attiva E si forma
    un pattern di inversione (2 candele consecutive), la zona si aggiorna da
    quella barra in poi:

      HoD update : candela[i-1] bullish (C>O)  +  candela[i] bearish (C<O)
                   con high[i] > hod_top attivo
                   → nuova zona HoD = wick di candela[i]

      LoD update : candela[i-1] bearish (C<O)  +  candela[i] bullish (C>O)
                   con low[i] < lod_btm attivo
                   → nuova zona LoD = wick di candela[i]

    La zona si resetta al valore base del giorno ad ogni cambio di giornata.

    Colonne prodotte
    ----------------
    pdl_hod_top  : bordo superiore zona HoD
    pdl_hod_btm  : bordo inferiore zona HoD
    pdl_lod_top  : bordo superiore zona LoD
    pdl_lod_btm  : bordo inferiore zona LoD
    pdl_inside   : 1.0 se il giorno è inside rispetto al rif., 0.0 altrimenti

    Il primo giorno non ha riferimento → colonne NaN (rimosso dal Researcher).
    """
    _static_name = "pdl"

    def __init__(
        self,
        field_high: str = "high",
        field_low: str = "low",
        field_open: str = "open",
        field_close: str = "close",
        plot_config: Optional[PlotConfig] = None,
    ):
        self.field_high  = field_high
        self.field_low   = field_low
        self.field_open  = field_open
        self.field_close = field_close
        if plot_config is None:
            plot_config = PlotConfig(
                panel="price",
                plot_type="zone",
                color="rgba(230,150,0,1.0)",         # bordi arancio-ambra pieno
                fill_color="rgba(230,150,0,0.18)",   # riempimento leggero
                line_width=1.5,
                opacity=1.0,
            )
        super().__init__(plot_config=plot_config)
        self.hod_top_name = f"{self.name}_hod_top"
        self.hod_btm_name = f"{self.name}_hod_btm"
        self.lod_top_name = f"{self.name}_lod_top"
        self.lod_btm_name = f"{self.name}_lod_btm"
        self.inside_name  = f"{self.name}_inside"

    def _build_name(self) -> str:
        return self._static_name

    def _process_ticker(
        self,
        dates: list,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
    ):
        """
        Calcola zone HoD/LoD (wick della candela giornaliera) con carry-forward
        inside-day per un singolo ticker.

        Zone calcolate sul giorno di riferimento:
          HoD zone : [day_high  …  max(open,close) della candela che ha fatto il high]
          LoD zone : [min(open,close) della candela che ha fatto il low  …  day_low]

        I dati in input devono essere già ordinati per date+time (come garantito
        da compute() che chiama sort("date","time") prima di invocare questo metodo).

        Returns
        -------
        hod_top, hod_btm, lod_top, lod_btm, inside  — array di lunghezza n
        """
        n            = len(dates)
        dates_arr    = np.array(dates)
        unique_dates = sorted(set(dates))

        # ── Step 1: aggregati giornalieri ─────────────────────────────────────
        # La wick viene calcolata sulla CANDELA SPECIFICA che ha generato il
        # massimo (per HoD) o il minimo (per LoD) del giorno:
        #   d_body_top = max(open, close) della candela con high == day_high
        #   d_body_btm = min(open, close) della candela con low  == day_low
        d_high     = {}
        d_low      = {}
        d_body_top = {}   # cima del corpo della candela che ha fatto il high
        d_body_btm = {}   # fondo del corpo della candela che ha fatto il low

        for d in unique_dates:
            idx_d     = np.where(dates_arr == d)[0]   # già ordinati per time
            d_highs   = highs[idx_d]
            d_lows    = lows[idx_d]

            d_high[d] = float(np.nanmax(d_highs))
            d_low[d]  = float(np.nanmin(d_lows))

            # Candela che ha fatto il massimo del giorno
            hod_pos        = idx_d[int(np.nanargmax(d_highs))]
            hod_o, hod_c   = float(opens[hod_pos]), float(closes[hod_pos])
            d_body_top[d]  = max(hod_o, hod_c)   # fondo dell'upper wick

            # Candela che ha fatto il minimo del giorno
            lod_pos        = idx_d[int(np.nanargmin(d_lows))]
            lod_o, lod_c   = float(opens[lod_pos]), float(closes[lod_pos])
            d_body_btm[d]  = min(lod_o, lod_c)   # tetto della lower wick

        # ── Step 2: riferimento con carry-forward (inside-day rule) ──────────
        day_zones  = {}
        day_inside = {}
        ref        = None

        for d in unique_dates:
            curr = dict(
                high     = d_high[d],
                low      = d_low[d],
                body_top = d_body_top[d],
                body_btm = d_body_btm[d],
            )

            if ref is None:
                day_zones[d]  = None
                day_inside[d] = False
                ref = curr
            else:
                day_zones[d] = ref
                is_inside = (curr['high'] <= ref['high'] and
                             curr['low']  >= ref['low'])
                day_inside[d] = is_inside
                if not is_inside:
                    ref = curr

        # ── Step 3: assegna i valori barra per barra con aggiornamento intraday ─
        #
        # La zona di base viene dal giorno di riferimento (step 2), ma può
        # aggiornarsi DURANTE il giorno corrente se si verifica un pattern di
        # inversione oltre il livello attivo:
        #
        #   HoD update → candela[i-1] bullish  +  candela[i] bearish  + high[i] > hod_top attivo
        #                 nuova zona HoD = wick della candela[i]: top=high[i], btm=max(o[i],c[i])
        #
        #   LoD update → candela[i-1] bearish  +  candela[i] bullish  + low[i]  < lod_btm attivo
        #                 nuova zona LoD = wick della candela[i]: top=min(o[i],c[i]), btm=low[i]
        #
        # La zona attiva si resetta al cambio di giorno.
        hod_top = np.full(n, np.nan)
        hod_btm = np.full(n, np.nan)
        lod_top = np.full(n, np.nan)
        lod_btm = np.full(n, np.nan)
        inside  = np.full(n, np.nan)

        # Zone "attive" correnti (si aggiornano intraday)
        act_hod_top = np.nan
        act_hod_btm = np.nan
        act_lod_top = np.nan
        act_lod_btm = np.nan
        current_day = None

        for i in range(n):
            d = dates[i]
            z = day_zones[d]

            # Cambio giorno: reset della zona attiva alla zona base del giorno
            if d != current_day:
                current_day = d
                if z is not None:
                    act_hod_top = z['high']
                    act_hod_btm = z['body_top']
                    act_lod_top = z['body_btm']
                    act_lod_btm = z['low']
                else:
                    act_hod_top = act_hod_btm = act_lod_top = act_lod_btm = np.nan

            # Controllo pattern di inversione intraday (richiede barra precedente
            # dello stesso giorno e zona base già disponibile)
            if i > 0 and dates[i - 1] == d and not np.isnan(act_hod_top):
                prev_bull = closes[i - 1] > opens[i - 1]
                prev_bear = closes[i - 1] < opens[i - 1]
                curr_bear = closes[i]     < opens[i]
                curr_bull = closes[i]     > opens[i]

                # HoD: prev bullish → curr bearish con nuovo massimo sopra la zona
                if prev_bull and curr_bear and highs[i] > act_hod_top:
                    act_hod_top = highs[i]
                    act_hod_btm = max(opens[i], closes[i])

                # LoD: prev bearish → curr bullish con nuovo minimo sotto la zona
                if prev_bear and curr_bull and lows[i] < act_lod_btm:
                    act_lod_btm = lows[i]
                    act_lod_top = min(opens[i], closes[i])

            # Assegna la zona attiva alla barra corrente
            if not np.isnan(act_hod_top):
                hod_top[i] = act_hod_top
                hod_btm[i] = act_hod_btm
                lod_top[i] = act_lod_top
                lod_btm[i] = act_lod_btm
                inside[i]  = 1.0 if day_inside[d] else 0.0

        return hod_top, hod_btm, lod_top, lod_btm, inside

    def compute(self, market_data: pl.DataFrame) -> pl.DataFrame:
        df      = market_data.with_row_index("__row_idx__")
        n_total = len(df)

        hod_top_all = np.full(n_total, np.nan)
        hod_btm_all = np.full(n_total, np.nan)
        lod_top_all = np.full(n_total, np.nan)
        lod_btm_all = np.full(n_total, np.nan)
        inside_all  = np.full(n_total, np.nan)

        for ticker in df["ticker"].unique().to_list():
            df_t = df.filter(pl.col("ticker") == ticker).sort("date", "time")
            idx  = df_t["__row_idx__"].to_numpy()

            dates  = df_t["date"].to_list()
            highs  = np.array(df_t[self.field_high].to_list(),  dtype=float)
            lows   = np.array(df_t[self.field_low].to_list(),   dtype=float)
            opens  = np.array(df_t[self.field_open].to_list(),  dtype=float)
            closes = np.array(df_t[self.field_close].to_list(), dtype=float)

            ht, hb, lt, lb, ins = self._process_ticker(dates, highs, lows, opens, closes)

            hod_top_all[idx] = ht
            hod_btm_all[idx] = hb
            lod_top_all[idx] = lt
            lod_btm_all[idx] = lb
            inside_all[idx]  = ins

        # fill_nan(None) converte float NaN → null Polars, così il Researcher
        # può rilevare correttamente le righe di warmup tramite is_not_null().
        return df.drop("__row_idx__").with_columns([
            pl.Series(self.hod_top_name, hod_top_all.tolist()).fill_nan(None),
            pl.Series(self.hod_btm_name, hod_btm_all.tolist()).fill_nan(None),
            pl.Series(self.lod_top_name, lod_top_all.tolist()).fill_nan(None),
            pl.Series(self.lod_btm_name, lod_btm_all.tolist()).fill_nan(None),
            pl.Series(self.inside_name,  inside_all.tolist()).fill_nan(None),
        ])

    def get_column_names(self) -> List[str]:
        """Tutte le colonne prodotte — usate dal Researcher per il null-check warmup."""
        return [
            self.hod_top_name,
            self.hod_btm_name,
            self.lod_top_name,
            self.lod_btm_name,
            self.inside_name,
        ]

    def get_plot_column_names(self) -> List[str]:
        """Solo le colonne zona (esclude pdl_inside che è un flag binario)."""
        return [
            self.hod_top_name,
            self.hod_btm_name,
            self.lod_top_name,
            self.lod_btm_name,
        ]

    def get_render_instructions(self) -> List[RenderInstruction]:
        """
        Due zone accoppiate:
          HoD : (hod_top, hod_btm) — rosso
          LoD : (lod_top, lod_btm) — verde
        """
        cfg = self.plot_config
        base_kw = dict(
            plot_type='zone',
            panel='price',
            line_width=cfg.line_width,
            opacity=cfg.opacity,
            show_in_legend=True,
        )
        return [
            RenderInstruction(
                [self.hod_top_name, self.hod_btm_name],
                label='pdl_hod',
                color='rgba(220,50,50,1.0)',
                fill_color='rgba(220,50,50,0.18)',
                **base_kw,
            ),
            RenderInstruction(
                [self.lod_top_name, self.lod_btm_name],
                label='pdl_lod',
                color='rgba(40,180,80,1.0)',
                fill_color='rgba(40,180,80,0.18)',
                **base_kw,
            ),
        ]

    def warmup(self) -> int:
        # Il primo giorno avrà zone NaN; il Researcher le rimuove automaticamente.
        return 1