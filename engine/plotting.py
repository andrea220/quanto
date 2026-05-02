"""
Plotting module for financial data visualization.

This module provides tools for creating interactive plots of price data and technical indicators.
Supports Polars DataFrames with OHLCV data and custom factors/indicators.
"""

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .factor import Factor, RenderInstruction
    from .engine import Researcher


class PlotRequest:
    """
    Riferimento a un factor (per nome/colonna o istanza) con override opzionali di stile.

    Passare a ``researcher.plot(plot_factors=[...])`` in alternativa a stringhe semplici
    quando si vogliono sovrascrivere colore, opacità, ecc. al volo.

    Parametri
    ---------
    ref : str | Factor
        Nome del factor (es. ``"pdl"``), nome di una colonna (es. ``"pdl_hod_top"``),
        o istanza di Factor.
    **style_overrides
        Qualunque campo di ``RenderInstruction`` da sovrascrivere
        (es. ``color``, ``fill_color``, ``opacity``, ``line_width``).

    Esempi
    ------
    >>> PlotRequest("pdl")                              # config di default
    >>> PlotRequest("pdl", fill_color="rgba(255,0,0,0.2)")
    >>> PlotRequest("ms_5_swing_high", color="#FF6600")
    >>> PlotRequest(PrevDayHighLow(), opacity=0.5)
    """
    def __init__(self, ref: Union[str, 'Factor'], **style_overrides):
        self.ref       = ref
        self.overrides = style_overrides


class Plotter:
    """
    Crea grafici finanziari interattivi con prezzi e indicatori.

    Può essere inizializzato con un ``Researcher`` (consigliato) o un DataFrame raw.

    Il flusso di rendering è:

        plot_factors items
            ↓  _resolve()
        List[RenderInstruction]          ← sorgente unica di verità per lo stile
            ↓  _render()  (dispatch)
        _render_lines / _render_markers / _render_zone / _render_vlines
    """

    # ── Palette di fallback per colonne senza colore esplicito ─────────────
    _FALLBACK_COLORS: List[str] = [
        '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
        '#BC4B51', '#5E548E', '#E07A5F', '#3D5A80',
    ]
    _MULTI_COLORS: Dict[str, str] = {
        'upper':  '#26A69A',
        'lower':  '#EF5350',
        'middle': '#FFA726',
    }
    _DASH_MAP: Dict[str, str] = {
        'solid': 'solid', 'dash': 'dash', 'dot': 'dot', 'dashdot': 'dashdot',
    }

    def __init__(
        self,
        researcher: Optional['Researcher'] = None,
        df: Optional[pl.DataFrame] = None,
        ticker: Optional[str] = None,
    ):
        if researcher is not None:
            self.researcher = researcher
            self.factors    = researcher.factors
            self.df         = researcher.get_data()
            if self.df is None:
                raise ValueError("Researcher returned no data")
            self.default_ticker = ticker if ticker is not None else (
                researcher.tickers[0] if researcher.tickers else None
            )
            if self.default_ticker and self.default_ticker not in researcher.tickers:
                raise ValueError(f"Ticker '{self.default_ticker}' not in researcher's tickers list")
        elif df is not None:
            self.researcher     = None
            self.factors        = []
            self.df             = df
            self.default_ticker = ticker
        else:
            raise ValueError("Either 'researcher' or 'df' must be provided")

        # Identify time column
        self.time_col = None
        for col in ['date', 'timestamp', 'time']:
            if col in self.df.columns:
                self.time_col = col
                break
        if self.time_col is None:
            raise ValueError("DataFrame must contain a 'date' or 'timestamp' column")

        # For intraday: combine date+time into a timestamp column
        if self.time_col == 'date' and 'time' in self.df.columns:
            n_rows        = self.df.height
            n_unique_dates = self.df.select(pl.col('date')).n_unique()
            if n_rows > n_unique_dates:
                self.df = self.df.with_columns(
                    pl.datetime(
                        year=pl.col('date').dt.year(),
                        month=pl.col('date').dt.month(),
                        day=pl.col('date').dt.day(),
                        hour=pl.col('time').dt.hour(),
                        minute=pl.col('time').dt.minute(),
                        second=pl.col('time').dt.second(),
                    ).alias('timestamp')
                )
                self.time_col = 'timestamp'

        if 'close' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        # Build factor registry {name_or_col → Factor}
        self._registry: Dict[str, 'Factor'] = self._build_registry()

    # ──────────────────────────────────────────────────────────────────────
    # Registry helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_registry(self) -> Dict[str, 'Factor']:
        """
        Costruisce un dizionario che mappa sia il nome del factor che ogni
        colonna che produce → Factor.  Usato da ``_resolve``.
        """
        if self.researcher is None:
            return {}
        registry: Dict[str, 'Factor'] = {}
        for f in self.factors:
            registry[f.name] = f
            for col in f.get_column_names():
                registry[col] = f
        return registry

    # ──────────────────────────────────────────────────────────────────────
    # Resolve: item → List[RenderInstruction]
    # ──────────────────────────────────────────────────────────────────────

    def _resolve(self, item) -> List['RenderInstruction']:
        """
        Converte un singolo elemento di ``plot_factors`` in una lista di
        ``RenderInstruction``.

        Tipi accettati
        --------------
        str         → cerca nel registry; se trovato usa Factor.get_render_instructions()
                      filtrato sulla colonna specificata; altrimenti linea generica
        Factor      → chiama direttamente get_render_instructions()
        PlotRequest → come str/Factor ma applica gli override di stile in cima
        """
        from engine.factor import RenderInstruction

        if isinstance(item, PlotRequest):
            overrides = item.overrides
            ref       = item.ref
        else:
            overrides = {}
            ref       = item

        if isinstance(ref, str):
            factor = self._registry.get(ref)
            if factor is None:
                # Colonna non mappata a nessun factor → linea generica
                instr = RenderInstruction(column_names=[ref], plot_type='lines')
                instrs = [instr]
            else:
                all_instrs = factor.get_render_instructions()
                if ref != factor.name:
                    # ref è una colonna specifica → filtra solo l'istruzione che la contiene
                    filtered = [i for i in all_instrs if ref in i.column_names]
                    instrs = filtered if filtered else [
                        # Colonna esiste nel factor ma non ha istruzione dedicata → linea generica
                        RenderInstruction(column_names=[ref], plot_type='lines')
                    ]
                else:
                    instrs = all_instrs
        else:
            # Factor instance
            instrs = ref.get_render_instructions()

        if overrides:
            return [i.with_overrides(**overrides) for i in instrs]
        return instrs

    # ──────────────────────────────────────────────────────────────────────
    # Render dispatch + individual renderers
    # ──────────────────────────────────────────────────────────────────────

    def _render(
        self,
        fig: go.Figure,
        df: pl.DataFrame,
        time_data: pl.Series,
        instr: 'RenderInstruction',
        row: int,
    ) -> None:
        """Dispatch a RenderInstruction al renderer corretto in base a plot_type."""
        pt = instr.plot_type
        if pt == 'vlines':
            self._render_vlines(fig, df, time_data, instr, row)
        elif pt == 'zone':
            self._render_zone(fig, df, time_data, instr, row)
        elif pt == 'markers':
            self._render_markers(fig, df, time_data, instr, row)
        else:  # 'lines' o 'lines+markers'
            self._render_lines(fig, df, time_data, instr, row)

    def _add_trace(self, fig: go.Figure, trace, row: int) -> None:
        """Helper: aggiunge una trace alla figura gestendo secondary_y solo per row==1."""
        if row == 1:
            fig.add_trace(trace, row=row, col=1, secondary_y=False)
        else:
            fig.add_trace(trace, row=row, col=1)

    def _pick_color(self, instr: 'RenderInstruction', col: str) -> str:
        """Restituisce il colore da usare per una colonna, con fallback intelligente."""
        if instr.color is not None:
            return instr.color
        for suffix, c in self._MULTI_COLORS.items():
            if f'_{suffix}' in col:
                return c
        return self._FALLBACK_COLORS[hash(col) % len(self._FALLBACK_COLORS)]

    def _render_lines(
        self,
        fig: go.Figure,
        df: pl.DataFrame,
        time_data: pl.Series,
        instr: 'RenderInstruction',
        row: int,
    ) -> None:
        """Disegna una o più linee."""
        for col in instr.column_names:
            color = self._pick_color(instr, col)
            label = instr.label or col
            trace = go.Scatter(
                x=time_data,
                y=df.select(pl.col(col)).to_series(),
                name=label,
                mode=instr.plot_type,
                opacity=instr.opacity,
                showlegend=instr.show_in_legend,
                line=dict(
                    color=color,
                    width=instr.line_width,
                    dash=self._DASH_MAP.get(instr.line_style, 'solid'),
                ),
            )
            self._add_trace(fig, trace, row)

    def _render_markers(
        self,
        fig: go.Figure,
        df: pl.DataFrame,
        time_data: pl.Series,
        instr: 'RenderInstruction',
        row: int,
    ) -> None:
        """Disegna marker (triangoli per swing, cerchi per default)."""
        for col in instr.column_names:
            color = self._pick_color(instr, col)
            label = instr.label or col
            trace = go.Scatter(
                x=time_data,
                y=df.select(pl.col(col)).to_series(),
                name=label,
                mode='markers',
                opacity=instr.opacity,
                showlegend=instr.show_in_legend,
                marker=dict(
                    color=color,
                    symbol=instr.marker_symbol,
                    size=instr.marker_size,
                    line=dict(color='rgba(0,0,0,0.4)', width=1),
                ),
            )
            self._add_trace(fig, trace, row)

    def _render_zone(
        self,
        fig: go.Figure,
        df: pl.DataFrame,
        time_data: pl.Series,
        instr: 'RenderInstruction',
        row: int,
    ) -> None:
        """
        Disegna una banda riempita tra coppie di colonne (top, btm).
        column_names deve essere pari: [top0, btm0, top1, btm1, …]
        """
        border_color = instr.color      or 'rgba(255,215,0,0.55)'
        fill_color   = instr.fill_color or 'rgba(255,215,0,0.12)'

        for i in range(0, len(instr.column_names) - 1, 2):
            top_name = instr.column_names[i]
            btm_name = instr.column_names[i + 1]
            top_data = df.select(pl.col(top_name)).to_series()
            btm_data = df.select(pl.col(btm_name)).to_series()
            label    = instr.label or top_name.rsplit('_', 1)[0]

            self._add_trace(fig, go.Scatter(
                x=time_data, y=top_data,
                mode='lines', fill=None,
                line=dict(color=border_color, width=instr.line_width),
                showlegend=False,
                name=f'{label}_top',
            ), row)
            self._add_trace(fig, go.Scatter(
                x=time_data, y=btm_data,
                mode='lines', fill='tonexty', fillcolor=fill_color,
                line=dict(color=border_color, width=instr.line_width),
                showlegend=True,
                name=label,
            ), row)

    def _render_vlines(
        self,
        fig: go.Figure,
        df: pl.DataFrame,
        time_data: pl.Series,
        instr: 'RenderInstruction',
        row: int,
    ) -> None:
        """
        Disegna linee verticali per ogni evento non-zero nel segnale.
        Aggiunge una trace invisibile alla legenda per colore pos/neg.
        """
        pos_color = instr.vline_positive_color
        neg_color = instr.vline_negative_color
        min_abs   = instr.vline_min_abs
        vwidth    = instr.vline_width
        vopacity  = instr.vline_opacity
        yref      = 'y domain' if row == 1 else f'y{row} domain'

        for col in instr.column_names:
            values = df.select(pl.col(col)).to_series().to_list()
            times  = time_data.to_list()
            legend_shown = {'pos': False, 'neg': False}
            base_label   = instr.label or col

            for t, v in zip(times, values):
                if v is None or v == 0 or abs(v) < min_abs:
                    continue
                color   = pos_color if v > 0 else neg_color
                leg_key = 'pos' if v > 0 else 'neg'

                if not legend_shown[leg_key]:
                    legend_shown[leg_key] = True
                    direction = '↑' if v > 0 else '↓'
                    self._add_trace(fig, go.Scatter(
                        x=[t], y=[None], mode='lines',
                        name=f'{base_label} ({direction})',
                        line=dict(color=color, width=vwidth),
                        opacity=vopacity, showlegend=True,
                    ), row)

                fig.add_shape(
                    type='line', xref='x', yref=yref,
                    x0=t, x1=t, y0=0, y1=1,
                    line=dict(color=color, width=vwidth),
                    opacity=vopacity,
                    row=row, col=1,
                )

    # ──────────────────────────────────────────────────────────────────────
    # Main plot method
    # ──────────────────────────────────────────────────────────────────────

    def plot(
        self,
        ticker:       Optional[str]  = None,
        plot_factors: Optional[List] = None,
        title:        Optional[str]  = None,
        height:       int            = 600,
        width:        Optional[int]  = None,
        chart_type:   str            = 'line',
        theme:        str            = 'plotly_white',
        start_date:   Optional[str]  = None,
        end_date:     Optional[str]  = None,
        show_eod:     bool           = False,
    ) -> go.Figure:
        """
        Crea un grafico interattivo con prezzo e indicatori.

        Parametri
        ---------
        ticker       : ticker da plottare (se None usa il primo disponibile)
        plot_factors : lista di str | Factor | PlotRequest
                       (se None usa tutti i factor del researcher)
        title        : titolo del grafico
        height/width : dimensioni in pixel
        chart_type   : 'line' | 'candlestick'
        theme        : template Plotly
        start_date   : data di inizio del range da plottare (es. '2026-01-01'), opzionale
        end_date     : data di fine del range da plottare (es. '2026-03-31'), opzionale
        show_eod     : se True aggiunge una linea verticale alla fine di ogni giornata
                       (solo per dati intraday)

        Restituisce
        -----------
        go.Figure
        """
        from engine.factor import RenderInstruction

        # ── 1. Ticker ────────────────────────────────────────────────────
        plot_ticker = ticker or self.default_ticker
        if plot_ticker is None:
            avail = self.df['ticker'].unique().to_list() if 'ticker' in self.df.columns else []
            plot_ticker = avail[0] if avail else "Unknown"
        if self.researcher is not None and plot_ticker not in self.researcher.tickers:
            raise ValueError(
                f"Ticker '{plot_ticker}' not in researcher's tickers: {self.researcher.tickers}"
            )

        # ── 2. Filtra e ordina i dati ─────────────────────────────────────
        df = (
            self.df.filter(pl.col('ticker') == plot_ticker)
            if 'ticker' in self.df.columns else self.df
        )
        df = df.sort(self.time_col)

        # Filtra per start_date / end_date se specificati
        date_col = 'date' if 'date' in df.columns else (self.time_col if self.time_col != 'timestamp' else None)
        if date_col is not None:
            if start_date is not None:
                sd = pl.lit(start_date).str.to_date()
                df = df.filter(pl.col(date_col).cast(pl.Date) >= sd)
            if end_date is not None:
                ed = pl.lit(end_date).str.to_date()
                df = df.filter(pl.col(date_col).cast(pl.Date) <= ed)
        elif self.time_col == 'timestamp':
            if start_date is not None:
                sd = pl.lit(start_date).str.to_date()
                df = df.filter(pl.col('timestamp').cast(pl.Date) >= sd)
            if end_date is not None:
                ed = pl.lit(end_date).str.to_date()
                df = df.filter(pl.col('timestamp').cast(pl.Date) <= ed)

        time_data = df.select(pl.col(self.time_col)).to_series()

        # ── 3. Validazione chart_type ─────────────────────────────────────
        if chart_type not in ['line', 'candlestick']:
            raise ValueError("chart_type must be 'line' or 'candlestick'")
        if chart_type == 'candlestick':
            missing = [c for c in ['open', 'high', 'low', 'close'] if c not in df.columns]
            if missing:
                raise ValueError(f"Candlestick requires OHLC columns. Missing: {missing}")

        # ── 4. Risolvi le istruzioni di rendering ─────────────────────────
        items = plot_factors if plot_factors is not None else (
            self.factors if self.researcher else []
        )
        all_instrs: List[RenderInstruction] = []
        for item in items:
            all_instrs.extend(self._resolve(item))

        # Valida che le colonne esistano
        for instr in all_instrs:
            for col in instr.column_names:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")

        # ── 5. Partiziona per pannello ─────────────────────────────────────
        price_instrs     = [i for i in all_instrs if i.panel == 'price']
        indicator_instrs = [i for i in all_instrs if i.panel != 'price']

        # ── 6. Crea subplots ───────────────────────────────────────────────
        n_ind  = len(indicator_instrs)
        n_rows = 1 + n_ind
        if title is None:
            chart_name = "Candlestick" if chart_type == 'candlestick' else "Price"
            title = f"{plot_ticker} - {chart_name} and Indicators"

        if n_ind > 0:
            row_heights = [0.5] + [0.5 / n_ind] * n_ind
            specs       = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * n_ind
            sub_titles  = [title] + [i.label or i.column_names[0] for i in indicator_instrs]
            fig = make_subplots(
                rows=n_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=row_heights,
                specs=specs,
                subplot_titles=sub_titles,
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=[title],
            )

        # ── 7. Trace del prezzo ────────────────────────────────────────────
        if chart_type == 'line':
            fig.add_trace(go.Scatter(
                x=time_data, y=df.select(pl.col('close')).to_series(),
                name='Close', line=dict(color='#2E86AB', width=2), mode='lines',
            ), row=1, col=1, secondary_y=False)
        else:
            fig.add_trace(go.Candlestick(
                x=time_data,
                open=df.select(pl.col('open')).to_series(),
                high=df.select(pl.col('high')).to_series(),
                low=df.select(pl.col('low')).to_series(),
                close=df.select(pl.col('close')).to_series(),
                name='OHLC',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350',
            ), row=1, col=1, secondary_y=False)
            fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

        # ── 8. Trace degli indicatori ──────────────────────────────────────
        for instr in price_instrs:
            self._render(fig, df, time_data, instr, row=1)
        for idx, instr in enumerate(indicator_instrs):
            self._render(fig, df, time_data, instr, row=2 + idx)

        # ── 8b. Linee EOD ──────────────────────────────────────────────────
        if show_eod:
            self._render_eod_lines(fig, df, n_rows)

        # ── 9. Layout ──────────────────────────────────────────────────────
        fig.update_layout(
            template=theme, height=height, width=width,
            hovermode='x unified', showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        rangebreaks = self._get_rangebreaks(df)
        for row in range(1, n_rows + 1):
            fig.update_xaxes(
                rangeslider=dict(visible=False),
                type='date',
                rangebreaks=rangebreaks or None,
                row=row, col=1,
            )
        fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
        for idx, instr in enumerate(indicator_instrs):
            fig.update_yaxes(
                title_text=instr.label or instr.column_names[0],
                row=2 + idx, col=1,
            )

        return fig

    # ──────────────────────────────────────────────────────────────────────
    # EOD vertical lines
    # ──────────────────────────────────────────────────────────────────────

    def _render_eod_lines(
        self,
        fig: go.Figure,
        df: pl.DataFrame,
        n_rows: int,
        color: str = 'rgba(90,90,90,0.65)',
        width: int = 1,
        dash: str = 'dashdot',
    ) -> None:
        """
        Aggiunge una linea verticale alla prima barra di ogni giornata (= inizio sessione).

        Funziona solo con dati intraday (time_col == 'timestamp').
        La linea è posizionata sulla prima barra di ogni giornata (tranne la prima),
        coerentemente con lo standard dei chart finanziari professionali.
        """
        if self.time_col != 'timestamp' or 'date' not in df.columns:
            return

        # Prima barra di ogni giornata = inizio nuova sessione
        eod_times = (
            df.group_by('date')
            .agg(pl.col('timestamp').min().alias('eod_ts'))
            .sort('date')
            .slice(1)  # salta il primo giorno (non ha sessione precedente)
        )

        for ts in eod_times['eod_ts'].to_list():
            for row in range(1, n_rows + 1):
                yref = 'y domain' if row == 1 else f'y{row} domain'
                fig.add_shape(
                    type='line',
                    xref='x', yref=yref,
                    x0=ts, x1=ts,
                    y0=0, y1=1,
                    line=dict(color=color, width=width, dash=dash),
                    row=row, col=1,
                )

    # ──────────────────────────────────────────────────────────────────────
    # Range breaks per intraday
    # ──────────────────────────────────────────────────────────────────────

    def _get_rangebreaks(self, df_filtered: pl.DataFrame) -> list:
        """
        Compute Plotly rangebreaks to hide non-trading periods in intraday charts.
        
        For intraday data (time_col == 'timestamp') automatically removes:
        - Weekend gaps (Saturday–Monday)
        - Overnight gaps (market close → market open next day)
        
        Trading hours are inferred from the actual times present in the data.
        
        Returns an empty list for EOD (daily) data.
        """
        if self.time_col != 'timestamp':
            return []  # EOD data – no rangebreaks needed

        rangebreaks = []

        # 1. Remove weekends
        rangebreaks.append(dict(bounds=["sat", "mon"]))

        # 2. Remove overnight gaps using trading hours derived from the data
        if 'time' in df_filtered.columns:
            open_t = df_filtered['time'].min()   # earliest bar start time
            close_t = df_filtered['time'].max()  # latest bar start time

            if open_t is not None and close_t is not None:
                open_hour  = open_t.hour  + open_t.minute  / 60.0
                close_hour = close_t.hour + close_t.minute / 60.0

                # Estimate bar duration to push the break past the last bar's end
                times_sorted = df_filtered['time'].unique().sort()
                if len(times_sorted) >= 2:
                    t0 = times_sorted[0]
                    t1 = times_sorted[1]
                    bar_minutes = (t1.hour * 60 + t1.minute) - (t0.hour * 60 + t0.minute)
                    close_hour += bar_minutes / 60.0

                # Only add overnight break when market doesn't run 24h
                if open_hour > 0 or close_hour < 23.5:
                    rangebreaks.append(dict(bounds=[close_hour, open_hour], pattern="hour"))

        return rangebreaks

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
        ticker:       Optional[str]  = None,
        plot_factors: Optional[List[Union[str, 'Factor']]] = None,
        title:        Optional[str]  = None,
        height:       int            = 600,
        width:        Optional[int]  = None,
        theme:        str            = 'plotly_white',
        start_date:   Optional[str]  = None,
        end_date:     Optional[str]  = None,
        show_eod:     bool           = False,
    ) -> go.Figure:
        """
        Convenience method to create candlestick chart.
        Wrapper around plot() with chart_type='candlestick'.
        """
        return self.plot(
            ticker=ticker,
            plot_factors=plot_factors,
            title=title,
            height=height,
            width=width,
            chart_type='candlestick',
            theme=theme,
            start_date=start_date,
            end_date=end_date,
            show_eod=show_eod,
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
    theme: str = 'plotly_white',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_eod: bool = False,
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
        theme=theme,
        start_date=start_date,
        end_date=end_date,
        show_eod=show_eod,
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
