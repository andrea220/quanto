"""
Normalizzazione e persistenza dati OHLCV su file parquet annuali.

Usato da tutti i downloader (IB, BBG) per produrre uno schema uniforme.

Schema parquet:
    date, time, ticker, open, high, low, close, volume, insertion_time, type
"""

import os
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "database"
DATA_DIR.mkdir(exist_ok=True)

SCHEMA_COLS = ["date", "time", "ticker", "open", "high", "low", "close",
               "volume", "insertion_time", "type"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ticker_dir(ticker: str) -> Path:
    return DATA_DIR / ticker.upper()


def ticker_exists(ticker: str) -> bool:
    """True se esiste almeno un file YYYY.parquet per il ticker."""
    d = ticker_dir(ticker)
    if not d.exists():
        return False
    return any(f.stem.isdigit() for f in d.glob("*.parquet"))


def get_last_saved_date(ticker: str, data_type: str) -> Optional[date]:
    """
    Restituisce la data più recente salvata per il ticker e il tipo specificato
    ('eod' o 'intraday'). Ritorna None se non ci sono dati.
    """
    d = ticker_dir(ticker)
    if not d.exists():
        return None

    years = sorted(
        int(f.stem) for f in d.glob("*.parquet") if f.stem.isdigit()
    )
    if not years:
        return None

    path = d / f"{years[-1]}.parquet"
    try:
        df = pd.read_parquet(path, columns=["date", "type"])
    except Exception:
        return None

    df = df[df["type"] == data_type]
    if df.empty:
        return None

    dates = pd.to_datetime(df["date"]).dt.date
    return dates.max()


def normalize(df: pd.DataFrame, ticker: str, data_type: str) -> pd.DataFrame:
    """
    Converte un DataFrame grezzo OHLCV (con indice o colonna datetime)
    nello schema standard. Il timezone di output è Europe/Rome.

    Args:
        df:        DataFrame grezzo con colonne open/high/low/close/volume
        ticker:    Simbolo interno (es. 'SPY')
        data_type: 'eod' oppure 'intraday'
    """
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = df.copy()

    # Trova la colonna datetime (potrebbe essere l'indice o una colonna)
    if df.index.name in ("date", "datetime") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    datetime_col = next(
        (c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
         or c in ("date", "datetime")),
        None
    )
    if datetime_col is None:
        raise ValueError("Nessuna colonna datetime trovata nel DataFrame")

    # Converte in Europe/Rome
    cal = pd.to_datetime(df[datetime_col], utc=True).dt.tz_convert("Europe/Rome")
    df["date"] = cal.dt.date
    df["time"] = cal.dt.time
    df["ticker"] = ticker.upper()
    df["type"] = data_type
    df["insertion_time"] = datetime.now().replace(second=0, microsecond=0)

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    return df[SCHEMA_COLS]


def save(df: pd.DataFrame, ticker: str) -> None:
    """
    Salva il DataFrame normalizzato in file parquet annuali.
    Effettua merge incrementale e deduplicazione su (date, time, type).
    """
    if df.empty:
        return

    d = ticker_dir(ticker)
    d.mkdir(parents=True, exist_ok=True)

    # Determina il tipo per il filtro sul cutoff
    data_type = df["type"].iloc[0] if "type" in df.columns else None
    last_date = get_last_saved_date(ticker, data_type) if data_type else None

    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"]).dt.date
    df["_year"] = pd.to_datetime(df["date"]).dt.year

    if last_date:
        df = df[df["_date"] >= last_date]
    if df.empty:
        return

    for year, group in df.groupby("_year"):
        path = d / f"{int(year)}.parquet"
        new_rows = group[SCHEMA_COLS].copy()

        if path.exists():
            old = pd.read_parquet(path)
            if set(SCHEMA_COLS) != set(old.columns):
                old = old.reindex(columns=SCHEMA_COLS, fill_value=pd.NA)

            # Rimuovi le righe del giorno di cutoff per il tipo corrente
            # (potrebbero essere barre incomplete)
            if last_date and data_type and int(year) == last_date.year:
                mask = (pd.to_datetime(old["date"]).dt.date == last_date) & \
                       (old["type"] == data_type)
                old = old[~mask]

            combined = pd.concat([old, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined = (
            combined
            .drop_duplicates(subset=["date", "time", "type"], keep="last")
            .sort_values(["date", "time"], kind="stable")
        )
        combined.to_parquet(path, index=False, compression="zstd")
