# WORKFLOW.md — Quanto

> Documento operativo. Descrive i flussi di lavoro per gestire l'anagrafica e scaricare dati.

---

## Indice

1. [Anagrafica ticker](#1-anagrafica-ticker)
2. [Download dati da IB](#2-download-dati-da-ib)
3. [Researcher — calcolo fattori](#3-researcher--calcolo-fattori-sulluniverso)
4. [Lettura dati di prezzo — DataFeed e MarketData](#4-lettura-dati-di-prezzo--datafeed-e-marketdata)

---

## 1. Anagrafica ticker

Il file `config/anag.json` è il registro centrale di tutti i ticker. Struttura:

| Campo | Descrizione | Obbligatorio per BBG |
|---|---|---|
| `bbg_ticker` | Ticker Bloomberg completo (es. `"SX5E Index"`, `"SPY US Equity"`) | ✅ |
| `ib_ticker` | Ticker IB — valorizzato automaticamente col simbolo alla creazione | |
| `name` | Nome descrittivo | |
| `exchange` | Borsa di quotazione | |
| `currency` | Valuta (`EUR`, `USD`, …) | |
| `asset_type` | Tipo di asset (`Index`, `ETF`, `Equity`, …) | |
| `country` | Paese / area geografica (derivato dall'exchange) | |
| `sector` | Settore (da IB `category`) | |
| `industry` | Industria (da IB `industry`) | |

### Creare un'entry manuale

```bash
python examples/anag_create.py MSFT
```

Crea una entry vuota con `ib_ticker = "MSFT"` e tutti gli altri campi a `null`. Lancia `ValueError` se il ticker esiste già.

> **Nota:** per IB non è necessario creare l'entry a mano — `ib_download.py` la crea e la popola automaticamente al primo utilizzo.

---

## 2. Download dati da IB

Prerequisito: IB Gateway o TWS aperto e connesso su `localhost:4001`.

### Scaricare un ticker (caso standard)

```bash
python examples/ib_download.py SPY
```

- Se il ticker **non è in anag.json**: lo registra automaticamente (popola tutti i campi da IB) e poi scarica i dati.
- Se il ticker **è già in anag.json**: scarica direttamente senza toccare l'anagrafica.

Al primo download vengono scaricati:
- EOD: 20 anni di barre giornaliere
- Intraday: 30 giorni di barre a 1 minuto (limite IB)

Agli aggiornamenti successivi, viene scaricato solo il delta dall'ultima data disponibile.

---

## 3. Researcher — calcolo fattori sull'universo

`Researcher` è il motore di research: carica i dati tramite `DataFeed`, calcola i fattori in modo vettorializzato su tutto il dataset e restituisce un `pl.DataFrame` pronto per l'analisi o il plot.

### Costruzione

```python
from engine.datafeed import DataFeed
from engine.engine import Researcher
from engine.factor import MovingAverage, Return

feed = DataFeed(prices_base="database")

researcher = Researcher(
    factors=[MovingAverage(20), MovingAverage(50), Return(1)],
    feed=feed,
    start_date="2022-01-01",
    end_date="2026-12-31",
    frequency="eod",          # eod | 1m | 5m | 1h | 4h
    tickers=["AAPL"],
)
```

### Metodi principali

| Metodo | Descrizione |
|--------|-------------|
| `get_data()` | Carica dati, calcola fattori, rimuove warmup rows. Risultato **cached**: la seconda chiamata è istantanea. |
| `get_data_eod()` | Barre EOD raw senza fattori (non cached). Usato da `backtest_refactor` per il contesto del giorno precedente in modalità intraday. |
| `plot(...)` | Grafico Plotly interattivo con price + tutti i fattori. |
| `factor_registry` | Dizionario `{nome: Factor}` per ispezione o override stile. |

### Esempio d'uso

```python
# Prima chiamata: load da parquet + compute + cache (~30ms su EOD)
df = researcher.get_data()
print(df.select(["date", "close", "ma_20", "ma_50", "ret_1"]).tail(5))

# Seconda chiamata: da cache (< 1 ms)
df = researcher.get_data()

# Plot candlestick con tutti i fattori
fig = researcher.plot(chart_type="candlestick", theme="plotly_dark")
fig.show()

# Plot con range visibile ristretto
fig = researcher.plot(
    chart_type="candlestick",
    start_date="2026-01-01",
    end_date="2026-03-31",
)
```

> **Nota warmup:** le righe iniziali dove i fattori sono `null` vengono rimosse automaticamente. Con `MovingAverage(50)` le prime 49 righe vengono scartate.

> **Nota intraday:** il pattern è identico per frequenze intraday. Per dati grandi (molti ticker × mesi di 1m) Polars gestisce efficacemente in memoria grazie al backend colonnare.

### Script di esempio

```bash
python examples/researcher.py
python examples/researcher.py --ticker AAPL --start 2024-01-01 --freq eod
python examples/researcher.py --no-plot   # senza apertura browser
```

---

## 4. Lettura dati di prezzo — DataFeed e MarketData

I dati scaricati vengono letti tramite due classi in `engine/datafeed.py`:

| Classe | Responsabilità |
|--------|---------------|
| `DataFeed` | I/O puro: scansione parquet, filtro per data/frequenza, export Excel |
| `MarketData` | Accesso bar-by-bar con offset temporali (T, T-1, T-N, slice) |

### Struttura directory attesa

```
database/
    <TICKER>/
        2024.parquet
        2025.parquet
        ...
```

### DataFeed — caricamento dati

```python
from engine.datafeed import DataFeed

feed = DataFeed(prices_base="database")

# Carica tutto in memoria come DataFrame Polars
df = feed.get_market_data("2024-01-01", "2025-12-31", "eod", ["AAPL"])

# Lazy scan (efficiente su dataset grandi — non carica in RAM finché non serve)
lf = feed.scan_prices("2024-01-01", "2025-12-31", "eod", ["AAPL"])
df = lf.collect(engine="streaming")

# Export Excel
path = feed.get_data_excel("2024-01-01", "2025-12-31", "eod", ["AAPL"])
# oppure con path esplicito:
path = feed.get_data_excel("2024-01-01", "2025-12-31", "eod", ["AAPL"],
                           output_path="output/aapl.xlsx")
```

Frequenze supportate: `"eod"`, `"1m"`, `"5m"`, `"1h"`, `"4h"`.  
Per intraday le barre a 1m vengono ricampionate alla frequenza richiesta con aggregazione OHLCV standard.

### MarketData — accesso bar-by-bar

`MarketData` è il layer usato dal backtest engine per accedere ai prezzi ad ogni bar. Supporta offset temporali e slice:

```python
from engine.datafeed import DataFeed, MarketData

feed = DataFeed(prices_base="database")

# Costruzione — usa sempre from_datafeed o from_dataframe
md = MarketData.from_datafeed(feed, "2024-01-01", "2025-12-31", "eod", ["AAPL"])

# Informazioni generali
md.symbols()           # ["AAPL"]
md.has_symbol("AAPL")  # True
md.current_date()      # date(2025, 12, 31)

# Prezzi singoli con offset
md.price("AAPL", "close")      # close corrente (T)
md.price("AAPL", "close", 1)   # close T-1
md.price("AAPL", "close", 10)  # close T-10

# OHLC e volume
md.ohlc("AAPL")        # {"open": ..., "high": ..., "low": ..., "close": ...}
md.volume("AAPL")      # volume corrente

# Serie storiche (slice)
md.price("AAPL", "close", slice(None, 20))   # ultimi 20 close come pl.Series
md.price("AAPL", "close", slice(1, 11))      # close da T-10 a T-1

# Righe complete come DataFrame
md.get("AAPL")                       # riga corrente
md.get("AAPL", 5)                    # riga T-5
md.get("AAPL", slice(None, 10))      # ultime 10 righe

# Bar per tutti i ticker contemporaneamente
md.get_current_bar()   # tutti i ticker al timestamp corrente
md.get_bar(1)          # tutti i ticker al bar T-1 (calendario globale)
```

> **Offset negativo e lookahead bias:** un offset negativo (es. `-1`) accede a barre nel FUTURO rispetto al `current_timestamp`. Per default `MarketData` solleva un `ValueError` se si tenta questa operazione. Per abilitarla intenzionalmente in contesti di research (es. calcolo dei forward return per il labeling), passare `allow_lookahead=True` al momento della costruzione.

> **Nota `get_bar` multi-ticker:** `get_bar(N)` calcola "N barre fa" sul calendario unione di tutti i ticker. Se un ticker ha barre mancanti, potrebbe non comparire nel risultato. Per accesso per-ticker usa `get()`.

### Esempio completo

```bash
python examples/datafeed_last5.py AAPL
python examples/datafeed_last5.py AAPL --start 2024-01-01 --end 2025-12-31 --freq 1h
python examples/datafeed_last5.py AAPL --excel
```
