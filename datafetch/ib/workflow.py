"""
IB workflow: registrazione anagrafica e download dati storici.

Funzioni pubbliche:
    register_ticker(ticker)  — one-time: crea/popola anag da IB
    download_ticker(ticker)  — ricorrente: scarica EOD + intraday
                               (auto-registra se il ticker non è in anag)
"""

import logging
from datetime import date

from ib_insync import IB, Stock, util

from datafetch.storage import normalize, save, get_last_saved_date
from datafetch.ticker_manager import TickerManager

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 4001
_DEFAULT_CLIENT_ID = 1

# Mapping exchange → paese (ISO 2)
_EXCHANGE_COUNTRY = {
    "NASDAQ": "US", "NYSE": "US", "ARCA": "US", "BATS": "US", "CBOE": "US",
    "BVME": "IT", "MIL": "IT",
    "LSE": "GB", "LSEETF": "GB",
    "XETRA": "DE",
    "SBF": "FR",
    "AEB": "NL",
    "TSX": "CA",
    "ASX": "AU",
    "HKEX": "HK",
    "TSE": "JP",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_ticker(
    ticker: str,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    client_id: int = _DEFAULT_CLIENT_ID,
) -> None:
    """
    Operazione one-time: crea l'entry in anag.json e la popola da IB.

    Utile per registrare un nuovo ticker senza scaricare i dati.
    Se il ticker è già in anag, non sovrascrive i campi già valorizzati.

    Raises:
        RuntimeError: Se IB non trova il contratto.
    """
    ticker = ticker.upper().strip()
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    try:
        contract = _qualify(ib, ticker)
        _update_anag(ib, ticker, contract)
    finally:
        ib.disconnect()


def download_ticker(
    ticker: str,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    client_id: int = _DEFAULT_CLIENT_ID,
) -> None:
    """
    Scarica EOD + intraday 1-min per il ticker e salva in database/<TICKER>/.

    Se il ticker non è in anag.json, lo registra automaticamente prima
    di procedere con il download (purché IB trovi il contratto).

    Raises:
        RuntimeError: Se IB non trova il contratto.
    """
    ticker = ticker.upper().strip()
    manager = TickerManager()

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    try:
        contract = _qualify(ib, ticker, currency=_currency(manager, ticker))

        # Registra in anag se non presente
        if not manager.get_ticker(ticker):
            _update_anag(ib, ticker, contract)

        # EOD
        eod_duration = _eod_duration(ticker)
        logger.info(f"Fetching EOD ({eod_duration}) per {ticker}...")
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=eod_duration,
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=True, formatDate=1, keepUpToDate=False,
        )
        if bars:
            save(normalize(util.df(bars), ticker, "eod"), ticker)
            logger.info(f"EOD: {len(bars)} barre salvate")

        # Intraday 1-min (IB limit: 30 giorni)
        logger.info(f"Fetching intraday (30 D) per {ticker}...")
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="30 D",
            barSizeSetting="1 min", whatToShow="TRADES",
            useRTH=True, formatDate=1, keepUpToDate=False,
        )
        if bars:
            save(normalize(util.df(bars), ticker, "intraday"), ticker)
            logger.info(f"Intraday: {len(bars)} barre salvate")

    finally:
        ib.disconnect()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _qualify(ib: IB, ticker: str, currency: str = "USD"):
    """Qualifica il contratto su IB. Solleva RuntimeError se non trovato."""
    contract = Stock(ticker, "SMART", currency)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"IB non ha trovato il contratto per '{ticker}'")
    return qualified[0]


def _update_anag(ib: IB, ticker: str, contract) -> None:
    """Crea l'entry in anag (se assente) e popola i campi null da IB."""
    manager = TickerManager()
    anag = manager.get_ticker(ticker)

    if anag is None:
        manager.create_anag(ticker)
        anag = manager.get_ticker(ticker)
        logger.info(f"Creata nuova entry in anag per '{ticker}'")

    updates = {}
    if not anag.get("exchange"):
        updates["exchange"] = contract.primaryExchange or contract.exchange
    if not anag.get("currency"):
        updates["currency"] = contract.currency
    if not anag.get("asset_type"):
        updates["asset_type"] = contract.secType

    if not anag.get("name") or not anag.get("sector") or not anag.get("industry"):
        details = ib.reqContractDetails(contract)
        if details:
            d = details[0]
            if not anag.get("name") and getattr(d, "longName", None):
                updates["name"] = d.longName
            if not anag.get("sector") and getattr(d, "category", None):
                updates["sector"] = d.category
            if not anag.get("industry") and getattr(d, "industry", None):
                updates["industry"] = d.industry

    if not anag.get("country"):
        exchange = updates.get("exchange") or anag.get("exchange") or ""
        country = _EXCHANGE_COUNTRY.get(exchange.upper())
        if country:
            updates["country"] = country

    if updates:
        manager.update_ticker(ticker, **updates)
        logger.info(f"Anag aggiornata per '{ticker}': {updates}")


def _currency(manager: TickerManager, ticker: str) -> str:
    anag = manager.get_ticker(ticker)
    return (anag or {}).get("currency") or "USD"


def _eod_duration(ticker: str) -> str:
    last = get_last_saved_date(ticker, "eod")
    if last is None:
        return "20 Y"
    days = (date.today() - last).days + 2
    return f"{min(days, 365)} D"
