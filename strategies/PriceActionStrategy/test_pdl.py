"""
Test per PrevDayHighLow factor.

Esegui dalla root del progetto o da questa cartella:
    python strategies/PriceActionStrategy/test_pdl.py
oppure:
    cd strategies/PriceActionStrategy && python test_pdl.py
"""

import sys
from pathlib import Path
from datetime import date, time
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.factor import PrevDayHighLow
from engine.datafeed import DataFeed
from engine.engine import Researcher

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"
INFO = "\033[36mINFO\033[0m"

def check(cond: bool, msg: str):
    if not cond:
        raise AssertionError(f"  {FAIL}  {msg}")
    print(f"  {PASS}  {msg}")

def approx_eq(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — dati sintetici EOD (1 barra per giorno)
# ─────────────────────────────────────────────────────────────────────────────
#
# Le zone si basano sulla WICK della candela che ha generato il max/min del giorno:
#   HoD zone : [day_high  …  max(open,close) della candela con high==day_high]
#   LoD zone : [min(open,close) della candela con low==day_low  …  day_low]
#
# Con 1 barra/giorno la candela che fa il high è la stessa che fa il low.
#
# Scenario (O=open, H=high, L=low, C=close):
#
#  Day 1  O=99  H=105 L=95  C=102
#           candela HoD = stessa barra → hod_btm = max(99,102) = 102
#           candela LoD = stessa barra → lod_top = min(99,102) = 99
#           → primo rif, zone NaN
#  Day 2  O=101 H=108 L=92  C=106
#           hod_btm=max(101,106)=106  lod_top=min(101,106)=101
#           → non-inside vs D1 → ref diventa D2
#           zone D2 = D1: hod_top=105, hod_btm=102, lod_top=99, lod_btm=95
#  Day 3  O=105 H=107 L=93  C=104  → inside D2
#           zone D3 = D2: hod_top=108, hod_btm=106, lod_top=101, lod_btm=92
#  Day 4  O=103 H=107 L=94  C=105  → inside D2 (ref ancora D2)
#           zone D4 = D2: stesse di D3
#  Day 5  O=104 H=110 L=91  C=108  → NON inside D2
#           zone D5 = D2 (ref era ancora D2): stesse; dopo D5 ref → D5

def test_eod_synthetic():
    print("\n" + "="*60)
    print("TEST 1 — dati sintetici EOD (1 barra/giorno)")
    print("="*60)

    rows = [
        # date             time      O      H      L      C
        (date(2026, 1, 5), time(0, 0), 99.0, 105.0,  95.0, 102.0),
        (date(2026, 1, 6), time(0, 0), 101.0, 108.0,  92.0, 106.0),
        (date(2026, 1, 7), time(0, 0), 105.0, 107.0,  93.0, 104.0),
        (date(2026, 1, 8), time(0, 0), 103.0, 107.0,  94.0, 105.0),
        (date(2026, 1, 9), time(0, 0), 104.0, 110.0,  91.0, 108.0),
    ]
    df = pl.DataFrame({
        "date":   [r[0] for r in rows],
        "time":   [r[1] for r in rows],
        "open":   [r[2] for r in rows],
        "high":   [r[3] for r in rows],
        "low":    [r[4] for r in rows],
        "close":  [r[5] for r in rows],
        "ticker": ["TEST"] * len(rows),
    })

    factor = PrevDayHighLow()
    result = factor.compute(df)

    # ── Day 1: nessun riferimento → tutto NaN ─────────────────────────────────
    r0 = result.row(0, named=True)
    check(r0["pdl_hod_top"] is None, "Day1 hod_top = None (nessun riferimento)")
    check(r0["pdl_lod_btm"] is None, "Day1 lod_btm = None (nessun riferimento)")

    # ── Day 2: zone da D1 ─────────────────────────────────────────────────────
    # D1 (unica barra): la candela del high = la candela del low = stessa
    #   hod_btm = max(open=99, close=102) = 102
    #   lod_top = min(open=99, close=102) = 99
    r1 = result.row(1, named=True)
    check(approx_eq(r1["pdl_hod_top"], 105.0), "Day2 hod_top = D1_high = 105")
    check(approx_eq(r1["pdl_hod_btm"], 102.0), "Day2 hod_btm = max(open=99,close=102) = 102")
    check(approx_eq(r1["pdl_lod_top"],  99.0), "Day2 lod_top = min(open=99,close=102) = 99")
    check(approx_eq(r1["pdl_lod_btm"],  95.0), "Day2 lod_btm = D1_low = 95")
    check(approx_eq(r1["pdl_inside"],    0.0), "Day2 NON inside")

    # ── Day 3: inside D2 → zone da D2 ────────────────────────────────────────
    # D2 (unica barra): candela del high e del low = stessa
    #   hod_btm = max(open=101, close=106) = 106
    #   lod_top = min(open=101, close=106) = 101
    r2 = result.row(2, named=True)
    check(approx_eq(r2["pdl_hod_top"], 108.0), "Day3 hod_top = D2_high = 108")
    check(approx_eq(r2["pdl_hod_btm"], 106.0), "Day3 hod_btm = max(open=101,close=106) = 106")
    check(approx_eq(r2["pdl_lod_top"], 101.0), "Day3 lod_top = min(open=101,close=106) = 101")
    check(approx_eq(r2["pdl_lod_btm"],  92.0), "Day3 lod_btm = D2_low = 92")
    check(approx_eq(r2["pdl_inside"],   1.0),  "Day3 è inside (ref = D2)")

    # ── Day 4: inside D2 di nuovo → stesse zone di D3 ────────────────────────
    r3 = result.row(3, named=True)
    check(approx_eq(r3["pdl_hod_top"], 108.0), "Day4 hod_top = D2_high = 108 (carry-forward)")
    check(approx_eq(r3["pdl_hod_btm"], 106.0), "Day4 hod_btm = 106 (carry-forward)")
    check(approx_eq(r3["pdl_lod_top"], 101.0), "Day4 lod_top = 101 (carry-forward)")
    check(approx_eq(r3["pdl_lod_btm"],  92.0), "Day4 lod_btm = D2_low = 92 (carry-forward)")
    check(approx_eq(r3["pdl_inside"],   1.0),  "Day4 è inside (ref = D2)")

    # ── Day 5: NON inside D2, ma ref era ancora D2 → zone ancora da D2 ───────
    r4 = result.row(4, named=True)
    check(approx_eq(r4["pdl_hod_top"], 108.0), "Day5 hod_top = D2_high = 108 (ref era D2)")
    check(approx_eq(r4["pdl_hod_btm"], 106.0), "Day5 hod_btm = 106")
    check(approx_eq(r4["pdl_lod_top"], 101.0), "Day5 lod_top = 101")
    check(approx_eq(r4["pdl_lod_btm"],  92.0), "Day5 lod_btm = D2_low = 92")
    check(approx_eq(r4["pdl_inside"],   0.0),  "Day5 NON inside (110 > 108 rompe D2)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — dati sintetici intraday (2 barre per giorno)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Day 1  bar1 O=99  H=104 L=97  C=100
#         bar2 O=100 H=105 L=95  C=102
#           day_high=105 → candela bar2: hod_btm = max(open=100,close=102) = 102
#           day_low=95   → candela bar2: lod_top = min(open=100,close=102) = 100
#  Day 2  bar1 O=102 H=107 L=93  C=103
#         bar2 O=103 H=108 L=92  C=106  → NON inside D1
#           zone D2 = D1: hod_top=105, hod_btm=102, lod_top=100, lod_btm=95

def test_intraday_synthetic():
    print("\n" + "="*60)
    print("TEST 2 — dati sintetici intraday (2 barre/giorno)")
    print("="*60)

    rows = [
        (date(2026, 1, 5), time(9,  0),  99.0, 104.0,  97.0, 100.0),
        (date(2026, 1, 5), time(9, 15), 100.0, 105.0,  95.0, 102.0),
        (date(2026, 1, 6), time(9,  0), 102.0, 107.0,  93.0, 103.0),
        (date(2026, 1, 6), time(9, 15), 103.0, 108.0,  92.0, 106.0),
    ]
    df = pl.DataFrame({
        "date":   [r[0] for r in rows],
        "time":   [r[1] for r in rows],
        "open":   [r[2] for r in rows],
        "high":   [r[3] for r in rows],
        "low":    [r[4] for r in rows],
        "close":  [r[5] for r in rows],
        "ticker": ["X"] * len(rows),
    })

    factor = PrevDayHighLow()
    result = factor.compute(df)

    # D1: entrambe le barre → NaN
    for i in [0, 1]:
        r = result.row(i, named=True)
        check(r["pdl_hod_top"] is None, f"D1 bar{i} hod_top = None")
        check(r["pdl_lod_btm"] is None, f"D1 bar{i} lod_btm = None")

    # D2: entrambe le barre → zone da D1
    # D1: high=105 fatto da bar2 (O=100,C=102) → hod_btm=max(100,102)=102
    #     low=95  fatto da bar2 (O=100,C=102) → lod_top=min(100,102)=100
    for i in [2, 3]:
        r = result.row(i, named=True)
        check(approx_eq(r["pdl_hod_top"], 105.0), f"D2 bar{i} hod_top = D1_high = 105")
        check(approx_eq(r["pdl_hod_btm"], 102.0), f"D2 bar{i} hod_btm = max(open=100,close=102) = 102")
        check(approx_eq(r["pdl_lod_top"], 100.0), f"D2 bar{i} lod_top = min(open=100,close=102) = 100")
        check(approx_eq(r["pdl_lod_btm"],  95.0), f"D2 bar{i} lod_btm = D1_low = 95")
        check(approx_eq(r["pdl_inside"],    0.0), f"D2 bar{i} NON inside")

    # Entrambe le barre di D2 hanno la stessa zona base (nessun pattern di
    # inversione si verifica perché ci sono solo 2 barre e la prima è D1)
    check(
        (result.row(2, named=True)["pdl_hod_top"] ==
         result.row(3, named=True)["pdl_hod_top"]),
        "entrambe le barre dello stesso giorno hanno la stessa zona base",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — aggiornamento intraday (pattern di inversione)
# ─────────────────────────────────────────────────────────────────────────────
#
# Scenario:
#   Ref day (Day1): O=100 H=110 L=90 C=105
#     → candela HoD: max(100,105)=105  candela LoD: min(100,105)=100
#     → zona base: HoD=[110,105]  LoD=[100,90]
#
#   Day2 — 4 barre intraday:
#     bar0: O=108 H=109 L=106 C=107  (bearish — nessun pattern, prima barra)
#     bar1: O=107 H=111 L=106 C=108  (bullish — C>O; high=111 > hod_top=110)
#              → bullish ma NON c'è stato prev bearish+curr bullish per LoD
#              → però c'è SOLO bull, non aggiorna ancora
#     bar2: O=108 H=113 L=107 C=109  (bullish — C>O; high=113 > hod_top=110)
#              → bar1 era bullish + bar2 è bullish: NO HoD update (serve prev bull + curr bear)
#     bar3: O=109 H=114 L=105 C=106  (bearish — C<O; high=114 > hod_top=110)
#              → bar2 era bullish + bar3 è bearish + high=114 > 110: HoD UPDATE!
#              → nuova HoD: top=114, btm=max(109,106)=109
#
#   Verifica:
#     bar0: zona base HoD=[110,105]
#     bar1: zona base HoD=[110,105]  (bull+bull: nessun update)
#     bar2: zona base HoD=[110,105]  (bull+bull: nessun update)
#     bar3: zona AGGIORNATA HoD=[114,109]

def test_intraday_update():
    print("\n" + "="*60)
    print("TEST 3 — aggiornamento intraday (pattern inversione)")
    print("="*60)

    rows = [
        # date              time       O      H      L      C
        (date(2026, 1, 5), time(0, 0), 100.0, 110.0,  90.0, 105.0),  # Day1 ref
        (date(2026, 1, 6), time(9,  0), 108.0, 109.0, 106.0, 107.0),  # Day2 bar0 bearish
        (date(2026, 1, 6), time(9, 15), 107.0, 111.0, 106.0, 108.0),  # Day2 bar1 bullish
        (date(2026, 1, 6), time(9, 30), 108.0, 113.0, 107.0, 109.0),  # Day2 bar2 bullish
        (date(2026, 1, 6), time(9, 45), 109.0, 114.0, 105.0, 106.0),  # Day2 bar3 bearish → UPDATE
    ]
    df = pl.DataFrame({
        "date":   [r[0] for r in rows],
        "time":   [r[1] for r in rows],
        "open":   [r[2] for r in rows],
        "high":   [r[3] for r in rows],
        "low":    [r[4] for r in rows],
        "close":  [r[5] for r in rows],
        "ticker": ["TEST"] * len(rows),
    })

    factor = PrevDayHighLow()
    result = factor.compute(df)

    # Day1: nessun riferimento
    r0 = result.row(0, named=True)
    check(r0["pdl_hod_top"] is None, "Day1 hod_top = None")

    # Day2 bar0: zona base da Day1
    r1 = result.row(1, named=True)
    check(approx_eq(r1["pdl_hod_top"], 110.0), "Day2 bar0 hod_top = 110 (zona base)")
    check(approx_eq(r1["pdl_hod_btm"], 105.0), "Day2 bar0 hod_btm = 105 (zona base)")

    # Day2 bar1 (prev=bearish, curr=bullish): nessun HoD update (serve prev_bull+curr_bear)
    r2 = result.row(2, named=True)
    check(approx_eq(r2["pdl_hod_top"], 110.0), "Day2 bar1 hod_top = 110 (nessun update: bear+bull)")
    check(approx_eq(r2["pdl_hod_btm"], 105.0), "Day2 bar1 hod_btm = 105 (invariato)")

    # Day2 bar2 (prev=bullish, curr=bullish): nessun update (serve curr_bear)
    r3 = result.row(3, named=True)
    check(approx_eq(r3["pdl_hod_top"], 110.0), "Day2 bar2 hod_top = 110 (nessun update: bull+bull)")
    check(approx_eq(r3["pdl_hod_btm"], 105.0), "Day2 bar2 hod_btm = 105 (invariato)")

    # Day2 bar3 (prev=bullish, curr=bearish, high=114>110): HoD AGGIORNATO
    r4 = result.row(4, named=True)
    check(approx_eq(r4["pdl_hod_top"], 114.0), "Day2 bar3 hod_top = 114 (AGGIORNATO)")
    check(approx_eq(r4["pdl_hod_btm"], 109.0), "Day2 bar3 hod_btm = max(109,106) = 109 (AGGIORNATO)")
    # LoD invariata (nessun pattern di inversione al ribasso)
    check(approx_eq(r4["pdl_lod_btm"],  90.0), "Day2 bar3 lod_btm = 90 (invariata)")
    check(approx_eq(r4["pdl_lod_top"], 100.0), "Day2 bar3 lod_top = 100 (invariata)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — vincoli su dati reali (SX5E 15m)
# ─────────────────────────────────────────────────────────────────────────────

def test_real_data():
    print("\n" + "="*60)
    print("TEST 4 — vincoli su dati reali (SX5E 15m)")
    print("="*60)

    data_dir = Path(__file__).resolve().parents[2] / "database"
    if not data_dir.exists():
        print(f"  {SKIP}  directory database non trovata ({data_dir})")
        return

    feed = DataFeed(prices_base=data_dir)
    factor = PrevDayHighLow()
    researcher = Researcher(
        [factor], feed,
        start_date="2026-01-01",
        end_date="2026-03-28",
        frequency="15m",
        tickers=["SX5E"],
    )

    df = researcher.get_data()
    if df is None or df.height == 0:
        print(f"  {SKIP}  nessun dato restituito dal feed")
        return

    print(f"  {INFO}  righe dopo warmup trim: {df.height}")

    # Tutte le colonne presenti
    for col in ["pdl_hod_top", "pdl_hod_btm", "pdl_lod_top", "pdl_lod_btm", "pdl_inside"]:
        check(col in df.columns, f"colonna '{col}' presente")

    # Nessun null dopo il trim del Researcher
    for col in ["pdl_hod_top", "pdl_hod_btm", "pdl_lod_top", "pdl_lod_btm", "pdl_inside"]:
        nulls = df[col].null_count()
        check(nulls == 0, f"'{col}' senza null (trovati: {nulls})")

    # hod_top >= hod_btm (il massimo del giorno >= close massimo)
    n_inv = df.filter(pl.col("pdl_hod_top") < pl.col("pdl_hod_btm")).height
    check(n_inv == 0, f"HoD: hod_top >= hod_btm in tutte le righe (invertite: {n_inv})")

    # lod_top >= lod_btm (close minimo >= il minimo del giorno)
    n_inv = df.filter(pl.col("pdl_lod_top") < pl.col("pdl_lod_btm")).height
    check(n_inv == 0, f"LoD: lod_top >= lod_btm in tutte le righe (invertite: {n_inv})")

    # lod_btm <= hod_top (minimo <= massimo — sempre vero)
    n_cross = df.filter(pl.col("pdl_lod_btm") > pl.col("pdl_hod_top")).height
    check(n_cross == 0, f"lod_btm <= hod_top in tutte le righe (incrociati: {n_cross})")

    # inside flag binario (0.0 o 1.0)
    n_nb = df.filter(
        (pl.col("pdl_inside") != 0.0) & (pl.col("pdl_inside") != 1.0)
    ).height
    check(n_nb == 0, f"pdl_inside sempre 0.0 o 1.0 (anomalie: {n_nb})")

    # Intraday: hod_top può solo crescere (o restare uguale) durante il giorno,
    # lod_btm può solo scendere (o restare uguale) — verifica monotonia per giorno
    monotone_hod = (
        df.sort(["ticker", "date", "time"])
        .with_columns(pl.col("pdl_hod_top").shift(1).over(["ticker", "date"]).alias("prev_hod"))
        .filter(pl.col("prev_hod").is_not_null())
        .filter(pl.col("pdl_hod_top") < pl.col("prev_hod"))
    )
    check(monotone_hod.height == 0,
          f"hod_top non decresce intraday (violazioni: {monotone_hod.height})")

    monotone_lod = (
        df.sort(["ticker", "date", "time"])
        .with_columns(pl.col("pdl_lod_btm").shift(1).over(["ticker", "date"]).alias("prev_lod"))
        .filter(pl.col("prev_lod").is_not_null())
        .filter(pl.col("pdl_lod_btm") > pl.col("prev_lod"))
    )
    check(monotone_lod.height == 0,
          f"lod_btm non cresce intraday (violazioni: {monotone_lod.height})")

    # Statistiche riassuntive
    n_inside = int(df.filter(pl.col("pdl_inside") == 1.0).height)
    n_tot    = df.height
    pct      = 100 * n_inside / n_tot if n_tot else 0
    print(f"  {INFO}  barre inside-day: {n_inside}/{n_tot} ({pct:.1f}%)")

    hod_width = (df["pdl_hod_top"] - df["pdl_hod_btm"]).mean()
    lod_width = (df["pdl_lod_top"] - df["pdl_lod_btm"]).mean()
    print(f"  {INFO}  ampiezza media zona HoD: {hod_width:.3f}")
    print(f"  {INFO}  ampiezza media zona LoD: {lod_width:.3f}")

    # Mostra le prime righe come preview (safe encoding per Windows)
    preview_cols = ["date", "time", "close",
                    "pdl_hod_top", "pdl_hod_btm",
                    "pdl_lod_top", "pdl_lod_btm", "pdl_inside"]
    preview_str = str(df.select(preview_cols).head(6))
    sys.stdout.buffer.write(
        f"\n  Preview (prime 6 righe):\n{preview_str}\n\n".encode("utf-8")
    )


# ─────────────────────────────────────────────────────────────────────────────
# entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    failed = []
    for fn in [test_eod_synthetic, test_intraday_synthetic, test_intraday_update, test_real_data]:
        try:
            fn()
        except AssertionError as exc:
            print(exc)
            failed.append(fn.__name__)
        except Exception as exc:
            import traceback
            print(f"  \033[31mERROR\033[0m  {fn.__name__}: {exc}")
            traceback.print_exc()
            failed.append(fn.__name__)

    print("\n" + "="*60)
    if failed:
        print(f"\033[31mFAILED\033[0m — test falliti: {failed}")
        sys.exit(1)
    else:
        print(f"\033[32mALL TESTS PASSED\033[0m")
