# REFACTOR.md — Quanto

> Documento vivente. Aggiorna le checkbox man mano che gli interventi vengono completati.  
> Ultima revisione: 2026-05-02

---

## 0. Protocollo di lavoro — da seguire per ogni item

Quando avvii una sessione di refactor, passa questo file all'agente (`@REFACTOR.md`) e indica quale item vuoi completare (es. "completa A1"). L'agente seguirà il protocollo qui sotto per ogni item completato.

### Passi obbligatori per ogni item (A1, A2, … F5)

```
1. Leggi il codice coinvolto        →  capisci lo stato attuale prima di toccare nulla
2. Esponi dubbi e proponi soluzioni →  illustra opzioni con pro/contro; attendi conferma
3. Implementa la modifica           →  minima, focalizzata sull'item dichiarato
4. Scrivi / aggiorna i test         →  nella cartella tests/ con pytest
5. Lancia i test                    →  pytest tests/ -v  (tutti devono passare)
6. Aggiorna la documentazione       →  .cursor/rules/quanto-architecture.mdc
7. Segna l'item come completato     →  [ ] → [x] in questo file
```

### Regole di qualità

- **Leggi prima, proponi dopo.** Il passo 2 è obbligatorio: non iniziare l'implementazione senza aver esposto eventuali dubbi, ambiguità o approcci alternativi. Se ci sono più soluzioni valide, presentale con pro/contro e aspetta conferma.
- **Semplicità prima di tutto.** La soluzione preferita è sempre la più semplice che risolve il problema. Refactoring strutturali più ampi possono essere *suggeriti* come nota a margine, ma non devono essere il focus né essere implementati senza esplicita richiesta.
- **Un item alla volta.** Non iniziare l'item successivo finché quello corrente non ha test verdi e documentazione aggiornata.
- **Nessuna modifica a cascata non pianificata.** Se durante un item emerge un problema non previsto, aggiungilo alla sezione 2 o 4 di questo file invece di risolverlo sul momento.
- **Aggiorna `quanto-architecture.mdc` solo per cambiamenti strutturali** (nuove classi, nuovi moduli, contratti d'interfaccia modificati) — non per bug fix interni.
- **Commit atomici.** Un commit per item completato, con messaggio `refactor(XX): <descrizione breve>` (es. `refactor(A1): unify PositionType in portfolio.py`).

### Cosa aggiornare in `quanto-architecture.mdc`

| Tipo di modifica | Cosa aggiornare |
|------------------|-----------------|
| Nuovo modulo (`engine/labeling.py`, ecc.) | Aggiungi riga alla tabella "Main modules" |
| Interfaccia / ABC modificata | Aggiorna o aggiungi l'esempio di utilizzo nella sezione pertinente |
| Enum / tipo spostato | Aggiorna il commento nella colonna "Contents" |
| Nuovo fattore concreto | Aggiungi alla lista "Existing concrete factors" |
| Metodo deprecato | Aggiungi nota `[DEPRECATED]` e versione di rimozione prevista |

---

## 1. Obiettivi del refactor

Il refactor ha tre scopi principali, ciascuno mappato sui Business Goals del BRD:

| Obiettivo | Business Goals correlati |
|-----------|--------------------------|
| **O1 — Eliminare il debito tecnico bloccante** — fix di bug critici e inconsistenze che impediscono il corretto funzionamento del backtest (metriche corrotte, CLI rotta, enum duplicati). | BG-2, BG-4 |
| **O2 — Stabilire contratti d'interfaccia puliti** — un unico path di backtest canonico, import espliciti, packaging installabile; le strategie non devono manipolare `sys.path`. | BG-2, BG-3 |
| **O3 — Costruire le fondamenta analitiche mancanti** — PIT compliance, labeling (t0/t1), sample weights, validazione leakage-safe (Purged CV / CPCV), struttura modulare signal→sizing→risk. | BG-1, BG-2, BG-3, BG-5 |
| **O4 — Rendere il sistema production-grade** — reporting netto dei costi, audit trail per trade, monitoring, alerting, kill switch. | BG-4, BG-5 |

---

## 2. Problemi per modulo

Legenda severità: 🔴 CRITICO — blocca funzionamento o produce risultati scorretti · 🟡 MEDIO — limita scalabilità/correttezza in casi non banali · ⚪ BASSO — qualità del codice, leggibilità

---

### `engine/engine.py`

| # | Problema | Severità |
|---|----------|----------|
| E1 | **Due path di backtest** (`backtest` vs `backtest_refactor`): nessuna è dichiarata canonica, le strategie usano `backtest_refactor` ma `backtest` rimane nel codice, crea confusione sul contratto della `Strategy` ABC. | 🔴 |
| E2 | **`PositionType` importato due volte** — da `execution` e da `portfolio`; il secondo shadowing il primo. Fonte: duplicazione dell'enum. | 🟡 |
| ~~E3~~ | ~~**Caching `Researcher.get_data` disabilitato** — la riga `# self._data_cache = df` è commentata; il docstring afferma che il cache esiste; ogni chiamata ricalcola tutti i fattori.~~ **Cache riabilitato: `self._data_cache = df` decommentato.** | ~~🟡~~ ✅ |
| E4 | **`RiskManager` salvato ma mai chiamato** — `self.risk` è istanziato in `on_start` ma nessun punto del loop di backtest lo invoca; i risk check non avvengono mai. | 🟡 |
| ~~E5~~ | ~~**`get_data_eod` ignora `self.frequency`** — carica sempre `'eod'` a prescindere dalla frequenza configurata sul `Researcher`.~~ **Comportamento intenzionale documentato: aggiunto docstring esplicito che chiarisce lo scopo separato del metodo (EOD context per `backtest_refactor`).** | ~~⚪~~ ✅ |
| E6 | **`positions_summary` concat ogni barra** — accumulo `O(N²)` in memoria; scala male su backtest lunghi. | ⚪ |

---

### `engine/execution.py`

| # | Problema | Severità |
|---|----------|----------|
| X1 | **`OrderType.LMT` non implementato** — l'enum esiste ma il metodo `execute` gestisce solo il path market order; un ordine limite non viene mai eseguito. | 🔴 |
| X2 | **`PositionType` duplicato** — definito anche in `portfolio.py` con gli stessi valori; due fonti di verità. | 🟡 |
| X3 | **`assert` per validazione mode** — usa `assert mode in (...)` invece di un `ValueError` esplicito; disabilitabile con `python -O`. | ⚪ |

---

### `engine/portfolio.py`

| # | Problema | Severità |
|---|----------|----------|
| P1 | **Short selling: cash non accreditato all'apertura** — `add_position` sottrae sempre `notional + entry_costs` dal cash anche per posizioni short, invece di accreditare i proventi della vendita. I PnL sembrano corretti solo per strategie long-only. | 🔴 |
| P2 | **`PositionType` duplicato** — vedi X2. | 🟡 |
| P3 | **`TradeId` a 5 cifre random** — rischio collisione su run con molti trade o sessioni multiple; UUID v4 sarebbe deterministicamente unico. | 🟡 |

---

### `engine/risk.py`

| # | Problema | Severità |
|---|----------|----------|
| R1 | **Stub puro, mai applicato** — l'unico campo è `max_leverage: int = 4`; nessun metodo; nessun punto del codice lo consulta. Il modulo esiste ma non offre nessuna protezione reale (richiesta da BR-6). | 🔴 |

---

### `engine/reports.py`

| # | Problema | Severità |
|---|----------|----------|
| Rp1 | **`StrategyAnalytics.plot_price` broken** — referenzia `self.backtester` che non viene mai assegnato (la classe riceve `self.strategy`); qualsiasi chiamata genera `AttributeError`. | 🔴 |
| Rp2 | **`daily_equity` mutato in-place tra le chiamate alle metriche** — `sharpe_ratio`, `expected_return`, ecc. sostituiscono gli zeri con `np.nan` direttamente su `self.daily_equity`; la prima metrica chiamata corrompe i dati per tutte le successive. | 🔴 |
| Rp3 | **Due sistemi di reporting paralleli non unificati** — `StrategyAnalytics` (metriche + Excel per la `Strategy`) e `ReportWriter` (parquet/csv logging) coesistono senza un'interfaccia comune; chi sviluppa una nuova strategia non sa quale usare. | 🟡 |
| Rp4 | **Nessun report net PnL / cost breakdown** — BR-7 richiede sempre gross PnL, costs, net PnL e turnover; il report attuale non distingue costi dalla performance lorda. | 🟡 |
| Rp5 | **Nessun audit trail / decision card per trade** — BR-8/FR-7 richiedono tracciabilità "why/what/when/with-which-version" per ogni decisione; non implementato. | 🟡 |

---

### `engine/factor.py`

| # | Problema | Severità |
|---|----------|----------|
| ~~F1~~ | ~~**Nessun supporto labeling (t0/t1)** — i fattori producono segnali ma non espongono l'intervallo evento `[t0, t1]` necessario per Purged CV, sample weights e meta-labeling (BR-5).~~ **Hook `t1_offset() -> Optional[int]` aggiunto al Factor ABC; creato `engine/labeling.py` con `FixedHorizonLabeler` (preview di D3). `SignalExitLabeler` rimane in Blocco D.** | ~~🔴~~ ✅ |
| F2 | **Loop NumPy per-ticker in `BuySellLiquidity`, `MarketStructure`, `PrevDayHighLow`** — corretto ma non scalabile; su universe ampi o backtest lunghi il bottleneck è in Python. | 🟡 |
| ~~F3~~ | ~~**Lookahead non documentato uniformemente** — alcuni fattori usano `shift(-n)` per la conferma dei pivot (corretto in research ma richiede lag esplicito in backtest); il contratto "quante barre di lag devo aggiungere" non è uniforme tra fattori.~~ **Aggiunto warning ⚠️ esplicito in `MarketStructure` (allineato a quello già presente in `SwingHighLow`).** | ~~🟡~~ ✅ |
| ~~F4~~ | ~~**Firma `compute` astratta inconsistente** — il tipo annotato varia tra implementazioni (`market_data` vs `pl.DataFrame`).~~ **ABC e `MovingAverage` allineati: `compute(self, market_data: pl.DataFrame) -> pl.DataFrame`.** | ~~⚪~~ ✅ |

---

### `engine/datafeed.py`

| # | Problema | Severità |
|---|----------|----------|
| ~~D1~~ | ~~**`get_bar` usa calendario globale cross-ticker** — "N barre fa" è calcolato sul set unione di `(date, time)` di tutti i ticker; su dataset multi-ticker con barre mancanti si perde la corrispondenza per-ticker.~~ **Documentato nel docstring di `get_bar()` con nota esplicita sul comportamento e sulle implicazioni multi-ticker.** | ~~🟡~~ ✅ |
| ~~D2~~ | ~~**Semantica offset negativo non ovvia** — un offset negativo in `price()`/`get()` avanza nel futuro invece di tornare indietro; controintuitivo e non documentato chiaramente.~~ **Aggiunto `allow_lookahead: bool = False` su `MarketData`: offset negativo solleva `ValueError` per default; warning esplicito nei docstring.** | ~~⚪~~ ✅ |
| ~~D3~~ | ~~**Commenti misto italiano/inglese** — riduce leggibilità del codice.~~ **Tutti i commenti inline tradotti in inglese.** | ~~⚪~~ ✅ |

---

### `engine/plotting.py`

| # | Problema | Severità |
|---|----------|----------|
| Pl1 | **`_should_use_secondary_axis` definito ma mai chiamato** — dead code. | ⚪ |
| Pl2 | **Import style inconsistente** — alcune chiamate usano `from engine.factor import ...` (assoluto), altre `.factor` (relativo); funziona se `engine` è sul `sys.path` ma è fragile dopo il packaging. | ⚪ |

---

### `datafetch/`

| # | Problema | Severità |
|---|----------|----------|
| Df1 | **CLI BBG broken** — `datafetch/cli/download_data.py` importa `datafetch.bbg.download_data` e `BloombergConnection` che non esistono (il modulo BBG reale si chiama `downloader.py` e la classe `BloombergDataDownloader`). Il path CLI BBG solleva `ImportError` al primo utilizzo. | 🔴 |
| Df2 | **`blpapi` non in `requirements.txt`** — dipendenza implicita; chi installa il progetto da `requirements.txt` non ottiene Bloomberg. | 🟡 |
| Df3 | **Nessun "Data Contract" esplicito BBG/IB** — non è garantito che i parquet prodotti da Bloomberg e da IB abbiano lo stesso schema, timezone e naming convention; Phase 0 del BRD richiede un contratto documentato e verificato. | 🟡 |

---

### `strategies/`

| # | Problema | Severità |
|---|----------|----------|
| S1 | **`sys.path.insert` + path relativo `../../database`** — fragile; se la working directory cambia (CI, notebook aperto da cartella diversa) tutto si rompe. Risolvibile con packaging + entry point. | 🟡 |
| S2 | **Star imports da moduli engine** — `from engine.engine import *`, ecc.; inquina il namespace locale e rende difficile capire da dove vengono i simboli. | 🟡 |
| S3 | **`RiskManager` passato ma ignorato** — le strategie passano un `RiskManager` a `backtest_refactor` ma l'engine non lo usa mai (vedi R1). | ⚪ |
| S4 | **Nessun template `Strategy Specification`** — BR-3/FR-1 richiedono che ogni strategia abbia una specifica (ipotesi, universo, orizzonte, labeling, sizing, cost model) prima del backtest. Non esiste un template né una convenzione. | 🟡 |

---

## 3. Gap vs BRD — funzionalità mancanti

| Phase BRD | Descrizione | Stato attuale |
|-----------|-------------|---------------|
| **Phase 0** | Data Contract BBG/IB: schema unificato, timezone policy, campi obbligatori, normalizzazione | Parziale — due downloader separati senza contratto verificato |
| **Phase 1** | PIT checker, availability lag policy, as-of join PIT-safe | Assente |
| **Phase 2** | Volume/Dollar bars, event-driven bar engine | Assente — solo time bars |
| **Phase 3** | Labeling framework (t0/t1, fixed-horizon, signal-exit), sample weights (concurrency/uniqueness), class weights, meta-labeling | Assente |
| **Phase 4** | Walk-forward OOS, Purged CV + embargo, CPCV, report OOS standardizzato | Assente |
| **Phase 5** | Backtest cost-aware canonico con gross/net PnL, cost breakdown, turnover | Parziale — costi simulati ma non riportati separatamente |
| **Phase 6** | Sizing module indipendente, portfolio constraints (exposure, leverage, concentrazione), pre-trade risk checks | Assente — `RiskManager` è stub |
| **Phase 7** | OMS state, execution tactics (TWAP/VWAP/limit), riconciliazione ordini/fill | Assente |
| **Phase 8** | Dashboard KPI live, alerting (staleness/drift/decay), kill switch + runbook | Assente |
| **Phase 9** | Decision card per trade, feature attribution, audit trail "why/what/when/version" | Assente |

---

## 4. Piano ordinato di intervento

I blocchi sono ordinati per dipendenza: ogni blocco può iniziare solo se i blocchi precedenti sono completati.  
All'interno di ogni blocco, gli item contrassegnati con `(parallelo)` possono essere lavorati in parallelo.

```
A (debito critico)
│
├─► B (qualità ingegneristica)
│       │
│       └─► C (struttura modulare)
│               │
│               ├─► D (fondamenta analitiche)
│               │       │
│               │       └─► E (validazione avanzata)
│               │
│               └─► F (production readiness) [dopo E]
```

---

### Blocco A — Debito critico *(prerequisito di tutto)*

> Fix di bug che producono risultati scorretti o che impediscono l'uso di funzionalità dichiarate.

- [ ] **A1** — Unificare `PositionType` in un unico modulo (proposta: tenerlo in `portfolio.py`, importarlo da `execution.py` e `engine.py`)  
  _File: `engine/portfolio.py`, `engine/execution.py`, `engine/engine.py`_

- [ ] **A2** — Fix `StrategyAnalytics.plot_price`: sostituire la ref a `self.backtester` con `self.strategy` (o rimuovere il metodo se non supportato)  
  _File: `engine/reports.py`_

- [ ] **A3** — Fix mutazione `daily_equity` in-place: usare `.copy()` all'inizio di ogni metodo di metrica che modifica i dati  
  _File: `engine/reports.py`_

- [ ] **A4** — Fix CLI BBG: allineare `datafetch/cli/download_data.py` agli import reali (`datafetch.bbg.downloader.BloombergDataDownloader`)  
  _File: `datafetch/cli/download_data.py`_

- [ ] **A5** — Canonicalizzare un solo path di backtest: dichiarare `backtest_refactor` come unico metodo supportato; deprecare `backtest` con un warning e rimuoverlo nella release successiva  
  _File: `engine/engine.py`, documentazione_

---

### Blocco B — Qualità ingegneristica *(dipende da A)*

> Miglioramenti che non cambiano la logica ma rendono il progetto più robusto e manutenibile.

- [ ] **B1** — Aggiungere `pyproject.toml` (o aggiornare `setup.py`); rimuovere `sys.path.insert` dalle strategie usando import relativi al package installato *(parallelo)*  
  _File: `pyproject.toml` (nuovo), `strategies/*/strategy.py`_

- [ ] **B2** — Sostituire star imports con import espliciti in tutte le strategie e nei notebook *(parallelo)*  
  _File: `strategies/*/strategy.py`, `strategies/example/*.ipynb`_

- [ ] **B3** — `OrderType.LMT`: implementare un path minimo (limit order con price check al momento dell'esecuzione) oppure rimuovere il valore dall'enum e documentare la scelta *(parallelo)*  
  _File: `engine/execution.py`_

- [ ] **B4** — Fix short selling cash accounting in `Portfolio.add_position`: per posizioni short, accreditare `notional` al cash all'apertura (vendita allo scoperto)  
  _File: `engine/portfolio.py`_

- [ ] **B5** — Sostituire `TradeId` random 5-digit con `uuid.uuid4()` *(parallelo)*  
  _File: `engine/portfolio.py`_

- [ ] **B6** — Abilitare caching in `Researcher.get_data` (decommentare e completare la riga `self._data_cache`) *(parallelo)*  
  _File: `engine/engine.py`_

- [ ] **B7** — Aggiungere `blpapi` a `requirements.txt` (o documentare come dipendenza opzionale); scrivere un documento `docs/data_contract.md` che specifica lo schema parquet comune BBG/IB (timezone UTC, colonne obbligatorie, `type` values, naming convention ticker)  
  _File: `requirements.txt`, `docs/data_contract.md` (nuovo)_

---

### Blocco C — Struttura modulare *(dipende da B)*

> Separazione delle responsabilità richiesta da BR-6; habilita i blocchi D ed F.

- [ ] **C1** — `RiskManager`: implementare enforcement reale con almeno: `check_pre_trade(order, portfolio) -> bool` (leverage cap, notional limit per trade, max open positions); integrarlo nel loop di backtest prima di `execute_pending_orders`  
  _File: `engine/risk.py`, `engine/engine.py`_

- [ ] **C2** — Interfaccia signal → sizing → portfolio constraint: definire tre protocolli/ABC separati:  
  - `SignalInterface`: produce `score: float` e `side: PositionType`  
  - `SizingModel`: trasforma score + portfolio state in `quantity`  
  - `PortfolioConstraints`: applica caps post-sizing  
  _File: `engine/signal.py` (nuovo), `engine/sizing.py` (nuovo), aggiornare `engine/portfolio.py`_

- [ ] **C3** — Unificare i sistemi di reporting: definire una classe `BacktestReport` che consolida `StrategyAnalytics` e `ReportWriter`; deprecare `ReportWriter` standalone come API pubblica delle strategie  
  _File: `engine/reports.py`_

- [ ] **C4** — Aggiungere al report: gross PnL, costs totali, **net PnL**, turnover (numero trade e notional turnover rate) — sia nel sommario Excel che nel `summary()` a video (BR-7)  
  _File: `engine/reports.py`_

---

### Blocco D — Fondamenta analitiche *(dipende da C)*

> Funzionalità richieste dalla roadmap BRD (Phase 1-3) per passare da research tool a framework ML-ready.

- [ ] **D1** — PIT checker + availability lag: aggiungere a `DataFeed` / `Researcher` un meccanismo che associa a ogni feature un `availability_timestamp`; `PitChecker.validate(df)` deve fallire se rileva join future-looking  
  _File: `engine/datafeed.py`, `engine/pit.py` (nuovo)_

- [ ] **D2** — Volume/Dollar bars: implementare `BarEngine` con `TimeBars` (esistente, rinominare), `VolumeBars`, `DollarBars`; ogni bar type deve essere deterministico (stesso input → stessa barra)  
  _File: `engine/bars.py` (nuovo)_

- [ ] **D3** — Labeling framework:  
  - `FixedHorizonLabeler(horizon: int)` → label + t0/t1  
  - `SignalExitLabeler(exit_condition, max_holding, session_close_rule)` → label + t0/t1 + `exit_reason`  
  - Output obbligatorio per riga: `t0, t1, y, exit_reason, holding_time`  
  _File: `engine/labeling.py` (nuovo)_

- [ ] **D4** — Sample weights: `compute_sample_weights(events_df)` basato su concurrency/uniqueness degli intervalli `[t0, t1]`; `compute_class_weights(y)` per imbalance; entrambi loggati nell'audit  
  _File: `engine/weights.py` (nuovo)_

---

### Blocco E — Validazione leakage-safe *(dipende da D)*

> Implementazione della validazione richiesta da BR-2/FR-3; richiede D3 (t0/t1 disponibili).

- [ ] **E1** — Walk-forward OOS baseline: `WalkForwardCV(n_splits, gap)` — split temporali senza shuffle  
  _File: `engine/validation.py` (nuovo)_

- [ ] **E2** — Purged CV + embargo: `PurgedKFold(n_splits, embargo_pct)` che usa gli intervalli `[t0, t1]` per rimuovere dal training le osservazioni contaminate  
  _File: `engine/validation.py`_

- [ ] **E3** — CPCV stress-test: `CombPurgedKFold(n_splits, n_test_splits)` che genera distribuzioni di metriche OOS su molti path; output: `{median, p10, p90, worst_case}` per metrica  
  _File: `engine/validation.py`_

- [ ] **E4** — Report OOS standardizzato: funzione `oos_report(results)` che produce net PnL, Sharpe/Sortino, MaxDD, turnover, slippage proxy, hit rate, stabilità per sotto-periodi — comparabile tra strategie diverse  
  _File: `engine/reports.py`_

---

### Blocco F — Production readiness *(dipende da C ed E)*

> Funzionalità richieste per il passaggio Paper → Live (Phase 7-9 del BRD).

- [ ] **F1** — OMS state + riconciliazione: `OrderManagementSystem` con state machine per ordini (NEW→PENDING→FILLED/REJECTED/CANCELLED); riconciliazione fill vs posizioni vs PnL; accounting coerente con il backtest  
  _File: `engine/oms.py` (nuovo)_

- [ ] **F2** — Monitoring dashboard KPI: report live (o batch) con net PnL, drawdown, volatilità PnL, turnover, esposizioni gross/net, slippage medio, reject rate  
  _File: `engine/monitoring.py` (nuovo)_

- [ ] **F3** — Alerting: `AlertEngine` con regole configurabili per data staleness (feed fermo), outlier OHLCV, drift feature, performance decay; output su log + webhook configurabile  
  _File: `engine/alerting.py` (nuovo)_

- [ ] **F4** — Kill switch + runbook: `KillSwitch(conditions)` che blocca nuove operazioni e/o chiude posizioni al verificarsi di condizioni (drawdown max, slippage anomalo, staleness); `docs/runbook.md` con procedura triage/rollback  
  _File: `engine/kill_switch.py` (nuovo), `docs/runbook.md` (nuovo)_

- [ ] **F5** — Decision card per trade: ogni trade deve loggare `{t0, features_at_t0, signal_score, side, size, constraints_applied, veto_reason, model_version, data_version}`; `DecisionLog.export()` produce CSV/parquet auditabile  
  _File: `engine/decision_log.py` (nuovo), integrazione in `engine/engine.py`_

---

## 5. Dependency map riassuntiva

```
Blocco A  ──────────────────────────────┐
(debito critico)                        │
          │                             │
          ▼                             │
Blocco B  ──────────────────────────────┤
(qualità ing.)                          │
          │                             │
          ▼                             │
Blocco C  ──────────────────────────────┤
(struttura modulare)                    │
     │              │                   │
     ▼              ▼                   │
Blocco D          Blocco F ◄────────────┘
(fondamenta       (production)
 analitiche)       (dipende anche da E)
     │
     ▼
Blocco E
(validazione
 leakage-safe)
     │
     └──► Blocco F
```

---

## 6. Priorità consigliata di release

| Release | Blocchi inclusi | Obiettivo |
|---------|-----------------|-----------|
| **R0 — Stabilizzazione** | A completo | Backtest affidabile, nessun bug critico |
| **R1 — MVP Backtest** | A + B + C | Codice pulito, risk enforcement, report netto costi |
| **R2 — ML-Ready** | + D + E | Labeling, sample weights, validazione leakage-safe |
| **R3 — Paper Trading** | + F1 + F5 | OMS, decision log, riconciliazione |
| **R4 — Production** | + F2 + F3 + F4 | Monitoring, alerting, kill switch |
