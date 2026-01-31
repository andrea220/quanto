# TBD

- Rivedere anagrafica
- Rivedere funzioni di download da BBG
- Allineare download da IB


# Business Requirements (BRD) — AlgoTrading Library
**Ambito:** trading sistematico **intraday + medio termine**.
**Obiettivo:** creare una piattaforma/libreria che renda **riproducibile, validabile e governabile** il ciclo **Research → Backtest → Paper → Live**, applicando best practice econometria+AI (es. López de Prado: data structures, labeling, sample weights, purged CV/CPCV, bet sizing, backtest pitfalls).


---

## Obiettivi di business (Business Goals)
**BG-1 — Ridurre rischio di false discovery:** esecuzione di validazioni robuste e disciplina sperimentale.  \
**BG-2 — Accelerare il ciclo di sviluppo:** standardizzare processi e interfacce tra moduli per ridurre frizioni Research→Prod.  
**BG-3 — Scalabilità per più strategie/orizzonti:** supportare intraday e medio termine senza duplicare pipeline.  
**BG-4 — Governance e audit:** garantire tracciabilità di dati/modello/decisioni per debug e controllo operativo.  
**BG-5 — Data Quality:** la libreria deve essere robusta a missing/outlier e revisioni.

---

## Business Requirements

### BR-1 — Point-in-Time Correctness (No Data Leakage)
**Descrizione:** ogni output (feature, label, decisione) deve usare solo informazione disponibile fino al timestamp della decisione.  
**Razionale:** leakage inflaziona metriche e produce strategie non replicabili; Rischio strutturale. 

**Criteri di accettazione**
- Ogni dato/feature/label è associato a timestamp evento e timestamp disponibilità (PIT).
- Dataset builder impedisce join “future-looking” e segnala violazioni PIT.
- Stessa disciplina PIT tra training, backtest e live.

---

### BR-2 — Validazione leakage-safe (Purged CV + Embargo) + CPCV (stress-test)

#### Scopo
In finanza le osservazioni **non sono indipendenti** e spesso le **label coprono intervalli temporali** (es. “rendimento nei prossimi 30 minuti” o “trade chiuso quando tocca TP/SL”). Se faccio split train/test “standard”, rischio che parte dell’informazione del test finisca nel training (**data leakage**) ⇒ metriche gonfiate e strategie non replicabili.  
=> BR-2 impone una validazione che **evita contaminazioni** e riduce il rischio di “false discovery”.

---

#### Concetti chiave (definizioni operative)
- **Label span / t1**: per ogni osservazione *i* la label dipende da un intervallo \[t0(i), t1(i)\] (inizio evento → fine evento/uscita/orizzonte).
- **Overlap**: due osservazioni sono “contaminate” se i loro intervalli \[t0, t1\] si sovrappongono (condividono informazione sul futuro).
- **Purging**: quando un blocco è in **test**, rimuovere dal **training** tutte le osservazioni che hanno overlap con il test.
- **Embargo**: dopo la fine del test, escludere dal training un buffer temporale **h** (minuti/barre/giorni) per evitare dipendenze residue (anche senza overlap “perfetto”).
- **CPCV**: versione “stress” della CV che genera **molti percorsi out-of-sample** combinando blocchi temporali, applicando purging+embargo, e produce una **distribuzione** di metriche (non un singolo Sharpe).

---

#### Requirement statement (cosa deve fare il sistema)
1. **Split temporali obbligatori (no shuffle)** per training/validazione/test.
2. **Purged Cross-Validation default**: per ogni split, il training viene automaticamente “purged” rispetto al test usando \[t0,t1\].
3. **Embargo obbligatorio**: applicare un buffer h dopo ogni blocco test (h parametrico; intraday: minuti/barre; medio termine: giorni).
4. **HPO leakage-safe**: qualunque tuning/selection deve avvenire *dentro* lo stesso schema Purged+Embargo (mai CV standard).
5. **Modalità CPCV disponibile**: per valutazioni “top standard”, produrre metriche su più “paths” OOS e riportare quantili/worst-case.

---

#### Criteri di accettazione (verificabili)
- **AC1**: nessuna osservazione di training ha \[t0,t1\] che si sovrappone a qualunque \[t0,t1\] del test (purging corretto).
- **AC2**: nessuna osservazione di training inizia entro \[fine_test, fine_test + h\] (embargo applicato).
- **AC3**: ogni run salva in audit log: schema split, parametri purge/embargo, periodi, seed, versione dataset/modello.
- **AC4**: in modalità CPCV l’output include: numero di paths, distribuzione metriche (median, p10/p90, worst-case) + mapping path→periodi.

---

#### Note di applicazione (intraday vs medio termine)
- **Intraday**: h espresso in **minuti o numero di barre/eventi**; overlap spesso intenso ⇒ purging fondamentale.
- **Medio termine**: h espresso in **giorni**; overlap meno denso ma presente (label a 5–20 giorni).

---

### BR-3 — Disciplina sperimentale: “Backtest non è teoria”
**Descrizione:** il backtest è strumento di **falsificazione** e non sostituisce una teoria/causalità; serve a scartare modelli, non a “tunarli” ex-post. 

**Razionale:** evitare data snooping e false discovery quando si iterano molte ipotesi.

**Criteri di accettazione**
- Ogni strategia ha una **Strategy Specification** prima di backtest (ipotesi, universo, orizzonte, labeling, sizing, cost model, limiti).
- Tracciamento del numero di esperimenti/varianti per strategia (per controllare selection bias).
- Promozione a paper/live solo se passa gate predefiniti (stabilità su split, sensibilità costi, robustezza per sottoperiodi).

---

### BR-4 — Rappresentazione del dato coerente con l’orizzonte (Event-driven bars)
**Descrizione:** per intraday la libreria deve supportare data structures oltre alle time bars (volume/dollar/imbalance) per catturare “informazione” in modo più stabile. 
**Razionale:** la scelta della barra è parte del modello di mercato; influenza feature, labeling e validazione.

**Criteri di accettazione**
- Ogni strategia dichiara su quale struttura dati opera (es. dollar bars).
- Research/backtest/live usano la stessa struttura eventi (no mismatch).

---
### BR-5 — Problem Design prima del modello (Labeling + Sample Weights)

**Descrizione**  
Definire **target/labeling** e **pesi campione** (sample weights: concurrency/uniqueness; e class weights per imbalance) è un requisito funzionale primario, non opzionale. Il modello ML viene *dopo*: in finanza gran parte dell’edge dipende da **come** definisci evento, obiettivo e unità statistica.

**Razionale**  
Dati finanziari = non-IID, dipendenze temporali, eventi sovrapposti, classi sbilanciate. Se labeling e pesi non sono progettati bene, ottieni segnali apparentemente ottimi in backtest ma instabili live.

---

#### Requisiti funzionali

**BR-5.1 — Labeling versionabile e riproducibile**  
Il sistema deve supportare schemi di labeling espliciti e versionati, con output coerente con l’orizzonte (intraday o medio termine).  
Schemi minimi:
- **Fixed-horizon labeling**: label su orizzonte fisso (es. +30m, +1d, +5d).
- **Event-based labeling con uscita guidata da segnale (NO triple-barrier)**: la fine evento `t1` è determinata dalla regola di uscita del tuo fattore/segnale (es. crossing, inversione, regime filter, ecc.).
  - `t0`: ingresso/decision point
  - `t1`: primo timestamp futuro che soddisfa la condizione di uscita del segnale
  - **fallback**: `max_holding` e/o chiusura sessione (vertical cap operativo)
  - `exit_reason`: motivo uscita (signal_exit, timeout, session_close, ecc.)
- **(Opzionale) Barrier-based / triple-barrier**: disponibile solo se richiesto, non obbligatorio.

**BR-5.2 — Dataset “event-aware” (t0/t1 obbligatori)**  
Ogni osservazione deve rappresentare un evento/trade con un intervallo temporale associato:
- `t0` (inizio evento) e `t1` (fine evento/realizzazione label) **sempre presenti**
- metadata dell’evento (horizon, exit_reason, holding_time, ecc.) per audit e debug

**BR-5.3 — Sample weights per non-IID (concurrency/uniqueness)**  
Il sistema deve calcolare `sample_weight` per compensare:
- **concurrency** (quanti eventi sono attivi nello stesso tempo),
- **uniqueness** (quota di informazione non condivisa da un evento sul suo intervallo [t0,t1]).  
Obiettivo: evitare che periodi ad alta attività/overlap dominino il training.

**BR-5.4 — Class weights per imbalance (separati dai sample weights)**  
Il sistema deve supportare `class_weight` per gestire classi rare (es. pochi trade “buoni” o pochi eventi direzionali), indipendentemente dai `sample_weight`.

**BR-5.5 — Meta-labeling**  
Supportare un setup in cui:
- un processo decide il **side** (long/short) dal fattore/segnale,
- un meta-modello decide **se entrare e/o quanto size**, mantenendo la stessa regola di uscita (signal-driven).

---

#### Criteri di accettazione

- **AC-5.1 Dataset esplicito**: ogni riga include almeno  
  `features_at_t0, y, t0, t1, sample_weight` + `label_metadata` (es. horizon/exit_reason).
- **AC-5.2 Pluggable labeling**: è possibile selezionare e versionare lo schema (fixed-horizon o signal-exit event-based) e rigenerare lo stesso dataset.
- **AC-5.3 PIT compliance**: le feature usate per calcolare entry/exit e label rispettano il point-in-time (nessun uso di dati futuri).
- **AC-5.4 Weights separation**: `sample_weight` (uniqueness) e `class_weight` (imbalance) sono entrambi supportati e tracciati nei log.
- **AC-5.5 Auditability**: ogni training/backtest logga: schema labeling, parametri (horizon/max_holding/regole exit), regole di pesatura, versioni dati/feature.

---

---

### BR-6 — Separazione responsabilità: Model ≠ Sizing ≠ Execution ≠ Risk
**Descrizione:** architettura modulare con responsabilità disaccoppiate:
- **Model:** stima (score/probabilità + incertezza)
- **Sizing/Portfolio:** trasforma in posizioni sotto vincoli
- **Execution:** implementa ordini con frizioni
- **Risk:** limiti e kill switch indipendenti dal modello  
**Razionale:** riduce fragilità e permette sostituzioni locali senza riscrivere l’intera pipeline.

**Criteri di accettazione**
- Un modello può essere sostituito lasciando invariati sizing/risk/execution (e viceversa).
- Risk ed execution possono bloccare/vetare segnali (safety > alpha).

---

### BR-7 — Performance netta costi come metrica primaria
**Descrizione:** backtest e report devono mostrare sempre risultati **netti** (commissioni, slippage/impact) e indicatori operativi (turnover, shortfall).  
**Razionale:** specialmente intraday, la strategia è inseparabile dalle frizioni.

**Criteri di accettazione**
- Ogni report include gross PnL, costs, net PnL e turnover.
- Sensitivity test su costi/slippage disponibile come gate di robustezza.

---

### BR-8 — Osservabilità e Audit Trail end-to-end
**Descrizione:** ogni decisione deve essere ricostruibile: quali dati/feature, quale modello/versione, quale sizing, quali vincoli, quali ordini, quali fill, quale PnL.  
**Razionale:** senza auditability non esiste produzione affidabile; serve per debugging, drift, incident management.

**Criteri di accettazione**
- Ogni trade/decisione ha un “why/what/when/with-which-version”.
- Monitoring minimo: data staleness, slippage anomalo, drift di feature e performance decay.
- Kill switch con condizioni esplicite e testate.

---

## 7) Requisiti funzionali (Use-case oriented)

### FR-1 — Gestione ciclo di vita Strategia (Research → Backtest → Paper → Live)
- **FR-1.1** L’utente definisce una Strategy Specification (template standard).
- **FR-1.2** La strategia passa “gates” prima della promozione di fase:
  - Research → Backtest (specifica completa)
  - Backtest → Paper (robustezza su split, cost sensitivity)
  - Paper → Live (KPI operativi stabili, kill switch validato)

### FR-2 — Dataset “PIT-safe” per training/validazione
- **FR-2.1** Creazione dataset con regole PIT e availability lag.
- **FR-2.2** Supporto a labeling e sample weights (concurrency/uniqueness). 

### FR-3 — Validazione e selezione modelli (vale anche per regole “if x>y then buy”, non solo AI)

- **FR-3.1 — CV leakage-safe (purged/embargo)**  
  Quando valuti *o scegli* una strategia/regola (anche deterministica) devi usare split **out-of-sample** corretti per serie temporali.  
  Se i trade/eventi hanno un intervallo \[t0,t1\] (holding fisso o uscita da segnale), allora gli eventi possono **sovrapporsi**:  
  - **purging** = rimuovere dal training gli eventi che si sovrappongono temporalmente al test  
  - **embargo** = escludere un buffer dopo il test per evitare contaminazioni residue  
  Obiettivo: evitare performance gonfiate da leakage.

- **FR-3.2 — CPCV (stress-test su molti split)**  
  Oltre a un singolo split/walk-forward, CPCV genera **molti percorsi out-of-sample** combinando blocchi temporali (sempre con purging+embargo).  
  Serve quando confronti più varianti (soglie, finestre, filtri, exit rules) e vuoi stimare **robustezza**: non “un numero”, ma una **distribuzione** di risultati (median, quantili, worst-case).

- **FR-3.3 — Reporting standard di metriche OOS (ML + trading)**  
  Ogni strategia produce report **out-of-sample** standardizzati.  
  - Se c’è ML: metriche ML (es. logloss/AUC/calibration) + metriche trading.  
  - Se è una regola tipo `x>y`: metriche trading **netto costi** (net PnL, Sharpe/Sortino, maxDD, turnover, cost/slippage, hit rate, stabilità per sottoperiodi).  
  Obiettivo: confrontare strategie in modo coerente e “production-ready”.


### FR-4 — Trasformazione segnale → posizioni (Bet Sizing / Portfolio)

- **FR-4.1 — Il segnale produce uno “score” (non necessariamente AI)**  
  L’output può essere:
  - una **probabilità/score** (se c’è ML), oppure
  - un **segnale deterministico** (es. `x>y → +1`, altrimenti `0` o `-1`).  
  In ogni caso, questo output è **solo un’intenzione** di trading, non una posizione finale.

- **FR-4.2 — Layer di sizing indipendente (quanto comprare/vendere)**  
  Un modulo separato trasforma lo score in **dimensione posizione** (notional/numero contratti), tenendo conto di:
  - forza del segnale (es. score più alto → size maggiore),
  - rischio/volatilità (es. vol targeting),
  - **concorrenza di bet**: se più segnali/trade sono attivi insieme, il sizing deve evitare somma “esplosiva” delle esposizioni.  
  Obiettivo: rendere la size **controllata e coerente** anche quando i segnali si sovrappongono.

- **FR-4.3 — Vincoli di portafoglio separati dal segnale**  
  Dopo il sizing, un ulteriore layer applica vincoli e regole portfolio, ad esempio:
  - limiti **gross/net exposure**, leverage massimo,
  - concentrazione per strumento/settore,
  - limiti di turnover e liquidità,
  - budgeting del rischio e stop su drawdown/esposizioni.  
  Obiettivo: il segnale può essere “buono”, ma il portafoglio deve restare **entro guard-rail** operativi e di rischio.


### FR-5 — Execution e gestione ordini
- **FR-5.1** Strategie di esecuzione tipiche (TWAP/VWAP/limit logic).
- **FR-5.2** Simulazione realistica nel backtest (latenza, slippage parametrico).
- **FR-5.3** Riconciliazione ordini/fill/posizioni e calcolo PnL.

### FR-6 — Monitoring e gestione incidenti (operatività “production-grade”)

- **FR-6.1 — Dashboard KPI (stato salute strategia + esecuzione)**  
  Il sistema deve esporre una dashboard con KPI live e storici per capire rapidamente “come sta andando” e “perché”, includendo almeno:
  - **PnL netto** (dopo costi), **drawdown**, volatilità PnL
  - **turnover** ed esposizioni (gross/net, concentrazione)
  - **slippage / implementation shortfall** e cost breakdown
  - **reject rate** e anomalie ordini/fill  
  Obiettivo: osservabilità immediata di performance, rischio ed execution.

- **FR-6.2 — Alerting su degrado dati, drift e performance decay**  
  Il sistema deve generare alert automatici quando:
  - **dati degradano**: staleness (feed fermo), gap, outlier, ritardi, mismatch calendari
  - **drift**: cambiamento distribuzione feature o cambiamento relazione feature→target
  - **performance decay**: peggioramento persistente di metriche OOS/live vs baseline  
  Obiettivo: intercettare rapidamente problemi di data quality o cambi di regime prima che diventino drawdown.

- **FR-6.3 — Kill switch operativo + runbook (incident response)**  
  Il sistema deve supportare un **kill switch** che blocca nuove operazioni e/o chiude posizioni al verificarsi di condizioni predefinite (es. drawdown, slippage anomalo, data staleness, errori OMS).  
  Deve esistere un **runbook**: procedura standard di cosa fare quando scatta un alert o il kill switch (triage, rollback, escalation, ripartenza).  
  Obiettivo: ridurre rischio operativo e garantire ripristino controllato.

### FR-7 — Trasparenza e spiegabilità operativa (feature importance / attribution)

- **FR-7.1 — Spiegabilità “operativa” a livello di pipeline**  
  Il sistema deve permettere di capire *perché* una decisione di trading è stata presa, almeno a livello di:
  - quali feature/fattori hanno guidato il segnale,
  - quali passaggi della pipeline hanno trasformato segnale → posizione → ordini,
  - quali vincoli/risk rules hanno modificato o bloccato l’azione.

- **FR-7.2 — Feature importance / attribution (quando applicabile)**  
  Per strategie ML o score-based, il sistema deve produrre misure di importanza/attribuzione (es. feature importance globale e/o spiegazioni locali per singola decisione), con logging associato a:
  - versione modello,
  - versione feature set,
  - timestamp decisione.  
  Per strategie deterministiche (es. `x>y`), l’attribuzione equivale a tracciare esplicitamente:
  - valore di `x`, valore di `y`, soglia/condizione e stato dei filtri che hanno attivato l’ordine.

- **FR-7.3 — Audit trail “decision rationale”**  
  Ogni trade deve poter essere ricostruito con un “decision card” minimale:
  - input principali (feature/indicatori chiave),
  - output del segnale (score/side),
  - sizing risultante,
  - vincoli applicati (cap/limit/stop),
  - motivi di eventuale veto (risk/execution/data quality).

**Obiettivo:** ridurre rischio di “black box operativo”, accelerare debugging e rendere confrontabili strategie diverse con un linguaggio comune.


---

## Roadmap di sviluppo


### Phase 0 — Setup & Allineamento Dati (Foundation)
**Obiettivo:** rendere coerenti anagrafica e data ingestion (BBG + IB) per evitare incoerenze a valle.

- [x] Rivedere anagrafica strumenti (mapping univoco: ticker/vendor/IB, currency, exchange, contract specs)
- [ ] Rivedere funzioni di download da BBG (schema, timezone, calendario, corporate actions, revision policy)
- [ ] Allineare download da IB (stesso schema di output, stessi identificativi, stessa granularità)
- [ ] Definire “Data Contract” minimo:
  - timestamp policy (UTC vs local), sessioni, timezone
  - campi obbligatori (OHLCV, bid/ask se disponibile, corporate actions)
  - regole di normalizzazione e storage (raw vs cleaned vs curated)
- **Gate P0:** dataset BBG e IB risultano confrontabili (stesso schema + id) e riproducibili.

---

### Phase 1 — PIT by design (BR-1) + Dataset Builder v1
**Obiettivo:** costruire un “dataset builder” che non permetta leakage e renda ogni dataset auditabile.

- [ ] Implementare regole PIT (timestamp evento + timestamp disponibilità)
- [ ] Availability lag policy (fondamentali/azioni corporate/qualunque fonte con ritardo)
- [ ] As-of join standard (feature/label)
- [ ] PIT checker: fallisce se trova join future-looking
- **Gate P1:** puoi rigenerare lo stesso dataset e dimostrare PIT compliance con log/audit.

---

### Phase 2 — Data Structures: Bars Engine (BR-4)
**Obiettivo:** definire le strutture eventi coerenti intraday e medio termine.

- [ ] Time bars baseline (per debug e confronto)
- [ ] Volume/Dollar bars (intraday robust)
- [ ] Policy sessioni (open/close/auction, rollover se futures)
- [ ] Stesso input → stessa barra (determinismo)
- **Gate P2:** research/backtest/live possono usare la stessa “bar definition” senza mismatch.

---

### Phase 3 — Labeling & Weights (BR-5) con Exit Signal-Driven
**Obiettivo:** definire formalmente *che cosa* predici e come pesi i campioni.

- [ ] Fixed-horizon labeling (baseline)
- [ ] Event-based labeling con uscita da segnale (t0→t1) + `exit_reason` + `max_holding` fallback
- [ ] Calcolo `sample_weight` (concurrency/uniqueness) basato su intervalli [t0,t1]
- [ ] Supporto `class_weight` per imbalance (se necessario)
- **Gate P3:** ogni osservazione ha `t0,t1,y,sample_weight` + metadata e puoi ricostruire i trade/eventi.

---

### Phase 4 — Validazione leakage-safe (FR-3 / BR-2) + Reporting OOS v1
**Obiettivo:** rendere la validazione “istituzionale” anche per regole semplici (x>y).

- [ ] Walk-forward OOS standard (baseline)
- [ ] Purged CV + embargo (richiede t1)
- [ ] CPCV “stress mode” (quando confronti varianti o fai selezione)
- [ ] Report OOS standard (net PnL, DD, turnover, slippage proxy, stabilità per sottoperiodi)
- **Gate P4:** qualunque confronto/selection passa solo da split leakage-safe e produce report OOS replicabili.

---

### Phase 5 — Backtest Engine cost-aware (BR-7) + Execution Simulator v1
**Obiettivo:** backtest realistico (non-HFT) con costi e frizioni.

- [ ] Commissioni + spread/slippage parametrico (minimo)
- [ ] Latenza decision→order→fill (semplice ma presente)
- [ ] Market/limit logic essenziale
- [ ] Output: gross/net PnL, costs breakdown, turnover, exposures
- **Gate P5:** i risultati “net” sono stabili e spiegabili; niente backtest “gratis” senza frizioni.

---

### Phase 6 — Signal → Sizing → Portfolio Constraints (FR-4) + Risk Guardrails (BR-6)
**Obiettivo:** separare chiaramente segnale, sizing e vincoli.

- [ ] Signal interface (score/side deterministico o probabilistico)
- [ ] Sizing module indipendente (vol targeting / scaling / caps; gestione concurrency)
- [ ] Portfolio constraints (gross/net, leverage, concentrazione, turnover)
- [ ] Pre-trade risk checks (limiti base)
- **Gate P6:** stesso segnale con sizing diverso dà portafogli diversi in modo controllato; vincoli sempre rispettati.

---

### Phase 7 — OMS/Execution Non-HFT (FR-5) + Riconciliazione
**Obiettivo:** chiudere il loop operativo verso broker.

- [ ] OMS state (orders/fills/positions)
- [ ] Execution tactics non-HFT: TWAP/VWAP/limit rules
- [ ] Riconciliazione e PnL accounting coerente con backtest
- **Gate P7:** paper-trading end-to-end con riconciliazione e PnL affidabile.

---

### Phase 8 — Monitoring, Alerting, Kill Switch, Runbook (FR-6 / BR-8)
**Obiettivo:** passare da “research tool” a “production system”.

- [ ] Dashboard KPI live (net PnL, DD, turnover, slippage, reject rate)
- [ ] Alerting: data staleness, outlier, drift, performance decay
- [ ] Kill switch + runbook (triage, rollback, restart)
- **Gate P8:** sistema paper-trading operabile con incident response (nessun “black box live”).

---

### Phase 9 — Trasparenza / Explainability operativa (FR-7)
**Obiettivo:** decisioni ricostruibili e debugging rapido.

- [ ] Decision card per trade (input chiave, segnale, sizing, vincoli, eventuali veto)
- [ ] Attribution:
  - regole deterministiche: log x, y, soglie, filtri
  - ML: feature importance / spiegazioni locali (quando applicabile)
- **Gate P9:** ogni trade è auditabile “why/what/when/with-which-version”.

---

## Strategia consigliata di rilascio (incrementale)
- **Release R1 (MVP Backtest)**: P0 → P5
- **Release R2 (Portfolio + Risk)**: P6
- **Release R3 (Paper Trading)**: P7 → P9
- **Release R4 (Live)**: dopo paper trading stabile + gate definiti in FR-1

---
