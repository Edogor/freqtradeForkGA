# Plan für eine sichere Next-Wave-Automation

Status: Guarded-Auto-Suche ist auf `codex/ga-v2-p0-foundation` implementiert und als
Search-only-/Shadow-Pfad freigegeben. Automatische Strategie- oder Trading-Promotion bleibt
gesperrt. Bedienung: [AUTOMATION_RUNBOOK_V2.md](AUTOMATION_RUNBOOK_V2.md).

## Entscheidung

Die Next-Wave-Automation soll **nicht direkt auf `ga_auto_queue_v2.sh`, `registry.json`, Log-Regexen
oder den aktuellen Fitnesswert gesetzt werden**. Diese Quellen sind nicht transaktional,
reproduzierbar oder semantisch einheitlich genug.

Zuerst werden Runner, Resultatvertrag und Evaluation repariert. Danach wird die Automation in drei
Sicherheitsstufen eingeführt:

1. Shadow: analysieren und Vorschläge schreiben, nichts queueen.
2. Approval: validierte Vorschläge nach menschlicher Freigabe queueen.
3. Guarded Auto: nur Policies und Run-Klassen automatisch starten, die ihre Integrations- und
   Shadow-Gates nachweislich bestanden haben.

Implementierungsstand 24.07.2026: Stufe 3 ist für genau einen eng begrenzten Pfad umgesetzt:
`automation_island_v2` mit Generic Island, festem Train-/Validation-Pair-Split und vollständiger
Pair-Validation. Ein deterministischer Bootstrap erzeugt die Root-Control-Wave. Der langlebige
Controller führt Queue, Reconciliation, Analysis, Pareto-Selection, Control/Replication/Explore,
hashgebundene Auto-Approval und atomisches Child-Handoff aus. Eine gesunde Wave ohne eligible
Kandidaten darf als reine Scratch-Control fortgesetzt werden; technische/semantische Fehler
blockieren grundsätzlich. Für transiente technische Totalausfälle existiert eine auf drei
aufeinanderfolgende Scratch-Control-Waves begrenzte Recovery; danach wird fail-closed gestoppt.
Kill-Switch, GA-/Executor-Laufzeit, Wave-, Attempt-, Parallelitäts-, RAM-, Delta- und Disk-Budgets
werden vor weiteren Starts beziehungsweise Child-Proposals geprüft. WF, Holdout, MC, Surrogate, SIS, LLM,
NSGA-II, Classic Island, Regime und automatische Trading-Promotion sind nicht Teil dieser Freigabe.

## Zielablauf

```text
Parent-Wave vollständig und reconciled
  -> Resultate gegen Schema, Hashes und Splitrollen prüfen
  -> alle Kandidaten auf gemeinsamem Replay-Panel vergleichen
  -> Promotion-Gates anwenden
  -> Control/Replicate/Exploit/Explore/Validation-Arme planen
  -> resolved Configs fail-closed validieren
  -> deterministischen Proposal-Report erzeugen
  -> Freigabe oder Policy-Autorisierung
  -> Attempts atomar claimen und ressourcengesteuert starten
  -> Heartbeats, Resultate und Artefakte pro Attempt schreiben
  -> terminale Zustände reconciliieren
  -> nächste Analyse erst nach vollständigem Parent-Snapshot
```

## 1. Ein kanonischer Runner

Manueller Start, Queue, Retry und Resume benutzen denselben internen `execute_attempt()`-Pfad.

Verantwortlichkeiten:

- Der Orchestrator erzeugt Wave-, Experiment- und Attempt-ID.
- Der Child registriert dieselbe ID nicht erneut.
- Die resolved Config wird vor Start unveränderlich gespeichert.
- Jeder Attempt erhält ein eigenes Verzeichnis für Log, Tracker, Diagnostics, Checkpoints, HOF,
  Resultat und Manifest.
- Erfolg bedeutet: Exitcode 0 **und** valides terminales `result.json` **und** vollständiges
  Artifact-Manifest.
- `EVOLUTION COMPLETE` in einem Log ist kein Erfolgsnachweis.

Vorgeschlagene Artefaktstruktur:

```text
genetic_algorithm/data/attempts/<attempt_id>/
  resolved_config.yaml
  manifest.json
  code_manifest.json
  data_manifest.json
  split_manifest.json
  worker_spec.json
  final_test_usage.json
  promotion_policy.json
  candidates/<candidate_id>/
    frozen_candidate.json
    scenarios/<scenario_id>.json
    candidate.json
    decision.json
  events.jsonl
  heartbeat.json
  runtime/
    attempt.log
  checkpoints/
  hall_of_fame/
  diagnostics/
  result.json
  result.json.sha256
  decision_inputs.json
```

Implementierungsstand 21.07.2026: Der immutable V2-Artifact-Store, Shadow-Attempt-Recorder und
Shadow-Replay-Executor bilden diese Resultatgrenze bereits ab. Writes sind atomar, Wiederholung ist
nur byte-identisch erlaubt, und Readback verifiziert alle deklarierten SHA-256-Hashes. Der Executor
isoliert Pair/Timerange, skaliert das unterstützte Fee-/Slippage-Modell und blockiert bei
abweichender Perioden-Coverage.

Der aktuelle Foundation-Slice bindet außerdem den exakten Git-/Dirty-Stand und rohe OHLCV-Dateien
mit Candle-Coverage in immutable Manifeste. Legacy-`Individual`/HOF-Objekte werden ohne zufällige
Reparatur in exakten Strategiecode exportiert; Codegen-Mutation oder Nichtdeterminismus blockiert.
Frozen Code und externe Ausführungsparameter werden pro Candidate persistiert. Noch offen sind
die Umschaltung der Legacy-CLI/Scheduler auf den neuen Attempt-State, vollständige Isolation der
übrigen Legacy-Artefakte sowie Wave-/Decision-State. Die Final-Test-
Einmalnutzung wird inzwischen durch ein separates SQLite/WAL-Ledger erzwungen: überlappende
Börse-x-Pair-x-Zeiträume können nicht durch neue Scenario-Namen, Timeframe-Resampling,
Kostenfaktoren oder
geänderte Dateien erneut als blind deklariert werden. Vor dem ersten möglichen Final-Backtest wird
das gesamte reservierte Panel dauerhaft als exponiert markiert.

Der neue `AttemptExecutorV2` bildet inzwischen die gemeinsame Prozessgrenze für künftige CLI-,
Queue- und Controller-Adapter: exakte Argumentliste und Arbeitsverzeichnis werden gehasht, der
attempt-isolierte Runtime-Logpfad wird vor Spawn persistiert, und der Child läuft ohne Shell-
Auswertung. Heartbeats verlängern die Lease. Exit 0 ohne verifiziertes `result.json` ergibt
`INVALID_RESULT`; Erfolgsmarker im Log werden nicht ausgewertet. Der Legacy-Scheduler ist bewusst
noch nicht umgestellt, da der von ihm gestartete Runner bisher kein kanonisches V2-Resultat erzeugt.

Für Shadow-Replays ist diese Child-Lücke jetzt geschlossen: `ReplayWorkerSpecV2` enthält eine
geschlossene SHA-256-Matrix aus Manifest, resolved Config, Policy, Code-/Daten-/Split-Manifest und
Frozen Candidates. Ihr eigener Datei-Hash wird als Argument Teil des gehashten Commands. Der
separate `.venv`-Child lädt alle Eingaben erneut, verbietet unbekannte Pre-Execution-Artefakte,
prüft Pfade und Hashes und startet erst dann den `ShadowReplayRunnerV2`. Soweit das Manifest noch
vertrauenswürdig ist, werden auch Input- oder Ausführungsfehler als kanonisches
`INVALID_RESULT` abgeschlossen. Der komplette Pfad von SQLite-Claim bis terminalem Readback ist in
einem echten Subprozess getestet.

Für Standard-Evolution schließt `EvolutionWorkerSpecV2` denselben Pfad. Alle operativen Engine-
Outputs werden attempt-lokal abgeleitet; der Child registriert sich nicht in der JSON-Registry.
Optionaler Warm-Start kommt ausschließlich aus einem reproduzierbaren Frozen Evolution Seed und
ist fail-closed. Legacy-Fitness short-listet nur; Erfolg entsteht erst durch anschließendes V2-
Replay. Der Scheduler→Executor→Child→Result-Pfad ist ebenfalls in einem echten Subprozess getestet.
Der operative Standardpfad ist nun umgeschaltet: CLI, `run_ga.py`, Web-RunManager und die beiden
historischen Shell-Queue-Namen materialisieren bzw. starten ausschließlich SQLite/WAL-V2-Attempts.
Ein direkter Start claimt per Attempt-ID statt den beliebigen Queue-Head. Nichtstandardmäßige
Engines und veränderliche Laufzeitoperationen (Island/Generic Island, Resume, Pause, Injection,
Visualisierung) besitzen noch keinen immutable Worker-Vertrag und werden deshalb ohne
Legacy-Fallback abgewiesen.

Auch die Queue-Ausführung ist für diesen Replay-Pfad geschlossen: `WorkerBindingV2` wird bereits im
`DRAFT`-State persistiert und bindet Spec-Hash, exakte Argumentliste und Arbeitsverzeichnis vor
`VALIDATED`. `ShadowSchedulerV2` verwendet nur solche gebundenen Attempts, claimt sie atomar,
startet den kanonischen Executor in begrenzten Slots und reconciled stale Prozesse vor weiteren
Claims. Alle `CLAIMED`-/`RUNNING`-Replay-Attempts belegen einen Slot, auch wenn sie von einer
vorherigen Scheduler-Instanz stammen. Ein anderer Command nach Queueing wird vor Spawn blockiert.
`EvolutionSchedulerV2` verwendet denselben Vertrag für `STANDARD_EVOLUTION`.
`AttemptSchedulerV2` führt beide Workerarten in einem operativen Slotpool zusammen; der
CLI-Daemon zählt fremde/übernommene aktive Leases mit und reconciled abgelaufene Prozesse vor
neuen Claims. Die Shell-Shims besitzen keine eigene `pgrep`-/`done`-Semantik mehr.
Ressourcenbudgets über reine Slotanzahl hinaus bleiben offen.

Der Attempt-Name ist unveränderlich und eindeutig. `experiment_name` allein darf keine Wiederholung
überschreiben.

## 2. Transaktionaler Zustand

`registry.json` wird für den autonomen Pfad durch SQLite im WAL-Modus ersetzt. JSON kann als Export
erhalten bleiben, ist aber nicht Source of Truth.

### Entitäten

- `Wave`: Parent, Policyversion, Budget, Status, Resultat-Snapshot-Hash
- `ExperimentSpec`: beabsichtigter Arm, Factor-Delta, Seeds, Config-Hash
- `Attempt`: konkrete Ausführung, Lease, PID/Starttoken, Retrynummer, Status
- `Artifact`: Typ, Pfad, Hash, Schema, Vollständigkeit
- `Decision`: Eingaben, Gates, Score, Gründe, erzeugte Specs, Approval

### Attempt-Zustände

```text
DRAFT -> VALIDATED -> QUEUED -> CLAIMED -> RUNNING
                                      -> SUCCEEDED
                                      -> FAILED
                                      -> INTERRUPTED
                                      -> INVALID_RESULT
```

Nur erlaubte Übergänge werden atomar geschrieben. Claim erfolgt als Transaktion; zwei Scheduler
können nie denselben Attempt starten. Ein Lease/Heartbeat und ein Prozess-Starttoken verhindern,
dass PID-Wiederverwendung als laufender Versuch gilt.

Implementierungsstand: `AttemptStateStoreV2` bildet diese Attempt-Zustände in SQLite/WAL ab,
persistiert immutable Transition-/Heartbeat-Events und verwendet Claim-Fencing-Token, Lease,
PID und Prozess-Starttoken. `SUCCEEDED` ist nur nach verifiziertem `result.json`-Readback möglich.
60 parallele Queue-Claims und 60 parallele Heartbeats wurden ohne verlorene Versionen getestet.
CLI, operative Queue, Monitor, Data-Lifecycle und Web-Runliste verwenden diesen State über den
read-only `ExperimentCatalogV2`. Alte JSON-Registry-Snapshots können idempotent als immutable
Historie importiert werden, werden aber nie zu ausführbaren Attempts; ein atomarer JSON-Export
erhält State-Version und Provenienz. Die autorisierte Reconciliation für `RUNNING` prüft PID plus
Linux-Boot-ID/Prozess-Startticks. Bei identisch lebendem oder nicht sicher prüfbarem Prozess erfolgt
keine Zustandsänderung. Erst ein nachgewiesen verschwundener beziehungsweise wiederverwendeter PID
erlaubt den Übergang: valides Terminalresultat wird übernommen, fehlendes Resultat wird
`INTERRUPTED`, beschädigte Evidenz `INVALID_RESULT`. Ein abgelaufener `CLAIMED`-Eintrag ohne
persistierte Prozessidentität kann nur dann automatisch geschlossen werden, wenn er den neuen
`GUARDED_SPAWN_RECOVERY_V1` trägt. Dessen Parent-Death-Guard startet das echte Child erst nach dem
committed `RUNNING`-Übergang und einem passenden atomaren Release-Receipt. Vorheriger
Controller-Tod, Guard-Timeout oder ein nachweislich verschwundener Guard sind daher sicher
`INTERRUPTED`; ein lebender/unprüfbarer Guard und migrierte Alt-Claims ohne Vertrag bleiben
unangetastet.

### Wave-Zustände

```text
DRAFT -> COLLECTING -> RECONCILED -> ANALYZED -> PROPOSED -> APPROVED -> QUEUED
                  \-> BLOCKED                  \-> REJECTED
```

`WaveStateStoreV2` persistiert Wave-Budget/-Version, immutable `ExperimentSpec`-Arme, die exakte
Attempt-Erwartungsmenge, Decisions und Events in derselben SQLite-Transaktionsdomäne wie die
Attempts. `COLLECTING` startet nur, wenn genau die spezifizierten Attempts mit passendem
Manifest-Hash und jeweils genau einem erwarteten Seed gequeued sind.

`RECONCILED` verlangt, dass **alle** erwarteten Attempts terminal sind und jeder Attempt entweder
ein vollständig hash-verifiziertes `result.json` oder eine explizite immutable
`ATTEMPT_ABORT`-Decision besitzt. Ein beschädigtes vorhandenes Resultat darf nicht durch Abort
ersetzt werden. Die Reconciliation friert die kanonisch sortierte Evidenz atomar als
`WaveResultSnapshotV2` ein; parallele Wiederholungen sind idempotent. Analysis, Proposal und
Approval bilden anschließend eine Hashkette über Snapshot und Vorgänger-Decision.

Der Analyzer-Slice ist inzwischen ebenfalls implementiert: `WaveAnalyzerV2` lädt ausschließlich
den eingefrorenen Parent-Snapshot und die immutable Experiment-Spezifikationen. Jedes referenzierte
Resultat wird inklusive Datei-, Artifact-, Manifest-, Config-, Policy-, Experiment- und
Seed-Provenienz erneut verifiziert. Technische Experiment-Gesundheit, valide Messung und
Promotionsfähigkeit sind getrennt: Ein Kandidat mit gültigen Messdaten, aber gescheitertem Gate
bleibt vollständig in Return-/Risiko-/Trade-Aggregaten enthalten, ist jedoch nicht eligible.

Die Aggregation erfolgt pro Experiment und ausführbarem Phänotyphash über eindeutige Seeds. Sie
liefert unter anderem Worst-/Median-Return-LCB, DD-/ES-UCB, Expectancy-LCB, Profit Factor, Winrate,
effektive Stichprobe und explizit bezeichnete Szenario-Trade-Summe. Control-Fehlen, Abort- und
Non-Success-Quote, zu geringe Seed-Coverage sowie Gatefehler sind nicht kompensierbare Gründe. Die
versionierte Analyzer-Policy und das komplette Ergebnis werden gehasht in einer deterministischen
`ANALYSIS`-Decision gespeichert. Die aktuellen Defaultgrenzen sind konservative Startwerte und
noch kein empirisch kalibriertes Optimum. Pareto-Auswahl, Planner, produktives Queue-Handoff und
Legacy-Migration bleiben offen; der Analyzer startet weiterhin nichts.

## 3. Kanonischer Resultatvertrag

Der Planner konsumiert ausschließlich versionierte `result.json`-Dateien nach
[GA_EVALUATION_SPEC.md](GA_EVALUATION_SPEC.md). Pflichtfelder umfassen:

- Attempt-/Wave-/Parent-ID
- Config-, Code-, Daten-, Phänotyp- und Policy-Hash
- Seed, Budget und tatsächlich ausgeführter GA-Typ
- terminaler Status und strukturierter Fehlergrund
- vollständige Train-/Validation-/Final-Szenariomatrix
- kapitalgewichtete Equity-Metriken und Unsicherheit
- Gate-Ergebnisse mit maschinenlesbaren Gründen
- Candidate-Provenienz und echter Replay-Status

Alte HOF-Fitness, Registry-Nullwerte und aus Logs extrahierte Mischmetriken dürfen nur importierte
Rohhinweise sein. Vor einer Entscheidung werden Kandidaten auf demselben Replay-Panel neu bewertet.

## 4. Deterministischer Wave-Planner

Der Planner ist eine reine Funktion:

```text
plan = f(parent_result_snapshot, policy_version, search_space_version, budget)
```

Gleiche Eingaben erzeugen byte-identische Configs und dieselbe `wave_id`. Ein erneuter Aufruf startet
nichts doppelt.

Eine mögliche Modultrennung:

- `orchestration/result_contract.py`: Schema, Laden, Validierung
- `orchestration/wave_analyzer.py`: gemeinsames Replay und Kandidatenvergleich
- `orchestration/promotion.py`: harte Gates und Pareto-Rang
- `orchestration/wave_planner.py`: Arme, Seeds, Budget und Config-Deltas
- `orchestration/wave_controller.py`: State Machine, Approval und Queueing
- `config/policies/*.yaml`: versionierte Grenzen und erlaubte Mutationen

Die Namen sind Vorschläge; wichtiger ist die Trennung von Messung, Entscheidung und Ausführung.

Implementierungsstand: `candidate_selector_v2.py` berechnet deterministische Non-Dominated-Ränge
auf einem **identischen Vergleichspanel**. Maximiert werden Worst-Return-LCB und
Worst-Expectancy-LCB, minimiert werden DD-UCB und Daily-ES-UCB. Es gibt bewusst keinen
kompensierenden Gesamtscore. Kandidaten mit höherem Return und zugleich höherem Tail-Risiko bleiben
Pareto-Alternativen, statt dass Return das Risiko rechnerisch löscht. Die anschließende Auswahl
beginnt mit dem risikoärmeren Anker und ergänzt Kandidaten nach normalisierter Objective-Distanz;
ein Limit pro Experiment verhindert, dass ein Arm alle Plätze belegt. Tradezahl, Profit Factor und
Winrate bleiben Evidenz beziehungsweise Gates und werden nicht blind maximiert. Eine echte
Behavior-/Phänotyp-Cluster-Distanz jenseits eindeutiger Phänotyphashes ist noch offen.

`wave_planner_v2.py` bildet Analysis, Selection, immutable Parent-Experiment-Specs, verifizierte
Parent-Configs und eine versionierte Planner-Policy auf einen `ChildWavePlanV2` ab. Wave-,
Experiment- und Attempt-IDs, Configs und Hashes sind reproduzierbar. Alle Arme verwenden explizite
gepaarte Seeds; Control bleibt unverändert, Replication darf kein Factor-Delta besitzen und
Exploit/Explore ändern standardmäßig genau einen vorhandenen typkompatiblen Config-Pfad. Budget und
Per-Arm-Quellen werden vor Proposal geprüft. Ein blockierter Selection-Report erzeugt eine
`BLOCK`-Decision ohne ausführbare Experimente. Ein vollständiger Plan allein erzeugt noch kein
`PROPOSAL`: Erst eine erfolgreiche Materialisierung darf den Approval-Prozess öffnen.

Der Planner bleibt für sich Shadow/read-only. Der neue explizite `REPLAY_VALIDATION`-Modus kann
jedoch über `wave_materializer_v2.py` sicher ausgeführt werden: Unbekannte Config-Pfade und
Runtime-Inkompatibilitäten blockieren; Parent-Resultat und `frozen_candidate.json` werden erneut
gegen die eingefrorenen Hashes geprüft; Config, Promotion-Policy, Code-, Daten-, Split- und
Attempt-Manifest sowie Worker-Spec werden immutable geschrieben. Das Materialisierungs-Receipt
bindet `plan_hash`, sämtliche Manifest-/Worker-Hashes und die verifizierte Kandidatenquelle.
Proposal und Approval müssen exakt denselben Plan- und Materialisierungshash nennen.

Das anschließende Handoff schreibt Child-Wave, Experiment-Spezifikationen, Attempt-Erwartungen,
Attempt-State und Worker-Bindings gemeinsam mit dem Parent-Übergang `APPROVED -> QUEUED` in einer
SQLite-Transaktion. Direkte reine Statusmarkierung ist gesperrt. Ein Receipt-Tamper, Worker-Tamper,
Approval-Mismatch oder Datenbankkonflikt erzeugt daher keine partielle Queue.

Zusätzlich ist jetzt `STANDARD_EVOLUTION` unter dem erzwungenen `safe_v2`-Profil freigegeben.
`EvolutionWorkerSpecV2` bindet Config, Attempt-Seed, Workerzahl, Top-N, Manifestkette und optionale
Parent-Genome unveränderlich. Control verwendet keinen Parent-Seed; Replication, Exploit und Explore
benötigen ein hash-verifiziertes `evolution_seed.json`, dessen Genom unter Child-Config und aktuellem
Code exakt dasselbe Executable reproduziert. Legacy-/Replay-only-Kandidaten ohne dieses Artefakt
werden nicht still als Scratch-Run ausgeführt.

Die GA-Fitness dient in diesem Worker nur der Suche. Top-Genome werden zu Frozen Candidates
materialisiert und durch den kanonischen V2-Replay- und Gate-Pfad bewertet; erst dieses Resultat darf
`SUCCEEDED` werden. Evolutions-Waves enthalten absichtlich keine `FINAL_TEST`-Zellen. Blindes Final
wird erst in einer getrennten Replay-Validation nach der Seed-Aggregation verbraucht. Island,
Generic Island und NSGA-II bleiben fail-closed, bis sie eigene Ergebnis- und Isolationsverträge
besitzen.

## 5. Experimentdesign jeder Folge-Wave

Eine Wave darf nicht nur Gewinner kopieren. Sie braucht Kontroll- und Lernarme. Mindeststruktur:

- **Control:** unveränderte Baseline mit identischem Budget.
- **Replication:** gleiche Baseline/Topkandidat mit neuen, gepaarten Seeds.
- **Exploit:** eine klar benannte Änderung um einen robusten Kandidaten.
- **Explore:** begrenzte neue Suchraum-/Operatorhypothese.
- **Validation:** eingefrorene Kandidaten auf neuen Pair-/Zeit-/Kostenpanels, ohne weitere Evolution.

Der genaue Budgetanteil ist Policy-Sache. Nicht verhandelbare Regeln:

- gleiche Seeds und Backtestbudgets für verglichene Arme;
- pro Arm nur ein kausaler Faktor oder ein ausdrücklich geplantes Factorial;
- identischer gültiger Suchraum, sofern genau SIS/Operator/Policy getestet wird;
- mindestens ein unveränderter Control-Anker über aufeinanderfolgende Waves;
- vorher festgelegte Primary Metric, Kill-Gates und Analyse;
- keine Auswahl des „besten Seeds“ ohne Bericht aller Seeds;
- HOF-Duplikate nach ausführbarem Phänotyp, nicht transientem Gene-Dict erkennen.

## 6. Candidate-Auswahl

Ein Kandidat kann nur in Exploit/Promotion gelangen, wenn:

1. sein Resultat vollständig und replayed ist;
2. alle nicht kompensierbaren Gates bestehen;
3. Train-, Pair-Val-, Temporal-Val- und Kostenstress-Rollen eindeutig sind;
4. er auf dem gemeinsamen Panel Pareto-relevant ist;
5. seine Unsicherheit ausreichend klein oder der Status bewusst `INCONCLUSIVE` ist;
6. er nicht nur ein Phänotypduplikat eines höher gerankten Kandidaten ist;
7. seine Herkunft keine Surrogate-/SIS-Schätzung ohne echten Backtest ist.

Für Diversität wählt der Planner aus verschiedenen Phänotyp-Clustern und nicht nur die Top-N einer
einzigen skalaren Fitness.

Der aktuelle V2-Selector erfüllt davon gemeinsames Panel, Phänotyp-Deduplizierung,
Experiment-Limits und Objective-Diversität. Behavior-Cluster aus Strategieeigenschaften sind noch
nicht implementiert und dürfen daher nicht als bereits nachgewiesene Diversität interpretiert
werden.

## 7. Config-Erzeugung

Configs werden nicht als freie Dicts mit still akzeptierten Keys erzeugt. Der Generator arbeitet
auf einem typisierten, versionsgebundenen Schema.

Fail-closed Prüfungen:

- unbekannte und falsch verschachtelte Keys sind Fehler;
- `mode`, Feature-Kombinationen und GA-Typ sind valide;
- Train/Val/Final-Pairs und Timeranges sind disjunkt und vollständig verfügbar;
- jede gewünschte Funktion wird vom ausgewählten Engine-Pfad tatsächlich gelesen;
- Population, Generationen, Inselzahl und Backtestbudget stimmen mit dem Arm überein;
- Warm-Start-Quelle, Genome-Schema, Timeframe und Indicator-Parameter sind kompatibel;
- Output-, Tracker-, HOF-, Checkpoint- und Logpfade sind attempt-eindeutig;
- alle Seeds werden explizit aufgelöst;
- ein Dry-Run zeigt resolved Config, Delta zur Baseline, Ressourcenbudget und Entscheidung.

Implementierungsstand 22.07.2026: Der zentrale Vertrag weist `multi_objective` auf `nsga2` hin,
blockiert wirkungslose `direction`/`weight`-Objectives und validiert Name, Eindeutigkeit, Typ,
Skala und Goldilocks-Toleranz. Der YAML-Gesamtscan findet keine verbleibende fehlerhafte
NSGA-Quelldatei. Generic-Island-Seed-Rotation und Surrogate-/Short-Runtime-Keys sind korrigiert und
regressiongetestet. Diese Reparaturen erlauben noch keinen NSGA-/Island-V2-Worker: Der Automation-
Worker unterstützt weiterhin ausschließlich Standard-Single-Objective unter `safe_v2`; neue
Engine-Arten brauchen eigene Worker-, Ressourcen- und Replay-Akzeptanztests.

Implementierungsstand CFG-007 am 23.07.2026: `ga-config-invariants-v1` erzwingt nur mechanische
Ausführbarkeit und widerspruchsfreie Semantik. Population, Inselgröße, Pairzahl, Generationen,
Mutation, Elitequote und Tournamentdruck sind keine universellen „Safe Ranges“, sondern explizite
Planner-Faktoren. Ihre Wirkung darf nur mit gepaarten Seeds, identischem Panel und Budget bewertet
werden. Die widersprüchlichen Legacy-AP-Regeln und Editor-Sonderregeln sind aus dem ausführbaren
Pfad entfernt; Resolver, Compatibility-Validator und Hook teilen denselben Vertrag.

Implementierungsstand 23.07.2026: Evolution- und Replay-Worker binden einen gemeinsamen
`AttemptOutputLayoutV2` in ihre gehashte Spec. Diagnostics, Tracker, HOF, Checkpoints, Logs,
Caches, generierte Strategien und Freqtrade-Arbeits-/Exportpfade sind damit attempt-eindeutig.
`GA_OUTPUT_DIR` ist kein Runtime-Override mehr. Ein echter Mini-Evolutions-E2E-Test belegt die
Dateisystemisolation; Island-/Generic-Island-/NSGA-II-Automationsworker bleiben davon unabhängig
weiterhin gesperrt.

Ebenfalls 23.07.2026: Island-Ergebnisse besitzen nun einen zentralen Extraktionsvertrag. Alle
deklarierten Arme werden vollständig abgeflacht; nur deduplizierte Finalisten desselben
`COMMON_REPLAY`-Panels mit explizitem Profit und Trade-Count dürfen den finalen Rang bilden.
Dieser Vertrag verhindert die früheren Dict→Nullwert-Fehlinterpretationen, autorisiert aber noch
keinen Island-Automationsworker.

Ein angeforderter Warm-Start muss bei fehlender/ungültiger Quelle fehlschlagen. Stiller Scratch-Start
ist in Automation verboten. Resume ist nur bei identischem Config-/Code-/Daten-/Genome-Schema
zulässig; sonst ist es ein neuer Warm-Start-Attempt.

Implementierungsstand 23.07.2026: Checkpoint V3 erzwingt diese Resume-Kompatibilität. Ein
resume-fähiger Checkpoint trägt eine kanonische Pflichtchecksumme und die exakten Hashes von
resolved Config, Code- und Datenmanifest sowie die Genome-Schemaversion; Island-Engines binden
zusätzlich Engine-Typ und geordnete Islandnamen. Diagnose-Checkpoints ohne immutable Provenienz
bleiben schreibbar, sind aber ausdrücklich nicht resume-fähig. Alte Formate, Checksumfehler,
Population-/Island-Abweichungen und der beschädigte numerisch neueste Island-Checkpoint blockieren,
statt still weiterzulaufen oder auf einen älteren Stand zurückzufallen. Der Standard-V2-Worker
liefert die Provenienz bereits. Ein abgebrochener Attempt wird vom Scheduler noch nicht automatisch
als neuer Attempt mit gehashtem Checkpoint-Input materialisiert; das ist ein separater Recovery-
Handoff und keine erlaubte best-effort Wiederaufnahme.

Der Legacy-Warm-Start ist ebenfalls fail-closed: Eine aktivierte Population-/HOF-Quelle benötigt
einen expliziten Pfad, den erwarteten Datei-SHA-256 und `strategy-gene-v2`. Es gibt keine globale
mtime-/„latest“-Discovery mehr. Die exakt gehashten Bytes werden nur einmal gelesen und vollständig
geparst; ein fehlendes, leeres, verändertes oder beschädigtes Artefakt, ein einziges
nichtkanonisches Genom, nichtfinite Ranking-Fitness oder eine nur teilweise mögliche Injection
bricht vor der Scratch-Population ab. Bei `both` sind beide Quellen atomar erforderlich. Historische
HOF-Dateien mit migrationsbedürftigen Genomen sind nur über den versionierten `CFG-008`-Pfad
zulässig. Dieser bindet alte und migrierte Datei jeweils an ihren SHA-256, protokolliert jede
erlaubte Normalisierung pro Genom und reproduziert Migration sowie Report vor dem Warm-Start.
Unbekannte Parameterfamilien, mehrdeutige Referenzen oder semantische Defekte blockieren die ganze
Quelle. Der kanonische V2-Worker bleibt unabhängig davon beim stärkeren
Frozen-Seed-/Executable-Roundtrip und aktiviert den Legacy-Loader nicht.

Implementierungsstand 23.07.2026: Entscheidungen und Wave-Vergleiche lesen keine Log-Regex-
Metriken mehr. `wave_comparison.py` verlangt eine reconciled Parent-Wave, genau eine immutable
ANALYSIS-Decision und eine vollständig gültige Snapshot-/Decision-Hashkette. Vor Ausgabe werden
alle eingefrorenen Attempt-Resultate und Kandidatenartefakte erneut verifiziert und die Analyse
deterministisch rekonstruiert; jede Abweichung blockiert. Der Monitor nutzt Logs nur noch für
operativen Fortschritt und übernimmt wirtschaftliche Felder ausschließlich aus `CANONICAL_V2`.
Der nicht kandidatgebundene Aggregate-Vergleich ist stillgelegt; der Legacy-Benchmark ist
diagnostisch markiert und besitzt weder Rankings noch automatische Interpretationen oder
Vergleichscharts.

## 8. Ressourcen- und Fehlersteuerung

Der Controller begrenzt nicht per Prozessnamens-Regex, sondern anhand eigener aktiver Leases und
gemessener Ressourcen.

- globale und per-GA-Typ Slots
- Mindest-RAM und optional CPU-/I/O-Budget
- Retrybudget nach strukturiertem Fehlergrund
- kein automatischer Retry bei invalidem Resultat, Config- oder Datenfehler
- Backoff für transiente Ressourcenfehler
- Canary vor großem neuen Engine-/Feature-Arm
- Stop-Regeln bei hoher Fehlerrate, Control-Degradation oder Verlustbudgetverletzung

Nach Restart werden Leases, Heartbeats, Prozess-Starttoken und Resultate reconciled. Ein Attempt wird
nicht allein wegen fehlender PID als erfolgreich oder fehlgeschlagen markiert.

Implementierungsstand: Diese Reconciliation und reale Kill-/Restart-Tests existieren im V2-Pfad.
`ShadowSchedulerV2` und `EvolutionSchedulerV2` führen sie periodisch vor neuen Claims aus; beide
verwenden denselben Executor. Der Spawn-Handshake ist mit Parent-Death-Signal, atomaren
Ready-/Release-Receipts, Prozessgruppenbeendigung und einem echten SIGKILL im kritischen Fenster
implementiert. Offen sind ein langlebiger operativer Controller/Daemon und Ressourcenmessung
jenseits der Slotzahl.

## 9. Go/No-Go-Gates für die nächste Wave

Der Planner blockiert und erzeugt nur einen Decision-Report, wenn mindestens eines gilt:

- Parent-Wave nicht vollständig reconciled;
- Resultat-/Artefaktmanifest fehlt oder Hash stimmt nicht;
- Metric-/Policy-/Datenversionen sind inkompatibel;
- Pflichtmetriken sind missing/non-finite;
- zu wenige gültige Seeds, Kandidaten, Pairs oder Zeitblöcke;
- Control-Arm ist nicht vorhanden oder unerklärt degradiert;
- Fehler-/Timeoutquote überschreitet das Policybudget;
- kein Kandidat besteht wirtschaftliche und Risiko-Gates;
- Warm-Start-/Search-Space-Kompatibilität fehlt;
- Proposal wäre nicht idempotent oder kollidiert mit vorhandenem Attempt.

„Kein Kandidat besteht“ ist ein valides Ergebnis. Die Automation darf dann eine Diagnose-/Control-
Wave vorschlagen oder stoppen, nicht zwanghaft einen Verlierer promoten.

## 10. Einführungsphasen

### Phase 0: Messbasis einfrieren

- aktuelle Configs, Code-/Datenstände und Resultate nur referenzieren, nicht umschreiben;
- laufende Attempts nicht migrieren;
- bekannte Legacy-Resultate als `legacy/untrusted` markieren.

### Phase 1: P0-Korrektheit

- Cache-Roundtrip, Pflichtmetriken, Units und deterministischen Phänotyphash reparieren;
- Holdout-Leak, NSGA-II und Parallel/Sequential-Parität beheben;
- gemeinsame Diagnostics isolieren;
- End-to-End-Invarianten ergänzen.

Stand 22.07.2026: Der NSGA-`(μ+λ)`-Lifecycle und die hier genannte
Parallel-/Sequential-Parität sind mechanisch repariert und regressiongetestet. Das erweitert den
automatisch zugelassenen Worker-Typ bewusst noch nicht: NSGA benötigt weiterhin einen eigenen
V2-Worker, Ressourcenbudgets und gepaarte Blind-OOS-Akzeptanz gegen den Single-Objective-Control.

### Phase 2: Runner und State Store

- einen Runnerpfad herstellen;
- SQLite-State, atomaren Claim, Lease/Heartbeat und Attempt-Verzeichnisse einführen;
- Shell- und Python-Queue zunächst als Adapter, danach Legacy-Shell außer Betrieb nehmen;
- Kill/Restart/Retry/Doppel-Claim testen.

### Phase 3: Resultat- und Evaluation v2

- kanonische Equity-/Szenarioartefakte schreiben;
- bestehende Kandidaten auf festem Panel replayen;
- Legacy- und V2-Ranking vergleichen;
- Promotion-Policy im Shadow-Modus kalibrieren.

### Phase 4: Planner im Shadow-Modus

- Parent-Snapshot analysieren;
- deterministische Configs und Decision-Report erzeugen;
- keinen Queue-/Startschreibzugriff besitzen;
- Vorschläge gegen manuelle Entscheidungen und spätere Resultate messen.

### Phase 5: Approval-Modus

- freigegebenen Proposal-Hash atomar queueen;
- zuerst ein Canary, danach Rest der Wave;
- jede Abweichung oder Mutation nach Approval blockieren.

### Phase 6: Guarded Auto

- nur bewährte Policy-/Armtypen automatisch erlauben;
- neue Features, SIS, Surrogate, Search-Space-Ausweitungen und Final-Test-Verwendung bleiben
  approval-pflichtig;
- automatischer Kill-Switch und Budgetdeckel.

## 11. Integrations-Akzeptanztests

Vor Approval-Modus:

1. Zwei Scheduler können denselben Attempt nicht doppelt claimen.
2. 60 parallele Registry-/Event-Updates gehen nicht verloren.
3. Kill vor Spawn, während Run und nach Resultatschreiben wird korrekt reconciled.
4. Restart übernimmt echte Runs und erkennt stale/PID-reused Prozesse.
5. Gleicher Parent-Snapshot erzeugt dieselbe Wave und keine Doppelstarts.
6. Unbekannter Config-Key, ungültiger Mode und falsch verschachteltes Feature blockieren vor Spawn.
7. Jeder Attempt schreibt ausschließlich in sein Verzeichnis.
8. Exit 0 ohne valides Resultat ist `INVALID_RESULT`, nicht Erfolg.
9. Validiertes Resultat mit späterem Postprocessing-Fehler bleibt nicht unbemerkt.
10. Fehlende/inkompatible Warm-Start-Quelle blockiert statt Scratch-Fallback.
11. Unvollständige Parent-Wave und negative Validation blockieren Folge-Promotion.
12. Sequential/parallel und fresh/RAM/disk erzeugen dieselbe Entscheidung.

Vor Guarded Auto zusätzlich:

13. Mehrere komplette Shadow-/Approval-Waves wurden ohne Zustands- oder Artefaktabweichung
    reconciled.
14. Die V2-Policy verbessert blindes OOS-Risiko/Return gegenüber dem festen Control mit vorab
    definiertem Konfidenzkriterium.
15. Kill-Switch, Budgetlimit und manuelles Stoppen sind in einem Canary praktisch getestet.

## 12. Geplante Wirksamkeitstests

Nach den P0-Fixes:

1. **SIS-Factorial:** Control, Seed-Filter, Immigrants, Weights und Full; mindestens zehn gepaarte
   Seeds, gleicher gültiger Suchraum und gleiches Budget. Primary ist blinde
   Later-Time-x-Unseen-Pair-Utility, DD und Ruin.
2. **Pair-Validation:** keine Pair-Val vs fixes Single-Pair vs rotierendes LOPO/Cluster-Panel; finale
   Bewertung auf zwei nie gesehenen Pairs.
3. **Temporal:** aktuelles WF vs nicht überlappende blocked Temporal-Validation vs keine innere
   Zeitprüfung bei gleichem Backtestbudget; finaler später Zeitraum bleibt blind.
4. **Indicator-Priors:** uniform vs aktuelle SIS-Priors vs milde datenarme Priors. MACD, ROC,
   CDL_HAMMER und BBANDS sind Upweight-Testkandidaten; PSAR, KAMA, TEMA und VROC
   Downweight-Testkandidaten.
5. **Robustheitsdiagnostik:** Block-Bootstrap, repariertes DSR und Block-Rank-Diagnostik zunächst nur
   post-hoc. Aufnahme nur bei messbarem zusätzlichen Vorhersagewert pro Rechenzeit.

Erfolg eines Features wird nicht durch höhere Train-Bestfitness definiert, sondern durch gepaarte
Verbesserung des blinden OOS-Return-LCB ohne Verschlechterung von DD-/ES-UCB über mehrere Seeds,
Pairpanels und Timeframes.
