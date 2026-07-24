# GA TODO und Befundregister

Stand: 24.07.2026. Dieses Dokument ist die priorisierte Arbeitsliste aus
[GA_AUDIT_2026-07-20.md](GA_AUDIT_2026-07-20.md).

Achtundzwanzigster Slice (24.07.2026): Der begrenzte Diagnose-Root
`wave-root-98ffe8fc62db2a7ba0a6` endete nach 13:39 Minuten, 1,5 GiB Peak-RAM
und einem hashverifizierten erfolgreichen Attempt erwartungsgemäß an
`MAX_WAVES_REACHED`; kein Child wurde erzeugt. Alle fünf Finalisten blieben
korrekt ineligible. SOL erzielte je nach Kandidat 7,73–15,98 % bei 39–129
Trades, aber BTC erzeugte bei allen Kandidaten null Trades. ETH/BNB lagen nur
bei 6–20 beziehungsweise 4–15 Trades. Hohe Winrates und Teilrenditen sind
deshalb kein Robustheitsnachweis; zusätzlich lagen unter anderem `N_eff` und
Drawdown-Dauern einzelner SOL-Szenarien außerhalb der Promotion-Grenzen.

Die Artefaktanalyse fand zwei Ursachen im Suchpfad. Erstens serialisiert die
aktive Freqtrade-Version Trades als JSON-Recordliste; der Pair-Extractor
verarbeitete nur DataFrames und schrieb daher trotz 39–600 aggregierter Trades
für jedes Pair null. Zweitens hob der allgemeine 10-%-Penalty-Floor den
konfigurierten 1-%-Worst-Pair-Multiplikator wieder an. Der Extractor unterstützt
nun beide Payloadformen. Pair-Split-Fitness wird aus den unkompensierten
Train-/Validation-Komponenten gebildet und anschließend genau einmal mit der
schlechtesten Coverage über das gesamte Pairpanel multipliziert. Ein echter
Replay der fünf Finalisten bestätigt korrekte Pairzahlen und für alle wegen
BTC=0 den globalen Faktor 0,01; ihre korrigierten Search-Scores liegen nur noch
bei 0,0071–0,0085 statt ungefähr 0,08–0,09.

Der Diagnose-Root war außerdem keine unabhängige Seed-Wiederholung:
Manifest/GA trugen 2001, aber explizite Preset-Islands überschrieben dies mit
42–45. Der immutable Attempt-Seed ist nun auch für Generic Island
authoritativ und leitet die Island-Seeds 2001–2004 ab. Automationsberichte
persistieren diesen Seed-Roundtrip und markieren historische Abweichungen
explizit. Exitcode 2 bleibt ein bewusster Guard-Stopp, wird in der
`systemd`-Unit aber als erfolgreicher Exit klassifiziert statt als roter
Servicefehler. 71 fokussierte Worker-/Fitness-/Pair-/Controller-Tests laufen
nach den Korrekturen grün. Das begrenzte Preset ist für den nächsten, noch
nicht gestarteten Root auf Seed 3001 umgestellt, damit dessen immutable
Wave-Identität nicht mit dem historischen Seed-2001-Lauf kollidiert.

Siebenundzwanzigster Slice (24.07.2026): Die zweite Versuchslinie wurde
vollständig und recoverable unter
`data/v2/trial_archives/second-generic-island-20260724-root-579e04974-plan-6985528f2`
archiviert. Dazu gehören State-DB, Root-Artefakte, der nie gestartete
redundante Child sowie ein kompakter hashverifizierter JSON-/Markdown-Bericht.
Der Bericht enthält Entscheidungsgründe, Attempt-Provenienz, Gatezustände und
die wesentlichen Pair-Metriken, aber bewusst keinen Strategiecode, keine
Einzeltrades und keine Workerlogs. Künftige Wave-Analysen schreiben denselben
Report automatisch in `automation/reports`, einschließlich einer bequem
lesbaren `LATEST.md`.

Der Automation-Controller liest Root-Seeds und Wave-Limit nun aus dem strikt
validierten V2-Config-Vertrag. Der erste Diagnose-Root verwendete den damals
frischen Root-Seed `2001` und `max_waves: 1`: Der Root durfte vollständig laufen
und analysiert werden, danach blockierte `MAX_WAVES_REACHED` jede
Child-Materialisierung. So kann die neue Seed-Rotation separat von der
Strategieentwicklung beurteilt werden, ohne versehentlich eine lange Kampagne
zu starten. 146 fokussierte Vertrags-/Config-/Planner-/Controller-Tests sowie
die bereinigte GA-Suite mit 1.667 Tests laufen grün.

Sechsundzwanzigster Slice (24.07.2026): Der zweite echte Root-Versuch auf dem
korrigierten Fitnessvertrag endete mit einem hashverifizierten `SUCCEEDED`-
Attempt. Startpopulation und erzeugte Gen-0-Strategien waren bytegenau
reproduzierbar. Alle Checkpoints hatten eindeutige generation-lokale IDs, alle
Backtests liefen technisch erfolgreich und der neue Pair-Coverage-Penalty
reduzierte die zuvor irreführend hohen Search-Scores deutlich. Fachlich war
noch kein Finalist promotionsfähig: Die Aktivität konzentrierte sich weiterhin
auf einen Train-Pair, während die übrigen Pair-Szenarien zu wenige Trades
lieferten. V2 stufte diese Evidenz korrekt als `INCONCLUSIVE` statt als Erfolg
oder technischen Fehler ein. Der kleine Baseline-Budgetlauf belegt damit
korrekte Selektion/Gates, aber noch keine ausreichend generalisierende
Strategieentwicklung.

Der kontrollierte Stopp vor einer Folge-Wave deckte drei weitere
Produktionslücken auf: Bootstrap-Readback nach einem fertigen Child verwechselte
Outputs mit unerlaubten Pre-Execution-Dateien, Shared-OHLCV löste relative
Datapfade gegen das Attempt-CWD auf, und die Suchphase nahm eine zusätzliche
Candle des exklusiven Endtags auf. Diese Verträge sind unter `ORCH-021` bis
`ORCH-023` repariert. Pair-Split-Checkpoints persistieren nun außerdem
Train-/Validation-Tradezahlen, Worst-Pair-Aktivität und den tatsächlich
angewendeten Coverage-Multiplikator. Vor dem nächsten teuren Lauf ist die
kontrollierte Child-Planung auf dem bereinigten Commit zu prüfen; ein
Scratch-Control ist bei null eligible Candidates korrekt, ein Exploit wäre es
nicht. Die bereinigte GA-Suite besteht nach diesen Reparaturen mit 1.666 Tests
und 13 bereits bekannten Warnungen; 34 externe Generation-Step-Tests sind
ebenfalls grün.

Fünfundzwanzigster Slice (24.07.2026): Der erste echte
`automation_island_v2`-Produktionsversuch durchlief alle zwölf Generationen,
vier Islands, Migration, HOF-Export und zwanzig Pair-Replays ohne
Backtest-Enginefehler. Er deckte dabei vier reale Integrations-/Zielfehler auf:

- Der immutable Artifact-Readback verglich Engine-native
  `pandas.Timestamp`-Werte mit ihren kanonischen JSON-Strings als
  Python-Objekte und meldete dadurch fälschlich Manipulation. Recorder und
  Store vergleichen und halten nun die kanonische Persistenzform.
- Der Replay-Runner behandelte Freqtrades Stopzeit fälschlich als exklusiv.
  Dadurch wurde eine zusätzliche Candle beziehungsweise ein zusätzlicher
  Null-Rendite-Tag gemessen und jede Scenario-Coverage invalidiert. Der Runner
  verwendet jetzt exakt den letzten erwarteten Candle-Open aus Timeframe und
  geschlossenem Szenariozeitraum.
- Elite-Copies behielten alte generation-lokale IDs und kollidierten mit
  Immigrants/Offspring. Jede neue Generation erhält nun eindeutige Slot-IDs.
- Shared Memory lud fest `5m`, obwohl der freigegebene Pfad `1h` verwendet.
  Es lädt nun genau einen konfigurierten Timeframe; Multi-Timeframe fällt
  bewusst auf die korrekte Disk-Ladung zurück.

Der korrigierte Replay der fünf Finalisten zeigte außerdem eine wichtige
Zielabweichung: Aggregierte Train-/Validation-Tradezahlen verdeckten Pairs mit
null oder nur einzelnen Trades. Solche Kandidaten bleiben im strikten
V2-Panel korrekt `INCONCLUSIVE`, wurden von der Legacy-Suchfitness aber zu hoch
gerankt. `BacktestResult` persistiert deshalb jetzt Tradezahlen für jedes
deklarierte Pair einschließlich Null-Pairs. Das Automation-Preset bestraft
kontinuierlich den schlechtesten Pair in Richtung 60 Trades, entsprechend
zwölf aktiven Monaten mal fünf Trades. Dies erhält einen Evolutionsgradienten,
ohne den strikten Replay-Gate vorzutäuschen. Ein real beobachteter
Floating-Point-Überlauf der Pair-HHI über 1.0 wird an der mathematischen
Domänengrenze stabilisiert. Cache-Schema 8 trennt die neue Messsemantik von
alten Einträgen. 1.661 bereinigte GA-Tests laufen grün; der nächste Schritt ist
ein frischer Root-Versuch auf dem neuen Commit und anschließend der Nachweis
einer semantisch geplanten Folge-Wave.

Vierundzwanzigster Slice (24.07.2026): `GENERIC_ISLAND_EVOLUTION` ist jetzt ein kanonischer
immutable V2-Worker. Der äußere Vertrag erzwingt `automation_island_v2`, Shadow-only, mindestens
zwei explizite Islands, einen gemeinsamen Workerpool und vollständige Pair-Validation jedes
Kandidaten. Train- und Validation-Pairs sind im Split-Manifest tatsächlich getrennt; ihr disjunkter
Union-Vertrag muss exakt `backtesting.pairs` ergeben. Parent-Genome werden strikt in jedes Island
injiziert. Ein echter Mini-Island-Lauf belegt Worker-, Checkpoint-, Candidate- und Replay-Roundtrip.

`AutomationControllerV2` ergänzt deterministischen/idempotenten Root-Bootstrap, lineare
restartfähige Wave-Reconciliation, Analysis, Pareto-Auswahl, Control/Replication/Explore,
hashgebundene Search-only-Auto-Approval und atomisches Child-Queueing. Eine gesunde Wave ohne
eligible Kandidaten darf als reine Control weitersuchen; ungesunde Evidenz blockiert. Persistenter
Kill-Switch sowie Laufzeit-, Wave-, Attempt-, Parallelitäts-, Disk-, RAM- und Factor-Path-Limits sind
fail-closed. Der Subprozess-E2E deckte dabei einen realen Virtualenv-Fehler auf: `resolve()` hatte
`.venv/bin/python` auf den Systeminterpreter reduziert. Materializer bewahren den Venv-Launcher nun.
Der read-only Produktions-Preflight bindet Config-, Code-, Daten-, Split- und Policyhash und prüft
Venv sowie reale Disk-/RAM-Reserven. Das Preset deckelt jeden Island-Run auf 600 Minuten; eine
generierte `systemd --user`-Unit startet echte Controller-Crashes neu, aber keine absichtlichen
Guard-/Budget-Stopps. Ein separater 12-Stunden-Executor-Timeout beendet auch außerhalb der
Generationsgrenze hängende Prozessgruppen. Technische Totalausfälle führen zu höchstens drei
Scratch-Control-Recovery-Waves in Folge; danach blockiert der Controller.
206 fokussierte Config-, Island-, Worker-, Wave-, Controller-, CLI- und Materializer-Tests sowie ein
echter Root→Child-Controller-E2E laufen grün. Der breite bereinigte GA-Lauf besteht 1.646 Tests;
der unbereinigte Lauf besteht 1.647 Tests und endet ausschließlich an den sechs bereits als
`TEST-001` bekannten fehlenden Pseudo-Fixtures in `test_llm_providers.py` und
`test_llm_strategy_generation.py`. Offen bleiben empirische Policy-Kalibrierung,
hartes Beenden bereits laufender Worker, Retention/Tombstones und automatische Trading-Promotion
(weiterhin ausdrücklich gesperrt).

Aktiver Implementierungsbranch: `codex/ga-v2-p0-foundation`. Der erste Slice enthält das
erzwungene Shadow-Preset `safe_v2`, versionierte Cache-Schemas, vollständige `BacktestResult`-Roundtrips,
deterministische Fee-Noise-Seeds, wieder angeschlossene Penalty-Inputs sowie strikte V2-Resultatmodelle
mit fail-closed Legacy-Adapter. Die Freigabe-Gates bleiben bis zu ihren vollständigen
End-to-End-Akzeptanztests geschlossen.

Zweiter Slice: kalendergefüllte, kapitalgewichtete Realized-close-Equity und Tagesreturns,
referenzgetestete DD-/DD-Dauer-/Time-under-Water-/Ulcer-/ES5-/Sharpe-/Sortino-/Calmar-Metriken,
deterministische Moving-Block-Bootstrap-Grenzen und ein Shadow-Adapter. Realized-close ist explizit
`INCONCLUSIVE`; `VALID` erfordert weiterhin echte Mark-to-Market-Equity.

Dritter Slice: `safe_v2` erzeugt aus zeitgestempelten Fills eine tägliche, gebührenbereinigte
Mark-to-Market-Equity und reconciled jeden geschlossenen Trade sowie den finalen Walletstand. Der
Order-Ledger unterstützt Long, Short, Hebel, mehrere Entries/DCA und Teilexits; der Hebel wird nicht
doppelt auf den bereits gehebelten Fill-Betrag angewandt. Der Backtester exportiert Funding als
zeitgestempelte Wallet-Cashflows, einschließlich der Fill-time-Korrekturen bei DCA, und reconciled
deren Summe gegen `funding_fees`. Offene Endpositionen ohne terminalen Wallet-/Positionssnapshot
sowie fehlende Tageskerzen scheitern weiterhin fail-closed. Trade-Expectancy erhält eine
deterministische Zeitcluster-Moving-Block-LCB für gleichgewichtete Trades und gebundenes Kapital.
Pairkonzentration sowie serielle, kapitalgewichtete und zeitclustergewichtete `N_eff`-Komponenten
werden gemeinsam persistiert. Diese Messungen machen ein einzelnes vollständiges Szenario technisch
`VALID` und liefern die Messbasis für den folgenden Panel-/Artifact-Layer.

Vierter Slice: Ein `ShadowAttemptRecorderV2` schreibt resolved Config, Manifest, versionierte
Promotion-Policy, einzelne Backtests, Candidate-Auswertungen, Shadow-Entscheidungen und das
terminale `result.json` atomar und unveränderlich. SHA-256-Readback erkennt fehlende, zusätzliche
oder manipulierte Artefakte. Die vorab deklarierte Scenario-Matrix und nicht kompensierbare
Expectancy-/Return-/DD-/ES-/`N_eff`-/Aktivitäts-/Tail-Gates sind implementiert. Auch ein kompletter
Pass bleibt `WOULD_PASS` ohne Autorisierung. Der bestehende CLI-Evolutionspfad liefert noch keine
vollständigen Replay-Szenarien und ist deshalb bewusst noch nicht als V2-Success-Runner angebunden.

Fünfter Slice: `ShadowReplayRunnerV2` friert Strategiecode, Timeframe, `max_open_trades` und externe
Ausführungsparameter in einem SHA-256-Phänotyphash ein. Er führt jede vorab deklarierte
Pair-x-Zeit-x-Kosten-Zelle mit isoliertem Pair und Timerange aus, skaliert Fee/Slippage gemäß
Kostenmultiplikator, prüft tatsächliche gegen deklarierte Perioden-Coverage und übergibt echte
`BacktestRecordV2` an den Recorder. Dynamische Slippage, separate Spreads, Funding, Short und
mehrere Seeds scheitern aktuell fail-closed.

Sechster Slice: Das Datenmanifest hasht rohe Feather-Dateien und den kanonischen OHLCV-Inhalt jeder
vorab deklarierten Szenarioperiode und verlangt lückenlose, exakt begrenzte Candle-Coverage. Das
Code-Manifest bindet Git-Commit, alle tracked Änderungen und relevante untracked Source-Dateien;
die resolved Config besitzt ihren eigenen kanonischen Hash. Ein Builder erzeugt daraus konsistente
Attempt-Manifeste. Der Legacy-Candidate-Exporter blockiert zufällige HOF-/Codegen-Reparaturen,
Mutation und nichtdeterministische Ausgabe; Frozen Candidates werden als Replay-Artefakt
persistiert. Ein realer Smoke-Test auf
`BTC_USDT-1h.feather` bestätigte 48/48 Candles; alle 25 Einträge des Wave-39-A-1h-HOF waren unter
dem eingeschränkten Spot-Long-/Single-Timeframe-Vertrag deterministisch exportierbar. Offen blieben
an dieser Stelle Final-Test-Usage-Ledger und CLI-/Scheduler-Integration.

Siebter Slice: Ein SQLite/WAL-`FinalTestUsageLedgerV2` reserviert Final-Test-Zellen atomar und
sperrt überlappende Börse-x-Pair-x-Zeiträume attemptübergreifend. Scenario-Umbenennung,
Timeframe-Resampling, Kostenvariation oder nachträglich geänderte Datendateien erzeugen keine neue
Blindheit. Der
Replay-Runner markiert das gesamte Panel vor dem ersten möglichen FINAL_TEST-Datenzugriff als
`EXPOSED`; auch ein direkt anschließender Konstruktor-/Prozessfehler gibt es nicht mehr frei.
Unexponierte Reservierungen dürfen kontrolliert freigegeben werden. Das Receipt wird Teil des
hash-verifizierten Attempt-Artefakts. Vollständige Train-/Validation-/Final-Disjunktheit und die
Reconciliation eines nach hartem Prozess-Kill zurückbleibenden `RESERVED`-Claims sind noch offen.

Achter Slice: `EvaluationSplitPlanV2` trennt `TRAIN`, `PAIR_VALIDATION`,
`TEMPORAL_VALIDATION` und `FINAL_TEST` explizit. Freqtrade-Timeranges werden mit exklusivem
Enddatum modelliert. Pair-Validation muss unbekannte Pairs verwenden, Temporal-Validation bekannte
Pairs in einem späteren Zeitraum, und Final-Evidenz muss für dasselbe Pair nach allen vorherigen
Beobachtungen plus konfiguriertem Embargo liegen. Ein anderer Timeframe gilt nicht als andere
Marktbeobachtung. Das mehrdeutige Legacy-`INNER_VALIDATION` blockiert im kanonischen V2-Pfad.

Neunter Slice: `AttemptStateStoreV2` implementiert die erlaubte Attempt-State-Machine in
SQLite/WAL. Queue-Claims sind atomar und durch zufällige Claim-Token gefenced; Lease, Heartbeat,
PID und Prozess-Starttoken verhindern Updates eines stale Workers oder einer wiederverwendeten PID.
Alle Transitionen und Heartbeats erzeugen immutable Events. `SUCCEEDED` kann nur durch
hash-verifizierten Terminal-Readback erreicht werden. 60 parallele Claims sowie 60 parallele
Heartbeats verloren weder Attempts, Versionen noch Events. Legacy-Queue/Registry/CLI und die
autorisierte Reconciliation abgelaufener Leases sind noch nicht umgestellt.

Zehnter Slice: `AttemptExecutorV2` führt ein atomar geclaimtes Attempt als echte Child-
Argumentliste ohne Shell-Auswertung aus. Command-Hash, Arbeitsverzeichnis und attempt-isolierter
Runtime-Logpfad werden vor dem Spawn unveränderlich persistiert; PID plus Linux-Boot-ID und
Prozess-Startticks bilden die Prozessidentität. Der Controller verlängert die Lease per Heartbeat,
hasht den geschlossenen Log und akzeptiert Exit 0 nur zusammen mit einem hash-verifizierten
Terminalresultat. `AttemptReconcilerV2` lässt abgelaufene, identisch noch lebende Prozesse in Ruhe
und schließt einen Run erst nach `NOT_FOUND`-/PID-Reuse-Nachweis. Ein vorhandenes valides Resultat
wird dann übernommen, ein fehlendes Resultat wird `INTERRUPTED` und ein beschädigtes Resultat
`INVALID_RESULT`. Der damals noch offene Spawn-vor-`RUNNING`-Crashfall wird im zweiundzwanzigsten
Slice durch den Parent-Death-Guard geschlossen. CLI und operative Scheduler verwenden inzwischen
diesen gemeinsamen Pfad.

Elfter Slice: `ReplayWorkerSpecV2` persistiert den vollständigen Shadow-Replay-Auftrag vor dem
Queueing: Manifest, resolved Config, Promotion-Policy, Code-/Daten-/Split-Manifest und alle Frozen
Candidates sind eine exakte, geschlossene Input-Hashmatrix. Der SHA-256 der Worker-Spec wird als
eigenes Argument Teil des gehashten Commands. Der Child lädt ausschließlich kanonische relative
Pfade, prüft Spec-, Datei- und semantische Manifest-Hashes erneut und startet danach den bestehenden
`ShadowReplayRunnerV2`. Ausführungs- und Inputfehler erzeugen, soweit das Attempt-Manifest noch
vertrauenswürdig ist, ein verifiziertes `INVALID_RESULT` statt nur eines Log-Tracebacks. Ein echter
End-to-End-Test durchläuft SQLite-Claim, `.venv`-Subprozess, Replay, Final-Test-Ledger,
`result.json`-Readback und terminalen State. Spec-Manipulation nach Command-Erzeugung wird im echten
Subprozess blockiert. Offen bleiben der Shadow-Scheduler-Adapter und V2-Worker für die eigentliche
Evolution statt nur eingefrorener Candidate-Replays.

Zwölfter Slice: `WorkerBindingV2` bindet Worker-Art, exakte Argumentliste, Arbeitsverzeichnis,
Spec-Pfad und Spec-Hash bereits im `DRAFT`-State. Der daraus berechnete Binding-/Command-Hash ist
vor `VALIDATED` unveränderlich; ein nach dem Queueing substituierter Command wird vor Spawn durch
das State-Fencing blockiert. `ShadowSchedulerV2` arbeitet ausschließlich auf SQLite/WAL, claimt nur
gebundene `SHADOW_REPLAY`-Attempts, zählt alle nichtterminalen Claims als belegte Slots und führt
periodisch die konservative Prozess-Reconciliation aus. Ein vorheriger Scheduler mit noch lebendem
Child belegt daher weiterhin Kapazität. Queueing ist für byte-identische Bindings idempotent.
Gezielte Tests decken Scheduler→Executor→Worker→Replay, externe aktive Leases, Dead-Process-
Recovery und Command-Substitution ab. Offen bleiben ein langlebiger operativer CLI/Daemon-Adapter,
Ressourcenmessung jenseits der Slotzahl und der entsprechende Worker für echte GA-Evolution.

Dreizehnter Slice: `WaveStateStoreV2` erweitert dieselbe SQLite/WAL-Transaktionsdomäne um
versionierte Wave-Zustände, immutable Experiment-Spezifikationen, die exakte erwartete
Attempt-Menge, Decisions und Events. Collection blockiert fehlende, zusätzliche oder hinsichtlich
Wave/Experiment/Manifest/Seed abweichende Attempts. Reconciliation akzeptiert pro terminalem
Attempt ausschließlich ein vollständig hash-verifiziertes Resultat oder eine explizite immutable
Abort-Decision; ein beschädigtes vorhandenes Resultat kann nicht als Abort umdeklariert werden.
Der kanonisch sortierte Parent-Snapshot wird atomar gehasht und eingefroren. Analysis, Proposal und
Approval sind über Input-Hashes an Snapshot beziehungsweise Vorgänger-Decision gebunden. Zehn neue
Integritäts-, Gate-, Idempotenz- und Parallelitätstestfälle sowie die gesamte GA-Suite mit 1288 Tests
laufen grün; die zwei bekannten LLM-Collection-Dateien bleiben separat kaputt. Analyzer, Planner,
produktives Queue-Handoff und Legacy-Migration sind noch offen.

Vierzehnter Slice: `WaveAnalyzerV2` konsumiert ausschließlich einen eingefrorenen
`WaveResultSnapshotV2`, lädt jedes referenzierte Resultat erneut aus dem Artifact-Store und prüft
Datei-, Artifact-, Manifest-, Config-, Policy-, Experiment- und Seed-Provenienz. Die versionierte
Analyzer-Policy begrenzt Abort-/Non-Success-Quote, verlangt optional einen Control-Arm und setzt
eine Mindestzahl eindeutiger Seeds je Experiment-x-Phänotyp. Gültige Messdaten bleiben auch bei
gescheitertem Promotionsgate in den Return-, DD-, ES-, Expectancy-, Profit-Factor-, Winrate- und
Trade-Aggregaten; das Gate macht sie lediglich ineligible. Gleiche Inputs erzeugen dieselbe
gehashte Analysis und `ANALYSIS`-Decision. Acht neue Tests decken Determinismus, Mutation,
Abort-Gesundheit, Gate-Nichtkompensation, Control-, Config-/Policy-Bindung und idempotentes
State-Recording ab; die gesamte GA-Suite läuft mit 1296 Tests grün. Offen bleiben empirische
Policykalibrierung, Pareto-/Diversitätsauswahl und der eigentliche Child-Wave-Planner.

Fünfzehnter Slice: `candidate_selector_v2.py` rankt ausschließlich eligible Kandidaten auf demselben
Vergleichspanel. Return-LCB und Expectancy-LCB werden maximiert, DD-UCB und Daily-ES-UCB minimiert;
kein einzelner Score kann Risiko durch Profit kompensieren. Deterministisches Non-Dominated-Sorting,
normalisierte Objective-Distanz und ein Per-Experiment-Limit erzeugen eine reproduzierbare Auswahl.
Sieben Tests belegen Dominanz, Return-/Risiko-Trade-offs, Panel-Inkompatibilität, blockierte Analysis,
Experiment-Diversität, Ineligible-Ausschluss und Byte-Determinismus. Behavior-Cluster bleiben offen.

Sechzehnter Slice: `wave_planner_v2.py` erzeugt aus Analysis-/Selection-Hash, Parent-Specs,
hash-verifizierten Parent-Configs und versionierter Policy einen byte-identischen
`ChildWavePlanV2`. Control ist unverändert, Replication verbietet Deltas, Exploit/Explore verlangen
einen expliziten einzelnen Faktor oder ein bewusstes Factorial; alle Arme verwenden gepaarte Seeds.
Config-Deltas dürfen nur existierende typkompatible Pfade ändern. Wave-/Experiment-/Attempt-IDs,
Budgets und Quellen sind deterministisch. Blockierte Selection erzeugt `BLOCK` ohne Experimente,
ein vollständiger Plan bleibt bis zur Materialisierung read-only; unvollständige Proposals können
nicht approved werden. Acht Planner-Tests plus ein Approval-Gate-Test erhöhen die grüne GA-Suite
auf 1312 Tests. Offen bleiben produktive Schema-Validierung,
Frozen-Candidate-/Manifest-Materialisierung und Approval→Queue.

Siebzehnter Slice: Der Planner besitzt nun einen expliziten `REPLAY_VALIDATION`-Modus, der nur
ausgewählte Frozen Candidates ohne Factor-Delta akzeptiert. `wave_materializer_v2.py` prüft das
resolved Replay-Config fail-closed auf unbekannte Pfade, produktive Validatorfehler und den
Shadow-Runtime-Vertrag, verifiziert Parent-Resultat und Frozen Candidate erneut gegen ihre
SHA-256-Kette und materialisiert Attempt-Manifest, Config, Promotion-Policy, Code-, Daten- und
Split-Manifest sowie Worker-Spec unveränderlich. Erst das vollständige Materialisierungs-Receipt
erzeugt ein `PROPOSAL`; `APPROVAL` bindet sowohl `plan_hash` als auch `materialization_hash`. Das
Queue-Handoff registriert Child-Wave, Experiment-Specs, Attempts und Worker-Bindings gemeinsam mit
dem Parent-Übergang in genau einer SQLite-Transaktion. Receipt-/Candidate-Manipulation, unbekannte
Config-Pfade, Approval-Abweichung und ein Datenbankkonflikt hinterlassen keine partielle Child-Wave.
Sieben neue Tests erhöhen die grüne GA-Suite auf 1319 Tests. Bewusst offen bleibt der kanonische
V2-Evolutionsworker für Control/Replication/Exploit/Explore; diese Arme werden nicht fälschlich an
den Replay-Worker übergeben.

Achtzehnter Slice: `EvolutionWorkerSpecV2` bindet Standard-GA, Attempt-Seed, Top-N und optionale
Parent-Genome an dieselbe geschlossene Manifest-/Config-/Policy-/Code-/Daten-/Split-Hashkette wie
der Replay-Worker. Der Child registriert sich nicht selbst. Legacy-Fitness darf nur suchen und
shortlisten; ein `SUCCEEDED` entsteht erst, nachdem jedes erzeugte Executable denselben strikten
V2-Replay-/Gate-Pfad durchlaufen hat. `FrozenEvolutionSeedV2` verbindet das serialisierte Genom mit
seinem reproduzierten Executable-Phänotyp. Control startet bewusst ohne Parent-Genom;
Replication/Exploit/Explore scheitern bei fehlendem, manipuliertem oder unter der Child-Config
nicht reproduzierbarem Seed. Checkpoints, Tracker, HOF, Diagnostics, Engine-Log, Cache und temporäre
Strategiedateien liegen unter genau einem Attempt-Root. `EvolutionSchedulerV2` nutzt denselben
SQLite-Claim-/Lease-/Heartbeat-/Executor-/Recovery-Pfad. Ein echter Subprozesstest bestätigt, dass
nach Queueing veränderte Inputs als verifiziertes `INVALID_RESULT` enden. Evolutions-Waves dürfen
keine `FINAL_TEST`-Zellen enthalten; diese bleiben einer späteren Replay-Validation vorbehalten.
Neun neue Tests erhöhen die grüne GA-Suite auf 1328 Tests. Standard-Single-Objective ist damit
queuefähig. Island-/Generic-Island-/NSGA-II-Worker, Ressourcenbudgets und ein langlebiger
Controller/Daemon bleiben gesperrt.

Neunzehnter Slice: Der NSGA-II-Vertrag blockiert den historischen Scheinmodus
`multi_objective`, unbekannte/duplizierte Objectives, ungültige Typen/Skalen und die bisher still
ignorierten Schlüssel `direction`/`weight`. `trade_frequency` wird als Legacy-Name explizit auf
`num_trades` aufgelöst; ein fehlendes konfiguriertes Objective scheitert statt als Null in den
Pareto-Vektor einzugehen. Generic-Island-Seed-Rotation schreibt jetzt den tatsächlich gelesenen
`random_seed`, und rekursive Insel-/Prozessfeatures können über `extra_config` nicht wieder
aktiviert werden. Surrogate-Mutation liest die verschachtelte GA-Mutationsrate und bewertet das
mutierte Genom neu; bei fehlender Prognose fällt es auf einen echten Backtest zurück. Unabhängige
Short-Signale besitzen in Generator und Mutation eine explizite Side-Semantik: Short-Entry ist
bearish, Short-Exit bullish, Aktivitätsfilter bleiben Entry-/Exit-basiert. Ein Scan über 461 YAMLs
findet 17 valide NSGA-Konfigurationen und 65 kanonische Objective-Einträge. 1.398 GA-Tests laufen
grün; die zwei bekannten LLM-Collection-Dateien bleiben separat ausgeschlossen. Surrogate und
Short bleiben unter `safe_v2` weiterhin deaktiviert, bis ihre OOS-Wirksamkeit belegt ist.

Zwanzigster Slice: Der echte NSGA-II-Lifecycle erzeugt nun λ Kinder, evaluiert jedes Kind über den
kanonischen sequentiellen oder parallelen Pfad und führt erst anschließend `(μ+λ)`-Environmental-
Selection aus. Der alte Aufruf in `GenerationStep.execute()` ist entfernt; dadurch können
unbewertete Kinder nicht mehr hinter einer vollen Elternpopulation verschwinden. Ein
Lifecycle-Test mutiert RSI-Perioden nach Generation 0, verlangt Objective-Vektoren vor Selection
und weist die genetisch geänderten Kinder unter den Survivors nach. Parallel-Tasks übergeben
`nsga2.min_trades` jetzt tatsächlich an den Worker. Walk-forward bleibt im Worker aktiviert und
nutzt damit exakt dieselbe Aggregations-, Low-Trade-, Failure-, Gap- und Worst-DD-Semantik wie der
sequentielle `FitnessEvaluator`; die frühere abweichende Top-K-Flat-Nachbewertung wird im aktiven
Pfad nicht mehr ausgeführt. Deferred Pair-Validation erneuert anschließend die Objective-Vektoren
aus den ersetzten Metriken und scheitert bei unvollständigem Vertrag statt stale Train-Objectives
weiterzuverwenden. Normal-, Min-Trade- und Fehlerfälle sind zwischen Parallel und Sequential
identisch getestet. Alle 17 NSGA-YAMLs bestehen weiterhin den Vertrag; 1.410 GA-Tests laufen grün.
Das belegt mechanische Korrektheit, noch keinen OOS-Vorteil, daher bleibt NSGA für automatische
V2-Waves gesperrt.

Einundzwanzigster Slice: SQLite ist nun auch für Abfrage, Monitoring und historische Migration die
einzige produktive Lifecycle-Quelle. `ExperimentCatalogV2` aggregiert Attempts deterministisch zu
Experimenten, liest erfolgreiche Resultate nur hash-verifiziert und kennzeichnet jeden Datensatz
als `CANONICAL_V2` oder `LEGACY_IMPORT`. Alte `registry.json`-Snapshots werden als vollständige,
immutable Versionen idempotent importiert; sie erzeugen niemals ausführbare Attempts. Derselbe
Snapshot wird auch bei 16 parallelen Importen genau einmal angelegt. Neue Snapshots behalten die
Historie, während gleichnamige kanonische Experimente stets Vorrang haben. Ein atomarer,
sortierter JSON-Export enthält State-Schemaversion, Attempts, Experimente und Importprovenienz.
CLI-Experimentabfragen, Monitor, Data-Lifecycle und Web-Runliste lesen diesen Katalog. `/proc`-,
Log- und Run-Verzeichnis-Discovery entscheiden keinen Lifecycle-Zustand mehr. Der operative
Scheduler reconciled stale Prozesse kontrolliert, während der Monitor nur dessen autorisierten
SQLite-Zustand und unverbindliche Log-Fortschrittsmetriken anzeigt. Kanonische Artefakte bleiben
von der alten altersbasierten Löschpolicy ausgenommen, bis ein eigener hashgebundener
Retention-/Tombstone-Vertrag existiert. Katalog-Score und -Return stammen stets von demselben
Candidate und Attempt; der angezeigte Return ist dessen schlechtestes Szenario, kein über
Pairs/Zeiten/Kosten cherry-gepicktes Maximum. 86 fokussierte Registry-, CLI-, Monitor-, Lifecycle- und
Webtests sowie 1.584 breitere Tests ohne die zwei bekannten `TEST-001`-LLM-Dateien laufen grün.
Der unbereinigte Lauf besteht 1.585 Tests und endet ausschließlich an deren sechs fehlenden
Pseudo-Fixtures.

Zweiundzwanzigster Slice: `GUARDED_SPAWN_RECOVERY_V1` schließt die nicht atomar lösbare Lücke
zwischen Betriebssystem-Spawn und SQLite-`RUNNING`. Jeder neue Claim trägt einen versionierten
Recovery-Vertrag. Vor dem Spawn werden Command, Logpfad und eine immutable
`SpawnGuardBindingV1` mit Nonce-Hash sowie kanonischen Ready-/Release-Pfaden persistiert. Der neue
Linux-Guard wird als eigene Session gestartet, armiert zuerst `PR_SET_PDEATHSIG`, prüft danach die
unveränderte Parent-PID und veröffentlicht seine PID plus Boot-ID/Startticks atomar. Das eigentliche
GA-Child startet er erst, nachdem PID/Starttoken als `RUNNING` committed wurden und ein passendes
Release-Receipt atomar sichtbar ist. Stirbt der Controller vorher, tötet der Kernel den Guard,
welcher die gesamte Prozessgruppe beendet; ein Lease-Timeout ohne Release startet ebenfalls kein
Child. Der Reconciler kann dadurch neue Claims vor Vorbereitung, ohne Ready-Receipt oder mit
nachweislich verschwundenem/PID-reused Guard sicher als `INTERRUPTED` schließen. Ein lebender oder
nicht prüfbarer Guard bleibt unangetastet. Migrierte Alt-Claims ohne Vertrag bleiben bewusst
`NO_ACTION_CLAIM_UNPROVEN`. Ein echter SIGKILL-Test trifft exakt das Fenster nach Guard-Ready und
vor `RUNNING`: Der Zielprozess erzeugt keinen Startmarker, der Guard verschwindet und der Claim wird
deterministisch reconciled. Schema v5→v6 erhält bestehende Attempts. 24 fokussierte State-/
Executor-Tests, 159 Runner-/Worker-/Scheduler-/CLI-/Web-nahe Tests und 1.591 breitere Tests ohne
die zwei bekannten `TEST-001`-LLM-Pseudotestdateien laufen grün.

Dreiundzwanzigster Slice: `AttemptOutputLayoutV2` ist Bestandteil der gehashten Evolution- und
Replay-Worker-Spec. Er legt Diagnostics, Tracker, HOF, Checkpoints, Cache, Engine-Config/-Log,
Plots, generierte Strategien, Freqtrade-Userdata/Backtest-Exports und separate Replay-Pfade
kanonisch unter genau einem Attempt-Root fest; ein abweichender oder herausführender Pfad verletzt
die Spec. Beide Worker entfernen während der Ausführung den nicht gebundenen
`GA_OUTPUT_DIR`-Alias und wechseln in ein attempt-lokales Arbeitsverzeichnis, sodass auch
übersehene relative Writes nicht mit parallelen Attempts kollidieren. Der reine Replay-Worker
benutzt nun ebenfalls eine isolierte `DirectBacktester`-Config. `run_ga.py` und der Regime-Island-
Runner lösen Output nur noch aus der Config auf; die Umgebungsvariable ist aus Produktionscode
entfernt. Ein echter Standard-GA-Minilauf mit vier Individuen und einer Generation schreibt
Diagnostics, Runs, HOF, Strategien sowie Freqtrade-Verzeichnisse unter den Attempt und verändert
keines der bekannten globalen Legacy-Verzeichnisse. Alle erzeugten Dateien sind anschließend im
terminalen Artifact-Hashmanifest enthalten.
1.441 GA-Tests laufen grün, wenn die jetzt drei bekannten `TEST-001`-LLM-
Pseudotestdateien ausgeschlossen werden. 150 zusätzliche CLI-, Orchestration-, Web-, Checkpoint-
und Paralleltests sind grün; die beiden Paralleltests benötigen lediglich die Berechtigung, einen
lokalen `forkserver`-Unix-Socket anzulegen.

Vierundzwanzigster Slice: `IslandFinalistBatch` ersetzt die untypisierte Island-Dict-Grenze.
`extract_island_finalists()` verlangt exakt alle deklarierten Island-Arme plus `__global__`,
flacht jede lokale Kandidatenliste in deterministischer Island-Reihenfolge ab und akzeptiert als
Finalisten ausschließlich deduplizierte, absteigend sortierte, gemessene Individuen desselben
`COMMON_REPLAY`-Panels. Jeder globale Phänotyp muss in mindestens einer lokalen Island-Liste
enthalten sein; Profit und Tradeanzahl müssen explizite finite Messwerte sein. Fehlende Arme,
Legacy-Listen, fremde Kandidaten, gemischte Panels, lokale Search-Fitness, doppelte Phänotypen,
Nullmetriken und leere Finalisten scheitern fail-closed. Regime- und Generic-Island erzeugen jetzt
auch für leere Populationen den vollständigen Key-Satz und validieren vor erfolgreicher Rückgabe.
`IslandCoordinator` liefert den typisierten Batch statt beliebige Backend-Daten; der alte
Reportpfad schreibt Registry-Fitness/-Profit nur noch aus dem bewiesenen besten Finalisten und
erzeugt keine erfundenen Nullwerte. 1.452 GA-Tests und 161 fokussierte Island-/Config-/Checkpoint-
Tests laufen grün. Das schafft noch keinen automatischen V2-Island-Worker; dieser bleibt als
eigene gesperrte Worker-Art bestehen.

Prioritäten:

- **P0:** blockiert vertrauenswürdige Messung oder sichere Automation
- **P1:** kann Auswahl/Robustheit wesentlich verfälschen
- **P2:** experimentelles Feature unwirksam, unklar oder technisch fehlerhaft
- **P3:** Wartbarkeit, Dokumentation oder Komfort

## Freigabe-Gates

- [ ] **GATE-MEASURE:** Fresh-, RAM- und Disk-Resultate sind identisch; Units, Pflichtmetriken und
  Phänotyphash sind deterministisch.
- [ ] **GATE-VALIDATE:** Train/Pair-Val/Temporal-Val/Final sind disjunkt; negative oder fehlende
  Validation kann nie promoted werden.
- [ ] **GATE-RUN:** Ein kanonischer Runner, atomarer State Store und attempt-isolierte Artefakte
  bestehen Kill/Restart/Paralleltests.
- [ ] **GATE-SHADOW:** Evaluation v2 und Next-Wave-Planner laufen read-only im Shadow-Modus und
  reproduzieren ihre Entscheidung.
- [ ] **GATE-AUTO:** Canary, Budgetlimit, Kill-Switch und mehrere vollständige Approval-Waves sind
  ohne Zustands-/Artefaktfehler abgeschlossen.

Keine Next-Wave-Vollautomation vor `GATE-MEASURE`, `GATE-VALIDATE` und `GATE-RUN`.

## P0: Messung und Resultatvertrag

- [x] **EVAL-001 – `BacktestResult` vollständig serialisieren.** Alle evaluationsrelevanten Felder
  inklusive Trades, Pair-/Monatsdaten, Trade-Returns, No-Trade, Verlustserie und DD-Dauer
  roundtrippen; große OHLCV-DataFrames bleiben bewusst außerhalb des Resultatcaches. Akzeptanz:
  fresh == RAM-cache == disk-cache in Daten und Fitness.
- [ ] **EVAL-002 – Cacheversion und Schemahash einführen.** Alte unvollständige Cacheeinträge
  invalidieren; Key enthält Backtester-, Metric-, Config-, Daten- und Kostenmodellversion.
  **Teilstand:** Cache-Schema v7 invalidiert Altformat und ist Teil des Keys; Code-, Daten- und
  Kostenmodellversionen müssen noch als explizite Manifest-/Keybestandteile angebunden werden.
- [x] **EVAL-003 – Metric-Mapping vervollständigen.** `avg_profit`, `avg_duration`, `trades`,
  Pair-/Monats- und Tailfelder korrekt aus `BacktestResult` übernehmen. Akzeptanz: Duration-,
  Clustering- und Spread-Penalty haben erreichbare Integrationstests.
- [ ] **EVAL-004 – Spread-Unitfehler beheben.** `avg_profit` nicht doppelt durch 100 teilen; alle
  Return-/Kostenfelder auf kanonische Dezimaleinheit umstellen.
  **Teilstand:** Der konkrete doppelte `/100`-Fehler ist repariert und getestet; die systemweite
  Einheitenmigration bleibt offen.
- [ ] **EVAL-005 – Missing ist nicht Null.** Pflichtmetriken als `null/unavailable` modellieren;
  fehlende DD-Dauer/Verlustserie/Pair-/Monatswerte dürfen keinen perfekten oder neutralen Score
  erzeugen.
  **Teilstand:** V2-Equity-/Risikofelder sowie Verlustserie/DD-Dauer verwenden `None`; fehlende
  Tail-, Monats- und Cross-Pair-Evidenz erhält keinen Robustheitscredit mehr. Die restlichen
  Legacy-Pflichtfelder und vollständige Scenario-Coverage bleiben zu migrieren.
- [ ] **EVAL-006 – Deterministisches Fee-/Slippage-Modell.** Kein Python-`hash()`, keine
  Generation/Individual-ID im Seed. Stabilen Phänotyphash und explizite Szenarioseeds nutzen.
  **Teilstand:** Prozessabhängiges `hash()` ist durch SHA-256 ersetzt und versteckte Fee-Noise ist
  standardmäßig aus; der Seed muss noch vom transienten Strategienamen auf Phänotyphash wechseln.
- [x] **EVAL-007 – Vollständigen ausführbaren Phänotyp hashen.** Transiente Gene-Felder entfernen,
  aber `max_open_trades`, Timeframe, Short-, informative und alle ausführbaren Parameter aufnehmen.
  Der Replay-Vertrag hasht exakten Strategiecode, literales Timeframe, `max_open_trades` und
  explizite externe Ausführungsparameter. Der Legacy-Gene-/HOF-Export generiert zweimal unabhängig,
  verweigert Reparatur/Mutation/Nichtdeterminismus und persistiert den Frozen Candidate.
- [ ] **EVAL-008 – Kanonisches `result.json` definieren.** Schema-, Policy-, Code-, Daten-,
  Config-, Phänotyp- und Seed-Provenienz sowie vollständige Szenariomatrix speichern.
  **Teilstand:** Strikte Pydantic-Verträge und atomare, immutable Persistenz mit Hash-Readback
  existieren. Shadow-Recorder und Replay-Runner persistieren echte `BacktestRecordV2` über eine
  explizite Scenario-Matrix sowie Code-, Daten- und Frozen-Candidate-Artefakte;
  CLI-/Scheduler-Integration fehlt.
- [ ] **EVAL-009 – Erfolg fail-closed definieren.** Exit 0 oder Logtext allein genügt nicht;
  terminaler Erfolg erfordert valides Resultat und Artifact-Manifest.
  **Teilstand:** Der V2-Artifact-Store akzeptiert ein terminales Resultat nur mit konsistentem
  Manifest, Candidate-/Scenario-Artefakten und vollständigen Hashes. Der Legacy-Scheduler prüft
  weiterhin nur Prozess/Registry und muss auf diesen Readback umgestellt werden.
- [ ] **EVAL-010 – Scores versionieren.** Raw Search Score, Gate-Status, RobustScore und
  Promotion-Status getrennt speichern; Scores verschiedener Versionen nicht direkt vergleichen.
  **Teilstand:** Candidate-Messstatus, einzelne Gate-Ergebnisse, diagnostischer RobustScore und
  Shadow-Promotion-Outcome sind getrennte versionierte Felder. Der Legacy-Search-Score ist noch
  nicht mit einer expliziten Policyversion im kanonischen Runner verbunden.
- [x] **EVAL-011 – Pair-Coverage im echten Suchpfad nicht kompensierbar machen.**
  Freqtrade-Trade-DataFrames und serialisierte Recordlisten liefern dieselben Per-Pair-Profite und
  -Tradezahlen. Train-/Validation-Fitness kann ein Null-Pair nicht mehr durch ein aktives anderes
  Split kompensieren; der schlechteste Multiplikator des gesamten Panels wird nach dem allgemeinen
  Penalty-Floor genau einmal auf den Composite-Score angewandt. Reale Finalist-Replays und
  Regressionstests belegen den aktiven Datenfluss.
- [ ] **EVAL-012 – Zero-Loss-Profit-Factor versioniert modellieren.** Freqtrades Backtestreport
  kodiert eine positive Stichprobe ohne Verlusttrade als Profit Factor `0.0`, obwohl der Quotient
  mathematisch rechtszensiert/unendlich ist. Das darf weder als schlechtester PF noch als beliebige
  riesige Zahl in Search oder V2 eingehen. Einen endlichen Cap plus explizites
  `profit_factor_censored`-Feld definieren, Cache-/Resultatschema versionieren und Rankings gegen
  gemischte sowie ausreichend große Samples testen.

## P0: Validierung und Evolutionslogik

- [ ] **VAL-001 – Holdout-Leak schließen.** Evolution tatsächlich auf `evolution_tr` begrenzen;
  Holdout-Range vor Start disjunkt prüfen und final nur einmal benutzen.
- [ ] **VAL-002 – Räumliche und zeitliche Felder trennen.** `pair_validation_*`, `temporal_val_*`
  und `final_test_*` statt Wiederverwendung von `holdout_*`/`train_val_gap`.
  **Teilstand:** Der V2-Szenariovertrag besitzt explizite Pair-/Temporal-/Final-Rollen und ein
  hash-persistiertes Split-Manifest; der Legacy-Evolutionspfad verwendet weiterhin alte Felder.
- [ ] **VAL-003 – Deferred Pair-Validation zweistufig machen.** Unvalidierte Kandidaten dürfen nicht
  mit validierten Kandidaten in derselben finalen Rangliste konkurrieren. Selektion schließen oder
  alle potentiell selektierbaren Kandidaten nachvalidieren.
- [ ] **VAL-004 – Pair-Coverage vollständig machen.** Pairs ohne Trades explizit als 0-Trade-Segment
  ausgeben; leere Pairliste darf nicht still auf Full Config zurückfallen.
- [ ] **VAL-005 – Pair-Rotation/LOPO implementieren.** Kein einzelnes XRP-Resultat als generelle
  Robustheit interpretieren; Pair-/Cluster-Folds vorab festlegen.
- [ ] **VAL-006 – Finales Pair-x-Time-Set reservieren.** Es darf weder Evolution noch wiederholte
  Wavesteuerung sehen; Nutzung wird im Manifest protokolliert.
  **Teilstand:** FINAL_TEST ist eine explizite Scenario-Rolle und ein Pflichtgate. Das
  transaktionale Usage-Ledger verhindert wiederholte oder überlappende Replay-Verwendung und
  persistiert ein Usage-Receipt. Der Split-Vertrag prüft Evolution/Pair-/Temporal-/Final-
  Disjunktheit im V2-Pfad. Die tatsächliche Legacy-Evolution ist noch nicht auf diesen Pfad
  umgestellt; außerdem fehlt die administrativ sichere Recovery unexponierter Crash-Reservierungen.
- [x] **CORE-001 – NSGA-II-Lifecycle reparieren.** Neue Offspring werden vollständig evaluiert,
  bevor `(μ+λ)` Environmental Selection Eltern und Kinder gemeinsam sortiert. Ein Lifecycle-Test
  erzwingt genetisch geänderte RSI-Perioden, prüft `evaluate -> select` und weist Kinder nach
  Generation 0 unter den Survivors nach.
- [x] **CORE-002 – Parallel/Sequential-Parität herstellen.** `nsga2_min_trades` wird bis in den
  Worker übergeben. Walk-forward läuft über denselben per-Candidate-`FitnessEvaluator`, wodurch
  Aggregation, Low-Trade-Credit, Fehler, Gap-Penalty und Worst-DD nicht mehr durch eine abweichende
  Top-K-Nachbewertung ersetzt werden. Ergebnis-, Min-Trade- und Fehlerfälle sind auf identische
  Fitness/Metriken/Objectives getestet; Deferred Validation erneuert stale Objective-Vektoren.
- [x] **CORE-003 – Regime-Ausfälle nicht herausfiltern.** Jedes erwartete Segment bleibt mit
  `failed`/`invalid_fitness`/`zero_trades`/`low_trades`/`ok`, Raw- und Effective-Fitness sowie
  vollständigen Coverage-Zählern im Ergebnis. Failed und ungültige Segmente erhalten effective 0;
  positiver Low-/Null-Trade-Score wird auf 0 gekappt, ein negativer Score bleibt negativ. Mean,
  Min und CVaR verwenden alle Segmente; Harmonic Mean fällt bei irgendeinem Wert <= 0 auf den
  Worst Case zurück und filtert nichts. Fehlende Evidenz setzt Worst-DD konservativ auf 1.0.
  `exclusive` darf nicht bevorzugte Regime nicht mehr entfernen, weil dieses Gene die Runtime-
  Handelslogik nicht einschränkt. Enabled ohne Segmente und ungültige Gewichte/Schwellen schlagen
  fail-closed fehl. Vertragstests und die komplette GA-Suite (1.427 grün) sichern dies ab.
- [x] **CORE-004 – Surrogate standardmäßig deaktivieren.** Kein Surrogate-Kandidat in HOF/Finale;
  Mutation/Reinsertion reparieren und echtes OOS-Gütegate verlangen.
  `safe_v2` und der globale Default lassen das Feature aus. Jede Fitness trägt nun persistierte
  Evidenz (`UNMEASURED`, `BACKTEST_VALID`, `BACKTEST_FAILED`, `SURROGATE_ESTIMATE` oder
  `LEGACY_UNKNOWN`). Nur erfolgreiche, finite Backtests dürfen Best-Run, HOF, Holdout-/CPCV-
  Kandidaten, finalen Runner-Output, V2-Evolution-Shortlist und Candidate-Export erreichen.
  Historische Surrogate-HOF-Einträge werden beim Laden verworfen; HOF-Reinjektionen werden erneut
  evaluiert. Mutation ersetzt jetzt das tatsächliche Populationsgenom, löscht stale Fitness,
  prognostiziert den mutierten Phänotyp erneut und fällt bei Prognosefehler auf einen echten
  Backtest zurück. Training nutzt deduplizierte, gemessene Raw-Fitness statt Fitness-Sharing und
  aktiviert den Filter erst nach einem konfigurierbaren R²-Gate auf vollständig späteren
  GA-Generationen (`min_validation_r2`, `min_validation_samples`). Checkpoint-Restore kann weder
  das nicht serialisierte Modell noch ein altes Gate wiederverwenden. 1.446 GA-Tests sind grün.
  Das beweist die Sicherheitsgrenzen, nicht positiven OOS-Nutzen; Surrogate bleibt bis zu
  gepaarten Blind-OOS-Budgettests experimentell und aus.
- [x] **CORE-005 – Spezialisten vergleichbar machen.** Raw Fitness aus unterschiedlichen Pair-/
  Regime-/Inselpanels nie global sortieren; gemeinsames Replay-Panel vor HOF/Promotion.
  Jede reale Evaluation erhält eine persistierte `panel_v1_*`-ID aus Pair-/Zeit-/Kosten-, Fitness-
  und Evaluatorvertrag sowie gegebenenfalls der exakten Regime-Segmentmenge. Seeds, Population und
  Suchbudget ändern diese ID nicht. Lokale Inselwerte bleiben Search-Signale und gelangen weder
  während der Evolution in die Shared-HOF noch als Fallback in `run_ga.py`. Generic-Island und
  Regime-Island wählen zunächst lokal Top-K, deduplizieren den ausführbaren Phänotyp und replayen
  alle globalen Finalisten auf genau einer Base-Pair- beziehungsweise neutralen Union-of-Regimes-
  Vergleichsmatrix. Erst dieses `COMMON_REPLAY` wird global gerankt und in eine panelgebundene HOF
  geschrieben. Spezialistenboni sind dort aus. Replay-Ausfall liefert keine globalen Resultate.
  Merge/External-Export wählen bei verschiedenen Panels round-robin nach lokalem Rang;
  Tournament migriert bidirektional und erzwingt Ziel-Reevaluation statt fremde Scores zu
  vergleichen. Ein Shared-Parallel-Evaluator wird bei verschiedenen Panels abgeschaltet, weil er
  Pair-Rotation zuvor mit der Base-Config ausführte. Ohne content-addressed Datenmanifest bleibt
  eine HOF run-lokal; nur verifizierter Manifest-Hash erlaubt Cross-Run-Vergleich. Der V2-
  Promotionspfad besitzt bereits sein hashgebundenes gemeinsames Replay-Panel. 1.457 GA-Tests und
  explizite Gegenbeispiele für invertierte lokale/globale Ränge sichern den Vertrag ab. Das
  beweist Vergleichbarkeit, nicht den OOS-Nutzen von Insel- oder Regimespezialisierung.

## P0: Runner, Queue und State

- [x] **ORCH-001 – Einen kanonischen Runner festlegen.** CLI, Legacy-`run_ga.py`, Scheduler und
  manuelle Starts delegieren an denselben `execute_attempt()`-Pfad.
  Standard-`safe_v2`-Starts werden vor Ausführung vollständig materialisiert und gehen über
  `runner_v2` → Exact-ID-Claim → `AttemptExecutorV2` → immutable Child-Spec. CLI,
  `run_ga.py`, Web-RunManager und beide Shell-Queue-Namen verwenden diesen Vertrag. Der gemeinsame
  `AttemptSchedulerV2` verarbeitet Replay und Standard-Evolution. Noch nicht kanonisch
  implementierte Island-, Generic-Island-, Resume-, Injection-, Pause- und Visualisierungsmodi
  werden explizit abgewiesen und starten keinen Legacy-Fallback.
- [x] **ORCH-002 – Doppelregistrierung entfernen.** Orchestrator erzeugt/claimt Attempt; Child
  registriert denselben Namen nicht erneut. Integrationstest startet wirklich einen Queueeintrag.
  Replay- und Evolution-Child registrieren keinen Zustand. Direkte Starts claimen atomar exakt
  ihre eigene Attempt-ID, selbst wenn ein älterer/höher priorisierter Queueeintrag existiert.
  Echte Subprozesstests decken Queue → Claim → Executor → Child → terminales Resultat ab.
- [x] **ORCH-003 – JSON-Registry als Source of Truth ersetzen.** SQLite/WAL mit atomarem Claim,
  erlaubten Statusübergängen und migrations-/exportfähigem Schema.
  Attempt-, Wave-, Experiment- und Decision-State liegen in SQLite/WAL. Der Katalog aggregiert
  kanonische Attempts für CLI, Monitor, Lifecycle und Web. Legacy-JSON wird ausschließlich als
  immutable, idempotente Historie importiert und erzeugt keine Attempts; JSON-Ausgabe ist ein
  atomarer, provenance-erhaltender Export. Gleichzeitige Imports und v4→v5-Migration sind getestet.
- [x] **ORCH-004 – Nebenläufigkeit testen.** Mindestens 60 parallele Updates ohne verlorene
  Einträge, kaputte Tempdateien oder leeres Reinitialisieren.
  Der V2-State-Store besteht 60 parallele Claims und 60 parallele Heartbeats ohne verlorene
  Attempts, Versionen oder Events.
- [x] **ORCH-005 – Shell-Slotzählung reparieren/ablösen.** Nicht nach `run_ga.py` suchen, wenn
  `python -m genetic_algorithm run` gestartet wird; langfristig aktive Leases statt Prozessregex.
  Der operative Daemon zählt alle `CLAIMED`-/`RUNNING`-Leases aus SQLite. `ga_auto_queue.sh` und
  `ga_auto_queue_v2.sh` sind nur noch sichere Shims für die Python-V2-Queue; `pgrep` und
  Prozessnamenslotzählung sind aus diesem Startpfad entfernt.
- [x] **ORCH-006 – `done`-Semantik beseitigen.** Vor Spawn verschobene Config ist `claimed`, nicht
  abgeschlossen; Retry-/Failure-/Interrupted-Zustand explizit führen.
  Die kanonische Queue verschiebt keine Config vor dem Spawn. Immutable Inputs werden erst als
  `QUEUED`, danach atomar als `CLAIMED` und erst nach verifiziertem Resultat terminal geführt.
- [x] **ORCH-007 – Recovery vollständig machen.** PID plus Prozess-Starttoken, Lease, Heartbeat,
  Logpfad und Startzeit persistieren; PID-Wiederverwendung und stale Runs erkennen.
  Claim-Token, Lease, Heartbeat, PID, Linux-Prozess-Starttoken, Command-Hash,
  Arbeitsverzeichnis und Runtime-Logpfad/-Hash sind persistiert. Dead-Process-/PID-Reuse-
  Reconciliation ist implementiert und real getestet. Ein versionierter Parent-Death-Guard startet
  das echte Child erst nach committed `RUNNING`; SIGKILL vor diesem Commit, Guard-Timeout,
  Guard-Liveness, PID-Reuse und migrierte Claims ohne Recovery-Vertrag sind getrennt getestet.
- [x] **ORCH-008 – Registry automatisch reconciliieren.** Vorhandenes valides Resultat schließt
  stale `running`; Monitor zeigt nicht nur `CRASHED`, sondern erzeugt einen kontrollierten Übergang.
  Normaler Abschluss und stale `RUNNING` lesen das Terminalresultat fail-closed.
  Lebende, unbekannte und PID-wiederverwendete Prozesse werden getrennt behandelt. Der
  gemeinsame operative Scheduler führt diese Reconciliation vor neuen Claims periodisch aus.
  Monitor und Export zeigen ausschließlich den danach autorisierten SQLite-Zustand; Logregexe
  ergänzen nur Fortschrittsfelder und dürfen keinen Lifecycle-Übergang behaupten.
- [x] **ORCH-009 – Attempt-isolierte Artefakte.** Diagnostics, Tracker, HOF, Checkpoints, Log und
  Resultat ausschließlich unter eindeutiger Attempt-ID. Keine globale `generation_stats.csv`.
  Der hashgebundene Output-Layout-Vertrag umfasst Evolution und Replay einschließlich
  DirectBacktester-Strategien, Backtest-Exports und Freqtrade-Userdata. Der reale Mini-E2E-Test
  vergleicht bekannte globale Pfade vor/nach der Evolution und prüft das vollständige
  Artifact-Hashmanifest. Nichtkanonische direkte Legacy-Ausführung besitzt definitionsgemäß keinen
  Attempt und bleibt außerhalb dieses Automationsvertrags; der Island-Reportpfad respektiert aber
  ebenfalls `output.dir`.
- [x] **ORCH-010 – `GA_OUTPUT_DIR` vereinheitlichen oder entfernen.** Jeder Engine-Typ muss denselben
  aufgelösten Attempt-Pfad verwenden; leere Wave-Outputverzeichnisse verhindern.
  Die Variable wird von keinem Produktionspfad mehr gelesen. V2-Worker entfernen einen geerbten
  Wert während Evolution und Replay, und Config-/Spec-Pfade bleiben die einzige Quelle.
- [x] **ORCH-011 – Island-Resultate korrekt extrahieren.** Dict-Ergebnisse auf alle Individuen
  flatten/replayen; keine falschen Registry-Nullwerte.
  Der zentrale Extraktionsvertrag prüft vollständige Arme, Typen, lokale Herkunft,
  Phänotyp-Deduplizierung, gemeinsames Replay-Panel, Rangfolge sowie explizite Profit-/Trade-
  Metriken. Beide Island-Backends, Coordinator und Legacy-Reporter verwenden ihn. Automation
  startet Island weiterhin nicht, solange kein eigener V2-Worker samt Ressourcen- und
  Artifact-Vertrag implementiert ist.
- [x] **ORCH-012 – Resume kompatibilitätsgeprüft machen.** Numerisch neuesten Checkpoint wählen;
  Checksumfehler blockiert. Config-/Code-/Daten-/Genome-Schema-Hash und Islandnamen müssen passen.
  Checkpoint V3 trennt resume-fähige Artefakte von Diagnose-/Dashboard-Snapshots. Resume verlangt
  eine verpflichtende kanonische Checksumme und exakte Provenienz für Engine, resolved Config,
  Code- und Datenmanifest, Genome-Schema sowie geordnete Islandnamen. Alte V1/V2-Dateien,
  Population- oder Island-Abweichungen und beschädigte Dateien blockieren. Generic Island wählt
  Generationen numerisch und fällt bei einem defekten neuesten Stand nicht auf einen älteren
  zurück. Der Standard-V2-Worker injiziert die immutable Provenienz; ein Scheduler-Handoff, der
  nach einem Prozessabbruch einen Checkpoint als neuen gehashten Attempt-Input materialisiert,
  ist damit noch nicht implementiert und darf nicht mit diesem Kompatibilitätsvertrag verwechselt
  werden.
- [x] **ORCH-013 – Warm-Start fail-closed.** Fehlende/ungültige Quelle blockiert, statt still von
  Scratch zu starten.
  Der Standard-V2-Worker akzeptiert ausschließlich hashgebundene Evolution Seeds, reproduziert vor
  Start den Executable-Phänotyp und injiziert das Genom vollständig. Auch der Legacy-
  `WarmStartLoader` verlangt nun pro Population-/HOF-Quelle einen expliziten Pfad, SHA-256 und
  `strategy-gene-v2`; globale „latest“-Suche ist entfernt. Dateiinhalt wird einmal gelesen, dann
  gehashte Bytes geparst. Fehlende, veränderte, leere oder beschädigte Quellen, ein einziges
  nichtkanonisches Genom, nichtfinite Fitness, ein fehlerhafter interner V3-Checkpoint oder
  unzureichende Population-Slots blockieren den ganzen Start. Historische Quellen sind nur über
  den unter `CFG-008` beschriebenen reproduzierbaren Migrationspfad zulässig.
- [x] **ORCH-014 – Log-Regex aus Entscheidungen entfernen.** `wave_comparison.py` darf nur
  strukturierte Resultate vergleichen; alle Metriken müssen demselben Kandidaten/Szenario gehören.
  Der Vergleich akzeptiert nur eine reconciled Wave mit exakt einer immutable ANALYSIS-Decision,
  prüft deren Snapshot-/Policy-/Payload-Hashkette und baut die Analyse erneut aus allen
  hashverifizierten Attempt-Artefakten auf. Nur eine bytekanonisch identische Analyse wird
  angezeigt oder als JSON/CSV exportiert. Logs liefern ausschließlich operativen
  Generations-/Evaluationsfortschritt und Fehlerzähler; wirtschaftliche Monitorwerte stammen nur
  aus `CANONICAL_V2`-Katalogzeilen. Der frühere Aggregate-Vergleich ist fail-closed stillgelegt.
  Der alte Benchmark-Bericht ist ausdrücklich nicht entscheidungsfähig; Rankings, automatische
  Schlussfolgerungen und Vergleichscharts sind entfernt. Spoof-, Missing-Analysis-, Resultat-
  Mutation- und selbstkonsistent gefälschte Decision-Tests belegen den Vertrag.
- [x] **ORCH-015 – Wave-/Decision-State fail-closed machen.** Exakte Experiment-/Attempt-Menge,
  immutable Specs/Decisions/Events und atomarer Parent-Snapshot. Jeder erwartete Attempt muss
  terminal sein und verifiziertes Resultat oder dokumentierten Abort besitzen; Resultatkorruption
  ist niemals durch Abort kompensierbar. Analysis, Proposal und Approval sind als Hashkette an ihre
  unveränderlichen Inputs gebunden.
- [x] **ORCH-016 – Reconciled Waves deterministisch analysieren.** Resultate vor Analyse erneut
  hash-verifizieren, Experimentgesundheit von Kandidateneignung trennen und pro
  Experiment-x-Phänotyp über eindeutige Seeds aggregieren. Fehlende Controls, technische Ausfälle,
  zu geringe Seed-Coverage und gescheiterte Gates dürfen nicht durch gute Teilmetriken kompensiert
  werden; die Analysis-Decision ist an Snapshot und Analyzer-Policy gehasht.
- [x] **ORCH-017 – Pareto-/Diversitätsauswahl ohne Gesamtscore.** Nur Kandidaten auf identischem
  Panel vergleichen, Return-/Expectancy-LCB maximieren, DD-/ES-UCB minimieren und deterministische
  Non-Dominated-Ränge bilden. Objective-Distanz und Per-Experiment-Limit steuern die Auswahl;
  Ineligible Candidates gelangen nie in den Pareto-Pool.
- [x] **ORCH-018 – Deterministischen Shadow-Child-Plan erzeugen.** Explizite Arm-Templates,
  gepaarte Seeds, unveränderten Control, typ-/pfadgebundene Factor-Deltas, Budget und
  deterministische IDs/Hashes persistierbar machen. Blockierte Auswahl erzeugt kein ausführbares
  Proposal; ein erlaubter Plan braucht vor Proposal zusätzlich eine vollständige Materialisierung.
- [x] **ORCH-019 – Frozen-Replay-Plan materialisieren und atomar übergeben.** Parent-Resultat und
  Frozen Candidate erneut hash-verifizieren; resolved Replay-Config, Code, Daten, Split, Manifest
  und Worker-Spec unveränderlich einfrieren. Proposal/Approval an Plan- und Materialisierungshash
  binden. Child-Wave, Experiment-Specs, Attempts, Worker-Bindings und Parent-Queue-Übergang in
  einer SQLite-Transaktion schreiben; Konflikte und Manipulation rollen vollständig zurück.
- [x] **ORCH-020 – Kanonischen V2-Evolutionsworker bauen.** Control, Replication, Exploit und
  Explore dürfen erst materialisiert/queued werden, wenn der GA-Child nicht selbst registriert,
  exakt ein Attempt-Root nutzt, Config/Seed/Warm-Start aus einer immutable Worker-Spec bezieht und
  ein V2-Resultat über denselben Executor-/Lease-/Recovery-Pfad committed.
  **Erledigt für Standard-GA/single-objective unter `safe_v2`:** Legacy-Fitness ist nur
  Search-Signal, Top-Genome werden als Executable plus `evolution_seed.json` eingefroren und über
  V2 replayed. Control/Replication/Exploit/Explore werden vollständig materialisiert und atomar
  übergeben. Generic Island besitzt zusätzlich einen eng begrenzten
  `automation_island_v2`-Worker mit Pair-Split, gemeinsamem Replay, Ressourcenlimits und
  search-only Folge-Waves. Classic Island und NSGA-II bleiben eigene, nicht unterstützte
  Automations-Worker-Arten.
- [x] **ORCH-021 – Restart nach fertigem Child-Resultat reparieren.** Ein Neustart zwischen
  Resultat-Commit und SQLite-Reconciliation darf legitime Outputartefakte nicht als unerwartete
  Pre-Execution-Dateien ablehnen. Der idempotente Bootstrap prüft weiterhin alle immutable
  Input-Hashes, überspringt beim reinen Wiederlesen aber die nur vor dem ersten Spawn sinnvolle
  Root-Leerheitsprüfung. Der echte Generic-Island-Probelauf hat diese Lücke reproduziert; ein
  Regressionstest hält die strikte Ausführungsprüfung und die restartfähige Inspektion getrennt.
- [x] **ORCH-022 – Such- und Replay-Candlegrenzen identisch machen.** Date-only Freqtrade-
  Timeranges nehmen den Stop-Zeitpunkt inklusive und luden in der Suchphase eine zusätzliche
  Kerze des exklusiven Endtags. Der V2-Worker leitet deshalb aus dem Split-Manifest einen
  sekundengenauen inklusiven Endzeitpunkt für die letzte erlaubte Candle ab. Ein Real-Daten-Probe
  bestätigt identische erste/letzte Candle für Suche und Replay.
- [x] **ORCH-023 – Shared-OHLCV unabhängig vom Attempt-CWD laden.** Relative Datapfade werden
  gegen den Repository-Root statt gegen das isolierte Evolution-Ausgabeverzeichnis aufgelöst.
  Mehrere Timeframes bleiben fail-safe ohne Shared Cache. Der reale 1h-Probezugriff findet damit
  alle vier gebundenen Pairdateien; dies ändert Performance, nicht Backtest-Semantik.
- [x] **ORCH-024 – Scratch-Control-Seeds pro Parent-Wave rotieren.** Der erste reale
  Fallback-Plan vermied korrekt einen Exploit ohne eligible Candidate, hätte aber Config und Seed
  des Parent-Controls wiederholt. Das wäre nach unverändertem Code ein deterministisches Duplikat
  ohne neue Evidenz. Der Planner leitet nun aus Parent-Wave-ID, Basisseed und Ordinal ein
  reproduzierbares 32-Bit-Seedpanel ab. Alle Arme derselben Child-Wave behalten identische paired
  Seeds; verschiedene Parent-Waves erhalten verschiedene Panels. Das redundante materialisierte
  Child des Probelaufs wurde durch den Kill-Switch nicht gestartet und bleibt als Audit-Evidenz
  erhalten.
- [x] **ORCH-025 – Diagnosekampagnen begrenzen und Analyseberichte persistieren.**
  Root-Seeds und maximales Wave-Budget sind typisierte V2-Configwerte. Der nächste reale Lauf
  nach dieser Implementierung verwendete Seed 2001 und stoppte nach der Root-Analyse, bevor ein
  Child entstand; der folgende begrenzte Root ist mit der neuen Identität Seed 3001 vorbereitet. Jede
  Controller-Analysis persistiert einen kompakten immutable JSON-/Markdown-Bericht sowie
  `LATEST`-Kopien. Resultate werden vor der Zusammenfassung erneut hashverifiziert; Strategiecode,
  Genome, Einzeltrades und Runtime-Logs sind explizit nicht Bestandteil des Reports.
- [x] **ORCH-026 – Attempt-Seed bis in explizite Generic Islands binden.**
  Der Manifest-Seed ersetzt im derived Engine-Config sowohl
  `genetic_algorithm.random_seed` als auch alle festen Preset-Island-Seeds; Island `n` erhält
  deterministisch `attempt_seed + n` im 32-Bit-Raum. Der Report vergleicht Manifest-, GA- und
  Island-Seeds und markiert einen abweichenden historischen Lauf. Guard-/Budget-Exitcode 2 gilt in
  der generierten User-Unit als erfolgreicher, nicht neu zu startender Abschluss.

## P0: Config-Vertrag

- [x] **CFG-001 – Unbekannte Keys verbieten.** Typisiertes Schema mit `extra=forbid` oder
  äquivalenter fail-closed Validierung; resolved Config als Artefakt.
  **Erledigt versioniert:** Schema V2 prüft unbekannte Dict-/Listenpfade, strikte Typen, Presets und
  programmatische Overrides zentral und fail-closed. Schema V1 bleibt absichtlich als
  Migrationspfad mit deterministischen Warnungen lesbar; historische Dateien werden nicht
  fälschlich als V2 deklariert. Resolved V2-Configs bleiben gehashte Attempt-Artefakte.
- [x] **CFG-002 – Ein Validierungspfad.** CLI `run`/`config validate`/`queue add`, `run_ga.py`,
  Standard-Engine und V2-Materializer verwenden denselben Resolver und dieselben fatalen Regeln.
  Der frühere Raw-YAML-Fallback und das Verschlucken des erweiterten Validator-Resultats sind
  entfernt. Der V2-Manifest-Builder verlangt zusätzlich Schema 2 plus vollständig aufgelöste
  Defaults; Evolution- und Replay-Worker validieren die persistierte Config nach der Hashprüfung
  erneut. Damit ist auch direkte API-Nutzung fail-closed.
- [x] **CFG-003 – Engine-Config-Laden vereinheitlichen.** Standard, Island und Generic Island laden
  jetzt alle Presets, Defaults, Typen und semantische Regeln über `config.schema.load_config`.
- [x] **CFG-004 – Falschen Wave41-NSGA-Modus korrigieren.** `multi_objective` ist nicht der aktive
  Wert; echte NSGA-Config nutzt `mode: nsga2` und die tatsächlich gelesene Objective-Sektion.
  Die historische Wave41-Quelldatei ist nicht mehr im Repository; der zentrale Validator
  verhindert eine Wiederholung mit einer konkreten Migrationsmeldung. Zwei aktive Benchmark-
  Configs verwenden nun `type`/`scale` statt wirkungsloser `direction`/`weight`. Unbekannte,
  doppelte oder zur Laufzeit fehlende Objectives scheitern fail-closed. `trade_frequency` und
  `trade_count` sind dokumentierte Legacy-Aliase für die tatsächlich gelieferte Metrik
  `num_trades`; eine echte zeitnormalisierte Frequenz bleibt Aufgabe von `METRIC-007`.
- [x] **CFG-005 – Generator-Key-Mismatches korrigieren.** `surrogate_model`/`surrogate`,
  Short-Selling-Pfad, Generic-Island-Regime/Timeframe-Spezialisierung und Seed-Rotation prüfen.
  Canonical Runtime-Keys für MC, Surrogate, Generic-Migration, Regime und Seed sind im Schema
  korrigiert; alte Namen werden als Migration gemeldet. Der Generic-Island-Builder übergibt
  rotierte Seeds über `genetic_algorithm.random_seed`, bewahrt aufgelöste Sibling-Config und
  erzwingt Recursion-/Process-Pool-Grenzen nach `extra_config`. Timeframe-/Regime-Varianten sind
  keine eigenen magischen Insel-Keys; sie werden als normale, tief gemergte Runtime-Config in
  `extra_config` ausgedrückt. Surrogate-Mutation rescoret das mutierte Genom und fällt bei
  Prognosefehler auf echten Backtest zurück. Short-Codegen, Threshold-Mutation, Condition-
  Mutation, Reassignment und Indicator-Replacement sind side-aware. Genome-Roundtrips erhalten
  Instance-IDs einschließlich CDL, MTF und Short verlustlos.
- [x] **CFG-008 – Legacy-Genome semantisch migrieren.** Checkpoint/HOF brauchen eine explizite
  Schema-Version und einen persistierten `MigrationReport`; `from_dict_exact()` garantiert
  strukturelle Verlustfreiheit, ersetzt aber noch keinen vollständigen semantischen Validator für
  bekannte Indikatoren, Operatoren, Parametertypen und Risk-Bounds.
  `strategy-gene-v2` validiert nun zusätzlich bekannte Indicator-/Parameterfamilien, Operator-
  Kompatibilität, endliche Schwellen, strikte `between`-Intervalle, Timeframe-/MTF-Relationen,
  ROI-/Stoploss-/Trailing-/Regime-/Self-Adaptive-Grenzen und Condition-Referenzen. Der versionierte
  Migrator verlangt die exakten Legacy-Bytes samt SHA-256, erlaubt nur nachweislich
  bedeutungserhaltende Defaults und eindeutige ID-Zuordnungen und schreibt pro Genom Quell-/Zielhash
  plus Änderungsliste in einen gehashten Report. Warm-Start liest zusätzlich die alte Quelle,
  reproduziert die Migration und verlangt identisches Ziel samt Report; Teilmigration,
  Mehrdeutigkeit und Manipulation blockieren. Native HOFs schreiben ab Version 3 die explizite
  Genome-Schemaversion; aktive Archive verweigern Legacy-Dateien und reparieren bei Injection
  keine Operatoren mehr zufällig. Die dabei entdeckte Gleichheitslücke für `between`-Grenzen ist
  im Producer repariert. Anleitung: `GENOME_MIGRATION_GUIDE.md`.
- [x] **CFG-006 – Config-Snapshot vor Start.** Mutable Dateien unter `config/done` dürfen keine
  Reproduktion darstellen; Hash und exakte resolved Kopie pro Attempt speichern.
  Der kanonische V2-Runner löst die Quelldatei genau einmal auf und schreibt vor `VALIDATED` und
  `QUEUED` eine Attempt-lokale, immutable `resolved_config.yaml`. Ihr semantischer SHA-256 steht
  im Manifest; ihr exakter Datei-SHA-256 steht in der geschlossenen Worker-Inputmatrix und damit
  indirekt auch im unveränderlichen Worker-Binding. Replay- und Evolutionsworker laden
  ausschließlich diese Kopie, prüfen Datei-Hash, Config-Hash, Schema V2 und Promotion-Policy
  erneut. Der Child-Wave-Materializer wiederholt denselben Readback unmittelbar vor dem atomaren
  Queue-Handoff. Ein Regressionstest verändert nach dem Queueing gezielt eine Quelldatei unter
  `config/done`: der geladene Run bleibt byte- und semantikgebunden an den eingefrorenen Snapshot;
  eine Veränderung des Snapshots selbst endet dagegen fail-closed als `INVALID_RESULT`.
- [x] **CFG-007 – Validator-Regeln konsolidieren.** Widersprüchliche Empfehlungen für Population,
  Inselgröße und Pairzahl aus Code, `.github`-Hinweisen und Docs entfernen; nur nachgewiesene Gates
  erzwingen.
  `ga-config-invariants-v1` trennt jetzt mechanische Ausführbarkeit von wirtschaftlichen
  Hypothesen. Fatal sind unter anderem unmögliche Population-/Elite-/Tournament-/Immigrant-Slots,
  unbekannte Dispatchwerte, nichtfinite Wahrscheinlichkeiten, doppelte Pairs, ungültige
  Timeranges, Workerzahlen und Migrationen sowie Classic-Island+WF, weil die Engine WF dort
  tatsächlich verwirft. Es gibt bewusst keine magischen Grenzen 15, 6/60 oder zwei Pairs.
  Historische AP-Warnungen wurden aus dem Runtimepfad entfernt; Config-Resolver, Compatibility-
  Validator und Editor-Hook verwenden denselben Vertrag. `.github`-Prompts/Instructions sowie
  Validator-, Generic-Island-, Walk-forward-, Feature- und Config-Referenzdokumente behandeln
  Tuningwerte nun als gepaarte V2-Hypothesen. Der minimale Islandwert 2 deckte einen echten
  Builderfehler auf: `elite_size=2` war unmöglich; abgeleitete Elite-/Immigrant-Slots sind nun für
  jede erlaubte Population konsistent.

## P1: Evaluation v2

- [x] **METRIC-001 – tägliche kapitalgewichtete Equity erzeugen.** Alle Portfolio-Risikometriken
  aus derselben Netto-Serie berechnen.
  **Erledigt:** Tages-PnL wird über den vollständigen Kalender mit prior-day wallet weighting in
  Realized-close-Equity umgerechnet. Der MTM-Order-Ledger deckt Long/Short, Hebel, Multi-Entry/DCA
  und Teilexits ab, reconciled jeden Trade und die End-Equity und wurde gegen 1.000 reale
  Freqtrade-Trade-Artefakte geprüft. Funding wird bei jeder Backtest-Aktualisierung als
  zeitgestempeltes Wallet-Delta exportiert und sowohl synthetisch als auch mit realen Futures-DCA-
  Läufen (50 Entries mit Detail-Timeframe und 6 Entries ohne Detail-Timeframe) vollständig
  reconciled. Cache-Schema 7 verhindert, dass alte Futures-Artefakte ohne Funding-Ledger oder
  explizite Risiko-/Expectancy-Metrikverträge als neue Evidenz gelesen werden. Extern gelieferte offene
  Endpositionen ohne terminalen Snapshot bleiben
  erwartungsgemäß fail-closed; Freqtrade-Backtests erzwingen vor dem Resultat einen End-Fill.
- [x] **METRIC-002 – Sharpe/Sortino/Calmar referenztesten.** Kalenderfrequenz, Risk-free-Annahme,
  Downside-Semantik und Annualisierung dokumentieren.
  **Erledigt:** `calendar-effective-v1` definiert vollständige 365-Tage-Kalenderreihen,
  geometrische Umrechnung einer explizit konfigurierten effektiven Jahresrate, Sample-Volatilität,
  Full-Sample-Lower-Partial-Moment für Sortino, geometrische CAGR und Peak-to-Trough-Calmar.
  Unabhängige Handreferenzen decken positive/negative Returns, Risk-free-Umrechnung, Totalverlust,
  Zero-Variance/No-Downside und ungültige Inputs ab. Undefinierte Quotienten bleiben `null`; der
  frühere Ensemble-Sortino 10 und günstige Null-Sentinels sind entfernt. Vertrag und Annahmen
  werden in Cache-/V2-Artefakten persistiert. Umschalten der Search-Fitness erfolgt weiterhin erst
  nach Replay-Kalibration.
- [x] **METRIC-003 – Net Expectancy mit LCB.** Pro Trade und gebundenem Kapital; Gebühren, Slippage,
  Spread und Funding bereits enthalten.
  **Erledigt:** `net-expectancy-clustered-v1` reconciled vollständige Netto-Trade-Ledger gegen
  Backtester-PnL und Wallet. Er berichtet gleichgewichtete Trade-Expectancy und Return auf maximal
  gebundene Margin jeweils mit deterministischer LCB. Feste UTC-Schlusszeitcluster halten
  gleichzeitige Pair-Schocks zusammen; Moving Blocks erhalten ihre zeitliche Abhängigkeit.
  Pair-Kapital-HHI, effektive Pairzahl, maximale Trade-/Clusterkonzentration und das Minimum aus
  serieller, Trade-Kapital- und Zeitcluster-Kapital-`N_eff` sind Teil des Resultatvertrags.
  Widersprüchliche Felder sind `INVALID`, zu wenig Trade-/Clusterevidenz `INCONCLUSIVE`; Promotion
  verwendet den schlechteren der beiden Expectancy-LCBs. Gebundene Margin wird nicht als
  Stop-Loss-Kapital-at-Risk ausgegeben; letzteres bleibt bis zu einem auditierten Stop-Ledger offen.
- [ ] **METRIC-004 – Win/Loss-Paket.** Wilson/Bayes-Winrate, Payoff Ratio, avg/median win/loss,
  geglätteter PF, Tail-Loss und Verlustserie gemeinsam ausgeben.
- [ ] **METRIC-005 – PF-Grenzfälle definieren.** No-Loss, One-Trade, All-Winner, All-Loser und
  No-Trade mit Glättung/Cap und Konfidenz behandeln.
- [ ] **METRIC-006 – Downside-Paket.** DD-UCB, DD-Dauer, Time Under Water, Ulcer Index, Daily ES,
  Exposure, Konzentration und Ruinwahrscheinlichkeit.
  **Teilstand:** Punktwerte für DD, Dauer, Time Under Water, Ulcer und ES5 sowie Bootstrap-UCB für
  DD/ES sind vorhanden. MTM unterstützt Long/Short, Hebel und Positionsanpassungen; Exposure,
  Konzentration und Ruin fehlen.
- [ ] **METRIC-007 – Aktivität als Rate/Konfidenz.** Trades pro aktivem Monat/Pair, `N_eff`,
  Haltedauer, Turnover und Capacity; kein Gaussian-Bonus auf absolute Tradezahl.
  **Teilstand:** Der Expectancy-Vertrag liefert konservative serielle, Trade-Kapital- und
  Zeitcluster-Kapital-`N_eff`-Komponenten sowie Pairkonzentration. Offene Exposure-Überlappung,
  Haltedauer, Turnover, Capacity und Profilgates fehlen. Der kanonische Replay-Runner isoliert
  derzeit ein Pair je Szenario; pairübergreifendes gemeinsames Resampling greift deshalb nur bei
  echten Multi-Pair-/Portfolio-Ledgern, noch nicht als gepoolte Candidate-Panel-Metrik. Eine solche
  Aggregation benötigt zuerst einen kapital- und exposurekonsistenten Portfoliovertrag, damit
  getrennte Backtest-Wallets nicht fälschlich als unabhängiges Parallelkapital addiert werden.
- [ ] **METRIC-008 – Profile einführen.** Daytrade, Intraday, Swing und Mixed mit getrennten
  Aktivitäts-/Kosten-/Haltedauer-Gates, aber gemeinsamen Profit-/Risikodefinitionen.
- [ ] **METRIC-009 – Search Score von Promotion trennen.** Harte Ökonomie-/Risikogates, danach
  Pareto bzw. versionierter glatter Suchscore.
  **Teilstand:** Nicht kompensierbare Shadow-Gates sind getrennt vom diagnostischen RobustScore;
  ein Candidate kann valide gemessen sein und dennoch `FAIL` erhalten. Empirische Schwellen-
  kalibrierung und Pareto-Ranking über Candidates fehlen.
- [ ] **METRIC-010 – Doppelte Belohnung entfernen.** Profit/Sharpe/Sortino/PF nicht zugleich in
  Summe und Bonus mehrfach verstärken, sofern kein Kalibrationsnachweis vorliegt.
- [ ] **METRIC-011 – Penalty-Floor entfernen oder begründen.** Mehrere harte Verletzungen dürfen
  nicht pauschal 10 % der Ausgangsfitness retten.
- [ ] **METRIC-012 – Monats-/Pairreturns kapitalgewichten und datieren.** Keine bloßen Summen von
  Trade-Ratios und keine Korrelation undatierter Monatslisten nach Indexposition.
  **Teilstand:** Portfolio-Monatsreturns werden nun geometrisch aus der täglichen Wallet-Equity
  erzeugt und mit `YYYY-MM` persistiert. Ensemble-Portfolios verlangen identische lückenlose
  Monatsachsen, statt fehlende Monate als null zu imputieren. Kapitalgewichtete und datierte
  Pairreturns sowie Pair-Coverage bleiben offen.

## P1: Robustheitsfeatures

- [ ] **ROB-001 – Aktuelles Prozent-Holdout bis Fix deaktivieren.** Historische Werte als
  `in_sample/legacy` markieren, nicht als OOS.
- [ ] **ROB-002 – WF korrekt benennen.** Aktuell „rolling temporal validation“, nicht Walk-forward-
  Reoptimierung. Nicht überlappende Fenster und budgetgleiche Parallel-/Sequential-Semantik.
- [ ] **ROB-003 – `max_windows`-Auswahl festlegen.** Bewusst neueste/repräsentative Fenster statt
  unbeabsichtigt älteste; Purge/Embargo und Overlap prüfen.
- [ ] **ROB-004 – Monte Carlo aus Fitness entfernen.** Trade-Shuffle-Profit löschen; Block-Bootstrap
  für DD/ES/Ruin plus nur adverse Kostenstress-Szenarien neu implementieren.
  **Teilstand:** deterministischer Moving-Block-Bootstrap auf täglichen Returns liefert Return-LCB,
  DD-UCB und ES-UCB ausschließlich im Shadow-Resultat. Alte Trade-Shuffle-Fitness ist noch zu entfernen.
- [ ] **ROB-005 – CPCV umbenennen oder vollständig implementieren.** Purge/Embargo tatsächlich
  anwenden, Survivor Bias und Doppelberechnung entfernen; bis dahin nur Block-Rank-Diagnostik.
- [ ] **ROB-006 – DSR deaktivieren/referenzieren.** Daily Returns, effektive Beobachtungen, korrekte
  Kurtosisformel, Trialuniversum und deterministische Reihenfolge.
  **Bestätigter Blocker:** Der Legacy-Helfer annualisiert unregelmäßige Trade-Returns pauschal mit
  `sqrt(252)`, während der Runtime-DSR einen kalenderannualisierten Freqtrade-Sharpe mit
  `n_returns = num_trades` kombiniert. Beobachtungsfrequenz und Stichprobenumfang sind damit nicht
  derselbe Zufallsprozess. `safe_v2` verbietet DSR weiterhin; eine Reaktivierung erfordert tägliche
  V2-Returns, identische Sample-Semantik und eigene Referenz-/Kalibrationstests.
- [ ] **ROB-007 – Param-Sensitivity neu bauen.** Echten Produktions-Caller, eindeutige Varianten,
  Bool/Int/Zero/Bounds/Relationen und Interaktionen; fixe Kosten-/DSR-Zustände.
- [ ] **ROB-008 – Robustheitsfeatures nur bei inkrementellem OOS-Nutzen aktivieren.** Vorab
  Kandidaten/Thresholds festlegen und Rank-/Calibration-Gewinn pro Rechenzeit messen.

## P1: SIS

- [ ] **SIS-001 – SIS standardmäßig deaktiviert lassen.** Keine automatische Gewichtung,
  Immigration oder Promotion vor gepaartem Wirksamkeitsnachweis.
- [ ] **SIS-002 – Config-Keys vereinheitlichen.** `max_immigrants_fraction` und
  `immigrants_per_gen`; unbekannte Aliase müssen Fehler sein.
- [ ] **SIS-003 – Tatsächliche Immigration korrekt loggen.** Providerzahl, Core-Cap und wirklich
  injizierte Zahl getrennt; SIS nicht still durch `random_immigrants` kappen.
- [ ] **SIS-004 – `full profile` strukturell gültig machen.** Timeframe-Constraints respektieren,
  Indicator-Parameter beim Typwechsel neu erzeugen, Long/Short-Conditions und Risk Bounds prüfen.
- [ ] **SIS-005 – Ungültige bestehende Artefakte markieren.** 4h-HOFs mit 15m/1h sowie
  BBANDS/Ichimoku- und CCI/MACD-Parameterkombinationen nicht wiederverwenden.
- [ ] **SIS-006 – Credit kausal evaluieren.** Wiederholte Elites, gemeinsame Gene-Credits und
  selbstverstärkende Online-Retraining-Schleife beseitigen.
- [ ] **SIS-007 – Predictor-Güte fail-closed.** Echtes zeit-/wave-geblocktes OOS, Mindestgüte je
  Ziel, Calibration und Nutzenmetrik; negatives R² deaktiviert das Modell.
- [ ] **SIS-008 – Health als Nutzwirkung definieren.** Keine hohe Health nur wegen vorhandener
  Spalten/Cluster; OOS-Güte, Datenprovenienz und A/B-Effekt messen.
- [ ] **SIS-009 – A/B-Framework paaren.** Identische Seeds/Budgets, vollständige Arme, paired CI/
  Test und blindes OOS statt Train-Bestfitness.
- [ ] **SIS-010 – SIS-Factorial ausführen.** Control, Seed-Filter, Immigrants, Weights, Full mit
  mindestens zehn gepaarten Seeds auf gültigem identischem Suchraum.

## P2: Feature- und Suchraumforschung

- [ ] **EXP-001 – Indicator-Prior-A/B.** Uniform gegen milde Upweights für MACD/ROC/CDL_HAMMER/
  BBANDS und Downweights für PSAR/KAMA/TEMA/VROC; kein Hard-Ban aus Beobachtungsdaten.
- [ ] **EXP-002 – CCI-Konfundierung testen.** Globales Enrichment gegen within-run negatives Delta;
  statisches 3x-Gewicht bis dahin entfernen.
- [ ] **EXP-003 – Stoploss/ROI kontrolliert testen.** Keine Empfehlung aus schwachen globalen
  Korrelationen; Factorial über Seeds, Pair-/Zeitpanels und Kostenstress.
- [ ] **EXP-004 – Timeframe-Vergleich kontrollieren.** Identische Datenhorizonte, Pairs, Budget,
  Kosten und Policy; Corpus-Rohquoten nicht als Beweis nutzen.
- [ ] **EXP-005 – MAP-Elites-Felder korrigieren.** Nur kanonische Metriknamen verwenden und
  behavior dimensions mit Integrationstests belegen.
- [ ] **EXP-006 – AOS verdrahten oder entfernen.** `select_mutation()` im echten Pfad nutzen und
  Mutation-/Crossover-Credit kausal trennen.
- [ ] **EXP-007 – Regime-Aware redesignen.** Vollständige Segmentcoverage, gemeinsames Replay und
  belegte zusätzliche OOS-Prognose; vorher deaktiviert.
- [ ] **EXP-008 – Feature Importance kontextstratifizieren.** Wave, Pairset, Timerange, Timeframe,
  Scoreversion und Suchwahrscheinlichkeit kontrollieren.

## P2: Datenqualität und Analyse

- [ ] **DATA-001 – Corpus-Provenienz backfillen.** Wave, Origin, Train-/Val-Pairs, Timerange,
  Kostenmodell, Scoreversion, Config-/Datenhash.
- [ ] **DATA-002 – Wiederholungsmessungen behalten.** Phänotyp nicht global auf einen
  alphabetisch ersten Kontext deduplizieren; Kontext und Replicate als eigene Zeilen.
- [ ] **DATA-003 – Pair- und Temporal-„holdout“ trennen.** Legacy-Mischfelder migrieren oder klar
  als unbekannt markieren.
- [ ] **DATA-004 – Leere abgeleitete Spalten beheben.** `positive_months_ratio`, `pair_profit_std`,
  `monthly_return_std` korrekt füllen oder aus Modellfeatures entfernen.
- [ ] **DATA-005 – Detaillierte Reports deduplizieren.** Toplisten nach Phänotyp/Attempt eindeutig;
  keine mehrfachen IDs als verschiedene Ränge.
- [ ] **DATA-006 – `SAFE` umdefinieren.** Overfit-Degradation getrennt von ökonomischer
  `promotion_status`; negative Primary-/Validation-Ökonomie nie „deployable safe“.
- [ ] **DATA-007 – Legacy-Registrywerte klassifizieren.** Island-Nullen, stale Running-Zustände und
  fehlende Pfade markieren; nicht still als echte Resultate analysieren.
  **Teilstand:** Importierte Datensätze tragen zwingend `LEGACY_IMPORT`, unbekannte Zustände werden
  `unknown`, ungültige optionale Zahlen/PIDs/Pfade werden nicht zu kanonischen Messwerten
  umgedeutet und Altfehler erhalten `LEGACY_IMPORT_ONLY`. Eine fachliche Klassifikation der
  Island-Nullen und historisch stale `running`-Einträge bleibt offen.
- [ ] **DATA-008 – Attempt-ID in Tracker/HOF.** Wiederholungen dürfen Config/Generationsdateien nicht
  überschreiben oder Events an alte Läufe anhängen.
- [ ] **DATA-009 – Zehn historische Island+WF-Experimente reklassifizieren.**
  Der vollständige Config-Scan fand zehn als Island+Walk-forward bezeichnete Dateien in
  `config/done` beziehungsweise `config/exploration`. Die Classic-Island-Engine überschreibt WF
  jedoch vor dem Sub-GA-Start mit `enabled: false`; diese Runs sind daher keine WF-Evidenz und
  dürfen in Corpus, Rankings oder Parameterempfehlungen nicht als solche gelten. Quelldateien
  bleiben als historische Evidenz unverändert; ihre abgeleiteten Datensätze benötigen einen
  `FEATURE_NOT_EXECUTED`-/`INVALID_EXPERIMENT_DESIGN`-Status.

## P2: Runtime- und Testlücken

- [ ] **TOOL-001 – Phase-1-Testlauncher auf V2 migrieren.**
  `scripts/run_phase1_tests.py` ruft `run_ga.py` weiterhin mit dem im kanonischen V2-Entry-Point
  absichtlich gesperrten `--no-interactive` auf und erwartet frei gewählte Legacy-Outputordner.
  Damit kann das Tool keine gültige neue Messkampagne starten. Es muss Attempts über
  `runner_v2`/SQLite materialisieren, deren Artifact-Roots aus dem Katalog lesen und seine
  Vergleichsmetadaten an verifizierte Resultate statt Prozess-Exitcodes binden.
- [ ] **TOOL-002 – Historische Wave44/45-Generatoren stilllegen oder auf V2 portieren.**
  `scripts/generate_wave44_tracks.py` und `scripts/generate_wave45_tracks.py` erzeugen mutable
  Queue-YAMLs außerhalb des Planner-/Approval-Vertrags. Sie binden weder Config-, Daten- noch
  Quellartefakt-Hashes; Wave45 schreibt zusätzlich ein unversioniertes Population-Artefakt und
  übernimmt HOF-Metriken ohne Candidate-/Panel-Provenienz in einen kompensierenden Z-Score.
  Der Genome-Scan fand in dessen Nachfolgeartefakt unter anderem ungültige leere
  `between`-Intervalle. Der neue Config-, Warm-Start- und Genome-Vertrag blockiert diese Outputs
  deshalb zu Recht. Falls die Skripte behalten werden, dürfen sie nur noch einen
  `ChildWavePlanV2`-Vorschlag erzeugen; Migration, Pareto-Auswahl, Materialisierung und Queueing
  müssen über die kanonischen V2-Komponenten laufen.
- [ ] **TEST-001 – Pytest-Collection reparieren.** Manuelle LLM-Providerfunktionen umbenennen,
  markieren oder echte Fixtures bereitstellen; Gesamtsuite muss Exit 0 liefern. Betroffen sind
  mindestens `test_llm_modules.py`, `test_llm_providers.py` und
  `test_llm_strategy_generation.py`; die dritte Datei wurde beim ORCH-009-E2E-Lauf zusätzlich
  bestätigt und erzeugt allein vier fehlende `provider_config`-Fixturefehler.
  **Aktueller Nachweis 23.07.2026:** 1.625 Tests bestehen; die Suite endet ausschließlich wegen
  sechs Collection-/Setupfehlern in `test_llm_providers.py` (zwei) und
  `test_llm_strategy_generation.py` (vier) mit Exit 1.
- [x] **TEST-002 – Orchestration-Isolation reparieren.** `DataLifecycle(data_root=tmp)` darf keine
  realen Projektlogs scannen oder löschen; beide gezielten Fehler beheben.
  Projektlogs werden nur beim echten Default-Root einbezogen. Isolierte Report-/Cleanup-Tests und
  ein Schutztest gegen Legacy-Retention auf kanonischen Artefakten laufen grün. Generic-Island-
  Tests erhalten außerdem einen eigenen temporären HOF und lesen nicht mehr still das reale
  globale Projektarchiv.
- [ ] **TEST-003 – SIS-Restart-Semantik klären.** Test erwartet Restart in Gen 3, Code guardet bis
  Gen >= 5; gewünschtes Verhalten festlegen und integrieren.
- [ ] **TEST-004 – Cache-/Determinismus-Invarianten ergänzen.** Siehe
  `GA_EVALUATION_SPEC.md`, Abschnitt 10.
- [ ] **TEST-005 – Reachability für Undefined Names.** `StrategyGene` in Coevolution, `Population`
  in Incremental und `pd` im Island-Modell gezielt ausführen und beheben. **Teilstand:** Die
  literal-`\n`-Zeile, die `method` und `detector` im aktiven Regimepfad auskommentierte, der
  `DatasetPolicy`-Typimport und der fehlende Modul-`logger` in `run_ga.py` sind repariert. Ein
  Reachability-Test durchläuft Detector-Konstruktion, Klassifikation, Balancing und Split; ein
  echter Daten-/Backtest-E2E-Test fehlt noch.
- [ ] **TEST-006 – Doppelte Benchmark-Mapping-Keys beseitigen.** Reportvarianten eindeutig machen.
- [ ] **TEST-007 – Full Runner E2E.** Standard, Island und Generic Island je als Mini-Attempt durch
  kanonischen Runner; valides Resultat, Registry und isolierte Artefakte prüfen.

## P3: Dokumentation und Bereinigung

- [ ] **DOC-001 – Docs-Index korrigieren.** Nicht vorhandene `plans/`-Dokumente entfernen oder
  neu verlinken; Featurestatus nicht pauschal als „Complete“ ausgeben.
- [ ] **DOC-002 – Architekturdrift dokumentieren.** `core` enthält neben Shims weiterhin große aktive
  Implementierungen; tatsächliche Owner und Imports festlegen.
- [ ] **DOC-003 – Config-Referenz aus Schema generieren.** Defaultwerte und aktive Keypfade dürfen
  nicht zwischen Docs, `.github`-Hinweisen, Validator und Code divergieren.
- [ ] **DOC-004 – Deprecated Features markieren.** Holdout, WF, MC, CPCV, DSR, Surrogate,
  Regime-Aware, MAP-Elites, AOS und SIS mit aktuellem Gate/Status versehen.
- [ ] **CLEAN-001 – Legacy-Launcher erst nach Migration stilllegen.** Shell-Skripte nicht mitten in
  laufenden Waves entfernen; nach erfolgreichem Adapter-/State-Store-Rollout read-only archivieren.

## Empfohlene konkrete Arbeitsreihenfolge

1. `EVAL-001` bis `EVAL-010` plus `TEST-004`.
2. `VAL-001` bis `VAL-006`, `CORE-001` bis `CORE-003`.
3. `ORCH-001` bis `ORCH-020` und `CFG-001` bis `CFG-007`.
4. `METRIC-001` bis `METRIC-012` im Shadow-Replay.
5. `ROB-*` und `SIS-*` nur als abgeschottete Experimentarme.
6. Shadow-Planner nach [NEXT_WAVE_AUTOMATION_PLAN.md](NEXT_WAVE_AUTOMATION_PLAN.md).
7. Erst nach allen Freigabe-Gates Approval- und später Guarded-Auto-Modus.
