# Betriebs-Runbook: Guarded Generic-Island-Automation V2

Stand: 27.07.2026

Dieser Pfad automatisiert ausschließlich die **Strategie-Suche im Shadow-Modus**. Er deployt keine
Strategie und autorisiert kein Live-/Paper-Trading. Das Profil
`automation_island_v2` erzwingt Generic Island, vollständige Pair-Validation für jeden Kandidaten,
Mark-to-Market-V2 und deaktiviert unter anderem WF, Holdout, MC, Surrogate, SIS, LLM, NSGA-II,
Regime- und Classic-Island-Pfade.

## Vor dem Start

```bash
.venv/bin/python -m genetic_algorithm config validate \
  genetic_algorithm/config/presets/automation_island_week_v2.yaml --strict
```

Der mitgelieferte Datensatzvertrag erwartet aktuell BTC/SOL als Training und ETH/BNB als räumliche
Validation auf 1h für `20230325-20260327`. Fehlende Dateien, Candle-Lücken, abweichende Hashes oder
eine veränderte Config blockieren vor Ausführung.

Im Profil `automation_island_v2` wird jedes Pair bereits während der Suche
isoliert backgetestet. Gemeinsame Multi-Pair-Portfolios sind untersagt, weil
Slotkonkurrenz bei kleinem `max_open_trades` verlierende Pairs verdecken kann.
Die Such-Fitness verwendet dieselbe Aktivitätseinheit wie das Gate: mindestens
5 Trades pro aktivem Monat auf dem schwächsten Pair. Drawdown-Dauer behält
auch jenseits der Gategrenze einen monotonen Suchgradienten.
Vor einem unbeaufsichtigten Wochenlauf müssen zusätzlich alle Punkte in
[WEEK_RUN_READINESS_V2.md](WEEK_RUN_READINESS_V2.md) erfüllt sein.

Der read-only Produktions-Preflight prüft zusätzlich Venv, Daten-/Split-/Codehash, freien
Plattenplatz und verfügbaren RAM:

```bash
.venv/bin/python -m genetic_algorithm automation preflight automation_island_week_v2
```

Nur `ready: true` mit `PREFLIGHT_READY` ist startfähig. Ein vorhandener
Bootstrap-Intent einer älteren Kampagne blockiert bei abweichender Config,
Policy oder Seed. Die ältere Kampagne muss zuerst recoverable archiviert und
aus den aktiven State-/Automation-Pfaden entfernt werden.

## Start oder Wiederanlauf

```bash
.venv/bin/python -m genetic_algorithm automation start automation_island_week_v2
```

Der Befehl läuft absichtlich im Vordergrund. Für eine Urlaubs-Ausführung sollte er durch einen
Prozessmanager wie `systemd --user`, `tmux` oder `screen` gehalten werden. Derselbe Startbefehl ist
auch der Restart-Befehl: Bootstrap-Intent, Root-Wave, Worker-Specs, Decisions und Child-Waves sind
idempotent beziehungsweise hashgebunden und werden aus SQLite/WAL wieder aufgenommen.

Eine auf die aktuellen absoluten Repository-/Venv-Pfade gebundene User-Service-Datei kann ohne
Installation erzeugt werden:

```bash
.venv/bin/python -m genetic_algorithm automation service-unit \
  --output genetic_algorithm/data/v2/ga-automation.service
```

Danach die Datei bewusst nach `~/.config/systemd/user/ga-automation.service` übernehmen und mit
`systemctl --user daemon-reload` sowie `systemctl --user enable --now ga-automation.service`
aktivieren. Die Unit startet bei echten Prozessfehlern neu. Ein absichtlicher
Guard-/Budget-Exitcode 2 gilt für systemd als erfolgreicher Abschluss und wird nicht neu gestartet.

Ein einmaliger Reconciliation-Tick ohne dauerhafte Schleife:

```bash
.venv/bin/python -m genetic_algorithm automation start automation_island_v2 --once
```

Dieser Tick darf einen bereits gequeueten Attempt starten. Er ist kein Dry-Run.

## Status

```bash
.venv/bin/python -m genetic_algorithm automation status
```

Standardpfade:

- State: `genetic_algorithm/data/v2/orchestration.sqlite3`
- Automation-Artefakte: `genetic_algorithm/data/v2/automation/`
- persistenter Stop-Schalter:
  `genetic_algorithm/data/v2/automation/STOP_AUTOMATION`
- kompakter, automatisch aktualisierter Analysebericht:
  `genetic_algorithm/data/v2/automation/reports/LATEST.md`
- kampagnenweiter, sanitizierter Verlauf:
  `genetic_algorithm/data/v2/automation/reports/CAMPAIGN_SUMMARY.md`

Der Campaign-Report trennt zwei unterschiedliche Aussagen ausdrücklich:

- `diagnostic_top_candidate` ist nur eine Reporting-Rangfolge nach
  Gate-Alignment und anschließend Robust-Score. Er ist **nicht** die
  Planner-Auswahl.
- `planner_selection_events` und `planner_selected_input_parents` weisen den
  tatsächlich vom Pareto-basierten Planner gewählten Parent und die damit
  erzeugten Child-Arme aus.

Zusätzlich werden Gate-Abstände als signierte Schwellenmargen berichtet:
positive Werte liegen auf der bestandenen Seite, negative Werte zeigen den
noch fehlenden Abstand zur Schwelle. Der Report enthält weiterhin Planhash,
Seeds und konservative Return-/Expectancy-/DD-/ES-Metriken, aber keinen
Strategiecode, kein Genom, keine Einzeltrades und keine Workerlogs.

Eigene Pfade können mit `--state-db` und `--automation-root` gesetzt werden. Beide Optionen müssen
bei Start und Status konsistent verwendet werden.

## Sicher stoppen

```bash
.venv/bin/python -m genetic_algorithm automation stop
```

Der Kill-Switch verhindert vor dem nächsten Scheduler-Tick jeden weiteren Start und jede neue
Wave. Bereits laufende Worker werden nicht hart beendet; ihre immutable Resultate können noch
abgeschlossen werden. Zum späteren Fortsetzen den `STOP_AUTOMATION`-Marker bewusst entfernen und
denselben Startbefehl erneut ausführen.

## Default-Budgets und Entscheidung

- maximal 7 Tage Laufzeit
- das aktuelle Diagnose-Preset stoppt nach genau einem Root-Wave; erst nach dessen Review darf
  `automation_controller.max_waves` bewusst erhöht werden
- maximal 400 Attempts; die Canary-Messung von rund 28 Minuten und 15,5 MiB
  pro Attempt lässt damit den 7-Tage-Guard vor dem Attemptlimit greifen und
  projiziert nur etwa 6,2 GiB Artefakte
- maximal ein paralleler Attempt
- maximal 40 GiB unter dem Automation-Root
- mindestens 20 GiB müssen auf dem Dateisystem frei bleiben
- unter 2 GiB verfügbarem RAM werden keine neuen Attempts geclaimt
- ein Generic-Island-Run stoppt nach spätestens 600 Minuten an der nächsten abgeschlossenen
  Generationsgrenze
- der Executor beendet die gesamte Worker-Prozessgruppe spätestens nach 12 Stunden
- maximal drei vollständig fehlgeschlagene Waves werden als Scratch-Control recovered; danach
  blockiert der Controller
- derselbe nur bedingt geeignete Continuation-Phänotyp darf höchstens drei Child-Waves speisen
- pro Child-Wave höchstens drei Attempts: Control, Replication und genau ein
  anwendbarer Frequency-, Duration/Risk- oder Edge-Repair-Arm
- Repair darf ausschließlich die explizit freigegebenen Fitnessgewichte,
  `genetic_algorithm.mutation_rate` und
  `genetic_algorithm.search_seed_salt` verändern

Wenn ein robuster Kandidat alle Gates besteht, wird er Pareto-basiert anhand konservativer
Return-/Expectancy-LCBs und DD-/ES-UCBs ausgewählt. Replication und Repair erhalten sein
hashverifiziertes Genom. Ein Kandidat, der noch nicht promotionsfähig ist, darf ausschließlich
zur Fortsetzung der Suche dienen, wenn kein Szenario eine negative Nettorendite hat, mindestens
75 % der Szenarien profitabel sind, Evidenz und Stichprobe vollständig sind, DD-/ES- sowie eine
365-Tage-Drawdown-Dauergrenze eingehalten werden und nur ausdrücklich erlaubte Gategründe
fehlschlagen. Diese Continuation bleibt Search-only und ist auf drei Parent-Waves begrenzt.

Genome-Provenienz und Child-Search-Config sind bewusst getrennt: Das Genom
bleibt an seinen echten Producer gebunden, aber jede Folge-Wave baut ihre
Replication-/Repair-Config erneut auf der unveränderten Parent-Control-
Baseline auf. Ein Delta, das dort keine Änderung bewirkt, blockiert
fail-closed. Der Manifest-Seed bleibt für Replay und Bootstrap gepaart;
nur der evolutionäre Repair-Suchstream erhält einen deterministischen Salt.

Vor dem nächsten langen Lauf wird dieser Vertrag mit dem begrenzten Profil
`automation_island_gate_repair_canary_v2` über maximal drei Waves geprüft.
Existiert kein solcher Kandidat, läuft nur eine neue Scratch-Control weiter. Technische
Fehlerquoten, unvollständige Evidenz, unbekannte Deltas, unsichere Configs, Budgetgrenzen oder
Hashabweichungen blockieren fail-closed.
Ein einzelner technischer Totalausfall darf ebenfalls als Scratch-Control recovered werden, aber
nur bis zur fest gebundenen Serie von drei vollständig fehlgeschlagenen Waves.

## Was vor der Abreise geprüft werden muss

1. `automation status` zeigt genau eine lineare Wave-Kette.
2. `automation preflight` meldet `PREFLIGHT_READY`; 40 GiB Artefaktlimit und 20 GiB freie Reserve
   sind separate Stop-Gates.
3. Der Prozessmanager startet im Repository-Root und verwendet `.venv/bin/python`.
4. Es läuft kein zweiter Legacy-Scheduler gegen dieselbe Config-Queue.
5. Der Stop-Befehl und der Status-Befehl wurden einmal mit denselben Pfaden getestet.
