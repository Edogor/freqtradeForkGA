# Betriebs-Runbook: Guarded Generic-Island-Automation V2

Stand: 25.07.2026

Dieser Pfad automatisiert ausschließlich die **Strategie-Suche im Shadow-Modus**. Er deployt keine
Strategie und autorisiert kein Live-/Paper-Trading. Das Profil
`automation_island_v2` erzwingt Generic Island, vollständige Pair-Validation für jeden Kandidaten,
Mark-to-Market-V2 und deaktiviert unter anderem WF, Holdout, MC, Surrogate, SIS, LLM, NSGA-II,
Regime- und Classic-Island-Pfade.

## Vor dem Start

```bash
.venv/bin/python -m genetic_algorithm config validate \
  genetic_algorithm/config/presets/automation_island_v2.yaml --strict
```

Der mitgelieferte Datensatzvertrag erwartet aktuell BTC/SOL als Training und ETH/BNB als räumliche
Validation auf 1h für `20230325-20260327`. Fehlende Dateien, Candle-Lücken, abweichende Hashes oder
eine veränderte Config blockieren vor Ausführung.

Im Profil `automation_island_v2` wird jedes Pair bereits während der Suche
isoliert backgetestet. Gemeinsame Multi-Pair-Portfolios sind untersagt, weil
Slotkonkurrenz bei kleinem `max_open_trades` verlierende Pairs verdecken kann.
Vor einem unbeaufsichtigten Wochenlauf müssen zusätzlich alle Punkte in
[WEEK_RUN_READINESS_V2.md](WEEK_RUN_READINESS_V2.md) erfüllt sein.

Der read-only Produktions-Preflight prüft zusätzlich Venv, Daten-/Split-/Codehash, freien
Plattenplatz und verfügbaren RAM:

```bash
.venv/bin/python -m genetic_algorithm automation preflight automation_island_v2
```

Nur `ready: true` mit `PREFLIGHT_READY` ist startfähig.

## Start oder Wiederanlauf

```bash
.venv/bin/python -m genetic_algorithm automation start automation_island_v2
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
- maximal 148 Attempts
- maximal ein paralleler Attempt
- maximal 40 GiB unter dem Automation-Root
- mindestens 20 GiB müssen auf dem Dateisystem frei bleiben
- unter 2 GiB verfügbarem RAM werden keine neuen Attempts geclaimt
- ein Generic-Island-Run stoppt nach spätestens 600 Minuten an der nächsten abgeschlossenen
  Generationsgrenze
- der Executor beendet die gesamte Worker-Prozessgruppe spätestens nach 12 Stunden
- maximal drei vollständig fehlgeschlagene Waves werden als Scratch-Control recovered; danach
  blockiert der Controller
- pro Child-Wave höchstens drei Attempts: Control, Replication, Explore
- erlaubtes Config-Delta: nur `genetic_algorithm.mutation_rate`

Wenn ein robuster Kandidat alle Gates besteht, wird er Pareto-basiert anhand konservativer
Return-/Expectancy-LCBs und DD-/ES-UCBs ausgewählt. Replication und Explore erhalten sein
hashverifiziertes Genom. Wenn eine technisch gesunde Wave noch keinen eligible Kandidaten enthält,
läuft nur eine neue Scratch-Control weiter. Technische Fehlerquoten, unvollständige Evidenz,
unbekannte Deltas, unsichere Configs, Budgetgrenzen oder Hashabweichungen blockieren fail-closed.
Ein einzelner technischer Totalausfall darf ebenfalls als Scratch-Control recovered werden, aber
nur bis zur fest gebundenen Serie von drei vollständig fehlgeschlagenen Waves.

## Was vor der Abreise geprüft werden muss

1. `automation status` zeigt genau eine lineare Wave-Kette.
2. `automation preflight` meldet `PREFLIGHT_READY`; 40 GiB Artefaktlimit und 20 GiB freie Reserve
   sind separate Stop-Gates.
3. Der Prozessmanager startet im Repository-Root und verwendet `.venv/bin/python`.
4. Es läuft kein zweiter Legacy-Scheduler gegen dieselbe Config-Queue.
5. Der Stop-Befehl und der Status-Befehl wurden einmal mit denselben Pfaden getestet.
