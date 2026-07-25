# Readiness für den ersten einwöchigen Generic-Island-Lauf

Stand: 25.07.2026

## Entscheidung

**NO-GO für den sofortigen Wochenstart.** Der Seed-3001-Lauf war technisch
stabil, hat aber einen wesentlichen Search-/Replay-Semantikfehler sichtbar
gemacht. Die Korrektur ist implementiert. Vor einem siebentägigen Lauf fehlen
noch zwei reale Kurzlauf-Nachweise.

Der Wochenlauf bleibt ein Search-only-Shadow-Lauf. Auch ein vollständig
bestandener Wochenlauf autorisiert weder Paper- noch Live-Trading und ist kein
Ersatz für ein bisher unberührtes finales Pair-x-Zeit-Panel.

## Bereits nachgewiesen

- Seed-3001-Worker und alle Kandidatenbacktests endeten technisch erfolgreich.
- Manifest-, GA- und Generic-Island-Seeds stimmen exakt überein.
- Immutable Artefakte und terminales Resultat bestehen den Hash-Readback.
- Der Controller stoppt nach einem begrenzten Root sauber an
  `MAX_WAVES_REACHED`; systemd wertet den Guard-Exitcode als Erfolg.
- Die strikte V2-Auswertung erkennt hohe Verluste, Drawdowns, unzureichende
  profitable Pairquote und zu geringe belastbare Evidenz fail-closed.
- Search und V2 verwenden nach der Korrektur beide isolierte Pair-Replays.
  Ein Replay aller fünf Seed-3001-Finalisten reproduziert Pair-Profite und
  Tradezahlen exakt.
- Der korrigierte Seed-4001-Root endete ohne technischen Fehler. Alle fünf
  Finalisten waren auf allen vier Pairs profitabel; der Topkandidat erreichte
  BTC +11,89 %, SOL +11,60 %, ETH +8,56 % und BNB +8,53 %. Search und strikter
  Replay stimmen bei Pair-Profit und Tradezahl exakt überein. Die beste
  Search-Fitness stieg von 0,276 in Generation 4 auf 0,589 in Generation 11.
- Profitable Zero-Loss-Samples besitzen einen versionierten, endlichen und
  explizit rechtszensierten Profit Factor. Kleine zensierte Samples erhalten
  im Search-Score erst mit wachsender Tradezahl Vertrauen; Cache-Schema 9 und
  V2-Metrikschema 2.1 trennen die neue Semantik von Altdaten.
- Der beobachtete Root benötigte rund 23 MiB Artefaktspeicher und etwa
  1,54 GiB Peak-RAM. Die vorhandenen 40-GiB-/20-GiB-Reserve- und
  2-GiB-RAM-Gates bleiben deutlich konservativer.
- Restart-, Lease-, PID-, Idempotenz-, Kill-Switch- und synthetische
  Root-zu-Child-Verträge sind automatisiert getestet.

## Muss vor dem Wochenlauf erledigt werden

### 1. Korrigierten Seed-4001-Root ausführen – abgeschlossen

Das Diagnose-Preset bleibt auf `max_waves: 1`. Der Lauf muss den neuen
`independent_pairs`-Search-Pfad real ausführen.

Die abgeschlossene Seed-3001-Kampagne ist mit State-DB, Root-Artefakten und
immutable Bericht recoverable archiviert. Die aktiven State-/Automation-Pfade
sind für Seed 4001 frei. Der Preflight blockiert einen noch vorhandenen
Bootstrap-Intent mit abweichender Config, Policy oder Seed ausdrücklich; ein
bloß grüner Daten-/Ressourcencheck genügt nicht.

Go-Kriterien:

- keine technischen Backtestfehler oder fehlenden Pairmessungen;
- jedes Kandidatensplitting enthält alle vier isolierten Pairs;
- Search- und strikter Replay stimmen für eingefrorene Finalisten bei
  Pair-Profit und Tradezahl überein;
- der Score besitzt einen brauchbaren Evolutionsgradienten und fällt nicht
  für die gesamte Population auf denselben Wert;
- Finalisten mit weniger profitablen oder stark verlierenden Pairs können
  bessere Cross-Pair-Kandidaten nicht durch Gesamtprofit überholen;
- Laufzeit, RAM und Artefaktwachstum werden erneut gemessen, weil vier
  isolierte Backtests pro Kandidat mehr Arbeit als zwei gemeinsame
  Split-Portfolios verursachen.

Die Kriterien sind erfüllt. Der Lauf benötigte etwa 24:41 Minuten Workerzeit,
26:15 Minuten Dienstzeit, 1,55 GiB Peak-RAM und 16 MiB Artefakte. Kein
Kandidat war bereits promotionsfähig, weil konservative Expectancy-/
Annual-Return-LCBs, Drawdown-Dauer, Tradefrequenz und teilweise DD-UCB weiter
blockierten. Das ist ein fachliches Ergebnis, kein technischer Fehler.

### 2. Einen echten Zwei-Wave-Canary durchführen

Das separate Preset `automation_island_canary_v2` verwendet Seed 5001 und
`max_waves: 2`. Es muss einen Root analysieren, genau einen sinnvollen
Child-Plan materialisieren und den Child tatsächlich starten und abschließen.

Go-Kriterien:

- lineare Root-zu-Child-Kette ohne doppelte Wave oder doppelten Attempt;
- bei keiner eligible Strategie genau ein Scratch-Control mit rotiertem Seed;
- bei eligible Evidenz nur erlaubte Control-/Replication-/Explore-Arme;
- Child-Config-Delta, Parent-Genome und Seeds entsprechen dem persistierten
  Plan;
- ein Controller-Neustart erzeugt keinen zweiten Child und überspringt keinen
  fertigen Zustand;
- `LATEST.md`, State-DB und immutable Resultate berichten denselben Ausgang.

Die synthetischen E2E-Tests belegen diese Mechanik bereits, ein vollständiger
realer Generic-Island-Child wurde im aktuellen Produktionspfad aber noch nicht
ausgeführt. Das ist der wichtigste verbleibende Automationsnachweis.

### 3. Wochenprofil und Betriebsprobe separat freigeben

Das Diagnose-Preset wird nicht direkt zum Wochenprofil umgebaut. Nach den
beiden Canaries wird ein separates, strikt validiertes Wochenpreset angelegt.
Bewusst festzulegen sind:

- frischer Root-Seed, der noch zu keiner immutable Wave gehört;
- sieben Tage Gesamt-Wallclock;
- Wave-, Attempt-, Parallelitäts-, RAM- und Diskbudgets;
- maximaler Worker- und Executor-Timeout;
- Verhalten nach technischen Totalausfällen;
- Aufbewahrung und kompakte Berichte ohne Strategie-/Trade-/Logdaten im Git.

Unmittelbar vor dem Start:

1. alle beabsichtigten Source-/Configänderungen committen und pushen;
2. `git status` prüfen; bekannte Runtime-Dateien dürfen den Code-Manifesthash
   nicht verschmutzen;
3. die vollständige bereinigte GA-Suite und die externen Generation-Step-
   Tests grün ausführen;
4. `automation preflight <wochenprofil>` muss ausschließlich
   `PREFLIGHT_READY` melden;
5. Kill-Switch einmal setzen, Status prüfen und kontrolliert wieder entfernen;
6. systemd-Unit neu aus dem aktuellen Repository-/Venv-Pfad erzeugen und
   Start/Stop/Restart testen;
7. freien Speicher und RAM prüfen sowie sicherstellen, dass kein Legacy-
   Scheduler dieselbe Queue verwendet.

## Empfohlene Reihenfolge

1. Aktuelle Korrektur testen, committen und ohne Laufzeitdaten pushen.
2. Reales Zwei-Wave-Canary ausführen und Restart/Child-Vertrag prüfen.
3. Laufzeit- und Speicherbudgets aus Seed 4001 plus Canary kalibrieren.
4. Separates Wochenpreset erstellen, Preflight und Operatorprobe bestehen.
5. Erst dann den einwöchigen Search-only-Lauf starten.
