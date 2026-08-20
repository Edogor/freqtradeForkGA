# Legacy-Genome-Migration

Stand: 23.07.2026.

Historische Populationen und Hall-of-Fame-Dateien sind keine
`strategy-gene-v2`-Quelle, nur weil ein Configfeld diesen String nennt. Eine verwendbare Quelle
braucht entweder einen nativen versionierten Producer oder eine reproduzierbare Migration aus
exakt gebundenen Legacy-Bytes.

## Sicherheitsvertrag

Der Migrator:

- verlangt den SHA-256 der exakten Quelldatei;
- verändert die Quelle niemals und überschreibt kein Ziel;
- migriert die gesamte Population/HOF atomar oder gar nicht;
- erlaubt nur historisch definierte Defaultfelder, eindeutige Instance-ID-/Condition-Zuordnung und
  die bekannte CDL-Suffixnormalisierung;
- lehnt mehrdeutige Referenzen, unbekannte Indikatoren/Operatoren, falsche Parameterfamilien,
  nichtfinite Werte und ungültige Risk-/Timeframe-/ROI-/Trailing-/Regime-Grenzen ab;
- schreibt pro Genom Quellhash, Zielhash und jede erlaubte Änderung in einen gehashten
  `MigrationReport`;
- prüft jedes Ziel erneut mit `StrategyGene.from_dict_exact()`.

`from_dict()` bleibt ausschließlich der Legacy-Normalisierer. Persistenz, IPC, HOF und
Warm-Start verwenden `from_dict_exact()` mit dem zusätzlichen semantischen Vertrag.

## Migration ausführen

Zuerst den Quellhash unabhängig bestimmen:

```bash
sha256sum genetic_algorithm/data/hall_of_fame/hall_of_fame.json
```

Dann in eine neue Datei migrieren:

```bash
python -m genetic_algorithm.scripts.migrate_genome_artifact \
  genetic_algorithm/data/hall_of_fame/hall_of_fame.json \
  /immutable/path/hall_of_fame.strategy-gene-v2.json \
  --source-sha256 <sha256>
```

Die Ausgabe nennt Ziel-SHA-256, Reporthash und Eintragszahl. Bestehende Zieldateien werden nicht
überschrieben.

## Migrierten Warm-Start binden

Quelle und Migrationsergebnis müssen beide erhalten und content-addressed konfiguriert werden:

```yaml
warm_start:
  enabled: true
  source_type: hof
  top_n: 15
  genome_schema_version: strategy-gene-v2

  source_hof: /immutable/path/hall_of_fame.strategy-gene-v2.json
  source_hof_sha256: <sha256-des-migrierten-artefakts>

  source_hof_migration_source: /immutable/path/legacy-hall-of-fame.json
  source_hof_migration_source_sha256: <sha256-der-legacy-datei>
```

Für Populationen heißen die vier Felder entsprechend `source_checkpoint*`.

Der Loader verifiziert beide Dateien, rekonstruiert die Migration erneut und verlangt kanonisch
dieselbe Zielstruktur und denselben Report. Ein eingebetteter, aber nicht reproduzierbarer Report
blockiert den Start. Danach werden alte Fitnesswerte nur zur Auswahl der Seeds verwendet; vor dem
neuen Lauf werden Fitness, Panelbindung und Evaluationsstatus gelöscht.

## Projektbefund

Beim Scan von 11.708 vorhandenen Checkpoint-/HOF-Genomen erfüllten 7.207 bereits den exakten
strukturellen und semantischen Vertrag. 4.272 weitere Payloads benötigen mindestens eine explizite
strukturelle Migration. Daneben wurden echte Defekte gefunden: falsche Parameterfamilien,
unbekannte Indikatoren, ungültige Candlestick-Operatoren, verwaiste Referenzen, nicht ausführbare
Trailing-Stops und leere `between`-Intervalle.

Der globale HOF lässt sich deterministisch migrieren; drei eindeutige alte `RSI`-Referenzen werden
im Report ausgewiesen. Der Wave-44-Top-30-Seed benötigt keine Genomänderung, aber weiterhin die
explizite Schema-/Provenienzhülle. Fehlerhafte andere Archive werden bewusst nicht teilimportiert.
