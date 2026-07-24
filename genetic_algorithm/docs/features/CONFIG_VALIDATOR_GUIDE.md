# Kanonischer Config-Validator

Stand: 23.07.2026
Invarianten-Policy: `ga-config-invariants-v1`

Der Validator beantwortet ausschließlich die Frage, ob eine Config von der aktuellen
Implementierung eindeutig und widerspruchsfrei ausgeführt werden kann. Er behauptet nicht, welche
Parameter profitabel sind.

## Ein Validierungspfad

```bash
python -m genetic_algorithm config validate path/to/config.yaml
```

CLI, Runner, Engines, V2-Materializer, Worker und der Editor-Hook verwenden denselben Resolver aus
`config/schema.py`. Die mechanischen Regeln stehen in `config/invariants.py`.

Schema V2 verbietet unbekannte Pfade und falsche Typen. An einer V2-Artefaktgrenze müssen außerdem
alle Defaults aufgelöst sein. Die resolved Config wird vor dem Queueing immutable gespeichert und
zweifach gebunden:

- kanonischer semantischer SHA-256 im Attempt-Manifest;
- exakter Datei-SHA-256 in der Worker-Inputmatrix.

## Fatal erzwungene Invarianten

| Bereich | Vertrag |
|---|---|
| Population | `2 <= population_size <= 10_000`; die Obergrenze ist ein Implementierungs-/Ressourcenschutz, kein Optimum |
| Generationen | `1 <= generations <= 100_000` |
| Slots | `0 <= elite_size < population_size`; Immigrants passen in die verbleibenden Slots |
| Tournament | `1 <= tournament_size <= population_size` |
| Wahrscheinlichkeiten | endlich und in `[0, 1]`; Basismutation überschreitet nicht ihr Maximum |
| Dispatch | nur implementierte GA-Modi, Selection- und Crossover-Verfahren |
| Pairs | nicht leer, eindeutig, keine leeren Namen |
| Pair-Validation | nicht leere, disjunkte Train-/Validation-Pairs und positive Gesamtgewichtung |
| Parallelität | explizite Workerzahl ist eine positive Ganzzahl; `null` bedeutet Auto-Erkennung |
| Generic Islands | mindestens eine Insel, Population mindestens 2, Migration passt in die Quellpopulation |
| Classic Islands | Walk-forward ist gleichzeitig verboten, weil die Engine es sonst intern deaktiviert |
| Fitnessgewichte | bekannte Felder, endlich, nicht negativ und mindestens ein positives Gewicht |

`tournament_size: 1` ist erlaubt, erzeugt aber nachweislich eine uniforme zufällige Elternwahl.
Darum erscheint eine sachliche Warnung, keine vermeintliche Profitabilitätsempfehlung.

## Bewusst keine Validatorgrenzen

Folgende Werte sind Tuning-Faktoren und keine universellen Sicherheitsgrenzen:

- Population 10–15 oder eine andere feste Populationsspanne;
- Inselpopulation 6, 60 oder ein anderer magischer Wert;
- höchstens zwei Pairs;
- Elitequote von 10–15 %;
- Tournament 3–6;
- eine feste Mutation, Generationszahl, Walk-forward-Länge oder MC-Permutationszahl;
- bestimmte Fitnessgewichte oder Indikatoren.

Die früheren AP-1…AP-20-Warnungen vermischten einzelne historische Runs, wechselnde Panels,
Holdout-Semantiken, Bugs und mehrere gleichzeitig geänderte Faktoren. Einige lasen zudem veraltete
Config-Pfade. Sie sind deshalb aus dem ausführbaren Vertrag entfernt.

## Wie Tuningwerte nachgewiesen werden

Eine Empfehlung darf erst als projektspezifische Evidenz gelten, wenn:

1. Control und Treatment bis auf den deklarierten Faktor identisch sind;
2. Seeds, Budget, Code-, Daten- und Szenariohash gepaart sind;
3. die Auswertung ausschließlich auf demselben Common-Replay-Panel erfolgt;
4. Return-/Expectancy-LCB sowie DD-/ES-UCB und Aktivitäts-/`N_eff`-Gates berichtet werden;
5. ein Ergebnis über mehrere Seeds und relevante Pair-/Zeitregime repliziert wurde;
6. Final-Test-Daten nicht zur erneuten Parameterauswahl verwendet werden.

Bis dahin ist ein Befund eine Hypothese für den Planner, keine Warnung und kein Startverbot.

## Kontextueller Preflight

`utils.config_validator.preflight_check()` ergänzt den reinen Vertrag optional um lokale
Datenverfügbarkeit. Diese Prüfung darf melden, dass eine Datei oder ein Datenverzeichnis fehlt.
Die kanonische V2-Ausführung verlässt sich letztlich auf das content-addressed Datenmanifest und
die exakte Periodenabdeckung.
