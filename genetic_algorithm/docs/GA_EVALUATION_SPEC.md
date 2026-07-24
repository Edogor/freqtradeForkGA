# Evaluationsspezifikation v2

Status: Design mit partiell implementierter Shadow-Messbasis. Die Formeln und Startgewichte sind
noch kein empirisch bewiesenes Optimum. Sie müssen im Shadow-Replay und in gepaarten Experimenten
kalibriert werden.

## Ziel

Gesucht werden Strategien mit hohem **Nettoertrag nach realistischen Kosten**, deren Wahrscheinlichkeit
für große oder lang anhaltende Verluste klein ist und deren Edge mit ausreichend vielen unabhängigen
Beobachtungen belegt ist. Daytrading-Aktivität ist wichtig, darf aber nicht durch bloßes Erhöhen der
rohen Tradezahl belohnt werden.

Die zentrale Änderung gegenüber der aktuellen Fitness ist:

> Optimierungsrang, statistische Evidenz und Freigabe zur Promotion sind drei verschiedene Dinge.

Eine gewichtete Summe darf niemals einen Verlust, fehlende Risikodaten oder ein gescheitertes
Validation-Segment durch Winrate oder einen anderen guten Teilwert kompensieren.

## Begriffe

- **Attempt:** genau eine Ausführung einer unveränderlichen resolved Config mit eigenem Seed,
  Code-/Datenstand und Artefaktverzeichnis.
- **Phänotyp:** ausführbare Strategie ohne transiente Felder wie Generation oder Individual-ID.
- **Szenario:** definierte Kombination aus Pairgruppe, Zeitblock und Kostenstress.
- **Train:** Daten, die während Suche und Selektion benutzt werden dürfen.
- **Pair Validation:** während der Kandidatenauswahl verwendete, aber im Training ungesehene
  Pairs oder Pair-Cluster. Sie darf denselben Zeitraum wie das Training verwenden.
- **Temporal Validation:** während der Kandidatenauswahl verwendete, zeitlich spätere Daten
  eines bereits im Training gesehenen Pairs. Sie muss nicht überlappend und mindestens um das
  konfigurierte Embargo vom Trainingszeitraum getrennt sein.
- **Legacy `INNER_VALIDATION`:** Diese mehrdeutige Rolle sagt nicht aus, ob die Validierung
  räumlich oder zeitlich getrennt ist und ist deshalb im kanonischen V2-Split-Vertrag ungültig.
- **Final Test:** eine bis zum finalen Entscheid blinde räumliche oder zeitliche Testzelle. Wird
  ein zuvor gesehenes Pair verwendet, muss ihr Zeitraum strikt später liegen und das Embargo
  einhalten. Jede registrierte Zelle darf nur einmal konsumiert werden.
- **Promotion:** Freigabe für die nächste Prüf- oder Paper-Trading-Stufe, nicht nur HOF-Aufnahme.
- **LCB/UCB:** untere/obere Konfidenzgrenze einer Kennzahl.

## 1. Kanonischer Eingangsvertrag

Jeder Backtest muss dieselbe vollständige Datenstruktur liefern. Pflichtfelder dürfen nicht auf
„gute“ Nullwerte zurückfallen.

### Provenienz

- `metric_schema_version`, `fitness_policy_version`
- Attempt-, Wave-, Parent-Wave- und Experiment-ID
- Hash der resolved Config und des ausführbaren Phänotyps
- Git-Commit plus Dirty-Patch-Hash
- Datenmanifest mit Pair, Timeframe, Start/Ende, Candle-Anzahl und Dateihash
- Seed(s), Workerzahl und deterministische RNG-Ableitungen
- Fee, Spread, Slippage, Funding und Stress-Szenario
- explizite Train-/Pair-Validation-/Temporal-Validation-/Final-Rolle jedes Pair-Zeitblocks

### Rohdaten

- kalendergerichtete Portfolio-Equity und tägliche Netto-Returns
- Exposure, gleichzeitig gebundenes Kapital und Turnover
- Trades mit Pair, Open/Close-Zeit, Stake, Brutto-/Netto-PnL und Kosten
- vollständige Pair-Coverage, einschließlich Pairs mit null Trades
- Fehler-, Timeout-, Missing-Data- und No-Trade-Status je Segment

### Einheiten

- intern alle Returns und Drawdowns als Dezimalbruch, nicht gemischt mit Prozentpunkten
- Geldwerte separat und mit Währung
- Zeitdauern in Sekunden oder Tagen mit eindeutigem Feldnamen
- Anzeigeumrechnung in Prozent nur an der UI-/Report-Grenze

`null` bedeutet „nicht verfügbar“. Null bedeutet einen gemessenen Wert von null. Ein fehlendes
Pflichtfeld macht die Evaluation `INVALID`; es wird nicht neutral imputiert.

## 2. Kapitalgewichtete Equity als Metrikquelle

Sharpe, Sortino, Calmar, Drawdown, Expected Shortfall und Monatsreturns werden aus derselben
kalendergerichteten Portfolio-Equity nach allen Kosten berechnet. Einzelne Trade-Returns dürfen dafür
nicht gemittelt oder einfach über Pairs addiert werden.

Implementierungsstand 23.07.2026: Freqtrade-Tages-PnL wird kalendergefüllt und durch die jeweilige
Equity des Vortags geteilt. Zusätzlich rekonstruiert `safe_v2` eine gebührenbereinigte
End-of-Day-Mark-to-Market-Serie aus Trades, zeitgestempelten Order-Fills und OHLCV. Der Order-Ledger
unterstützt vollständig geschlossene Long- und Short-Trades, Hebel, mehrere Entries/DCA und
Teilexits. Der bereits im Fill-Betrag enthaltene Hebel wird nicht nochmals auf absoluten PnL
multipliziert. Jeder Trade wird gegen `profit_abs`, das Portfolio gegen den finalen Walletstand
reconciled. Futures-Funding wird bei jeder Backtest-Aktualisierung als zeitgestempeltes Wallet-Delta
persistiert. Das schließt DCA-bedingte Fill-Transfers und Recalculation-Korrekturen ein; die
Cashflows müssen sich exakt zu `funding_fees` summieren.

Historische Artefakte mit aggregiertem Funding ohne Cashflow-Zeitstempel und extern gelieferte offene
Positionen ohne terminalen Wallet-/Positionssnapshot bleiben bewusst nicht rekonstruierbar. Das
Cache-Schema 7 invalidiert entsprechende alte Backtest-Caches und bindet zusätzlich Risiko- und
Expectancy-Metrikverträge. Fehlende Tageskerzen, abweichende
Reconciliation oder unvollständige Semantik führen fail-closed zurück auf `INCONCLUSIVE`; die
Realized-close-Serie bleibt nur Diagnose und kann `VALID` nicht erfüllen. Reguläre
Freqtrade-Backtests schließen verbliebene Positionen am Periodenende zwangsweise und liefern damit
den benötigten terminalen Fill.

Dadurch werden gleichzeitig offene Positionen, unterschiedliche Stakes, Compounding, inaktive Tage
und Portfolio-Kapital korrekt berücksichtigt. Per-Trade- und Per-Pair-Werte bleiben zusätzliche
Diagnostik.

### 2.1 Risikometrikvertrag `calendar-effective-v1`

Alle folgenden Angaben verwenden Netto-Dezimalreturns. Die kanonische Tagesreihe enthält jeden
Kalendertag des Backtestzeitraums, einschließlich inaktiver Tage mit Return null. Für einen
kontinuierlichen Kryptomarkt gilt deshalb `P = 365`, nicht 252.

- `annual_risk_free_rate` ist eine effektive Jahresrate als Dezimalwert und muss endlich sowie
  größer als `-1` sein. Der Default von `safe_v2` ist bewusst `0.0`; er ist eine deklarierte
  Annahme, kein versteckter Fallback.
- Die periodische Zielrendite ist geometrisch:
  `MAR_P = (1 + annual_risk_free_rate)^(1/P) - 1`.
- Sharpe verwendet den arithmetischen Mittelwert von `r_t - MAR_P`, die
  Stichproben-Standardabweichung der Periodenreturns (`ddof=1`) und den Faktor `sqrt(P)`.
- Sortino verwendet denselben Zähler. Die Downside Deviation ist
  `sqrt(mean(min(r_t - MAR_P, 0)^2))` über **alle** Beobachtungen; positive Beobachtungen tragen
  null bei. Sie wird nicht nur durch die Anzahl negativer Tage geteilt.
- CAGR wird geometrisch aus der vollständigen Equity-Reihe berechnet:
  `exp(mean(log(1 + r_t)) * P) - 1`.
- Calmar ist `CAGR / MaxDrawdown`, wobei Max Drawdown der größte Peak-to-Trough-Verlust derselben
  Equity ist.

Weniger als zwei Beobachtungen, Null-Volatilität, kein messbares Downside oder Drawdown null ergeben
für den jeweils nicht definierten Quotienten `null`. Es gibt weder einen Sortino-Cap von 10 noch
günstige Nullwerte. Returns unter `-100 %`, nicht endliche Werte, eine Rate `<= -100 %` oder eine
intern nicht reconciliierende Equity machen die Messung ungültig.

Der Vertrag, `P`, effektive Jahresrate und periodische Rate werden im Backtestartefakt gespeichert.
Die Monatsreihe wird aus den kapitalgewichteten Tagesreturns geometrisch zusammengesetzt und trägt
explizite `YYYY-MM`-Labels. Der Portfolio-/Ensemble-Evaluator akzeptiert nur identische,
lückenlose Monatsachsen, verändert Eingangsgewichte nicht und nutzt denselben Sharpe-/Sortino-
Vertrag mit `P = 12`. Historische undatierte Monatslisten sind keine zulässige Ensemble-Evidenz.

## 3. Drei Evaluationsstufen

### Stufe A: Validität

Ein Kandidat ist nur gültig, wenn:

- Backtest und alle vorgeschriebenen Segmente erfolgreich und endlich sind;
- Datenbereiche vollständig, disjunkt und frei von Lookahead sind;
- benötigte Metriken und Kostenfelder vorhanden sind;
- alle konfigurierten Pairs in der Coverage auftauchen, auch mit null Trades;
- keine verbotene Config-/Gene-Kombination oder stille Fallback-Ausführung vorliegt;
- der effektive Stichprobenumfang `N_eff` berechnet werden kann.

Fehlschlag, Timeout, fehlendes Pair oder Null-Trades in einem Pflichtsegment dürfen die Fitness nie
erhöhen. Je nach Policy wird der Kandidat `INVALID`, `INCONCLUSIVE` oder `FAIL`, aber niemals `SAFE`.

### Stufe B: nicht kompensierbare Promotion-Gates

Die Grenzwerte sind profil- und risikobudgetabhängig. Sie werden versioniert und vor dem Run
festgelegt. Mindestgates:

1. Netto-Expectancy-LCB im gesamten Train-/Pair-/Temporal-Validation-Panel ist positiv.
2. Median der Validation-Szenario-Returns ist positiv.
3. Unteres Return-Quantil und Worst-Szenario bleiben über dem festgelegten Verlustbudget.
4. Max-Drawdown-UCB und Daily-ES-UCB bleiben unter ihren Risikobudgets.
5. Mindestanteil profitabler Pairs und Zeitblöcke wird erreicht.
6. `N_eff` und aktive Monate reichen für die gewünschte Konfidenz.
7. Baseline-, 1,5x- und 2x-Kostenstress verletzen keine festgelegten Kill-Gates.
8. Der finale gesperrte Test besteht; bei zu wenig Evidenz lautet das Ergebnis `INCONCLUSIVE`.

Ein Kandidat, der ein Gate verletzt, darf weiterhin wissenschaftlich interessant sein, wird aber
nicht promoted. Dadurch kann zum Beispiel eine hohe Winrate nie einen negativen Validation-Profit
„bezahlen“.

Implementierungsstand 21.07.2026: Die Shadow-Policy verlangt eine vor dem Replay vollständig
deklarierte Pair-x-Zeit-x-Kosten-Matrix. Fehlende oder zusätzliche Zellen ergeben
`INCONCLUSIVE`. Expectancy-LCB, Median-Return-LCB, Worst-Return, DD-/ES-UCB, `N_eff`, aktive Monate,
Trades pro aktivem Monat, profitable Szenarien, Verlustserie und DD-Dauer sind nicht
kompensierbare Einzelgates. Ein Gate-Fail lässt die Messung `VALID`, erzeugt aber eine
Shadow-Entscheidung `FAIL`; ein kompletter Pass erzeugt ausschließlich `WOULD_PASS` und niemals
Ausführungsautorität. Die derzeitigen Defaultschwellen sind Startwerte, keine empirische Freigabe.

Der Replay-Executor bindet diese Policy inzwischen an einen eingefrorenen ausführbaren Phänotyp.
Exakter Code, literales Timeframe, `max_open_trades` und externe Ausführungsparameter sind Teil des
Hashes. Freqtrade-Zeitranges werden aus der inklusiven Scenario-Periode erzeugt und der gemeldete
Backtestzeitraum anschließend exakt gegengeprüft. Ein verkürzter oder nicht parsebarer Zeitraum ist
`INVALID`, selbst wenn Return- und Risikowerte gut aussehen.

Der kanonische Runner erzeugt außerdem ein `split_manifest.json`, prüft exklusive
Freqtrade-Timeranges, Pair-/Zeittrennung und Embargo fail-closed und bindet dessen Hash an das
Attempt-Manifest. Die Final-Test-Ledger identifiziert eine blinde Zelle anhand von Exchange, Pair
und Zeitraum. Ein anderes Timeframe darf dieselbe Marktbeobachtung nicht erneut als blind erscheinen
lassen.

### Stufe C: Rang innerhalb der gültigen Kandidaten

Nach den Gates ist Pareto-Auswahl vorzuziehen:

- maximiere annualisierten Netto-Return-LCB;
- minimiere Max-Drawdown-UCB;
- minimiere Betrag des Daily Expected Shortfall 5 %;
- optional maximiere Netto-Expectancy-LCB, falls nur drei Ziele praktikabel sind.

Für Tournament-/Single-Objective-Selektion kann vorübergehend ein glatter Suchscore benutzt werden.
Ein sinnvoller Startpunkt je Szenario ist:

```text
U_s = 0,35 * ReturnQuality
    + 0,30 * DownsideControl
    + 0,20 * EdgeConfidence
    + 0,15 * ActivityAndStability
```

Dabei sind alle Teilwerte auf festen wirtschaftlichen Anchors gesättigt, nicht per Population
min-max-normalisiert:

- `ReturnQuality`: saturierender Wert aus Netto-Return-/CAGR-LCB und Calmar;
- `DownsideControl`: saturierender inverser Wert aus DD-UCB, Daily ES, Ulcer Index und DD-Dauer;
- `EdgeConfidence`: Netto-Expectancy-LCB, geglätteter Profit Factor und Payoff Ratio;
- `ActivityAndStability`: Stichprobenkonfidenz, Zeit-/Pair-Coverage und Monatsstabilität.

Mehr Aktivität verbessert den Konfidenzanteil nur bis zum Zielumfang. Danach gibt es keinen weiteren
Tradezahlbonus. Overtrading wird über echte Kosten, Turnover, Clustering und Capacity sichtbar.

Über alle Szenarien wird nicht einfach der Mittelwert genommen. Eine provisorische robuste
Aggregation ist:

```text
RobustScore = 0,50 * median(U_s)
            + 0,30 * q25(U_s)
            + 0,20 * q10(U_s)
```

Worst-Return, Worst-DD, positive Szenarioquote und Coverage bleiben separate Gates. Die Gewichte
müssen gegen blindes OOS kalibriert werden; sie sind keine „bewiesenen besten Werte“.

## 4. Profit-, Edge- und Win/Loss-Metriken

Pflichtausgabe:

- Netto-Return und CAGR/annualisierter Return mit LCB
- Netto-Expectancy pro Trade und pro gebundenem Risikokapital mit LCB
- Winrate mit Wilson- oder Bayes-Intervall
- durchschnittlicher/medianer Gewinn und Verlust
- Payoff Ratio `avg_win / abs(avg_loss)`
- geglätteter und gecappter Profit Factor mit Sonderbehandlung für keine Verlusttrades
- größter Einzelverlust und Verlustquantile
- längste Verlustserie und block-bootstrap-basierte Verteilung der Verlustserie

Winrate wird berichtet, aber nicht isoliert optimiert. Eine 90-%-Strategie mit kleinen Gewinnen und
seltenen großen Verlusten muss über Expectancy, Payoff, ES und Drawdown schlecht abschneiden.

### 4.1 Expectancy-Vertrag `net-expectancy-clustered-v1`

Die kanonische Netto-Expectancy wird ausschließlich aus vollständig geschlossenen, reconcilierten
Trade-Ledgern berechnet. `profit_abs` enthält Gebühren, Slippage, Spread und Funding bereits über
den Backtester- und MTM-Vertrag. Der Adapter verlangt, dass die Summe aller Trade-PnL sowohl mit
`BacktestResult.total_profit` als auch mit der Walletdifferenz übereinstimmt.

Zwei nicht austauschbare Sichtweisen werden gemeinsam berichtet:

- Gleichgewichtete Trade-Expectancy: arithmetischer Mittelwert der reconcilierten
  `profit_ratio`-Werte.
- Return auf gebundenes Kapital: Summe Netto-PnL geteilt durch die Summe der maximal gebundenen
  Stakes, für Long inklusive und für Short abzüglich Entry-Fee. Damit werden DCA, unterschiedliche
  Positionsgrößen und Hebel-Margin berücksichtigt.

„Gebundenes Kapital“ ist bewusst nicht gleich „Kapital am initialen Stop“. Eine Stop-Risk-Metrik
darf erst ergänzt werden, wenn initialer Stop, Stopänderungen, Slippage beim Stop und Gap-Risiko
zeitgestempelt und reconciled vorliegen.

Für beide Sichtweisen wird eine deterministische Moving-Block-Bootstrap-LCB berechnet. Alle Trades,
die in denselben festen UTC-Schlusszeitcluster fallen, bleiben pairübergreifend zusammen; Blöcke
bestehen aus aufeinanderfolgenden Clustern. Dadurch wird ein gemeinsamer Marktschock nicht als viele
unabhängige Trades gezählt. Der konservative Stichprobenumfang ist

```text
N_eff = min(N_eff_serial, N_eff_trade_capital, N_eff_temporal_capital)
```

Zusätzlich werden Pair-Kapital-HHI, effektive Pairzahl sowie größte Trade- und Cluster-Kapitalquote
persistiert. Unvollständige oder widersprüchliche Tradefelder sind `INVALID`; zu wenige Trades oder
Zeitcluster sind `INCONCLUSIVE`. Das Promotion-Gate verwendet pro Szenario den schlechteren der
beiden LCB-Werte. Ein positiver Kleinsttrade-Mittelwert kann dadurch keinen negativen
kapitalgewichteten Edge kompensieren.

## 5. Risiko

Pflichtmetriken aus der Portfolio-Equity:

- Max Drawdown samt UCB
- Drawdown-Dauer, Time Under Water und Ulcer Index
- Daily Expected Shortfall 5 % und optional 1 %
- Downside Deviation und Sortino
- Calmar
- Tail-Loss, größter Verlust und Verlustserien
- Exposure, Turnover, Kapitalkonzentration und Pair-Konzentration
- block-bootstrap-basierte Ruin-/Loss-Budget-Wahrscheinlichkeit

Max Drawdown allein reicht nicht: Ein kurzer tiefer Verlust, ein langer flacher Drawdown und seltene
Tail-Losses haben unterschiedliche operative Bedeutung.

## 6. Aktivität und Daytrading-Profil

Absolute Tradezahl wird ersetzt durch:

- Portfolio-Trades pro aktivem Monat
- Trades pro Pair und aktivem Monat
- effektive unabhängige Trades `N_eff`
- mediane Haltedauer und Quantile
- Turnover und Kostenanteil am Bruttogewinn
- Exposure-/Capacity-Metriken

`N_eff` berücksichtigt serielle Abhängigkeit, zeitlich geclusterte Trades und stark korrelierte
gleichzeitige Pair-Signale. Der Aktivitätsscore kann beispielsweise bis zum statistisch benötigten
Ziel wie `min(1, sqrt(N_eff / N_target))` wachsen. `N_target` wird pro Profil durch gewünschte
Konfidenzbreite festgelegt, nicht aus einer willkürlichen 5m/1h-Multiplikatortabelle.

Profile sollten mindestens `daytrade`, `intraday`, `swing` und `mixed` unterscheiden. Sie ändern
Aktivitäts-, Haltedauer-, Kosten- und Capacity-Gates, nicht die Bedeutung von Profit oder Risiko.
Mixed-Timeframe-Kandidaten erhalten gene-spezifische Profile; nicht alle Gene eines Runs teilen
dieselbe Tradezahl-Schwelle.

## 7. Datenaufteilung und Robustheit

Cross-Pair und Zeitrobustheit testen verschiedene Fehlerarten und ersetzen einander nicht.

### Empfohlener Ablauf

1. Evolution auf älteren Zeitblöcken und vorab festgelegten Train-Pairs.
2. Pair-Validation als rotierendes Leave-one-pair-/Pair-Cluster-out-Panel.
3. Budgetbegrenzte, nicht überlappende, purged Temporal-Validation nur für Top-K.
4. Kostenstress für dieselben eingefrorenen Phänotypen.
5. Final-Test-Matrix aus räumlich ungesehenen Pairs und/oder späteren Zeitblöcken genau einmal
   konsumieren. Für bereits verwendete Pairs ist ein späterer, embargo-getrennter Zeitraum Pflicht;
   besonders starke Evidenz entsteht, wenn beide Achsen kombiniert werden.

Das finale Set wird nicht periodisch beobachtet. Sobald seine Ergebnisse eine neue Konfiguration
beeinflussen, ist es Validation geworden und ein neues Final-Set muss reserviert werden.

### Szenarioaggregation

Jedes Pair-Zeit-Kosten-Segment bleibt als eigene Zeile erhalten. Fehler und No-Trade-Segmente werden
nicht herausgefiltert. Neben RobustScore werden mindestens ausgegeben:

- Median, q25, q10 und Worst Netto-Return
- Median und Worst DD/ES
- Anteil profitabler Pairs, Zeitblöcke und Gesamtszenarien
- Pair-/Zeit-Coverage
- Rangstabilität und Konfidenzintervalle

Implementierungsstand 22.07.2026 für den Legacy-Regime-Pfad: Jedes deklarierte Regime-Segment wird
als `segment_outcome` mit Status, Tradezahl, Raw-/Effective-Fitness und Fehler persistiert. Positive
Fitness aus Null- oder zu wenigen Trades wird auf 0 gekappt; negative Fitness bleibt erhalten.
Fehler und ungültige Fitness tragen ebenfalls 0 bei und bleiben in allen Nennern. Der Harmonic Mean
ist nur für ausschließlich positive Segmente definiert; sobald ein Segment <= 0 ist, gilt der
schlechteste Segmentwert. Das ist eine mechanische Fail-closed-Reparatur, kein Nachweis, dass die
Regimeklassifikation oder diese Legacy-Fitness OOS-Mehrwert liefert.

## 8. Monte Carlo und Unsicherheit

Reines Permutieren der Trade-Reihenfolge ist kein sinnvoller Profit-Stresstest, weil das Produkt der
Returns permutationsinvariant ist. Stattdessen:

- stationary oder moving block bootstrap auf Daily Returns bzw. Signal-/Trade-Clustern;
- Blocklänge datengetrieben oder konservativ festgelegt;
- Drawdown-, ES-, Ruin-, Expectancy- und Return-Verteilungen auswerten;
- ausschließlich adverse Kosten-/Slippage-Szenarien zusätzlich rechnen;
- Zufallsseed und Samples vollständig versionieren.

Diese Diagnostik ist zunächst post-hoc. Sie darf erst Fitness oder Promotion beeinflussen, wenn ihr
inkrementeller Vorhersagewert für blindes OOS nachgewiesen ist.

## 9. Vergleichbarkeit

Fitness ist nur vergleichbar, wenn folgende Felder identisch sind:

- Metric- und Policy-Version
- Datenmanifest und Rollen der Splits
- Kostenmodell
- Backtester-/Codeversion
- Profil und Risikobudget
- Szenariopanel

HOF-Werte aus unterschiedlichen Insel-Spezialisierungen, Waves oder alten Scoreversionen werden nie
direkt sortiert. Sie müssen auf einem gemeinsamen Replay-Panel neu bewertet werden.

Implementierungsstand 22.07.2026: Legacy-Fitness persistiert `fitness_panel_id` und
`fitness_panel_role`. Die ID bindet den deklarierten Pair-/Zeit-/Kostenvertrag, Scoring-Policy,
Evaluatortyp und die genaue Regime-Segmentmenge; Suchseed und Populationsbudget gehören nicht zur
Messsemantik. Standard-HOF-Dateien sind nach Panel getrennt. Ohne verifizierten Datenmanifest-Hash
bleiben sie zusätzlich run-lokal, da identische Pfade keine identischen Candle-Inhalte beweisen.

Generic- und Regime-Islands nutzen ihre lokalen Scores ausschließlich zur Suche und lokalen Top-K-
Auswahl. Vor globalem Ranking, HOF und Runner-Export werden eindeutige Phänotypen auf einer
gemeinsamen Base-Pair- beziehungsweise Union-of-Regimes-Matrix erneut bewertet. Der gemeinsame
Regime-Comparator deaktiviert Specialist-Boosts. Ein fehlgeschlagenes oder nicht verfügbares Replay
erzeugt keine globalen Finalisten; `run_ga.py` fällt nicht auf lokale Scores zurück. Merge- und
Tournament-Migration dürfen weiterhin Genome zwischen Inseln austauschen, vergleichen fremde
Score-Skalen aber nicht und erzwingen eine Evaluation im Zielpanel. Der V2-Promotionspfad bleibt
zusätzlich an Code- und Datenmanifest sowie seine deklarierte Pair-x-Zeit-x-Kostenmatrix gebunden.

## 10. Invarianten und Akzeptanztests

Vor Aktivierung müssen mindestens folgende Tests grün sein:

1. Vollständiger `BacktestResult`-Roundtrip; frischer, RAM- und Disk-Cache ergeben exakt dieselben
   Daten und Fitness.
2. Identischer Phänotyp ergibt unabhängig von ID, Generation, Prozess und Workerzahl dasselbe
   Resultat.
3. Sequenzielle und parallele Evaluation sind innerhalb definierter Toleranz identisch.
4. NSGA-II bewertet neue Offspring, bevor Environmental Selection sie verwerfen kann.
5. Train-, Pair-Validation-, Temporal-Validation- und Final-Ranges/Pairs erfüllen den expliziten
   Split-Vertrag und sind auf derselben Beobachtungsachse maschinell nachweisbar disjunkt.
6. Missing, NaN, Inf, Fehler und Null-Trades können keinen Score verbessern.
7. Mehr Nettoprofit bei identischem Risiko verschlechtert den Rang nie.
8. Mehr DD/ES bei identischem Profit verbessert den Rang nie.
9. Eine negative Validation-Ökonomie kann nie promoted werden.
10. Synthetische Equity-Kurven liefern gegen unabhängige Handreferenzen bekannte Return-, DD-,
    Sharpe-, Sortino-, Calmar-, ES- und DD-Dauerwerte; ungültige Risk-free-Raten sowie
    Zero-Variance-/No-Downside-Fälle scheitern definiert.
11. Long/Short-, Hebel-, DCA-, Teilexit- und Funding-Replays reconciliieren sowohl jeden Trade als
    auch die finale Wallet-Equity; Funding-Events summieren sich zum Backtester-Aggregat.
12. All-Winner, All-Loser, One-Trade, No-Trade und No-Loss-Profit-Factor-Grenzfälle sind definiert.
13. Gleiche Monats-/Pairraten bleiben über unterschiedliche Timeranges und Pairzahlen vergleichbar.
14. Surrogate-, SIS- oder importierte HOF-Kandidaten gelangen ohne echten Replay-Backtest weder in
    die finale HOF noch in Promotion.
15. Score- und Gate-Entscheidung können aus dem gespeicherten Resultat deterministisch reproduziert
    werden.

Implementierungsstand 22.07.2026 zu Invariante 13: Der Legacy-GA-Kern persistiert die Herkunft
jeder Fitness und lässt ausschließlich erfolgreiche, finite Backtests über HOF-, Best-Run-,
Holdout-/CPCV-, Finalisten- und Exportgrenzen. Surrogate-Schätzungen bleiben Search-only;
historische Surrogate-HOF-Einträge werden verworfen und injizierte HOF-Genome erneut evaluiert.
Der V2-Promotion-Pfad bewertet eingefrorene Kandidaten weiterhin auf seinem deklarierten Replay-
Panel. Noch offen ist `CORE-005`: Backtests aus unterschiedlichen Insel-/Regimepanels sind trotz
echter Messung nicht global vergleichbar und benötigen vor einer gemeinsamen HOF/Promotion ein
identisches Replay-Panel.

## 11. Einführung

1. Metrikvertrag und Tests implementieren, alte Fitness unverändert weiterlaufen lassen.
2. Vorhandene Topstrategien auf einem festen Replay-Panel mit `legacy` und `v2` parallel bewerten.
3. Rangverschiebungen, abgelehnte bisherige Champions und Risk-Calibrations dokumentieren.
4. V2 zunächst nur als Shadow-Entscheidung verwenden.
5. Erst nach stabiler Reproduzierbarkeit und blindem OOS-Nachweis als Such-/Promotion-Policy
   aktivieren.

Damit wird nicht versprochen, große Verluste auszuschließen. Es wird aber verhindert, dass das
System fehlende Daten, In-Sample-Holdout, hohe Winrate oder zufällige Fitnessunterschiede als
Risikobeweis missversteht.
