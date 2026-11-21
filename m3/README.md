# Milestone 3 — Part 1: intent classification and entity extraction

```bash
.venv/Scripts/python -m pip install -r m3/requirements.txt   # torch CPU wheel: --index-url https://download.pytorch.org/whl/cpu
python m3/entities.py        # extractor self-checks
python m3/intents.py         # rule-classifier self-checks
python m3/eval_intents.py    # both classifiers on the labelled sets -> m3/eval/results/
```

| File | What |
|---|---|
| `m3/intents.py` | label set + example phrasings, `RuleClassifier`, `EmbeddingClassifier`, `HybridClassifier`, `DEPLOYED` |
| `m3/entities.py` | gazetteer from `data/raw/*.csv`, `extract()`, `clarification()`, `resolve_choice()`, `merge()`, `mask()` |
| `m3/eval/intent_testset.csv` | 66 labelled questions (≥5 per intent, follow-ups, ambiguous names, OOS) |
| `m3/eval/intent_heldout.csv` | 22 more, written before the improvement round (generalisation check) |
| `m3/eval/results/` | `intent_report.md` (accuracy, confusion matrices, errors, sweep) + 3 confusion PNGs |

## API for the rest of the pipeline

```python
from m3.intents import DEPLOYED
from m3.entities import extract, clarification, resolve_choice, merge

clf = DEPLOYED()                                   # load once (embedding model ~90 MB, cached by HF)
intent = clf.classify(msg, prev=cache_intent)      # Intent(label, status, scores, reason)
ex = extract(msg)                                  # Extraction(entities, ambiguous, assumed, spans)
entities = merge(cache_entities, ex.entities)      # follow-up: replace only the slots the message names
ask = clarification(intent.label, ex, entities)    # str -> ask the user; None -> run the template
```

* `intent.status`: `ok`, `follow_up` (cached intent kept), `out_of_scope` (refuse, no trace), `clarify` (`label is None`: ask the user to rephrase).
* `entities`: `drivers`/`constructors`/`circuits` are lists of `{id, name}` with the graph's `driverId`/`constructorId`/`circuitId`; `grand_prix` is the race name exactly as in `Race.name`; `season` (int) or `season_range` ([from, to], inclusive). A race is `grand_prix` + `season`, or `circuit` + `season`. Prediction messages may also carry `grid`, `champ_pos`, `wins` and `points`.
* `ex.ambiguous` → `clarification()` lists the candidates; pass the user's reply to `resolve_choice(reply, candidates)`. `ex.assumed` holds notes like *"hamilton" read as Lewis Hamilton (356 entries vs 5)*; show them in the trace.

## Intents (10)

| Intent | Brief row | Entities required (else ask) | Example |
|---|---|---|---|
| driver_performance | Driver performance | driver | How many podiums did Leclerc get in 2022? |
| constructor_performance | Constructor performance | constructor | How reliable was Williams between 2014 and 2021? |
| head_to_head | Head-to-head | 2 drivers | Who finished ahead more often, Norris or Ricciardo at McLaren? |
| race_facts | Race and circuit facts | GP + year, or circuit | Who won the 2019 German Grand Prix? |
| qualifying_grid | Qualifying and grid | any of driver / team / circuit / race / season | Which driver gained the most places from the grid at Monza in 2021? |
| pit_stops | Pit stops and strategy | race, driver or season | What was the fastest pit stop at the 2023 Qatar Grand Prix? |
| season_history | Season and history | season or range | Which drivers changed teams between 2020 and 2021? |
| driver_profile | **extra type** | driver | Which teams has Fernando Alonso driven for? |
| podium_prediction | prediction path | driver | Predict Norris at the 2024 Miami Grand Prix |
| out_of_scope | refusal | — | What is Hamilton's salary? |

`driver_profile` (nationality, birth date, number, code, teams driven for, debut) is answerable from `Driver` properties plus `Result-[:FOR_CONSTRUCTOR]->Constructor`. **Refused (out of scope):** anything outside F1, and F1 topics with no data in the graph: salaries and contracts, tyres, weather, sprint races, lap times and lap records, penalties and safety cars, car technology, drivers' personal lives.

## The two classifiers

**Rule-based.** Weighted regexes per intent (keywords plus paraphrase families such as "fare against", "finished higher", "took victory", "dropped … spots"), plus entity-count cues: two drivers push toward head_to_head; a team and no driver push toward constructor_performance. A tie or no keyword means it asks the user to rephrase.

**Embedding similarity.** `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dimensional, CPU, no API key). It's trained for sentence similarity, so it suits short paraphrased questions. The ~100 example phrasings are embedded once at start-up. A message goes to the intent of its nearest example by cosine similarity. If the best score is below **0.45**, or within **0.03** of the second-best intent, it asks the user to rephrase. Both thresholds were fixed before evaluation; the sweep in the report is a sensitivity check, not tuning. Names and years are **masked** before embedding ("how many wins did *driver* have at *circuit* in *year*"), so similarity reflects the question's shape rather than who it's about.

**Shared rules (identical in both classifiers):**
- **Follow-ups.** If, after masking, a message has nothing left but filler ("And in 2020?", "What about Norris?", "2021?"), it names no new intent, so the cached one is kept.
- **Out of scope.** Decided only by rules: either no F1 entity or term at all, or an explicit topic the graph has no data for. **Similarity alone never marks a message out of scope.** When the nearest example is OOS but no rule confirms it, the classifier asks the user to rephrase.

## Results and the deployed classifier

| Classifier | Test (66) first run | Test final | Rephrase rate | Held-out (22) first run | Held-out final |
|---|---|---|---|---|---|
| rule-based | 71.2% | 97.0% | 1.5% | 59.1% | 72.7% |
| embedding (masked) | 74.2% | 84.8% | 9.1% | 63.6% | 63.6% |
| **hybrid: rules, embedding fallback** | — | **98.5%** | 0.0% | — | **81.8%** |

Confusion matrices: `m3/eval/results/confusion_*.png`. Ablation: the embedding classifier **without masking** scores 59.1% on the first run and 68.2% on the final run, against 74.2% and 84.8% with masking.

**Deployed: the hybrid.** Rules decide when they match; the embedding classifier only handles messages the rules can't decide.
- The test-set figures for the rules are optimistic, because the improvement round added paraphrase patterns for the families that failed there. The held-out set is the honest comparison: rules 72.7%, embedding 63.6%, hybrid 81.8%.
- Rules are precise when they match. Their errors are mostly "no keyword", which is exactly where the embedding classifier generalises, so the fallback recovers those without adding wrong labels.
- Most of the embedding classifier's own errors are the margin rule asking to rephrase between related intents (driver_performance vs head_to_head, pit_stops vs driver_performance). The sweep shows margin 0 would reach 92.4%, but the margin is a required safety check, so it stays.
- Both parts are local and deterministic: no API key, no network after the first model download, milliseconds per message.

## Improvements made (before → after)

1. **Masking entities before embedding:** 59.1% → 74.2% (first run). Without it, "Hamilton" pulled questions toward whichever examples mentioned Hamilton.
2. **Out-of-scope decided by rules in a shared gate.** The first embedding run sent "What's Lewis Hamilton's favourite food?" to driver_profile and "safety car in the 2016 Brazilian GP" to race_facts. The fix makes an explicit unanswerable topic decide out of scope for both classifiers, and also adds other sports (Tour de France) to the non-F1 list.
3. **Paraphrase families in the rules** ("against / fare / finished higher", "dependable / retire", "started first / starting slot / climber", "topped the table", "represent", "going to … podium", "forecast / estimate / the model"). Prediction cues also got a higher weight, because future tense has to beat the fact keywords it appears with ("podium at the … GP"). Rules: 71.2% → 97.0% on the test set and 59.1% → 72.7% on the held-out set.
4. **Ten more embedding examples** (pronouns, informal phrasing) for the weakest intents: 74.2% → 84.8% on the test set. Held-out stayed at 63.6%.
5. **Entity disambiguation without guessing.** A surname shared by several drivers is first narrowed by season ("Schumacher in 2021" → Mick) and by a team named in the message ("Verstappen at Red Bull" → Max). If several still remain, the user is asked ("Schumacher" → Michael, Ralf or Mick). The only automatic choice is when one driver has at least 5× the race entries of every namesake (Lewis vs Duncan Hamilton), and that choice is reported in `assumed`.

Honesty note: I read the held-out errors after the first improvement round. The prediction-weight change in (3) fixed a test-set error that has the same root cause as one held-out item, so that held-out item isn't fully independent.

## Entity extraction details

- **Sources:** gazetteers come from `drivers/constructors/circuits/races/results.csv`, the same rows the Milestone-2 graph was loaded from, so the ids match the graph. Text is normalised: lowercase, accents stripped ("Perez" = Pérez, "Raikkonen" = Räikkönen), and "GP" read as "Grand Prix". Matching takes the longest span first, so "Red Bull Ring" isn't read as Red Bull and "Mick Schumacher" isn't read as an ambiguous "Schumacher".
- **Aliases:**
  - Drivers: full name, surname, driverRef.
  - Constructors: name, ref, compact form ("redbull", "astonmartin") and a short sponsor list ("mercedes amg", "merc", "kick sauber", "vcarb").
  - Circuits: name, ref, city/location, name without generic words ("Yas Marina"), and country.
  - Races: Grand Prix name, plus "US GP" etc.
- **Surname vs team:** a surname that is also a team name means the team ("McLaren", "Williams"). Surnames that are ordinary English words ("Speed", "March", "Lewis" → Jackie Lewis) are never matched alone.
- **Seasons:** "in 2019", "between 2014 and 2021", "2014-2021", "from … to", "before 2020" → [1950, 2019], "after / since / until", "the 2010s".
- **Missing entities:** each intent has a required-slot check, and a failure gives a specific question ("Which year's German Grand Prix?", "Which two drivers should I compare?").

## Remaining limitations

- The held-out set still fails on two out-of-scope questions that use F1-sounding words but no listed topic ("Who will **win** the next election?", "How much **fuel** does an F1 car carry?"). The generic words "win" and "F1" count as F1 signal, and no rule names elections or fuel. An LLM check or a larger OOS example bank would be needed to catch these without hurting in-domain recall.
- The rules are a hand-maintained list. New phrasings ("top step", "stronger") fall through to the embedding classifier, which often asks to rephrase instead of answering.
- A follow-up that only names one driver replaces the whole driver list. In a head-to-head, "What about Bottas?" therefore leaves one driver, and the user gets asked for the second.
- There's no fuzzy matching for typos ("Verstapen") and no relative time ("last season").
