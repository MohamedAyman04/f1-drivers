# Milestone 3 — Part 2: Graph Retrieval, Podium Prediction Explainer, QA Test Set & Stage-by-Stage Error Analysis

## 1. Integration with Part 1 (`m3/intents.py` & `m3/entities.py`)

Member 2 (`m3/retrieval.py` and `m3/podium.py`) integrates directly with Member 1's deployed classifier (`DEPLOYED = HybridClassifier`) and entity extractor (`extract`, `merge`, `clarification`, `resolve_choice`):

```python
from m3.retrieval import PipelineSession
from m3.podium import PodiumPredictor

predictor = PodiumPredictor()
session = PipelineSession(predictor=predictor)
turn = session.step("How many wins did Lewis Hamilton have at Silverstone before 2020?")
```

### Alignment Summary
- **Shared 10-Intent Label Set (`m3.intents.LABELS`)**: `driver_performance`, `constructor_performance`, `head_to_head`, `race_facts`, `qualifying_grid`, `pit_stops`, `season_history`, `driver_profile`, `podium_prediction`, and `out_of_scope` (plus 11 granular sub-intent families routing across **30 parameterized Cypher templates**).
- **Entity-Cache Contract Enforced (`filter_active_entities`)**:
  - `m3.entities.merge(cache_entities, ex.entities)` keeps cached slots across turns and replaces only the slots named in the new message.
  - Each template reads only the slots its intent requires plus its valid optional filters.
  - On a full new question (`intent.status != "follow_up"`), leftover optional slots from earlier turns (such as a cached `Silverstone` circuit when the user asks *"How many times did Leclerc qualify on pole in 2022?"* or *"How many points did he score in 2019?"*) are cleared unless explicitly mentioned in `ex.entities` or deictically referenced (`"there"`, `"that circuit"`, `"same period"`).
  - On an elliptical follow-up (`intent.status == "follow_up"`, e.g. *"And in 2020?"*, *"What about Norris?"*), all merged cache slots for that intent remain active.
  - Out-of-scope questions (`intent.status == "out_of_scope"`) return a refusal with no retrieval trace and leave `cache_intent` and `cache_entities` untouched.

---

## 2. Question Types (Intents) & Parameterized Cypher Templates (Section 2)

### 2.1 Intent & Template Catalog

- **Total Intents**: 10 canonical intents (`m3.intents.LABELS`) subdivided into 12 retrieval/routing sub-families across **30 parameterized Cypher templates**.
- **Section 2.a Requirements Check**:
  - Covers all **7 table categories** plus **`driver_profile`** (extra type), **`podium_prediction`**, and **`out_of_scope`**.
  - **Multi-hop**: All **8** graph-retrieval intents (and all 10 retrieval sub-families) traverse **2 to 3 hops** across entity and fact nodes (minimum required: 3).
  - **Multi-entity**: **7** of the 8 graph-retrieval intents accept **2 to 4 entities** (minimum required: 2).
  - **New relative to Milestone 2**: **7** of the 10 intents (and 25 of the 30 Cypher templates) are new relative to the 5 Milestone 2 verification queries (minimum required: 6).

| Intent (`m3.intents.LABELS`) | Brief Row / Category | Entities Used | Hops | New vs. M2? | Parameterized Cypher Templates | Example Question & Live Neo4j Result |
| :--- | :--- | :--- | :-: | :-: | :--- | :--- |
| `driver_performance` | Driver performance | `drivers` (req); `circuits`, `grand_prix`, `season`, `season_range` | 3 | **Yes** (adds circuit + cutoff & career/season podiums) | `driver_season_or_career_stats`, `driver_at_circuit_stats`, `season_driver_leaderboard`, `circuit_top_drivers_by_avg_finish` | *"How many podiums did Charles Leclerc get in 2022?"* $\to$ `{Driver: 'Charles Leclerc', Entries: 22, Starts: 22, Points: 291.0, Wins: 3, Podiums: 11, Avg_Finish: 5.5}` |
| `constructor_performance` | Constructor performance | `constructors` (req, 1 or 2+ teams); `circuits`, `season`, `season_range` | 3 | Extends M2 Q2 | `constructor_period_performance`, `constructor_by_season_performance`, `all_constructors_reliability_ranking` | *"How reliable was Williams between 2014 and 2021?"* $\to$ `{Constructor: 'Williams', Entries: 320, Classified: 266, Reliability: 0.8313, Points: 829.0, Podiums: 16}` |
| `head_to_head` | Head-to-head | 2 `drivers` (req); `constructors`, `season`, `season_range` | 3 | Extends M2 Q3 | `teammate_race_head_to_head`, `any_shared_race_head_to_head` | *"Who finished ahead more often, Lando Norris or Daniel Ricciardo at McLaren?"* $\to$ `{Constructor: 'McLaren', Shared_Races: 44, Driver_1_Ahead: 32, Driver_2_Ahead: 12}` |
| `race_facts` | Race and circuit facts | `grand_prix` + `season`, or `circuits` / `country` (`season_range`) | 2–3 | **Yes** | `race_podium_and_results`, `circuit_profile_and_history`, `circuit_winners_in_period`, `circuits_by_country` | *"Who won the 2019 German Grand Prix?"* $\to$ `P1: Max Verstappen (Red Bull, Grid 2), P2: Sebastian Vettel (Ferrari, Grid 20), P3: Daniil Kvyat (Toro Rosso, Grid 14)` |
| `qualifying_grid` | Qualifying and grid | `drivers`, `constructors`, `circuits`, `grand_prix` + `season`, `season_range` | 3 | **Yes** | `race_grid_to_finish_gains`, `race_grid_to_finish_losses`, `qualifying_race_results`, `driver_qualifying_stats`, `constructor_qualifying_stats`, `circuit_front_row_to_podium_conversion` | *"Which driver gained the most places from the grid at Monza in 2021?"* $\to$ `{Driver: 'Valtteri Bottas', Constructor: 'Mercedes', Grid_Position: 19, Finish_Position: 3, Places_Gained: 16}` |
| `pit_stops` | Pit stops and strategy | `grand_prix` + `season`, `circuits` + `season`, `drivers`, or `season` | 3 | **Yes** | `race_fastest_pit_stops`, `race_slowest_pit_stops`, `race_pit_stop_averages`, `driver_season_pit_stop_stats`, `season_pit_stop_summary` | *"What was the fastest pit stop at the 2023 Qatar Grand Prix?"* $\to$ `{Driver: 'Oscar Piastri', Stop_Number: 3, Lap: 43, Duration_Text: '27.489', Duration_ms: 27489, Duration_Seconds: 27.489}` |
| `season_history` | Season and history | `season` or `season_range` (req) | 3 | Extends M2 Q5 | `driver_team_changes_between_seasons`, `season_overview_and_leader`, `period_points_leaders` | *"Which drivers changed teams between 2020 and 2021?"* $\to$ `{Drivers_Changed: 6, Drivers: ['Carlos Sainz', 'Daniel Ricciardo', 'Esteban Ocon', 'Lance Stroll', 'Sebastian Vettel', 'Sergio Pérez']}` |
| `driver_profile` | **Extra type** | `drivers` (req) | 3 | **Yes** | `driver_profile_and_career_summary` | *"List every team Kimi Raikkonen has raced for."* $\to$ `{Driver: 'Kimi Räikkönen', Nationality: 'Finnish', Date_Of_Birth: '1979-10-17', Code: 'RAI', Car_Number: 7, Teams_Driven_For: ['Sauber', 'McLaren', 'Ferrari', 'Lotus F1', 'Alfa Romeo'], Debut_Season: 2001, First_Grand_Prix: '2001 Australian Grand Prix'}` |
| `podium_prediction` | Prediction path (Section 4) | `drivers` (req); `grand_prix`, `circuits`, `season`, `grid`, `champ_pos`, `wins`, `points` | 3 | **Yes** | `pre_race_driver_circuit_context`, `pre_race_constructor_context` (`r.date < date($before_date)`) | *"Predict Norris at the 2024 Miami Grand Prix."* $\to$ $P(\text{podium}) = 0.2701$ vs. threshold `0.432` (**Not Podium**) + live SHAP/LIME + pre-race KG facts |
| `out_of_scope` | Refusal | — | 0 | **Yes** | None (short-circuited before Neo4j) | *"What is Hamilton's salary?"* $\to$ Refusal message, no retrieval trace |

---

## 3. Saved Milestone 1 Podium Model & Live SHAP/LIME + Pre-Race KG Explainer (Section 4)

### 3.1 Saved Model Bundle & Pre-Race Feature Table
- `scripts/export_m1_artifacts.py` exports the exact Milestone 1 preprocessing pipeline (`SimpleImputer(strategy="median")` + `StandardScaler()` fitted on train seasons 1950–2017) and tuned models to `artifacts/m1_podium_bundle.joblib`, plus the 24,933 leakage-free driver-race pre-race rows to `artifacts/pre_race_features.csv`.
- `m3/podium.py` (`PodiumPredictor`) loads the saved artifacts from disk without fitting or retraining any model inside the application.
- **Thresholds (from Milestone 1 validation F1)**:
  - `Random forest` (default, Test PR-AUC `0.720`): **`0.432`** (base value `0.1418`)
  - `Logistic regression` (Test PR-AUC `0.708`): **`0.399`** (base value `0.1398`)
  - `Baseline (LR on grid)` (Test PR-AUC `0.647`): **`0.427`**
- **Training Medians (used for grayed default pre-fill in Chat mode)**:
  - `grid = 12.0`, `drv_pos = 12.0`, `drv_pts_share = 0.076` (`0.0755`), `drv_wins = 0.0`.

### 3.2 Manual Feature Input Ranges & Justification (Section 4.a.2)

| Feature | Label | Type | Valid Range | Default (Median) | Data-Driven Justification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `grid` | Starting Grid Position | Integer | `[1, 34]` | `12` | Pole position is 1; largest championship F1 starting grid (excluding Indy 500) in 1950–2024 is 34 cars. |
| `drv_pos` | Driver Championship Position Before Race | Integer | `[1, 108]` | `12` | Championship standings ranks at `round - 1` span 1 to 108 across 1950–2024 (1 to 102 in train). |
| `drv_pts_share` | Share of Championship Leader's Points Before Race | Float | `[0.0, 1.0]` | `0.076` | Ratio `points / max(points)` at `round - 1` is mathematically bounded in $[0.0, 1.0]$. |
| `drv_wins` | Season Grand Prix Wins Before Race | Integer | `[0, 24]` | `0` | Prior wins in the current season cannot be negative and a season has at most 24 rounds. |

### 3.3 Evaluated Driver-and-Race Podium Explanation Cases (`8/8 PASSED — 100.0%`)

| Case ID | User Query / Scenario | Extracted Pre-Race Features | $P(\text{podium})$ (Thr `0.432`) | Predicted / Actual | Live SHAP Contributions (`grid`, `drv_pos`, `drv_pts_share`, `drv_wins`) | Live LIME ($R^2$) | Pre-Race KG Context (`r.date < race_date`) |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **POD-01** | *"Predict Max Verstappen at the 2024 Saudi Arabian Grand Prix."* | `grid=1`, `drv_pos=1`, `drv_pts_share=1.000`, `drv_wins=1` | **0.6441** | **Podium** / P1 (Podium) | `+0.3422`, `+0.0792`, `+0.0757`, `+0.0052` | `+0.2631`, `+0.0739`, `+0.0573`, `+0.0200` ($0.818$) | **Jeddah** (< `2024-03-09`): 3 starts, 1 win, 3 podiums, avg 1.67. **Red Bull**: 83/90 classified (92.2%). |
| **POD-02** | *"Predict Charles Leclerc at the 2022 Bahrain Grand Prix."* (Round 1 medians) | `grid=1`, `drv_pos=12*`, `drv_pts_share=0.076*`, `drv_wins=0*` | **0.4767** | **Podium** / P1 (Podium) | `+0.3362`, `+0.0040`, `-0.0031`, `-0.0022` | `+0.2625`, `-0.0136`, `-0.0054`, `-0.0158` ($0.753$) | **Bahrain** (< `2022-03-20`): 5 starts, 0 wins, 1 podium, avg 10.2. **Ferrari**: 70/78 classified (89.7%). |
| **POD-03** | *"Predict Fernando Alonso at the 2023 Monaco Grand Prix."* | `grid=2`, `drv_pos=3`, `drv_pts_share=0.630`, `drv_wins=0` | **0.5566** | **Podium** / P2 (Podium) | `+0.3125`, `+0.0744`, `+0.0321`, `-0.0042` | `+0.2666`, `+0.0725`, `+0.0545`, `-0.0177` ($0.785$) | **Monaco** (< `2023-05-28`): 17 starts, 2 wins, 4 podiums, avg 6.47. **Aston Martin**: 83/98 classified (84.7%). |
| **POD-04** | *"Predict Norris at the 2024 Miami Grand Prix."* | `grid=5`, `drv_pos=5`, `drv_pts_share=0.527`, `drv_wins=0` | **0.2701** | **Not Podium** / P1 (Surprise Win) | `+0.0775`, `+0.0425`, `+0.0104`, `-0.0021` | `+0.2606`, `+0.0719`, `+0.0541`, `-0.0213` ($0.828$) | **Miami** (< `2024-05-05`): 2 starts, 0 wins, 0 podiums, avg 18.0. **McLaren**: 89/98 classified (90.8%). |
| **POD-05** | *"Predict Nico Hülkenberg at the 2024 British Grand Prix."* | `grid=6`, `drv_pos=12`, `drv_pts_share=0.059`, `drv_wins=0` | **0.1192** | **Not Podium** / P6 (Not Podium) | `+0.0231`, `-0.0165`, `-0.0263`, `-0.0030` | `+0.2623`, `-0.0124`, `-0.0283`, `-0.0157` ($0.767$) | **Silverstone** (< `2024-07-07`): 12 starts, 0 wins, 0 podiums, avg 9.67. **Haas**: 93/110 classified (84.5%). |
| **POD-06** | *"Predict Pierre Gasly at the 2024 Monaco Grand Prix."* | `grid=10`, `drv_pos=19`, `drv_pts_share=0.000`, `drv_wins=0` | **0.0217** | **Not Podium** / P10 (Not Podium) | `-0.0677`, `-0.0285`, `-0.0215`, `-0.0024` | `-0.0563`, `-0.0276`, `-0.0150`, `-0.0197` ($0.054$) | **Monaco** (< `2024-05-26`): 5 starts, 0 wins, 0 podiums, avg 7.2. **Alpine**: 83/102 classified (81.4%). |
| **POD-07** | Manual validation: `grid=42, drv_pos=1, drv_pts_share=1.35, drv_wins=0` | Refused | `None` | Refused | — | — | Refused with clear range error message on `grid` $[1, 34]$ and `drv_pts_share` $[0.0, 1.0]$. |
| **POD-08** | Manual validation: `grid=3, drv_pos=null, drv_pts_share=0.5, drv_wins=1` | Refused | `None` | Refused | — | — | Refused with clear missing-value message on `drv_pos`. |

---

## 4. QA Test Set Evaluation & Stage-by-Stage Error Analysis (Section 6)

### 4.1 Evaluation Summary Across All Test Sets
1. **Member 2 QA Test Set (`m3/eval/qa_testset.json` — 60 items)**: **`60/60 PASSED (100.0%)`**
   - **44 Core In-Domain Queries**: 4+ distinct questions per retrieval intent (40 core items exported to `m3/eval/results/qualitative_sheet_40_items.csv` + 4 `driver_profile` items). Gold facts are dynamically computed at evaluation time by executing the parameterized Cypher template against Neo4j.
   - **4 Follow-Up Queries**: Tests session entity cache retention and slot replacement.
   - **4 Ambiguous-Entity Queries**: Tests multi-driver surname disambiguation (`Schumacher`, `Verstappen`, `Rosberg`, `Hill`).
   - **4 Empty-Result Queries**: Tests valid entity combinations with 0 matching graph paths (Hamilton in 1955, Verstappen at Pescara, Norris and Fangio as teammates, constructor transfers between 1950 and 2024).
   - **4 Out-of-Scope Queries**: Tests refusal on salaries, tire surface telemetry, sprint shootout points, and non-F1 topics.
2. **End-to-End Member 1 $\to$ Member 2 Integration (`m3/eval/intent_testset.csv` — 66 items & `m3/eval/intent_heldout.csv` — 22 items)**:
   - All **66** test-set questions and **22** held-out questions execute end-to-end through `DEPLOYED` + `extract` + `merge` + `filter_active_entities` + `clarification` + `GraphRetriever` / `PodiumPredictor` with zero unhandled exceptions (`66/66` and `22/22`).

### 4.2 Stage-by-Stage Error Analysis

| Stage | Failure Mode / Edge Case | Concrete Example from Evaluation | Root Cause | Fix / Mitigation Applied |
| :--- | :--- | :--- | :--- | :--- |
| **1. Intent Classification** | **1. Keyword collision across intents.**<br>**2. Margin abstention on close embedding neighbours.** | 1. *"Number of Grand Prix victories for Nico Rosberg in 2016?"* routed to `driver_profile` by `RuleClassifier` (matched keyword `"number"` = car number).<br>2. *"On average where did Alonso finish at the Hungaroring?"* got `clarify` in `EmbeddingClassifier` (`driver_performance` `0.80` vs. `head_to_head` `0.79`, margin `< 0.03`). | 1. Polysemous words (`"number"`, `"country"`, `"stops"`).<br>2. Masked sentence shapes for short questions sit near the 0.03 margin boundary. | Deployed `HybridClassifier` (`RuleClassifier` primary + masked `EmbeddingClassifier` fallback), lifting test accuracy from `81.8%` (embedding) / `97.0%` (rules) to **`98.5%`** (`65/66`) and held-out accuracy to **`77.3%`**. |
| **2. Entity Extraction & Resolution** | **1. `driverRef` vs. `surname` collision on multi-driver F1 families.**<br>**2. Stale optional slots leaking across non-follow-up turns in session cache.** | 1. Querying `"Verstappen"` (`QA-50`), `"Rosberg"` (`QA-51`), or `"Hill"` (`QA-52`) without a season initially matched `Jos Verstappen` (`driverRef='verstappen'`), `Nico Rosberg` (`driverRef='rosberg'`), and `Graham Hill` (`driverRef='hill'`) instead of asking the user to clarify.<br>2. After *"How many wins did Lewis Hamilton have at Silverstone before 2020?"*, asking *"How many points did he score in 2019?"* kept `circuits=[Silverstone]` in `merge(cache, ex.entities)` and filtered 2019 points to Silverstone only (`44.0` pts instead of `413.0` pts). | 1. In `drivers.csv`, the earlier driver in a family often has the bare surname as `driverRef`.<br>2. `merge()` preserves all non-overwritten cached slots so pronouns (`"he"`) work, which also leaves old optional venue/time filters in the dictionary. | 1. Both `m3/entities.py` and `GraphRetriever.resolve_driver` check all surname candidates first and narrow only by facts explicitly present in the message (active season or constructor); if $>1$ candidate remains, the user is asked to clarify.<br>2. Added `filter_active_entities` in `m3/retrieval.py` to drop unreferenced optional venue/time slots on full new questions (`not is_follow_up_turn`) while retaining required pronoun-referenced slots. |
| **3. Cypher Query & Empty-Result Retrieval** | **1. Un-grouped Cypher aggregations returning 1 row of zeros on empty matches.**<br>**2. String sorting on `PitStop.duration`.**<br>**3. Historical pre-1994 pole positions missing from `Qualifying` nodes.** | 1. *"How many Grand Prix points did Lewis Hamilton score in 1955?"* (`QA-53`) or *"And in 2018?"* after *"Which events have been staged at Imola?"* (Imola hosted no race in 2018).<br>2. Lexicographical sorting on `p.duration` ordered `"1:02.345"` before `"27.489"`.<br>3. *"How many pole positions did Senna take?"* returned fewer poles if only `Qualifying` nodes (populated from 1994+) were counted. | 1. Cypher aggregations over 0 paths emit `{Entries: 0, Points: 0}` if not grouped or post-checked.<br>2. `PitStop.duration` is a string while `PitStop.milliseconds` is an integer.<br>3. `qualifying.csv` starts in 1994, whereas `Result.grid = 1` covers 1950–2024. | 1. Grouped aggregations by entity node AND added zero-count checks in `GraphRetriever._run_template` so 0-entry rows return `is_empty = True` and `"The knowledge graph has no data for that question."`<br>2. Sorted all pit-stop templates by integer `p.milliseconds`.<br>3. Included both `Grid_Poles` (`res.grid = 1`) and `Qualifying_Poles` (`q.position = 1`) in `driver_qualifying_stats` and `constructor_qualifying_stats`. |
| **4. LLM Misreading Retrieved Rows** | **Misreading decimal proportions (`Reliability`, `Front_Row_Podium_Rate`) or swapping `Driver_1_Ahead` / `Driver_2_Ahead`.** | Given `Reliability: 0.8313` for Williams (2014–2021), an unguided zero-shot model can report `"0.83% reliability"` instead of `"83.13% (0.8313)"`. | Ratios in $[0, 1]$ without explicit column names or prompt instructions can be misread as pre-multiplied percentages. | Named all returned Cypher columns explicitly (`Driver_1`, `Driver_1_Ahead`, `Reliability`, `Classified_Start_Share`, `Duration_Seconds`) and added explicit unit notes in the prompt task. |
| **5. Hallucination & Causal Language in Explanations** | **1. Adding sprint points from parametric memory.**<br>**2. Using causal verbs (`"caused"`, `"because of"`) when explaining SHAP/LIME outputs.** | 1. Reporting Max Verstappen's 2021 points as `395.5` (which includes sprint points) instead of the Knowledge Graph's `388.5` (Grand Prix results only).<br>2. Stating *"Starting on pole caused Verstappen to finish on the podium."* | 1. LLM pre-training includes official FIA totals with sprints.<br>2. LLMs naturally translate positive feature attributions into causal claims. | 1. Short-circuited empty results before calling the LLM and instructed the prompt to quote only context numbers and state that figures cover Grand Prix results only.<br>2. Enforced strict non-causal wording in `PodiumPredictor` and verified zero matches against `CAUSAL_BANNED_PATTERNS` across all explanation test cases. |
