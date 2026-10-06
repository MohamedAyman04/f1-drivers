# CSEN903 Milestone 1: F1 Podium Prediction Companion. Project Plan

> **Grounded in the course labs.** Tools and methods come from Lab 1 (Data Wrangling & Visualization), Lab 2 (ML Foundations), Lab 3 (Neural Networks with TensorFlow/Keras) and Lab 4 (Explainable AI). See §0.3 for the mapping. Anything beyond the labs is marked **[beyond labs]** with a reason.
>
> **Living plan.** This plan does not fix every choice up front. Each phase has **Decision Gates**: a check to run, the evidence (plot or statistic) it produces, and a rule mapping each possible outcome to an action. Record every gate outcome in the **Decision Log** (§12) so the notebook, the report and the oral evaluation can trace each choice back to its evidence.

> ### ⚠️ Hard constraint: one standalone Python notebook on Kaggle
> - **The deliverable is a single `.ipynb` that runs fully on Kaggle with Run All.** All code lives in the notebook: no external `.py` modules, no `src/` folder, no helper scripts, no SQL database, no shell steps, no manual preprocessing done outside the notebook.
> - **Python only.** Every step (loading, cleaning, joins, statistics, plots, models, XAI, inference) is written in Python inside the notebook, using pandas for joins (no SQL).
> - **The only input is the 14 raw CSVs**, attached to the Kaggle notebook as a Kaggle dataset (*Formula 1 World Championship 1950–2024*, `rohanrao/formula-1-world-championship-1950-2020`) and read from `/kaggle/input/...`. `data/f1.db` is **not used** in the project.
> - **Nothing is pre-computed offline.** Cleaned tables, the modeling table, and trained models are rebuilt in memory on every run. Anything the notebook writes goes to `/kaggle/working/` and is regenerated each run, so nothing depends on it existing beforehand.
> - **Libraries: the course toolkit from Labs 1–4** (§0.3). That's pandas, numpy, matplotlib, seaborn, missingno, scikit-learn, TensorFlow/Keras, shap and lime. The first cell runs `!pip install -q shap lime missingno`, exactly as Lab 4 does, so Internet must be **On** in the Kaggle notebook settings. Anything outside the labs is used only when the brief explicitly requires it, and is marked **[beyond labs]** with a one-line justification in the notebook.
> - **Follow the lab coding patterns** so every team member can explain each cell in the evaluation:
>   - Inspect with `df.info()`, `describe()`, `isnull().sum()` and `value_counts()`.
>   - Do one step per cell.
>   - Fit scalers and imputers on train only, then `transform` val/test.
>   - Set seeds with `np.random.seed(42)` and `tf.random.set_seed(42)`.
>   - Build the NN with `keras.Sequential` plus `EarlyStopping(restore_best_weights=True)`.
>   - Plot confusion matrices with `sns.heatmap`.
>   - Use `shap.summary_plot`, `force_plot` and `waterfall` for SHAP, and `LimeTabularExplainer` for LIME.
> - **The decision log, report text, and all justifications** are written as Markdown cells inside the notebook. The PDF report and GitHub repo are just exports and copies of the same notebook and its figures.

---

## 0. Ground Truth: Spec Summary and Known Data Facts

### 0.1 Task definition (from the brief)
| Item | Value |
|---|---|
| Unit of analysis | one driver entering one race, `(raceId, driverId)` unique |
| Target | `podium = positionOrder <= 3` (binary) |
| Prediction cutoff | **after qualifying, before race start** |
| Forbidden as features | current-race results, status, points, laps, pit stops, lap times, post-race standings |
| Models | ≥2 statistical ML models + 1 shallow NN (1–2 hidden layers), same rows and same features |
| Metrics | ROC-AUC, PR-AUC, F1. The brief also says "all five metrics", so we add **Precision** and **Recall** (as in Lab 2's `classification_report`), plus Accuracy for reference only, since Lab 2 shows it misleads on imbalanced data. All reported on train, validation **and** test |
| Rules | thresholds and hyperparameters chosen on train/val only; raw CSVs immutable; notebook runs top to bottom with Run All |
| Deadline | **18 Oct 2026, 23:59**. Submit Kaggle notebook link + GitHub repo + PDF report |

### 0.2 Facts already observed in `data/` (quick read-only profile)
| Fact | Implication |
|---|---|
| 14 CSVs in `data/raw/` (the local copy of the Kaggle dataset) | These are the **only** input. Read-only; never overwrite. (`data/f1.db` exists locally but is ignored, see the hard constraint) |
| Seasons 1950–2024, 1,125 races, 26,759 result rows | Long history, so era effects are significant |
| **85** duplicate `(raceId, driverId)` pairs in `results`, mostly 1950–1958 (shared drives), last one in 1978 | Need a dedup rule (Gate 2.C) |
| `qualifying` starts in **1994**, very sparse in 1996–2002 (e.g. 2001: 1/17 races), full from 2003 | Main driver of the **modeling start-year decision** (Gate 3.A) |
| `lap_times` from 1996, `pit_stops` from 2011 | Only usable as *historical* aggregates, and only in those eras |
| `\N` sentinel is widespread (e.g. 10,953 in `results.position`, 6,819 in `qualifying.q3`) | Sentinel conversion needed in every table |
| 1,638 rows with `grid = 0` | Mix of pit-lane starts and non-starters/DNQ, so semantics must be resolved (Gate 2.D) |
| Entries per race range 10–55 (avg 23.8) | Indy 500 (1950–60) and DNQ-era fields inflate counts (Gate 2.E) |
| Overall podium rate ≈ **12.7%** | Imbalanced classes, so PR-AUC matters and thresholds need tuning |
| Circuit countries are inconsistent (`USA` vs `United States`, `UK`); driver `nationality` is a demonym (`British`) | Need a nationality→country map for Q2 |
| `status` has 139 values, mixing mechanical, accident, driver, regulatory and "+N Laps" | Need a status taxonomy for Q3 and for the "started" definition |

### 0.3 Course toolkit: what each lab gives us and where we use it
**Rule:** the default tool for every step is the one taught in the labs. A tool from outside the labs is allowed only if (a) the brief requires it (e.g. PR-AUC, ROC-AUC, permutation importance) or (b) no lab tool can answer the question. It must come from a library we already use (sklearn/numpy/pandas), and it's tagged **[beyond labs]** with the reason.

| Lab | Taught tools / methods | Where we use them in this project |
|---|---|---|
| **Lab 1: Data Wrangling & Visualization** | `pd.read_csv` from `/kaggle/input` via *Add Input*; `info/describe/isnull().sum()/value_counts()`; `missingno.matrix`; impute-or-remove reasoning (drop if most of a column is missing, **median** for skewed numerics, **mode** for categoricals); encoding: **label/ordinal** for ordered categories, **one-hot** (`get_dummies(drop_first=True)`) for nominal; plot types by purpose (comparison, distribution, composition, relationship): `plt.hist`, `plt.scatter`, `sns.heatmap` on `pivot_table` and on correlations, `sns.barplot`, `sns.boxplot`; `pd.cut` binning | Phases 1–3 (audit, cleaning, before/after EDA), Q1–Q3 plots, Phase 5 feature EDA, Phase 7 encoding/imputation choices |
| **Lab 2: ML Foundations** | `train_test_split`; `LogisticRegression(max_iter=1000)`; `accuracy/precision/recall/f1_score`, `classification_report`, `confusion_matrix` shown with `sns.heatmap`; "accuracy is misleading on imbalanced data"; **baseline → improved model + comparison visualization** | Phase 8 baseline (LR) and the improvement story; Phase 9 metrics, confusion matrices, comparison bar charts; metric-choice justification |
| **Lab 3: Neural Networks (TensorFlow/Keras)** | `StandardScaler` **fit on train, transform test**; `keras.Sequential` + `layers.Input`/`Dense(relu)`; output layer **1 neuron + sigmoid** for binary tasks; `binary_crossentropy`; optimizers **Adam vs SGD**; `validation_split` / validation data, `epochs`, `batch_size=32`; `EarlyStopping(monitor, patience=15, restore_best_weights=True)`; train/val **accuracy and loss curves** (2×2 subplots); comparison table (test metric, test loss, epochs trained); seeds | Phase 8 shallow NN, its variants (with/without early stopping, Adam vs SGD, widths), Phase 9 training curves, preprocessing-order experiment E1 |
| **Lab 4: Explainable AI** | Intrinsic vs post-hoc and model-specific vs model-agnostic; global vs local explanations; `DecisionTreeClassifier` + `plot_tree` (intrinsically interpretable); `RandomForestClassifier`; `shap.TreeExplainer` → `summary_plot` (beeswarm), `force_plot`, `waterfall`; SHAP for the NN; `lime.lime_tabular.LimeTabularExplainer(...).explain_instance(row, model.predict_proba)`; fairness / bias / trust discussion | Phase 8 (RF as 2nd statistical model, shallow decision tree as an interpretable reference), Phase 11 XAI, fairness note on the nationality-based `home_race` feature |

**[beyond labs] items we still need, and why:**
| Item | Why it's needed | Implementation (no new libraries) |
|---|---|---|
| ROC-AUC, PR-AUC | Required by the brief | `sklearn.metrics.roc_auc_score`, `average_precision_score`, `RocCurveDisplay`, `PrecisionRecallDisplay` |
| Permutation importance | Required by the brief | `sklearn.inspection.permutation_importance` |
| Season-based (temporal) split instead of the labs' random `train_test_split` | Brief asks for a justified split; random rows would leak future seasons and split races across sets | Boolean masks on `year` in pandas; the labs' random split is kept as a **comparison experiment** (Gate 6.A) |
| Class imbalance handling | Podium rate ≈ 13%; Lab 2 warns accuracy misleads on imbalance | `class_weight='balanced'` (sklearn) / `class_weight=` in `model.fit` (Keras) |
| Threshold tuning for F1 | Brief: threshold chosen on validation | `precision_recall_curve` on validation predictions |
| Confidence intervals (Q1–Q3) | Small samples per circuit/constructor | Wilson interval written as a short numpy function |
| Gradient boosting *(optional 3rd statistical model)* | Common strong tabular baseline; only added if time allows | `sklearn.ensemble.HistGradientBoostingClassifier`: no LightGBM/XGBoost, no extra installs |

**Not used** (removed from the earlier plan to stay within the course toolkit): SQL/SQLite, statsmodels, LightGBM/XGBoost, PyTorch, imbalanced-learn/SMOTE, isotonic/Platt calibration.

---

## 1. Notebook Layout (single standalone Kaggle notebook)

**On Kaggle (what gets evaluated):**
```
/kaggle/input/<f1-dataset>/*.csv   # the 14 raw CSVs, attached Kaggle dataset (read-only by design)
/kaggle/working/                   # anything the notebook writes (figures, optional exports), regenerated on each run
f1_podium.ipynb                    # THE deliverable: all code + all markdown explanations
```

**GitHub repo (a copy for submission, not a dependency):** `f1_podium.ipynb`, `README.md` (with the Kaggle link), `report.pdf`, and optionally the exported figures. The notebook must never import anything from the repo.

**Path handling (first code cell):** look for the CSVs under `/kaggle/input` (glob for `results.csv`). Locally, fall back to `data/raw/`. That's the *only* environment difference, and it's purely Python.

**Notebook sections (fixed skeleton; contents flex with the gates):**
0. Setup: `!pip install -q shap lime missingno` (Lab 4 pattern); imports grouped as in Labs 1–4 (pandas/numpy/matplotlib/seaborn, sklearn, tensorflow.keras, shap, lime); `np.random.seed(42)` + `tf.random.set_seed(42)` (Lab 3); data path detection; library versions printed
1. Data loading and raw audit (EDA *before* cleaning)
2. Cleaning (sentinels, typing, keys/FKs, dedup)
3. Post-cleaning EDA (before/after comparisons)
4. Data-engineering questions Q1, Q2, Q3
5. Modeling table construction (leakage-safe features)
6. Feature justification EDA
7. Split strategy
8. Preprocessing pipelines and preprocessing-order experiments
9. Models: baseline, then statistical models, then shallow NN, plus tuning
10. Evaluation and model comparison table
11. Feature-group ablation
12. XAI (permutation importance, SHAP global/local, LIME)
13. Inference function demo (complete + partially-missing record)
14. Limitations and conclusions

**Engineering rules:** a single `SEED`; all thresholds and hyperparameters are stored in one config dict; every figure is displayed inline **and** saved to `/kaggle/working/figures/` with a numbered name (`fig_04_q1_frontrow.png`) so the report can reuse them; all helper functions (parsers, feature builders, evaluation, inference) are defined in notebook cells, in order of use.

---

## Phase 1: Load and Raw Audit (EDA *before* cleaning)

**Tasks**
1. Load all 14 CSVs with `dtype=str` first (so sentinels aren't silently coerced), then record shape, dtypes, and memory.
2. Per-table profile using the Lab 1 inspection sequence: `df.head()`, `df.shape`, `df.info()`, `df.describe()`, `df.isnull().sum()`, plus **`(df == '\\N').sum()`**, since `\N` isn't seen as null until it's converted. Run `value_counts()` on categoricals (`status`, `nationality`, `country`) to find inconsistent spellings, like Lab 1's "Male/male/M" example.
3. Compute a SHA-256 checksum of each raw CSV (Python `hashlib`), then assert it again at the end of the notebook. This shows the raw files were never modified (Kaggle inputs are read-only anyway); all cleaning works on in-memory copies (`df.copy()`).
4. Plots, each chosen by its Lab 1 purpose (comparison / distribution / composition / relationship):
   - `missingno.matrix` per key table after temporarily mapping `\N` to NaN on a *copy*, to show missingness patterns (Lab 1).
   - `sns.heatmap` of `\N`-rate **by year × column** (pivot table) for qualifying q1/q2/q3, results.time, fastestLap and pit stops (relationship over time).
   - `sns.barplot` / line plot: races per season, entries per race over time, podium rate over time (comparison).
   - `plt.hist` / `sns.boxplot` of numeric columns to spot outliers (distribution).

**Gate 1.A: dataset version check**
| Outcome | Action |
|---|---|
| Kaggle dataset has 14 tables and covers 1950–2024 (matches the local copy) | Proceed |
| Kaggle dataset version differs (e.g. includes 2025 races or extra columns) | Pin the analysis to seasons ≤ 2024 in code (`races.year <= 2024`) so results are reproducible, or attach the exact version used; document it |

**Gate 1.B: unexpected sentinels**
Scan for other placeholder values (`""`, `"0"` times, `"NA"`, `-1`, `0` lat/lng, `position=0`).
| Outcome | Action |
|---|---|
| Only `\N` found | Standard handling |
| Other sentinels found (e.g. `grid=0`, `number` blank, `q1=""`) | Add to a per-column sentinel registry with justification; handle in Phase 2 |

---

## Phase 2: Cleaning (on copies)

### 2.1 Sentinels and semantic typing
- Replace `\N` (and the registered sentinels) with `NaN`/`NaT`.
- Dates: `races.date`, `fp*_date`, `quali_date`, `sprint_date`, `drivers.dob` → `datetime64`. Times → combined datetime where both exist.
- Durations: `q1/q2/q3`, `fastestLapTime`, `lap_times.time` (`m:ss.mmm`) → **float seconds**; `pit_stops.duration` (`ss.mmm`, sometimes `m:ss.mmm` for red-flag stops) → seconds; `results.milliseconds` → int.
- Numerics: `position`, `rank`, `fastestLap`, `number`, `fastestLapSpeed` → nullable numeric types.
- Categoricals: country/nationality strings normalized (strip, unify `USA`/`United States`, `UK`/`United Kingdom`).
- Write a unit-check cell that asserts the parsed duration equals `milliseconds / 1000` wherever both exist.
- **Missing values: impute or remove? Use Lab 1's reasoning.** For each column, decide based on % missing, the missingness pattern (`msno.matrix`), and how important the column is:
  - Most of a column missing *and* not needed (e.g. `fp*_date`, `sprint_*`, `url`, `fastestLapSpeed` for old years) → drop it from the working copy, like Lab 1 drops `Cabin`.
  - Missing *because the event didn't happen* (`q2/q3` for drivers knocked out in Q1, `position` for DNFs) → **don't impute in cleaning.** Keep NaN, since it's informative, and turn it into features later (e.g. `reached_q3`).
  - Imputation for *model features* happens later, inside the training split only (Phase 7): **median** for skewed numerics, **mode** for categoricals.

**Gate 2.A: duration parsing anomalies**
| Outcome | Action |
|---|---|
| Parsed values plausible (q times 50–150 s, pit 15–60 s) | Proceed |
| Outliers (e.g. pit durations > 10 min from red flags, q times > 200 s) | Flag them, don't delete. For features, cap or exclude based on a plotted distribution; document the threshold |

### 2.2 Keys and referential integrity
- Primary keys: `circuitId`, `constructorId`, `driverId`, `raceId`, `resultId`, `qualifyId`, `statusId`, `year`, `(raceId, driverId, lap)` for lap_times, `(raceId, driverId, stop)` for pit_stops, `constructorResultsId`, `driverStandingsId`, `constructorStandingsId`, `sprint resultId`. Check that each is unique and non-null.
- Foreign keys: every `raceId`, `driverId`, `constructorId`, `circuitId`, `statusId`, `year` reference exists. Output an FK-reconciliation table (child.col → parent, #orphans, %).

**Gate 2.B: PK/FK violations**
| Outcome | Action |
|---|---|
| All reconcile | Report the table with zero orphans as evidence |
| Orphan FKs in tables we use | Drop the orphans if < 0.1% and irrelevant; otherwise investigate (e.g. races scheduled with no results yet) and document |
| Races in `races` with no results (future/cancelled) | Exclude from modeling (no target); document the count |
| Duplicate natural keys besides the known `(raceId, driverId)` | Treat with the same documented logic as Gate 2.C |

### 2.3 Grain: one row per `(raceId, driverId)`
**Gate 2.C: duplicate rule (85 pairs)**
Before choosing, plot and tabulate: duplicates by year, how many involve a podium, and how often the duplicate rows differ in `grid`, `positionOrder`, and `points`.

| Candidate rule | Pros | Cons |
|---|---|---|
| **R1: keep the row with best (min) `positionOrder`** | The driver *was* classified at that position, so the target reflects their real podium credit | Selection uses outcome info. That's OK for the *target*, but grid/constructor features must come from a consistent row |
| R2: keep the row where the driver started the car (lowest non-zero `grid` / first `resultId`) | Matches the pre-race view (who started where) | May label a driver as no-podium when they actually got shared podium credit |
| R3: drop all shared-drive races/drivers | Cleanest | Loses data (small anyway, mostly pre-1960) |

**Decision rule:**
- If the modeling window (Gate 3.A) **excludes ≤1978**, duplicates don't affect the model. Still apply R2 (pre-race consistent) to the full cleaned table for Q1/Q2/Q3 and document it.
- If the modeling window includes the 1950s, use **R2 for features + target = any of the driver's rows has `positionOrder ≤ 3`** (hybrid), and report how many labels change across R1/R2/hybrid.
- Always show the count of affected rows and the before/after uniqueness assert.

### 2.4 Other cleaning decisions
**Gate 2.D: `grid = 0` semantics**
Cross-tab `grid==0` against `status` and `laps`.
| Outcome | Action |
|---|---|
| `grid=0` & laps>0 → pit-lane start | Re-code grid as `last_grid + 1` (or the field size), and add a flag `pitlane_start=1` |
| `grid=0` & status ∈ {DNQ, DNPQ, Withdrew, Not restarted} & laps=0 | **Not a race entry/start.** Exclude from the modeling table (a pre-race model wouldn't be asked to predict these drivers), and from the Q3 denominator |
| Ambiguous | Keep it, flag it, and run a sensitivity check |

**Gate 2.E: non-standard events**
| Issue | Default action | Re-evaluate if… |
|---|---|---|
| Indianapolis 500 (1950–60) counted as WC races | Exclude from Q2 (it heavily biases American "home" effect) unless sensitivity shows no difference; irrelevant if modeling starts ≥ 1994 | Q2 effect changes sign with/without it, then report both |
| DNQ/DNPQ entries (huge 1980s–90s fields) | Not race starts, so drop from modeling and from Q3 denominators | |
| Half-points / shortened races (e.g. 2021 Spa) | Keep (podium is still positionOrder ≤ 3) | |

### 2.5 Post-cleaning EDA (required "before vs after")
- Side-by-side plots: `msno.matrix` before/after, row counts before/after per table (`sns.barplot`), `plt.hist` of parsed durations (q times, pit durations), `sns.boxplot` of pit durations before/after outlier flagging, entries per race after DNQ removal, podium rate before/after dedup.
- Re-run `df.info()` / `df.describe()` after cleaning to show the dtypes changed (strings → datetime/float).
- A summary "cleaning impact" table: step → rows affected → % → justification.

---

## Phase 3: Data-Engineering Questions (multi-hop joins)

Each answer must include: **the answer, a plot, and a short note on grain, missingness, and pre-race cutoff**.
Tools: joins with `pd.merge` (state `how=` and check row counts before/after each merge to prove the grain is preserved); aggregation with `groupby` / `pd.pivot_table`; plots with `sns.barplot` / `sns.heatmap` (Lab 1). Confidence intervals use a small numpy Wilson function **[beyond labs: small samples per circuit/team]**.

### Q1: Front-row → podium conversion by circuit, pre- vs post-2014
Joins: `results` ⨝ `races` ⨝ `circuits` ⨝ `qualifying` (on raceId+driverId).
- Define two front-row notions: **qualifying front row** (`qualifying.position ≤ 2`) and **grid front row** (`results.grid ∈ {1,2}`). Show how often they differ (penalties) and report the conversion separately for each.
- Conversion = P(podium | front row) per circuit, per era (`<2014`, `≥2014`).
- Statistics: Wilson 95% CI per circuit. To compare rankings between eras, rank the circuits per era (`rank()`) and use the pandas rank correlation `df[['rank_pre','rank_post']].corr(method='spearman')` on circuits present in both.
- Plots: `sns.barplot` of conversion per circuit with error bars, hue = era; `sns.heatmap` pivot (circuit × era → conversion rate), the Lab 1 pivot-heatmap pattern; and a rank-change chart.

**Gate Q1.A: qualifying coverage**
| Outcome | Action |
|---|---|
| Qualifying sparse before 2003 | Qualifying-based analysis is restricted to years with full coverage (≈2003+). Grid-based analysis can use the full history, so state it explicitly. Pre-2014 era for the quali view = 2003–2013 |
| Coverage OK | Use the full range |

**Gate Q1.B: small samples**
| Outcome | Action |
|---|---|
| Circuit has < *k* front-row starts in an era (pick *k* from the sample-size `plt.hist`, likely 10–20) | Exclude it from the ranking and list it separately with its count. Keep the rule simple and explainable; no shrinkage models |
| Ranking dominated by CI overlap | Report a "no significant difference" conclusion honestly, alongside the top-N ranking |

### Q2: Home-country podium effect, controlling for starting position
Joins: `results` ⨝ `drivers` (nationality) ⨝ `races` ⨝ `circuits` (country).
- Build a **nationality→country mapping** (e.g. British→UK, American→USA, Dutch→Netherlands, Monegasque→Monaco). Assert 100% mapping coverage; list unmapped demonyms.
- `home = driver_country == circuit_country`.
- Control for grid via two approaches, both built from lab tools. Report the conclusion they agree on:
  1. **Stratified comparison (Lab 1 pattern):** `pd.cut` grid into bins (1, 2–3, 4–6, 7–10, 11+). Make a `pd.pivot_table(index=grid_bin, columns=home, values=podium, aggfunc='mean')` and show it as a `sns.heatmap` plus a `sns.barplot` (x = grid_bin, hue = home) with CIs. Also compute a **grid-weighted average difference**: weight the home-minus-away gap in each bin by the number of home starts in that bin. That gives one adjusted number.
  2. **Logistic regression (Lab 2 model):** `LogisticRegression` on `[home] + one-hot(grid_bin) [+ era dummies]` (`get_dummies(drop_first=True)`, Lab 1). Report the `home` coefficient as an odds ratio `exp(coef)`. Bootstrap the races (numpy resampling, ~500 reps) for a 95% CI **[beyond labs: CI on the coefficient]**.
- Grain note: one row per driver-race; races with no grid (pit-lane/DNQ) are handled per Gate 2.D.

**Gate Q2.A: confounding and sensitivity**
| Outcome | Action |
|---|---|
| Effect driven by a few nationalities (e.g. Indy 500 Americans, early-era Italians) | Report with/without Indy 500, and per-era (pre-1980 / 1980–2013 / 2014+) |
| Strata sparse for home drivers at front positions | Coarsen bins (e.g. 1–3, 4–10, 11+) and rely more on the logistic-regression estimate |
| Effect significant but small | State it with effect size; emphasize "association, not causation" |
| Multiple countries per circuit name issue (e.g. "USA" vs "United States") | Fix via the normalization in 2.1 |

### Q3: Mechanical-retirement rate by constructor, 2014–2021 vs 2022–2024
Joins: `results` ⨝ `status` ⨝ `constructors` ⨝ `races`.
- **Status taxonomy** (documented table mapping every statusId to one class):
  - `finished`: Finished, +N Laps
  - `mechanical`: Engine, Gearbox, Transmission, Clutch, Hydraulics, Electrical, Brakes, Suspension, Power Unit, ERS, Turbo, Water/Oil/Fuel pressure/leak/pump, Battery, Drivetrain, Driveshaft, Halfshaft, Overheating, Exhaust, Electronics, Throttle, Steering, Cooling system, Mechanical, Technical, Wheel bearing, etc.
  - `accident`: Accident, Collision, Collision damage, Spun off, Fatal accident, Damage, Debris
  - `driver`: Physical, Illness, Driver unwell, Injury, Stalled(?)
  - `regulatory`: Disqualified, Excluded, Underweight, 107% Rule, DNQ, DNPQ, Withdrew, Not restarted, Safety concerns
  - `ambiguous`: Retired, Puncture/Tyre/Wheel/Wheel nut (could be crash-induced or failure), Out of fuel (strategy/team), Front/Rear wing, Broken wing, Undertray, Brake duct
- Denominator = **race starts** (exclude DNQ/DNPQ/Withdrew/non-starters).
- Rate = mechanical DNFs / starts, per constructor per period. Require a minimum number of starts (e.g. ≥ 30) and show Wilson CIs.
- Plots: horizontal `sns.barplot` of rate per constructor (hue = period, with CIs). A `sns.heatmap` pivot (constructor × period). A stacked bar of retirement *composition* (mechanical / accident / driver / regulatory per period), which is Lab 1's "composition" plot type.
- Constructor identity: rebrands (Toro Rosso→AlphaTauri→RB, Force India→Racing Point→Aston Martin, Sauber→Alfa Romeo→Sauber, Renault→Alpine). **Primary analysis by `constructorId` (as the data defines it)**, with a secondary view by team lineage, clearly labelled.

**Gate Q3.A: taxonomy sensitivity**
| Outcome | Action |
|---|---|
| Ranking stable between **strict** (excluding ambiguous) and **broad** (including ambiguous) definitions | Report strict as primary, mention stability |
| Ranking changes | Report both, explain which statuses drive the difference, and keep strict as primary (it matches the brief's definition: "component failed, not crash/driver/regulation") |
| "Retired" is common in 2014–2024 | Check by inspecting the year distribution of `Retired`; if large, call it out as a key limitation |

---

## Phase 4: Modeling Window and Table Construction

### Gate 3.A: modeling start year (most important structural decision)

> **Outcome (team meeting 2026-10-06, decision M1 in `docs/decisions.md`):** window **1950–2024**, `grid` as the only position feature (no qualifying features), no era/year feature, add `field_size` + `grid_rel`, Indy 500 removed, R2 dedup kept. The 2003+ table below is superseded; 2003+ with qualifying is a late comparison experiment.
Evidence to produce first: (a) qualifying coverage % by year, (b) `grid` vs `qualifying.position` agreement by year, (c) podium base rate and field size by year, (d) dataset size per candidate window.

| Option | When to choose |
|---|---|
| **A: 2003–2024** (full quali coverage, single-lap → knockout formats) | **Default.** Qualifying features are near-complete and give enough rows (~9k) |
| B: 2006–2024 (knockout Q1/Q2/Q3 format) | If q1/q2/q3-derived features (e.g. gap to pole using best session, reached-Q3 flag) prove valuable in Phase 5 EDA **and** format inconsistency in 2003–05 hurts |
| C: 1994–2024 with missingness indicators | If the extra rows show measurable validation gain in a quick experiment (keep the trial in the notebook) |
| D: 1950/1980–2024 using grid only | Only if qualifying features turn out weak versus grid (unlikely). Note the larger era drift |

Record: rows, podium rate, and % missing quali for the chosen window. **Data before the window is still used to compute history/form features** (e.g. career stats), so "start year" only limits *labelled rows*.

### 4.1 Leakage-safe construction principles
- Every feature for race *r* must be computable from **(a) races strictly before r** (by date, then round) or **(b) this weekend's pre-race sessions** (qualifying, grid, sprint per Gate 4.C).
- Standings tables are **post-race**, so use the standings **after the previous race in the same season** (shift by one round per driver/constructor). For round 1, use 0 / NaN + a flag `season_start=1`.
- Rolling features via `groupby(driverId).shift(1).rolling(...)` on a date-sorted table. Never use `rolling` without `shift`.
- **Leakage unit tests** (asserts in the notebook):
  1. For a sampled race, recompute features using only data with date < race date and assert equality.
  2. Shuffle the current-race outcome and assert features are unchanged.
  3. Make sure no column of the feature matrix correlates > 0.95 with the target in training (red flag check).

### 4.2 Candidate feature groups (to be confirmed by Phase 5 evidence)
| Group | Features | Pre-race valid because… |
|---|---|---|
| **Grid** | `grid`, `pitlane_start`, `grid_penalty = grid − quali_pos` | Grid is published before lights out |
| **Qualifying** | `quali_pos`, `gap_to_pole_%` (best of q1/q2/q3 vs pole), `gap_to_teammate_%`, `reached_q3`, quali position relative to teammate | Qualifying precedes the race |
| **Driver form** | last-*N* (N ∈ {3,5,10}) avg finish, podium rate, points/race, DNF rate; exponentially weighted variants | Uses only past races (shifted) |
| **Constructor form** | same as above aggregated over both cars; constructor mechanical-DNF rate (past) | Past races only |
| **Standings** | driver/constructor championship position & points **after previous round**, points gap to leader, wins so far | Shifted standings |
| **Circuit history** | driver's past podium rate / avg finish at this circuit, constructor's at this circuit, circuit front-row conversion (computed on training years only) | Past only; circuit-level priors computed inside training fold to avoid target leakage |
| **Driver profile** | experience (career starts), age at race, `home_race` (from Q2) | Static/known |
| **Context** | era flags (hybrid ≥2014, ground-effect ≥2022), round number / season progress, field size | Known before the race |
| *(optional)* **Sprint** | sprint finish position (2021+) | See Gate 4.C |
| *(optional)* **Pit/lap history** | past-race avg pit duration, past race-pace vs field (2011+/1996+) | Historical only; likely weak, so include only if EDA shows signal |

**Gate 4.A: target encoding / circuit priors**
| Outcome | Action |
|---|---|
| Circuit-level podium priors computed on all years | **Not allowed**: leaks test info. Compute on training years, or in an expanding window by date |

**Gate 4.B: first-race / rookie missingness**
| Outcome | Action |
|---|---|
| Form features NaN for rookies or season openers | Impute inside the pipeline (median or a "no-history" constant) **plus** a missing-indicator flag; compare in a preprocessing experiment |

**Gate 4.C: sprint results**
| Situation | Action |
|---|---|
| Sprint is held before the main race on the same weekend (true 2021–2024) | It's pre-race information, so it's **allowed**. But it only exists for ~6 races/yr, so add it as an optional feature with a `has_sprint` flag. Keep it only if the ablation shows gain; otherwise drop it for simplicity |

**Final assert:** `modeling_table[['raceId','driverId']].is_unique`, no NaN target, documented column list with "why valid at cutoff".

---

## Phase 5: Feature Justification EDA

For each candidate group, produce evidence **before** training:
For each candidate group, produce the evidence with the Lab 1 plots:
- **Comparison:** `pd.cut` each feature into bins → `sns.barplot(x=bin, y='podium')` gives podium rate per bin, like Lab 1's "survival by age group". Examples: P(podium | grid), P(podium | gap_to_pole), P(podium | last-5 podium rate).
- **Distribution:** `sns.boxplot(x='podium', y=feature)` and overlaid `plt.hist` per class, to see whether the classes separate.
- **Relationship:** a correlation `sns.heatmap(df[features + ['podium']].corr(), annot=True)`, then **rank features by |correlation| with `podium`** (exactly Lab 1 graded task 4). Use `corr(method='spearman')` as well for monotonic but non-linear features. Both heatmaps also flag redundancy (e.g. `grid` vs `quali_pos` ρ ≈ 0.9+).
- **Pivot heatmap:** `pd.pivot_table(index=grid_bin, columns=form_bin, values='podium')` → `sns.heatmap` shows how two features interact.
- **Era drift:** the same barplots with `hue=era` (pre-2014 / 2014–21 / 2022+).
- **Empirical check:** a single-feature `LogisticRegression` per group, scored by validation PR-AUC (Lab 2 model). This gives a quick, consistent "signal strength" number for each group.

**Gate 5.A: redundancy**
| Outcome | Action |
|---|---|
| |ρ| > 0.9 between two features | Linear/NN models keep one (the more interpretable or less missing); trees can keep both. **The feature set must be identical across models (brief)**, so pick one global set and choose the version that hurts least on validation |
| Many redundant form windows (N=3/5/10) | Keep the window with the highest |corr| with podium / best single-feature val PR-AUC, plus maybe one long window |

**Gate 5.B: weak features**
| Outcome | Action |
|---|---|
| |corr| ≈ 0, flat barplot, and single-feature PR-AUC ≈ no-skill | Drop, or keep only if there's a domain reason (document it). Ablation will confirm |
| Strong signal but heavy missingness (> 40%) | Lab 1 reasoning (the `Cabin` example): if most values are missing, don't "guess" them. Keep only a binary "available" flag, or drop. Keep with median imputation + indicator only if missingness is low and not era-dependent |

**Gate 5.C: era drift**
| Outcome | Action |
|---|---|
| Relationships stable across eras | Single model, no era features needed |
| Relationship shifts (e.g. grid matters more post-2014) | Add era flags / interactions; consider limiting the training window (re-check Gate 3.A); report as a limitation |

---

## Phase 6: Splitting Strategy

**Default: temporal split by season, grouped by race** (no race straddles two sets; the future is never used to predict the past).

| Set | Seasons (window 1950–2024; **decided in meeting 2026-10-06, M2**) |
|---|---|
| Train | 1950–2017 (21,749 rows / 2,897 podiums) |
| Validation | 2018–2021 (1,619 / 243) |
| Test (unseen, touched once) | 2022–2024 (1,356 / 204) |

**Why not the labs' `train_test_split(test_size=0.2, random_state=42)`?** It shuffles rows. That (1) splits one race's drivers across train and test, and (2) trains on future seasons to predict past ones. Both inflate scores, and neither is possible at real prediction time. We keep the labs' 3-way idea (Lab 3: train / validation / untouched test) but cut by **season**. Implemented with pandas boolean masks on `year`. **[beyond labs, justified]**
- The **validation block** plays the role of Lab 3's `validation_split`. For the Keras NN, pass `validation_data=(X_val, y_val)` instead of `validation_split=0.3`, because `validation_split` takes the *last* 30% of rows, which isn't a clean season boundary.
- Hyperparameters and thresholds are chosen on the validation block only. The final model is either kept as trained on train (simplest) or refit on train+val with the same settings (decide in Gate 8.A). It's then evaluated **once** on test.

**Gate 6.A: regulation shift in test (2022 ground-effect era)**
| Outcome | Action |
|---|---|
| Validation performance much higher than test (ΔPR-AUC > ~0.1) | Don't tune on test. Report it as **era drift** (honest finding), and show per-season test metrics |
| Want an alternative / to justify the split choice empirically | ~~Random `train_test_split` comparison~~ **Dropped (meeting 2026-10-06, M2).** Justify the temporal split in text only. Tuning uses the single validation block, no CV |

**Gate 6.B: set sizes / class balance**
| Outcome | Action |
|---|---|
| Any split < ~1,000 rows or podium count < ~150 | Shift boundaries (e.g. val = 2020–2021) and document it |
| Podium rate differs a lot between sets | Expected (field size changes); report the rates in a table |

---

## Phase 7: Preprocessing and Preprocessing-Order Experiments

Follow the Lab 3 rule: **fit on train, transform val/test.** Done explicitly, step by step, in separate cells so each one is visible and explainable:
1. **Impute (Lab 1 choices):** `SimpleImputer(strategy='median')` for numeric features (robust to outliers, Lab 1's argument for `Age`). `strategy='most_frequent'` (mode) for categoricals. Add 0/1 "was missing" flags for features where missingness means something (e.g. rookie has no form history).
2. **Encode (Lab 1 rule):** ordered categories → ordinal/label encoding (`OrdinalEncoder` with an explicit order, as in Labs 2–3). Unordered categories (e.g. era) → one-hot (`pd.get_dummies(drop_first=True)`, applied with the train columns reused for val/test). No raw IDs (driverId/constructorId) as features; form features carry that information instead.
3. **Scale (Lab 3):** `StandardScaler().fit(X_train)` → `transform` train/val/test. Required for LR and the NN; harmless for trees, so the **same scaled matrix is fed to every model** (brief: same rows, same features).
4. **Imbalance:** `class_weight='balanced'` (sklearn) and `class_weight={0:w0, 1:w1}` in Keras `fit` versus none, decided by validation PR-AUC.

The fitted imputer + encoder + scaler are wrapped into one `sklearn.pipeline.Pipeline`/`ColumnTransformer` object only *after* they've been shown step by step. That way the inference function (Phase 12) reuses exactly the same fitted transformations.

**Required: ≥2 preprocessing-order experiments** (each in its own cell, same model and split, before/after table):
| # | Experiment | Correct | Leaky / alternative | Expected finding |
|---|---|---|---|---|
| E1 | `StandardScaler` + median imputer **fit before split (on all rows)** vs **fit on train only** (Lab 3's pattern) | fit on train | fit on full data | Leaky version inflates val/test slightly; shows why order matters |
| E2 | **Split → then build rolling form features** vs **compute target-derived features (e.g. circuit podium priors) on all years → then split** | priors from train years only | priors from all years | Leaky priors inflate test scores noticeably. This is the most F1-specific leakage lesson |
| E3 | **Impute → scale** vs **scale → impute** (and median vs mean imputation, Lab 1's comparison) | — | — | Usually a small difference; documents robustness of the chosen order |
| E4 *(optional)* | **Simple random oversampling of podium rows (numpy, train only)** vs **oversampling before split** vs `class_weight` | train-only / weights | oversampling on full data | Leaky oversampling duplicates test rows into train, so it's heavily inflated |

Each experiment is shown with a results table plus a grouped `sns.barplot` of val/test PR-AUC, ROC-AUC and F1 (Lab 2's "baseline vs improved comparison visualization").
Also plot the effect of each preprocessing step on the distributions, as the checklist requires:
- `plt.hist` before/after imputation, showing the spike at the median.
- `plt.hist` / `sns.boxplot` before/after scaling (same shape, mean 0 / std 1).
- `df.describe()` before/after each step.

**Gate 7.A**
| Outcome | Action |
|---|---|
| Leaky variant scores higher | Expected; keep the correct pipeline and use it as a teaching point in the report |
| No difference | Still report it; explain why (e.g. tree models are scale-invariant, small missingness) |
| Correct variant worse by a lot | Inspect for bugs (e.g. a feature that's missing almost entirely in train but not test) |

---

## Phase 8: Models

### 8.1 Reference points
- **Heuristic reference** (not a "model", but a sanity floor): score = −grid (podium if grid ≤ 3). Every model must beat it.
- **Baseline model (required): Logistic Regression**: `LogisticRegression(max_iter=1000, class_weight='balanced')` on the full scaled feature set (Lab 2/3 setup). It's also intrinsically interpretable (Lab 4): coefficients are reported as odds ratios.

### 8.2 Required models (same rows, same features)
| Model | Lab source | Tuning (validation block only) | Notes |
|---|---|---|---|
| **Logistic Regression** (baseline) | Labs 2–3 | `C` ∈ {0.01, 0.1, 1, 10} | Interpretable coefficients (Lab 4: intrinsically interpretable) |
| **Random Forest** (2nd statistical model) | Lab 4 | `n_estimators` ∈ {100, 300}, `max_depth` ∈ {None, 6, 10}, `min_samples_leaf` ∈ {1, 5, 20}, `class_weight` | Bagging of trees; a "black box" explained with SHAP TreeExplainer (Lab 4) |
| **Decision Tree** *(interpretable reference)* | Lab 4 | `max_depth` ∈ {2, 3, 4}, `min_samples_split=20` | Shown with `plot_tree`: human-readable "if grid ≤ 3 and …" rules. Counts as an extra modeling attempt |
| **Shallow NN** (TensorFlow/Keras) | Lab 3 | see 8.2.1 | Scaled inputs, sigmoid output, class weights |
| *(optional)* Gradient Boosting | **[beyond labs]** | `learning_rate`, `max_depth`, `max_iter` | `sklearn.ensemble.HistGradientBoostingClassifier` (no new install). Add only if time allows; justify it as a common strong tabular model |

Hyperparameter search: a simple loop over a small grid, scoring each setting on the **validation block** by PR-AUC, with results stored in a DataFrame and shown as a table/barplot. This is the same "try, compare, visualize" pattern as Lab 2's graded improvement task and Lab 3's model comparison table.

#### 8.2.1 Shallow NN: Lab 3 recipe adapted to binary classification
```python
def build_model(units=(32,), optimizer='adam', dropout=0.0):
    model = keras.Sequential([layers.Input(shape=(X_train_scaled.shape[1],))])
    for u in units:                                   # 1–2 hidden layers only ("shallow")
        model.add(layers.Dense(u, activation='relu'))
        if dropout: model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))  # binary → 1 neuron + sigmoid (Lab 3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(curve='PR', name='pr_auc'), keras.metrics.AUC(name='roc_auc')])
    return model
early_stop = EarlyStopping(monitor='val_pr_auc', mode='max', patience=15, restore_best_weights=True)
model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
          epochs=200, batch_size=32, class_weight=cw, callbacks=[early_stop], verbose=0)
```
NN variants to compare (each its own cell, one comparison table like Lab 3's, with test metric, test loss and epochs trained):
| Variant | What changes | Lab link |
|---|---|---|
| NN-1 | 1 hidden layer (32), Adam, **no** early stopping | Lab 3 "overfit" model |
| NN-2 | same + EarlyStopping (patience 15, restore best) | Lab 3 early-stopping model |
| NN-3 | 2 hidden layers (64 → 32) + EarlyStopping | Lab 3 graded task A (different structure) |
| NN-4 | NN-2 with **SGD** instead of Adam | Lab 3 graded task B (optimizer comparison) |
| NN-5 *(if overfitting)* | + Dropout / fewer units | Regularization |
Show the Lab 3 2×2 plot (train/val metric and loss per epoch) for the variants. Pick the final NN on validation PR-AUC.

**≥3 distinct modeling attempts are documented:** LR baseline → Decision Tree → Random Forest (+ tuned variant) → NN variants (→ optional HistGB). Each is in its own cell with results.

### 8.3 Threshold selection
- Lab 2 uses the default 0.5 threshold (`model.predict`). With ~13% positives, 0.5 is usually too high. So choose the F1 threshold **on validation**: `precision_recall_curve(y_val, p_val)` → the threshold that maximizes F1. Store it, then apply it unchanged to test. Report the 0.5-threshold F1 too, as the before/after evidence.
- Alternative rule to compare (domain-aware): **"top-3 per race"**: predict podium for the 3 highest-probability drivers in each race. Report its F1 too; it's a natural constraint (exactly 3 podiums per race).

**Gate 8.A: model choice**
| Outcome | Action |
|---|---|
| Complex model beats LR by a meaningful margin (≥ ~0.02 val PR-AUC, consistent across the validation seasons) | Prefer the complex model as the final model |
| Gains within noise | Prefer the simpler LR (justify via interpretability, Lab 4's interpretability-vs-performance trade-off); still report all |
| NN underperforms trees | Expected on tabular data; discuss it. Try one Lab 3 improvement (early stopping, Adam vs SGD, different width/structure, class weights) and document it |
| Overfitting (train ≫ val; val loss rising in the Lab 3 curves) | Early stopping / dropout for the NN; `max_depth` / `min_samples_leaf` for trees; smaller `C` for LR. Re-run and show the before/after |
| NN without early stopping keeps improving on train but not val | Use it as the Lab 3 overfitting illustration; final NN uses early stopping |

### 8.4 Metric choice
- **Primary: PR-AUC**. Positives are rare (~13–15%), and it measures ranking of podium finishers, which is what a "companion" app needs.
- **Secondary: ROC-AUC** (overall separability; comparable across eras with different base rates), plus **F1** at the val-chosen threshold (operational decision quality).
- Also report Precision, Recall (Lab 2's `classification_report`) and the no-skill PR baseline (= positive rate) next to every PR-AUC.
- **Accuracy is shown but explicitly not used for decisions.** Lab 2's cats/dogs example applies here: a model that always says "no podium" scores ~87% accuracy and is useless.

---

## Phase 9: Evaluation and Comparison
- **Comparison table** (pandas DataFrame, Lab 3 style): model × {train, val, test} × {ROC-AUC, PR-AUC, F1, Precision, Recall, Accuracy*} + threshold used (+ epochs trained / test loss for NNs).
- Plots:
  - Confusion matrix per model at the chosen threshold, via `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')` (Lab 2).
  - `classification_report` per model on val and test (Lab 2).
  - Grouped `sns.barplot` comparing models across metrics (Lab 2's comparison visualization).
  - ROC and PR curves (val & test, all models overlaid) via `RocCurveDisplay` / `PrecisionRecallDisplay` **[beyond labs: required by the brief]**.
  - Per-season test metrics (line plot).
  - NN training curves (Lab 3).
- **Uncertainty** *(optional)*: resample races with numpy (~500 reps) for 95% CIs on test PR-AUC. Use it only if two models are close.
- Error analysis: which races/drivers are most mispredicted (e.g. chaotic wet races, first-lap incidents), shown as a table and a plot. This feeds the Limitations section.

---

## Phase 10: Feature-Group Ablation
- Groups: **Grid, Qualifying, Driver form, Constructor form, Standings, Circuit history, Driver profile/home, Context** (+ Sprint if kept).
- For each group: drop its columns → re-run the **same preprocessing (fit on train) and the final model (and the LR baseline)** with fixed hyperparameters → evaluate on **validation** (primary) and test (reported). For RF/NN, repeat with 3 seeds and report mean ± std.
- Implemented as a simple loop over a `FEATURE_GROUPS = {group_name: [cols]}` dict, with results collected into a DataFrame.
- Also run "only this group" for intuition (optional).
- Plot: horizontal `sns.barplot` of ΔPR-AUC (and ΔROC-AUC) vs the full model.

**Gate 10.A**
| Outcome | Action |
|---|---|
| Removing grid **or** qualifying barely hurts | Expected because they're redundant; run a joint "grid+quali removed" ablation to show their shared importance |
| Removing a group *improves* val | Consider dropping it from the final feature set. If so, re-run the final models and **keep both versions visible** in the notebook (before/after) |
| Ablation contradicts the Phase 5 EDA | Discuss it (correlated features, model substitution) |

---

## Phase 11: Explainability (XAI)
Framed with Lab 4's vocabulary. The notebook's intro cell classifies each method:
- **Intrinsically interpretable:** LR coefficients (as odds ratios, with a `sns.barplot`) and the shallow Decision Tree (`plot_tree`). These are explanations "for free" and a sanity check for the post-hoc results.
- **Post-hoc, model-agnostic:** permutation importance, SHAP, LIME.
- **Global vs local:** global = permutation importance + SHAP summary; local = SHAP force/waterfall + LIME for single rows.

Steps (Lab 4 code patterns):
1. **Permutation importance** **[beyond labs, required by the brief]:** `sklearn.inspection.permutation_importance(model, X_val, y_val, scoring='average_precision', n_repeats=10, random_state=42)` for the final model and the NN. Show it as a `sns.barplot` with error bars. For the Keras NN, wrap `predict` in a small class with `fit`/`predict_proba`, or permute columns manually in a numpy loop (same algorithm).
2. **SHAP global:**
   - RF: `shap.TreeExplainer(rf)` → `shap_values` for the podium class → `shap.summary_plot(...)` beeswarm (Lab 4), plus `plot_type='bar'`.
   - NN: `shap.KernelExplainer(nn_predict, shap.sample(X_train_scaled, 100))` on ~200 val rows → beeswarm (Lab 4's graded task applies SHAP to the NN).
3. **SHAP local:** `shap.force_plot(explainer.expected_value[1], shap_values[1][i], X_val.iloc[i])` (Lab 4) and a waterfall plot for a *different* row (Lab 4 graded task 4).
4. **LIME local:** `lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=..., class_names=['No podium','Podium'], discretize_continuous=True).explain_instance(X_val.values[i], model.predict_proba, num_features=8)` → `exp.show_in_notebook()` **and** `exp.as_pyplot_figure()` (so the plot also appears in the saved/exported notebook and the report). Explain **the same row** as one of the SHAP plots and compare them (Lab 4 graded task 5).
5. Rows to explain (≥2): (1) a confident correct podium (e.g. a pole-sitter); (2) a surprising case (high-probability miss, or a back-of-grid driver predicted podium).

- Interpret each plot in markdown.
- **Explicit statement (in notebook and report):** importance ≠ causation. The model relies on features correlated with podiums; SHAP/LIME explain the *model*, not F1 physics. Correlated features split credit.
- **Fairness / bias note (Lab 4's transparency-trust-bias section):** `home_race` comes from driver **nationality**, which is a sensitive attribute. Discuss whether it's a fair feature here. It describes the race *context* (home crowd, familiarity), not a judgement about a person, and the stakes are a sports prediction, not a life decision. Still, check its SHAP contribution, and state that the dataset's historical make-up (e.g. many British/Italian drivers in early decades) can bias it.

**Gate 11.A**
| Outcome | Action |
|---|---|
| SHAP and permutation rankings disagree | Explain why: permutation measures performance drop, SHAP measures contribution to output, and correlated features differ between them |
| LIME unstable across runs | Fix the seed, increase num_samples, and show stability over a few runs |
| `!pip install shap lime` fails (Internet off) | Turn Internet **On** in Kaggle settings (same as Lab 4). If still impossible, use local SHAP only (the brief accepts "LIME, local SHAP, or both") and say so in markdown |
| `shap_values` shape differs by shap version (list per class vs 3-D array) | Handle both in one helper: `sv[1]` if list, else `sv[..., 1]`; print `shap.__version__` in setup |
| SHAP too slow for the NN on Kaggle | Smaller background (`shap.sample(..., 50)`) and fewer explained rows (≤ 100); the global NN explanation then leans on permutation importance |
| `force_plot` doesn't render in the exported notebook/PDF | Use `shap.force_plot(..., matplotlib=True)` or a waterfall plot instead |
| A suspicious feature dominates (possible leakage) | **Stop.** Re-run the leakage tests from §4.1 before continuing |

---

## Phase 12: Inference Function
```python
predict_podium(record: dict | DataFrame, history=cleaned_tables, model=final_pipeline, threshold=VAL_THRESHOLD)
# defined in a notebook cell; uses the in-memory cleaned DataFrames and the fitted pipeline from earlier cells (no saved files)
  -> {"probability": float, "podium_pred": bool, "missing_fields": [...], "top_reasons": [...]} 
```
- Accepts either a **raw pre-race record** (raceId/driverId/grid/quali times) and builds features from history using the *same* feature code, **or** a pre-built feature row.
- Validates that no forbidden fields are present (raises/warns if `positionOrder`, `points`, `status`, ... are passed).
- Missing fields → handled by the same fitted median/mode imputer + indicators (Lab 1 choices, fit on train in Phase 7). The function reports which fields were imputed.
- Optionally return the top-3 SHAP/LIME reasons for the prediction, reusing the Phase 11 explainer.
- Demonstrate on (1) a **complete** record from the test set, and (2) a **partially missing** record (e.g. no q2/q3 times, rookie with no form history), and compare the outputs.

---

## Phase 13: Report (PDF) and Submission
The report is written **from** the notebook. Its figures are the ones the notebook saves to `/kaggle/working/figures/` (download them from the Kaggle output tab), and its text expands the notebook's Markdown cells. No figure or number may appear in the report unless the notebook produces it.

Report structure:
1. Introduction and task framing (cutoff, leakage definition)
2. Data audit and cleaning (with before/after figures and the decision log summary)
3. Data-engineering answers Q1–Q3 (figures + grain/missingness/cutoff notes)
4. Features: list + why each is valid at the cutoff + EDA evidence
5. Split, preprocessing, preprocessing-order experiments
6. Models, how each works, limitations; comparison table; primary/secondary metric justification
7. Ablation
8. XAI (with the non-causal disclaimer)
9. Inference demo
10. Limitations: qualifying-coverage drift, era sensitivity (2014, 2022), constructor rebrands, status taxonomy ambiguity, sample size per circuit, unobserved factors (weather, tyres, strategy, safety cars), first-lap chaos
11. Team contributions (optional)

**Submission checklist**
- [ ] Notebook runs end-to-end on **Kaggle** in a fresh session (dataset attached as Kaggle input), with Restart & Run All verified and no errors
- [ ] Standalone check: no `import` of local modules, no reads outside `/kaggle/input`, no `f1.db`/SQL, no files assumed to exist in `/kaggle/working` before the run
- [ ] Toolkit check: every import is in the §0.3 table; every **[beyond labs]** use has its one-line justification in markdown
- [ ] Lab-mapped evidence present: missingno before/after (L1), confusion-matrix heatmaps + classification reports (L2), NN training curves + NN comparison table (L3), SHAP beeswarm + force/waterfall + LIME on the same row (L4)
- [ ] Kaggle notebook visibility set to **Public** (or shared with the evaluators) so the submitted link is accessible
- [ ] GitHub repo public/accessible, with the same notebook, README (Kaggle link + dataset name), figures, report
- [ ] Report exported to PDF
- [ ] Raw-checksum assert passes at notebook end
- [ ] Form submitted before **18 Oct 2026 23:59**

---

## 12. Decision Log (lives as a Markdown table inside the notebook, updated as we go)
| ID | Gate | Evidence (fig/stat/cell) | Options considered | Decision | Why | Date |
|---|---|---|---|---|---|---|
| D01 | 1.A | | | | | |
| D02 | 2.C dedup | | R1/R2/R3/hybrid | | | |
| D03 | 2.D grid=0 | | | | | |
| D04 | 3.A window | | A/B/C/D | | | |
| D05 | Q3 taxonomy | | strict/broad | | | |
| D06 | 4.C sprint | | | | | |
| D07 | 5.A redundancy | | | | | |
| D08 | 6.A split | | | | | |
| D09 | 8.A final model | | | | | |
| … | | | | | | |

**General rule for unforeseen discoveries:** (1) quantify it (count, %, plot), (2) check if it touches the modeling window / target / cutoff, (3) if it doesn't, document it and move on; if it does, add a new gate here with ≥2 options, pick using validation evidence (never test), and keep both attempts visible in the notebook.

---

## 13. Risk Register
| Risk | Signal | Mitigation |
|---|---|---|
| Hidden leakage (standings, rolling without shift, circuit priors on all data) | Suspiciously high AUC (> 0.97), single feature dominates SHAP | Leakage unit tests (§4.1); review every feature against the cutoff |
| Test-set contamination | Any decision referencing test scores | Lock test evaluation into one final cell; log the test read-once |
| Era drift (2022 regs) | Val ≫ test | Report per-season metrics; discuss; optional grouped-random comparison |
| Kaggle environment differences | Import/version errors | Develop on Kaggle from the start (or test there at least every few days); print library versions in the setup cell; use only pre-installed packages, with in-notebook fallbacks (§ hard constraint) |
| Accidental dependence on local files | Works locally, fails on Kaggle | Only path is the auto-detected CSV folder; run the standalone check in §Phase 13 |
| Notebook too slow for Run All | > 15–20 min | Keep intermediate DataFrames in memory (compute once, reuse); load `lap_times` (17 MB) only if a feature needs it; limit SHAP background sample; modest hyperparameter grids; turn on Kaggle GPU for the NN only if needed |
| Nationality→country mapping errors | Unmapped demonyms | Assert full coverage; manual map documented in a cell |
| Every team member must defend every step | Oral evaluation | Walkthrough session using the Decision Log as the script; for each step, be able to name the lab it comes from (§0.3), or the reason it goes beyond the labs |
| Library API differences on Kaggle (shap/lime/keras versions) | Errors in XAI or NN cells | Print versions in setup; version-tolerant helper for `shap_values`; test XAI cells early on Kaggle |

---

## 14. Timeline (today: 3 Oct 2026 → deadline 18 Oct 2026)
| Dates | Work | Exit criterion |
|---|---|---|
| Oct 3–4 | Phase 1–2 (audit, cleaning, dedup, keys) | Clean tables + before/after EDA; D01–D03 logged |
| Oct 5–6 | Phase 3 (Q1–Q3) | 3 answers with figures + notes |
| Oct 6–8 | Phase 4–5 (window, modeling table, feature EDA, leakage tests) | Unique table, tests pass; D04–D07 |
| Oct 8–10 | Phase 6–8 (split, preprocessing experiments, models, tuning) | Comparison table on val |
| Oct 11 | Phase 9 (test eval, once) + Phase 10 ablation | Final metrics + ablation chart |
| Oct 12 | Phase 11–12 (XAI, inference) | Plots interpreted; demo works |
| Oct 13–14 | Final fresh-session Run All on Kaggle + standalone check + copy notebook to GitHub | Clean Run All on Kaggle, notebook public |
| Oct 15–16 | Report PDF | Draft complete |
| Oct 17 | Team rehearsal / buffer | Everyone can defend each gate |
| Oct 18 | Submit | Form sent |
