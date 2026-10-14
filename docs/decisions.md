# Decision log (moved out of the notebook; D1–D16, then team-meeting decisions M1–)

Evidence lives in `notebooks/f1_podium_prediction.ipynb`. Dates are 2026-10-03 unless noted.

## #26 — Features chosen by adding one group at a time (2026-10-11; supersedes the M3 feature list and `field_size`/`grid_rel` in M1)

Notebook §14.4 (new candidates), §16 (selection), §17 (preparing the chosen set).

- **Candidates (15)**, grouped by the team's four questions: start position (`grid`, `field_size`,
  `grid_rel`); this season (driver and team standings after the previous round); career (`driver_te`,
  `constructor_te`); circuit (`circuit_te`, new `drv_circuit_te` = driver's own smoothed past podium rate at
  this track); recent races (new `drv_podium_last5`, `con_podium_last5`); rivals (new `rivals_ahead` = starters
  in this race with more championship points after the previous round). All past-only; brute-force checks pass.
- **Exploratory evidence (§16.1):** podium rate by fifth of each candidate + Spearman correlations (Lab 1).
  `circuit_te` is flat (~13% in every fifth, ρ = 0.007); many candidates repeat each other (`grid`/`grid_rel`
  0.97, driver position/points share 0.92, `rivals_ahead`/driver position 0.89).
- **Empirical evidence (§16.2):** start from `grid`; add one group at a time; LR (C = 1) and RF (300 trees,
  depth 6, leaf 20) with fixed settings. **Keep a group only if it improves validation PR-AUC of both models in
  ≥ 80% of 500 redraws of the validation races** (bootstrap by race, [beyond labs]).

  | Step | LR | RF | LR better in | RF better in | Kept |
  |---|---|---|---|---|---|
  | grid only (baseline) | 0.668 | 0.668 | – | – | yes |
  | + field size | 0.669 | 0.671 | 61% | 91% | no |
  | + driver standings | 0.717 | 0.724 | 100% | 100% | yes |
  | + team standings | 0.713 | 0.714 | 15% | 6% | no |
  | + career podium rates | 0.729 | 0.738 | 88% | 94% | yes |
  | + circuit conversion | 0.727 | 0.736 | 18% | 39% | no |
  | + driver at this circuit | 0.730 | 0.734 | 80% | 18% | no |
  | + recent form (last 5) | 0.713 | 0.724 | 1% | 0% | no |
  | + rivals ahead | 0.727 | 0.731 | 17% | 1% | no |

  All 15 at once: LR 0.712 / RF 0.734, below the selected set.
- **Chosen:** `grid`, `drv_pos`, `drv_pts_share`, `drv_wins`, `driver_te`, `constructor_te` + 3 `no_history`
  flags = **9 model inputs** (was 16).
- **Baseline changed** from the `−grid` ranking to **logistic regression on `grid` only** (the brief asks for a
  statistical baseline; same validation PR-AUC 0.668).
- **Validation after tuning (§18–19):** LR (C = 10) PR-AUC 0.729 / F1 0.719; RF (depth 8, leaf 20) 0.741 / 0.725;
  NN (16) 0.736 / 0.724; baseline 0.668 / 0.712. Test still untouched.
- Known limits: order-dependent, dropped groups not retried (the "all at once" row is a partial check); the
  post-training ablation measures the kept groups again.

## Team-meeting decisions (2026-10-06) — these supersede D8 and the 2003+ parts of D1–D3/D10

### M1 — Modeling window and position feature
- **Window 1950–2024** (all starters), not 2003+. 1994 rejected: qualifying covers only 46% of
  1994–2002 entries (22 of 373 starters in 2001) and q2/q3 do not exist there.
- **`grid` is the only position feature** for every row. No `quali_pos`, no q1/q2/q3 times, no gap
  to pole — they do not exist before 1994 and copying grid into a qualifying column would hide
  noise (grid equals quali position in only 73% of 2003+ starters; within one place in 90%).
- **No era flag and no year feature** (era options compared: none / raw year / coarse bucket;
  none chosen). Known cost: pole→podium is ~50% pre-1990 vs ~80% since 2010 and starters
  reaching the flag rise 50% → 86%, so the model blends eras. Revisit in the late experiment.
- **Added features:** `field_size` (starters in the race) and `grid_rel = grid / field_size`.
- **Indianapolis 500 (1950–60, 11 races) removed entirely** from modeling, not just flagged.
- **Shared drives:** keep R2 — the car the driver started in (D1–D3 unchanged; 1950–1978 rows
  are now in scope).
- **Late experiment (end of project):** compare against 2003+ with qualifying features, same
  models, validation only.
- History/rolling features are feature engineering and are out of scope for this decision.

### M2 — Split (replaces D10)
Temporal split by season on the 1950–2024 window (24,724 starters, 3,342 podiums; counts from the notebook's R2 dedup; Indy 500 out,
R2 dedup):

| Set | Seasons | Rows | Podiums | Podium rate |
|---|---|---|---|---|
| Train | 1950–2017 | 21,749 (88.0%) | 2,895 (86.6%) | 13.3% |
| Validation | 2018–2021 | 1,619 (6.5%) | 243 (7.3%) | 15.0% |
| Test (touched once) | 2022–2024 | 1,356 (5.5%) | 204 (6.1%) | 15.0% |

Why B: validation holds twice the podiums of the 2020–21 option (243 vs 117) for steadier
threshold/model choice, at the cost of 2018–19 (4% of rows) in training; test stays a clean
"next regulation era". Rejected: A (2020–21 val, only 117 podiums), C (val 2014–18 / test
2019–24: training would end before the hybrid era), and a per-era split (tests earlier eras
with a model trained on later ones; not a forward test). Gate 6.B satisfied (val ≥ 1,000 rows
and ≥ 150 podiums).
- **Tuning:** single validation block (2018–2021) for hyperparameters and threshold; no CV
  inside train (D10's CV was only needed because the old validation set was small; 243 podiums
  is enough). Mention this in the report.
- **No random `train_test_split` comparison** (team decision). Gate 6.A's optional random-split
  experiment is dropped; the time-split rationale is argued in text only.

### M3 — Pre-race feature list (2026-10-06) — feature list superseded by #26
- **`pitlane_start`: dropped.** Only 72 rows in the whole history, and `grid = 0` cannot separate
  pit-lane starts from non-starters without the post-race `laps`. The 72 rows stay in the data
  (D4), just without the flag.
- **`sprint_result`: excluded** (Gate 4.C closed). Exists only from 2021, so almost all rows
  would be empty.
- **Standings: used**, lagged by one round (`docs/leakage.md`). Built in the feature-engineering step.
- **Identity (driver, constructor, circuit): target encoding, one numeric column each.** Raw
  one-hot rejected: 689 drivers + 167 constructors + 77 circuits = 933 columns for ~21,700
  training rows, and in the test seasons 42% of rows have a driver, 20% a constructor and 15% a
  circuit never seen in training. Value = that entity's podium rate in races strictly before the
  current one, smoothed towards the overall podium rate; unseen entities get the overall rate.
  Computed from past races only — never fit on the full table or on val/test rows (leakage).
  The smoothing strength is set in the feature-engineering step (tune on validation only).
- **Overlap note:** this is the same quantity as `driver_prior_podium_rate`,
  `constructor_prior_podium_rate` and `driver_circuit_prior_podium_rate` in notebook 02. Build
  once in feature engineering and do not include both versions.
- **As built (notebook §15.3):**
  - `driver_te`, `constructor_te`: `(past podiums + m·prior) / (past starts + m)`, prior = train
    podium rate (0.133), m = `SMOOTH_M` = 10 (tune on validation only). A constructor's two cars
    are aggregated per race, so teammates share one value and nothing from the same race leaks.
  - **`circuit_te` deviates from the literal plan:** a plain circuit podium rate is ~3/field size
    everywhere (std 0.0085 across 32 circuits, range 0.117–0.148), so it carries no signal. It is
    the past podium rate of the **top-3-grid starters** at that circuit instead (std 0.087, range
    0.333–0.704; prior = train top-3 conversion 0.533).
  - **Lagged standings:** driver `drv_pos`, `drv_pts_share` (points ÷ leader's), `drv_wins`;
    constructor `con_pos`, `con_pts_share`; all from the previous round of the same season (round 1
    has none; constructor standings start in 1958). Indy 500 rounds count as rounds because they
    carry championship points.
  - **Entity history includes earlier val/test races** (those results are known before a later
    race — walk-forward); only the priors, fill values, scaler and any tuned m use train/val.
  - **Fills for no history:** target encodings → the prior; points share and wins → 0;
    positions → train median (`drv_pos` 12, `con_pos` 6).
  - Check: brute-force recompute of 150 random rows per encoding from earlier rows only matches.
  - Known oddity: `drv_pos` reaches 102 (the 1950s standings list every Indy 500 entrant).
  - Rows with no history: standings 12.8% of train vs ~5% of val/test; `circuit_te` 7.0% / 4.9% /
    2.9%; `driver_te` 3.1% / 0.7% / 0.6%.

### M4 — Preprocessing (2026-10-06)
- **Scaling: one scaler for all models** (same inputs for every model keeps the comparison
  fair; trees are unaffected). Fit on train only, then `transform` val/test (Lab 2 pattern).
- **Missing values: fill and flag, no row dropping.** Where a feature has no history (first
  race of a driver/team/circuit, first round for standings) fill with a neutral value
  (target encodings → overall train podium rate, standings → 0 or last-place equivalent, to be
  fixed when the table is built) and add a binary `no_history` flag per affected group.
  Fill values and any imputer are fit on train only.
- **Built in the notebook (§15–16):** modeling table, season split with pinned counts, history
  features (§15.3, see M3 below), fill-and-flag helpers (toy self-check) and the train-only
  `StandardScaler`: 16 model inputs = 11 scaled + 5 `no_history_*` flags. The scaler is fit on the
  *filled* train data (fitting before the fill leaves train off mean 0 / std 1).
- **Era drift visible after scaling:** `field_size` has a train mean of 23.0 but ~20 in val/test
  (~1 SD below train) and `circuit_te` sits ~+0.9 SD above train in val/test (modern grids convert
  better). Expected, and the consequence of choosing no era feature in M1.
- **Run-All fixes made while testing (pre-existing, not decisions):** added the missing
  `from scipy import stats` and `LogisticRegression` imports to the setup cell, and dropped the
  duplicate `year` before the Q1/Q2/Q3 re-merges (they raised NameError / KeyError on a fresh
  kernel). Full notebook now runs top to bottom with no errors.

### M5 — Models and class imbalance (2026-10-06)
- **Statistical models: logistic regression and random forest**, plus a shallow Keras NN still to
  come (the brief needs ≥2 ML + 1 NN, same rows and the same 16 inputs). Baseline: score = −grid.
- **Selection on validation PR-AUC only** (class weight none vs `balanced`, small grids: LR
  C ∈ {0.01, 0.1, 1, 10}; RF depth {4, 6, 8} × min leaf {20, 50}, 300 trees). Threshold = the
  F1-optimal one on validation. Test is **not** scored until the NN exists (scored once).
- **Class imbalance: no class weights.** `balanced` did not help validation PR-AUC (LR 0.7203 vs
  0.7250, RF 0.7355 vs 0.7370), so the plain models stay; the F1 threshold handles the 13% base rate.
- **Selected:** LR C = 1 (val PR-AUC 0.725); RF depth 6, min leaf 20 (val PR-AUC 0.737).
- **Validation metrics (train in notebook §17):**

  | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
  |---|---|---|---|---|---|
  | −grid baseline | 0.668 | 0.918 | 0.712 | 0.712 | 0.712 |
  | Logistic regression | 0.725 | 0.932 | 0.727 | 0.674 | 0.790 |
  | Random forest | 0.737 | 0.939 | 0.726 | 0.684 | 0.774 |

- **Reading:** both models beat the baseline on PR-AUC (+0.06 / +0.07) but F1 barely moves
  (0.712 → 0.727), and LR and RF are nearly tied. Validation PR-AUC (0.67–0.74) is above train
  (0.49–0.60) because the modern era is easier to predict (podium rate 15% vs 13%; pole→podium
  ~80% vs ~50% early on) — era drift, not a bug.

### M6 — Shallow neural network (2026-10-06) — **written, not yet run**
- Keras `Sequential`: `Dense(relu)` × 1–2 + `Dense(1, sigmoid)`, Adam (1e-3), binary
  cross-entropy, batch 256, ≤200 epochs, `EarlyStopping(monitor="val_loss", patience=15,
  restore_best_weights=True)`, `validation_data=(X_val, y_val)`, seeds via
  `keras.utils.set_random_seed(42)`, no class weights (M5), same 16 inputs as LR/RF.
- Candidates (16), (32), (32, 16); pick on validation PR-AUC; threshold F1-optimal on validation.
- Notebook §18 adds it to the model comparison table and asserts it beats the `−grid` baseline.
- **Status:** the cells were only syntax-checked. TensorFlow has no wheel for the project's
  Python 3.14 `.venv`, so they have not been executed. Run the notebook on Kaggle (TensorFlow
  preinstalled) or in a Python 3.12 environment with `tensorflow-cpu`. `environment.txt` was
  deliberately left unchanged (team decision) and still lists only CPU torch.
- The test set is still untouched: scoring all four models once on test is the next step.

## D1 / D2 / D3 — Grain and dedup

Duplicates are the 1950s shared drives (85 pairs, 61 in 1950–1958, last case 1978), where two
entries describe the same driver in the same race but different stints of one car. The §6 audit
used R1 (best `positionOrder`) only to count the grain gap (91 rows).

Rule kept: **R2 — the row for the car the driver STARTED in** (lowest non-zero `grid`,
tie-break first `resultId`; fallback best `positionOrder` when neither row started — 3 pairs).

Why: the pre-race view (who started where) is what Q1/Q2/Q3 and the model reason about; a
driver who took over a winning car mid-race did not "convert" anything from their grid slot.
Using post-race fields here only chooses which record represents the driver — it is not a
model feature. Sensitivity: R1↔R2 flips 17 podium labels, all pre-1979, so the 2003+ modeling
window is unaffected (podium rate 0.1273 → 0.1267).

## D4 — `grid == 0` split

`grid = 0 & laps > 0` (72 rows: Finished/+N Laps) = pit-lane start → flag + re-coded grid
(`grid_eff`). `grid = 0 & laps == 0` (1,566 rows: DNQ 1013 / DNPQ 331 / Withdrew 164,
`positionText` F/W) = never started → excluded from modeling and the Q3 denominator. This
refutes the `f1.pdf` blanket claim that 0 means pit-lane start.

## D5 — Q1 uses `qualifying.position` AND `results.grid

Grid differs from qualifying position in a non-trivial share of entries (penalties), and the
rate changes by era (§10.1) — so both columns feed the analysis instead of grid-as-proxy.
**Superseded by D23** (grid only).

## D6 — Q2 mapping rules

Strip whitespace (`Argentinian ` → Argentina); hyphenated → first part (American-Italian →
USA); full 43-nationality map in the §14.2 cell with a 100%-coverage assert; circuits
`USA`/`United States` → USA; home = nationality (no birth-country column exists); Indy 500
excluded from the primary Q2 view (Gate 2.E, oval + US-only distortion).

## D7 — Q3 taxonomy

Strict mechanical primary (wings Broken/Rear/Front → ambiguous; Supercharger/ERS →
mechanical; `Out of fuel` → ambiguous as strategy, not failure). Denominator = starts (drop
non-starters). Constructor identity = `constructorId` (renames reset history, per `f1.pdf`),
plus a team-lineage secondary view.

## D8 — q2/q3 structural missingness pre-2003

Documented coverage limitation; needs era indicator or restricted window (§8).
**Superseded by M1:** quali features are dropped, so the limitation no longer applies.

## D9 — Leakage allow/deny lists

§11 / `docs/leakage.md`. `sprint_results` excluded by default pending Gate 4.C.

## D10 — Split (superseded by M2 for the 1950+ window; text below is the old 2003+ plan)

Per `f1.pdf` season boundaries on window 2003+: train 2003–2019 (6,831 rows / 963 podiums),
val 2020–2021 (780 / 117), test 2022–2024 (1,359 / 204) — counts re-verified when the Phase 4
table is built. Val is just under the ~1k/150 rule
of thumb → expanding-window CV inside train for tuning; the labs' random `train_test_split`
kept as non-primary comparison (Gate 6.A).

## D11 — Metrics

PR-AUC primary (base rate 12.7%, ~15% modern era); ROC-AUC + F1@val-threshold secondary;
+Precision/Recall (the brief's five); Accuracy reference-only (Lab 2: ~87% no-skill).

## D12 — Phase 2 cleaning on copies

Durations unit-checked (ms/1000), PK/FK 0 orphans, 72 pit-lane kept / 1,566 non-starters out,
Indy flagged, strict status taxonomy; impact table + before/after figures in §12.

## Gate outcomes 2.A–2.E

- **2.A**: quali q1 median in the 50–150 s band, pits median 15–60 s; long red-flag stops
  (>600 s) flagged, not deleted; `milliseconds/1000` unit check passed.
- **2.B**: all 14 PKs unique/non-null, all FKs reconcile (0 orphans); every race has results.
- **2.C**: R2 applied (91 rows removed, grain unique); 17-label R1 delta, all pre-1979.
- **2.D**: 72 pit-lane starts kept with `pitlane_start=1` + `grid_eff`; 1,566 non-starters out.
- **2.E**: Indy-500 rows flagged (`is_indy500`) for Q2 exclusion; shortened races kept
  (podium = `positionOrder <= 3` regardless).

## D13 — Gate Q1.A: dual front-row view

Qualifying sparse pre-2003: grid front row covers 1950–2024, quali front row covers the full
modern era. Both reported.

## D14 — Gate Q1.B: small-sample threshold k = 10

16 circuits meet k ≥ 10 front-row starts in both eras; small samples excluded from rankings
(and reported separately) to avoid noise.

## D15 — Gate Q2.A: Indy 500 exclusion, dual-method agreement

Indy 500 excluded from the primary Q2 view (oval, US-only). Stratified (−0.70pp gap) and
logistic regression (OR 0.915, CI [0.785, 1.082]) agree: no home advantage.

## D16 — Gate Q3.A: strict mechanical taxonomy confirmed primary

Strict excludes crashes/regulations; broad sensitivity check (strict + ambiguous) confirms
rankings highly stable (Spearman ρ > 0.95).

## D23 — Q1 uses `results.grid` only (2026-10-09; supersedes D5, D13)

Front row = grid slot 1 or 2. Grid is where the car actually starts (after penalties) and
exists for every race since 1950; qualifying is missing for most seasons before 2003. Results
unchanged: 55.5% → 76.0%, 16 circuits ranked, Spearman ρ = −0.03.

## D24 — Q3: unclear retirements count as mechanical, except Refuelling (2026-10-09; supersedes the strict/broad split in D7, D16)

Punctures, tyre and wing failures, fire, safety belt, out of fuel and "Retired" (no reason)
→ mechanical; Refuelling → other. One rate per period: **11.2% (2014–21) vs 6.9% (2022–24)**.
Worst: Caterham 32% (34 starts), Lotus F1 24%, Toro Rosso 18% / Alfa Romeo 12.5%, Alpine 11%.
The ≥30-starts rule for the per-team chart removes no team (smallest: Marussia, 31 starts).

## D25 — One spelling per value, fixed during cleaning (2026-10-09; moves the D6 mapping)

Cleaning step 3: Argentinian → Argentine, double nationalities → first part, East German →
German (drivers and constructors); circuits United States → USA; `drivers.home_country` in the
circuits' spelling with a 100%-coverage assert. Q2 numbers unchanged (−0.70pp; OR 0.915,
95% range [0.785, 1.082]).

## D27 — XAI: permutation importance on all models, SHAP and LIME on the random forest (2026-10-12)

Section 20 of the notebook. Permutation importance (PR-AUC drop, 10 shuffles) is computed for all four
models. The detailed SHAP plots (`TreeExplainer`, exact), the local waterfalls and LIME use the random
forest, which has the best validation PR-AUC (0.741 vs 0.737 NN, 0.729 LR: effectively tied). A
cross-model SHAP (KernelExplainer, same 100 rows and background) compares LR, RF and NN, as PLAN Phase 11
asked: all three put `grid` first, and they split the credit among the correlated standings columns
differently. All explanations use the **validation rows (2018–2021)** so
the test set stays unseen until the final scoring step. Features are displayed in original units
(scaling undone for display only). Local rows are picked by rule, not by hand: the true podium the
forest was most sure about (Hamilton, 2020 Bahrain GP, P = 0.86) and the one it was least sure about
(Kvyat, 2019 German GP, grid 14, P = 0.007).

Findings: `grid` dominates (permutation drop ≈ 0.30 PR-AUC, mean |SHAP| 0.088 vs 0.036 for the next
input); the other inputs are small and their order is within the error bars. SHAP and LIME agree on
direction and top input for both rows; LIME's local fit is poor on the miss row (R² 0.15). Explanations
describe the model, not causes (stated in the notebook and for the report).

Implementation note: permutation importance is a manual loop (same algorithm as
`sklearn.inspection.permutation_importance`) because sklearn's function requires an object with `.fit`
and would fail on the Keras network.

## D28 — Simplification pass (2026-10-12; supersedes parts of D3, D19/M3, D20/M4, D26, D27)

Team review: keep the simplest option that still meets the brief. The notebook was rebuilt (126 → 107 cells).

- **Loading:** plain `pd.read_csv` with pandas' type guessing. The columns read as text because of `\N`
  are the sentinel evidence; cleaning replaces `\N` with NaN and converts every text column whose
  values are all numbers.
- **Dedup:** only rule R2 (keep the car the driver started in; otherwise the better finish). The R1
  comparison code was removed.
- **Candidates:** `circuit_te`, `drv_circuit_te` and `rivals_ahead` were removed (never selected).
  The team standings group now also has `con_wins`. 13 candidates in 6 groups.
- **Selection rule:** keep a group if validation PR-AUC rises by ≥ 0.005 for both LR and RF. This
  replaces the 500-redraw bootstrap / 80% rule. Same result: `grid`, `drv_pos`, `drv_pts_share`,
  `drv_wins`, `driver_te`, `constructor_te`.
- **Preprocessing:** `SimpleImputer(median)` + `StandardScaler`, fit on train only. The custom
  fill values and the `no_history_*` flags were removed (their importance was ≈ 0).
- **Q1–Q3:** Wilson intervals were kept. The Q2 bootstrap, the Q1 rank scatter and the Q3 lineage
  heatmap were removed. Q2 conclusion: weighted gap −0.70pp, OR 0.915, and home/away intervals
  overlap in every starting group.
- **XAI:** the cross-model KernelSHAP and the LIME 5-seed stability check were removed.
- **Decision log:** kept only in this file. The in-notebook DECISIONS cells, the SHA-256
  fingerprint cells and the "Gate"/"M" labels were removed from the notebook.
- **Added (brief items that were missing):** test-set scoring (once), confusion matrices,
  feature-group ablation, two preprocessing-order experiments (fit-before-split, impute/scale order),
  and an inference function demonstrated on a complete and a partially-missing record.
- **Qualifying:** still grid only, 1950–2024. Qualifying times are a stated limitation (coverage drift).

## D29 — Career podium rates removed (2026-10-13; supersedes the `driver_te` / `constructor_te` part of D19/M3, D26, D28)

`driver_te` / `constructor_te` (smoothed past podium rates, m = 10) are removed from the candidates.
The smoothing constant was never tuned, and the D28 ablation showed the group helped on validation
(+0.013) but not on test (−0.026 LR, +0.002 RF). Recent form (`drv_podium_last5`, `con_podium_last5`)
stays as the leakage-safe form candidate. 11 candidates; selection keeps `grid` + driver standings
(`drv_pos`, `drv_pts_share`, `drv_wins`). `drv_wins` is kept with its group (team decision), although its
own contribution is small for RF/LR (the network leans on it most).

Results: validation PR-AUC LR 0.714, RF 0.726, NN 0.727 (baseline 0.668); test LR 0.708, RF 0.718,
NN 0.699 (baseline 0.647). The final model is chosen on validation, so it is the NN (tied with RF).
XAI stays on the RF (TreeExplainer exact). The chosen forest predicts single-threaded so a score that
equals the threshold cannot flip between tables.

Preprocessing-order significance (paired bootstrap over 2,000 redraws of the validation races, seed 0;
cell added to notebook §15.2 on 2026-10-14): leaky fit vs train-only, LR diff −0.00007 (95% CI [−0.00034, +0.00024], p = 0.69),
RF −0.00040 ([−0.00120, +0.00032], p = 0.29); scale→impute vs impute→scale, LR +0.00002 (p = 0.99),
RF 0 (p = 1). None is significant.
