# Decision log (moved out of the notebook; D1–D16, then team-meeting decisions M1–)

Evidence lives in `notebooks/f1_podium_prediction.ipynb`. Dates are 2026-10-03 unless noted.

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
Temporal split by season on the 1950–2024 window (24,724 starters, 3,344 podiums; Indy 500 out,
R2 dedup):

| Set | Seasons | Rows | Podiums | Podium rate |
|---|---|---|---|---|
| Train | 1950–2017 | 21,749 (88.0%) | 2,897 (86.6%) | 13.3% |
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

### M3 — Pre-race feature list (2026-10-06)
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
