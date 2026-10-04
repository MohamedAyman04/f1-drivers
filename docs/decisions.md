# Decision log (moved out of the notebook; D1–D16)

Evidence lives in `notebooks/f1_podium_prediction.ipynb`. Dates are 2026-10-03 unless noted.

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

## D9 — Leakage allow/deny lists

§11 / `docs/leakage.md`. `sprint_results` excluded by default pending Gate 4.C.

## D10 — Split

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
