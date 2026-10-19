# Phase 3 answers (moved out of the notebook §13)

> **Old working document (kept for history).** Written during the project; numbers, cell references and some decisions (e.g. a "final model") are out of date. The final work is `notebooks/f1_podium_prediction.ipynb` and `report/main.pdf`.

Joins: `clean["results"]` ⨝ `clean["races"]` ⨝ `clean["circuits"]`
(Q1) / ⨝ `clean["drivers"]` (Q2) / ⨝ `clean["status"]` + `clean["constructors"]` (Q3).
Every merge asserts the row count is unchanged (grain preserved). Rates use Wilson 95% CIs
(`wilson_interval` helper: small-N circuits). Figures 8–15 are regenerated
by the notebook.

## Q1 — Front-row → podium by circuit, pre- vs post-2014

Front row = `grid ∈ {1,2}` (D23; qualifying not used). Circuits need ≥ 10 front-row starts in
both eras for the ranking (Gate Q1.B; 16 qualify; 21 / 16 / 13 / 7 for k = 5 / 10 / 15 / 20).
Left-out circuits typically have 2 starts and 95% ranges like [34%, 100%]; they still count in
the era-wide rates.

- **Answer**: conversion rose **55.5% → 76.0%**. Top 2014+: **Suzuka 88.9%**, **Yas Marina
  86.4%**, **Monaco 85.0%**, Barcelona and Spa 81.8%. Rank order reshuffled completely
  (Spearman ρ = −0.03; Monaco was 48.8% pre-2014, Hungaroring fell from 67.9% to 59.1%).
- **Why**: modern turbo-hybrid reliability + large top-team/midfield pace gaps.
- **Grain**: one row per driver-race (26,668 rows preserved across all merges).
- **Cutoff**: grid slots are pre-race; podium is the retrospective outcome.

## Q2 — Home-country effect, controlling for grid

100% nationality→country map (asserted in code); starts only (`grid_eff`, pit-lane → 11+
bin); Indy 500 excluded from the primary view (Gate Q2.A). Two methods:

- **Answer**: **no home advantage.** Stratified grid-weighted gap **−0.70pp** (pole:
  home 58.3% vs away 64.2%; slots 2–3/4–6 ≈ flat). Logistic regression
  (`is_home` + one-hot grid bins + era dummies): **OR 0.915, 95% race-cluster bootstrap CI
  [0.785, 1.082]** — spans 1.0. With Indy 500: −0.69pp / OR 0.909, same conclusion.
- **Reading**: F1 pace is car + grid + strategy; home weeks add media/sponsor load that can
  offset familiarity. Association only — not a causal claim.

## Q3 — Mechanical-retirement rate by constructor

Denominator = starts (non-starters out). Every unclear retirement counts as mechanical except
Refuelling (D24). Periods 2014–2021 vs 2022–2024 follow the brief and the two rule resets
(2014 turbo-hybrid power units; 2022 new ground-effect cars with frozen engine development).
`constructorId` primary + team-lineage view. ≥30 starts per team per period removes no team.

- **Answer**: overall **11.2% → 6.9%**. Worst 2014–21: **Caterham 32.4%** (34 starts, wide
  range), Lotus F1 23.7%, Toro Rosso 18.2%, Renault 16.0%, McLaren 15.9%. Worst 2022–24:
  **Alfa Romeo 12.5%**, Alpine 11.0%, RB 10.4%.
- **Cutoff note**: in-race retirements are forbidden as predictors, but past constructor
  DNF rates are valid pre-race reliability features.

