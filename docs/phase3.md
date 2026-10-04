# Phase 3 answers (moved out of the notebook §14)

Joins: `clean["results"]` ⨝ `clean["races"]` ⨝ `clean["circuits"]` ⨝ `clean["qualifying"]`
(Q1) / ⨝ `clean["drivers"]` (Q2) / ⨝ `clean["status"]` + `clean["constructors"]` (Q3).
Every merge asserts the row count is unchanged (grain preserved). Rates use Wilson 95% CIs
(`wilson_interval` helper, [beyond labs]: small-N circuits). Figures 8–15 are regenerated
by the notebook.

## Q1 — Front-row → podium by circuit, pre- vs post-2014

Two front-row notions (Gate Q1.A): grid front row (`grid ∈ {1,2}`, 1950–2024 complete) and
qualifying front row (`qualifying.position ∈ {1,2}`, 2003+). Circuits need k ≥ 10 front-row
starts in both eras for the ranking (Gate Q1.B); 16 qualify.

- **Answer**: conversion rose **55.5% → 76.0%** (+20.5pp; 68.0% on 2003–2013 quali baseline,
  so +8.0pp within the modern era). Top 2014+: **Suzuka 88.9%** (16/18, CI [67.2, 96.9]),
  **Yas Marina 86.4%**, **Monaco 85.0%**, Barcelona and Spa 81.8%. Rank order reshuffled
  completely vs full history (Spearman ρ = −0.031; Monaco was 48.8% pre-2014 on attrition),
  but moderately stable 2003–13 → 2014+ on quali front rows (ρ = 0.450).
- **Why**: modern turbo-hybrid reliability + large top-team/midfield pace gaps.
- **Grain**: one row per driver-race (26,668 rows preserved across all merges).
- **Missingness**: quali absent pre-1994/sparse to 2003 → grid view primary historically.
- **Cutoff**: grid/quali slots are pre-race; podium is the retrospective outcome.

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

Denominator = starts (non-starters out); strict taxonomy primary (D7) + broad sensitivity
(strict + ambiguous); `constructorId` primary + team-lineage secondary view; ≥30 starts in both windows (verified:
every 2022–24 constructor has ≥48 starts, so ≥30 vs ≥15 admits the same 12).

- **Answer**: overall **9.5% → 5.9%** (−3.6pp) as hybrid units matured through the 2022
  chassis reset. Worst P1: **Caterham 23.5%**, Lotus F1 17.1%, Toro Rosso 16.1%, McLaren
  14.4% (Honda years), Renault 14.0%; best P1: Mercedes 4.1%. Worst P2: **Alfa Romeo
  12.5%**, Alpine 11.0%; best P2: McLaren/Red Bull 3.7%. Broad adds ~+1.6pp everywhere;
  rankings stable (Spearman ρ > 0.95) — strict stays primary.
- **Cutoff note**: in-race retirements are forbidden as predictors, but past constructor
  DNF rates are valid pre-race reliability features.
