# Review findings

Written 2026-10-14 for the CSEN903 Milestone 1 notebook. Cell numbers are 0-based in the **fixed** notebook on branch `review-fixes` (117 cells). Plan: `REVIEW_PLAN.md`. Out-of-scope audit: `OUT_OF_SCOPE.md`. Independent check: `review/verify.ipynb` (rebuilds the modeling table from the raw CSVs with separate code; its split sizes and PR-AUCs match the notebook to four decimals).

## Summary

31 items checked. Outcome: **fixed** 15 (V1, V4, V5, V8, V9, V11, V12, V18, V20, N1, N2, N5, N6, X1, X2); **verified** 12 (V2, V6, V7, V10, V13, V14, V15, V16, V17, V19, V21, N7); **needs answer** 1 (V3); **resolved by team** 2 (N3, N4); **open** 1 (N8).

**What matters most**

1. The results hold up. An independent rebuild reproduces the row counts (24,724 before the V1 fix), the split sizes and every model score; the leakage checks found no leakage; Q1–Q3 numbers reproduce.
2. Real defects, all fixed on `review-fixes`: entries that never raced counted as starters (V1, small effect); the neural network never reached its stopping rule (V4); the ablation was too narrow (V8); wrong statements in the text (N1, N2, the 'within 0.03' claim in the test-results paragraph, the Q1 correlation wording); Q1 used different rows from the rest (V12); grain/missingness/cutoff missing from the notebook answers (X2); metrics not plotted or discussed (§5).
3. One thing to be honest about in the evaluation: the selected feature set depends on the order of the groups (V9). What is stable is 'driver standings on top of the grid'; the exact extra features are not. The noise in a validation PR-AUC difference is larger than the 0.005 selection rule (the driver-standings gain of +0.045 has a 95% range of 0.008 to 0.080).
4. One decision waits for the instructor: which model SHAP/LIME should explain (V3, office-hours question 2).

## Requirements and checklist traceability

| Requirement | Where | Status | Evidence | Fix / note |
|---|---|---|---|---|
| EDA before and after cleaning, with plots | §3–4, §6; c7–c37 | met | missingno before/after (fig 01, 04), coverage by season (fig 02), grid and entries (fig 03), after-cleaning plots (fig 05), dtypes before/after. | — |
| EDA not manual inspection; choices justified and visualised | throughout | met | Every cleaning step is computed, printed and (where it matters) plotted. | — |
| Sentinels cleaned, semantic typing of dates and durations | c16, c34 | met | 160,076 \N cells in 30 columns reproduced; numeric conversion only when nothing is lost; durations checked against the milliseconds column. | — |
| Exactly one row per (raceId, driverId), rule explained | c30, c35 | met | 85 pairs reproduced; keep the lowest real grid slot; assert after. | — |
| Primary keys unique and non-null; foreign keys reconcile; raw files immutable | c27, c28 | met | 14 primary keys, 22 foreign-key relations, 0 orphans; cleaning on copies. | — |
| Q1 circuits, front-row conversion, 2014 split, qualifying vs grid | §8.1 | met | Instructor confirmed grid only; numbers reproduced; Indy rows removed for consistency. | Fixed (V12). |
| Q2 home advantage controlling for starting position | §8.2 | met | Stratified gap and odds ratio reproduced; bootstrap range includes 0. | — |
| Q3 mechanical-retirement rate, 2014–2021 vs 2022–2024 | §8.3 | met | Reproduced; sensitivity to the taxonomy added. | Fixed (V11). |
| Each answer: result, visualisation, short justification of grain, missingness, cutoff | §8 | met after fix | Grain/missingness/cutoff line was missing in the notebook, present in the report. | Fixed (X2). |
| Feature choices justified with evidence; trials kept in the notebook | §11 | met | Fifths plots, Spearman, forward ladder, order check, noise estimate. | Order dependence documented (V9). |
| Appropriate train/validation/test split, justified | §9 | met | By season: 1950–2017 / 2018–2021 / 2022–2024; random split would leak across a race. | — |
| At least two statistical models + a shallow network on the same rows and features | §12–13 | met | LR, RF, NN with identical inputs; baseline LR on grid. | — |
| All metrics for every model on train/validation and on test | c85, c90 | met after fix | Tables had all six; now plotted and discussed, with an 'always no podium' reference. | Fixed (§5). Office-hours question 1. |
| Thresholds and hyperparameters chosen without the test set | c72, c85 | met | Test is scored once in §14; ablation reads test but decides nothing. | — |
| Primary / secondary metric choice justified | c41, c86 | met | PR-AUC primary; text corrected (N1). | — |
| Baseline: simplest reasonable model, compared | §12 | met | LR on grid alone; threshold also tuned on validation. | Office-hours question 13. |
| At least three modeling attempts compared | §11–§13 | met | Baseline, LR, RF, NN, hyperparameter grids, feature ladders. | — |
| Feature-group ablation for each group | §15.1 | met after fix | Previously 2 groups and 2 models, test only. | Fixed (V8). |
| At least two preprocessing-order experiments | §15.2 | met | Leaky fit and scale/impute order, plus a paired bootstrap. | — |
| Leakage-safe form feature implemented | c62, c63 | met | Built with shift(1); perturbation test on 120 races found no leakage. Not selected by the ladder. | Office-hours question 7. |
| Effect of each preprocessing step on distribution/range/values | §10–11.3 | met | Imputation and scaling shown; cleaning steps shown by counts and before/after plots. | — |
| Global explanations: permutation importance and SHAP; local: LIME and/or SHAP; all plotted and interpreted | §16 | met | Permutation for all four models, SHAP on the forest, LIME and waterfall for a hit and a miss. | Which model to explain: question 10 (V3). |
| Explanations stated as non-causal | c102, Limitations; report | met | Boxed statement in the report; markdown in the notebook. | — |
| Inference function on a complete and a partially missing record | §17 | met | Verstappen (complete) and a round-1 rookie with only grid. | — |
| Run-all ready, no manual cells, sectioned | whole notebook | met locally | Four full runs on 2026-10-14 completed with 0 errors in 57 code cells (about 14 min each on a 20-core CPU) with identical outputs. | Test on Kaggle before submission. |
| Report (PDF) with answers, figures, selected features and why valid, limitations | report/main.tex | partly met | Draft complete (names and IDs filled in by the team); placeholders for the Kaggle and GitHub links; numbers re-synced with the fixed notebook; main.pdf is stale until recompiled. | Recompile main.pdf; fill N8. |
| GitHub repo accessible; Kaggle link | — | open | Repo is private (404 when not signed in). | Make accessible to graders before submitting. |

## Verification findings (V) and text findings (N, X)

Severity: **should fix** / **note** / **ok**. Status: fixed on the branch, verified (no change needed), needs answer (office hours), resolved by team.

### V1. Entries that never raced were counted as starters

- **Area / severity / status:** Cleaning / should fix / fixed
- **Claimed or found:** Rows with status Withdrew, Did not qualify or 107% Rule, 0 laps, but a grid slot in the data were counted as starters. The notebook text (c25) says withdrawn drivers are non-starters; the code only caught grid 0 with 0 laps.
- **Evidence:** 79 rows (61 Withdrew, 11 Did not qualify, 7 107% Rule; 77 train, 2 test, none in validation). Re-fitting without them moves PR-AUC by at most 0.004 (RF test +0.0036, RF validation −0.0011, LR ±0.0002), so it is a correctness fix, not a result changer. 3 of them were front-row rows used in Q1. Do not drop all laps == 0 rows: 353 Collision and 224 Accident rows with 0 laps are real starters who crashed at the start.
- **What was done / to do:** Rule added in c35 (status in Withdrew / Did not qualify / Did not prequalify / 107% Rule and 0 laps); grid_eff now uses the last real starter; counts reprinted. Brief wording is 'one driver entering one race', so this is also office-hours question 3.

### V2. Qualifying position vs grid

- **Area / severity / status:** Q1 / note / verified
- **Claimed or found:** Q1 never uses qualifying position.
- **Evidence:** Closed: the instructor confirmed grid only. In 2003+ grid differs from qualifying position in 27% of rows; 36 drivers qualified on the front row and started further back.
- **What was done / to do:** One sentence added in c44 saying grid-only was confirmed.

### V3. XAI explains the forest, the final model is the network

- **Area / severity / status:** XAI / should fix / needs answer
- **Claimed or found:** SHAP and LIME are run on the random forest (c103); the model used by the prediction function (chosen on validation) is the neural network (0.728 vs 0.725, a tie). Lab 4's graded task explains a neural network with SHAP (beeswarm, force, waterfall) and LIME, so explaining the network is probably expected.
- **Evidence:** The brief does not say which model must be explained. The notebook states why the forest is used (exact TreeExplainer, tie on validation), and permutation importance covers all four models.
- **What was done / to do:** Office-hours question 2. Likely next step whatever the answer: add SHAP (shap.Explainer on the network's predict function, or KernelExplainer) and LIME for the neural network too. The lab guide notebook shows no code for this, only the forest with TreeExplainer.

### V4. Early stopping never really ran

- **Area / severity / status:** Neural network / should fix / fixed
- **Claimed or found:** The network used a 200-epoch cap; the three sizes ran 200, 200 and 198 epochs, so training was cut by the cap and not by the stopping rule.
- **Evidence:** In the fixed notebook (cap 1,000) the three sizes stop at epochs 266, 282 and 220 (best epochs 251, 267, 205) with validation PR-AUC 0.728 / 0.725 / 0.726, so early stopping now ends training. Larger learning rate (3e-3) stops earlier at 157. SGD at 0.01 had not converged after 600 epochs.
- **What was done / to do:** Cap raised to 1,000 epochs (c83); table now shows the best epoch; the loss figure has one panel per size with the best epoch marked. SGD and no-early-stopping variants from Lab 3 were tried in scratch work and not added, to keep run time down.

### V5. Validation is used for everything

- **Area / severity / status:** Evaluation / note / fixed
- **Claimed or found:** Validation chooses features, settings, early stopping, the final model and the thresholds, so validation scores are optimistic.
- **Evidence:** The test set is the honest number: test PR-AUC is 0.006 to 0.028 below validation.
- **What was done / to do:** Stated in the metrics discussion (c86).

### V6. Why train PR-AUC is below validation

- **Area / severity / status:** Evaluation / note / verified
- **Claimed or found:** Claim: old seasons are less predictable.
- **Evidence:** Confirmed. Forest PR-AUC on the training seasons by decade: 0.50 (1950s), 0.46 (1960s), 0.47 (1970s), 0.47 (1980s), 0.56 (1990s), 0.68 (2000s), 0.73 (2010–2017); on validation it is 0.725. Podium rate also differs (13.3% train vs 15.0% validation and test), but the lift over no-skill is 4.3 (train) vs 4.8 (validation) and 4.8 (test).
- **What was done / to do:** A decade table was added after the comparison (c87) so the claim is backed by a statistic in the notebook.

### V7. Round-1 rows and rookies get the training median

- **Area / severity / status:** Features / note / verified
- **Claimed or found:** Missing standings are filled with training medians (grid 12, position 12, points share 0.08, wins 0).
- **Evidence:** 5.2% of validation and 5.0% of test rows are missing (95% / 88% of them are round 1). Using last season's final standings for round 1 instead: validation +0.011 (LR) / +0.007 (RF); test +0.004 (LR) / −0.010 (RF). Mixed, so the median was kept.
- **What was done / to do:** Sentence added to Limitations. Same limitation applies to the inference demo with a rookie.

### V8. Ablation was too narrow

- **Area / severity / status:** Ablation / should fix / fixed
- **Claimed or found:** Only the 2 kept groups, for 2 models, and the plot showed only test drops.
- **Evidence:** Now covers the kept groups for logistic regression, forest and network, and each of the 5 groups removed from the all-candidates model (LR and RF), on validation and test. No decision uses the test column.
- **What was done / to do:** Rewritten in section 15.1 (c93); figure fig_20 has four panels. Office-hours questions 5 and 6 decide whether this is the expected scope.

### V9. The selected feature set depends on the order of the groups

- **Area / severity / status:** Feature selection / should fix / fixed
- **Claimed or found:** Forward selection adds groups in a fixed order and never retries a dropped group.
- **Evidence:** All 120 orders lead to 8 different selected sets. Every set contains the driver standings and grid information (grid or grid_rel), the other groups come and go. The noise in a validation PR-AUC difference is larger than MIN_GAIN (0.005): redrawing validation races, the driver-standings gain is +0.045 (95% range 0.008 to 0.080) and the field-size gain +0.002 (0.000 to 0.008). Backward elimination keeps field size + driver standings and drops the grid group (grid is still in grid_rel × field size).
- **What was done / to do:** Two cells added after the ladder (c74) and the text now says which part of the result is stable (driver standings on top of grid) and which is not (the exact extra features).

### V10. Home effect

- **Area / severity / status:** Q2 / note / verified
- **Claimed or found:** Home drivers are 0.70 percentage points less likely to podium for the same start group (odds ratio 0.915).
- **Evidence:** Recomputed independently: −0.70 pp and 0.915 (my first run gave −0.68 because I had not applied the notebook's United States → USA spelling fix, which matters for American drivers). X_q2 column 0 is home. After the V1 fix the notebook gives −0.69 pp and 0.921. Race-bootstrap 95% range of the gap: −1.82 to +0.48 pp, so 'no detectable home effect' is right. All 43 nationalities map; 14 mapped countries never host a race.
- **What was done / to do:** None.

### V11. Mechanical taxonomy sensitivity

- **Area / severity / status:** Q3 / note / fixed
- **Claimed or found:** Unclear statuses (Retired, Puncture, Tyre, wings, ...) count as mechanical.
- **Evidence:** 21% of 'mechanical' starts in 2014–2024 have an ambiguous status; 'Retired' alone is 50 rows. Strict definition (cell c57): 11.2% → 9.0% (2014–2021) and 6.9% → 5.5% (2022–2024), so the conclusion (cars got more reliable) holds. Team ranking: 2014–2021 top 3 unchanged; 2022–2024 the third place changes from RB (48 starts) to AlphaTauri. Team names are as in the data (Toro Rosso, AlphaTauri, RB are separate teams).
- **What was done / to do:** Strict-vs-broad cell added to Q3 (c57). Office-hours question 14.

### V12. Q1 recomputation

- **Area / severity / status:** Q1 / note / fixed
- **Claimed or found:** 55.5% → 76.0% front-row conversion; circuit ranking correlation −0.03.
- **Evidence:** Reproduced exactly. The original also counted 24 Indianapolis 500 front-row rows (the modeling table and Q2 exclude Indy). Without them the pre-2014 rate is 55.8%. K_MIN counts 21/16/13/7 reproduced; Spearman is +0.06 (K=5), −0.03 (K=10), −0.13 (K=15). With only 16 circuits the 95% bootstrap range of the Spearman is about −0.64 to +0.55, so 'unrelated' is too strong: the data cannot show agreement or disagreement. 2014+ rate is 74.6–76.0% for era boundaries 2012–2018.
- **What was done / to do:** Q1 now uses the same rows as the rest of the analysis; answer text says 'no detectable agreement' (c49).

### V13. Duplicates, non-starters, pit-lane starts

- **Area / severity / status:** Cleaning / note / verified
- **Claimed or found:** 85 shared-drive pairs, 1,563 non-starters, 72 pit-lane starts, 381 Indy rows, 24,724 modeling rows.
- **Evidence:** All reproduced independently from the raw CSVs with different code (groupby-idxmin instead of sort + drop_duplicates). 176 rows: 79 pairs of two and 6 triples, all 1950–1978. 3 pairs have grid 0 in every row; the tie is broken by positionOrder. The independent modeling table has identical split sizes and identical PR-AUCs to four decimals.
- **What was done / to do:** None (the V1 fix changes the non-starter count; see the notebook output).

### V14. Sentinels

- **Area / severity / status:** Cleaning / note / verified
- **Claimed or found:** 160,076 \N cells in 30 columns; isnull() finds 68.
- **Evidence:** Reproduced by counting the literal text in every raw file.
- **What was done / to do:** None.

### V15. Leakage checks on every row

- **Area / severity / status:** Leakage / ok / verified
- **Claimed or found:** Form uses earlier races only; standings are taken after the previous round.
- **Evidence:** Perturbation test on 120 races: flipping every podium outcome of a race never changed that race's own form features (0 of 120), and does change later races (133 rows, all later). The driver standings at a raceId already include that race (points equal the cumulative results through that race in 100% of checked rounds, including sprint points from 2021), so joining on the previous round is correct. No duplicate (raceId, driverId) in driver_standings.
- **What was done / to do:** None. The 150-row check in c63 could be extended to all rows; the perturbation test in review/verify.ipynb already covers it.

### V16. grid_eff for pit-lane starters

- **Area / severity / status:** Features / note / verified
- **Claimed or found:** grid_eff = last starter's slot + 1.
- **Evidence:** 54 races have a pit-lane starter, 15 have two or more; those starters share the same grid_eff (ties). Immaterial (72 rows).
- **What was done / to do:** None.

### V17. Two full runs give the same numbers

- **Area / severity / status:** Reproducibility / should fix / verified
- **Claimed or found:** Run-all must be reproducible.
- **Evidence:** Four complete runs of the fixed notebook gave identical printed results in every shared cell: two started in parallel, a third later, and a fourth on the notebook rebased onto the team's latest commit (117 cells, about 14 minutes each); only process IDs in TensorFlow log lines differed. This is on one machine (CPU, TensorFlow 2.21); TensorFlow's oneDNN warns that other CPUs can change the last digits.
- **What was done / to do:** None needed. Seeds are fixed for numpy, TensorFlow and every model; run it once on Kaggle before submitting.

### V18. Numbers in the markdown vs the outputs

- **Area / severity / status:** Text vs output / should fix / fixed
- **Claimed or found:** Markdown was written by hand.
- **Evidence:** Automated check: 10 numbers in the markdown were not found in any output; 8 were section numbers or a rounding slip (0.385 shown as 0.38), 2 were real errors (N1 13.7%, N2 96%). Reading every interpretive paragraph against its output found two more wrong claims: 'test scores are within 0.03 of validation' (true for PR-AUC only; forest F1 drops 0.057, recall 0.105) and 'rank correlation about 0 means unrelated' (V12). All other claims (ablation, SHAP, LIME, Q3 teams, selection table) match.
- **What was done / to do:** All corrected; every text number that depends on the re-run was refreshed from the new outputs (review/update_text.py).

### V19. Thresholds are sensible

- **Area / severity / status:** Thresholds / note / verified
- **Claimed or found:** F1-optimal thresholds chosen on validation.
- **Evidence:** LR 0.399, RF 0.431; the share predicted positive (15.3%, 15.4%) is close to the true 15.0%. F1 moves between 0.64 and 0.71 for thresholds within ±0.1 of the optimum, so the reported F1 depends on the threshold.
- **What was done / to do:** Mentioned in the metrics discussion.

### V20. environment.txt

- **Area / severity / status:** Environment / should fix / fixed
- **Claimed or found:** Listed PyTorch, not TensorFlow; extra unused packages.
- **Evidence:** Resolves for Python 3.13; full run had 0 errors.
- **What was done / to do:** Done 2026-10-14 (pushed).

### V21. Non-causal wording

- **Area / severity / status:** Interpretation / ok / verified
- **Claimed or found:** Explanations must not be presented as causes.
- **Evidence:** Present in the notebook (c102, Limitations) and in the report (boxed statement). No sentence says the grid or a feature causes podiums; the phrase 'the fast car is what produces podiums' is about the car.
- **What was done / to do:** None.

### N1. Wrong claims in the podium-rate paragraph

- **Area / severity / status:** Text / should fix / fixed
- **Claimed or found:** 'PR-AUC … regardless of shifts in field size' is wrong (its no-skill level equals the podium rate); 13.7% should be 13.5%; 1980s–90s fields averaged 24–25 starters (max 28) and the podium rate was about 12%, not 10–11%.
- **Evidence:** Recomputed by decade in review/verify.ipynb.
- **What was done / to do:** The team rewrote this paragraph in commit 294ceea and dropped the wrong claims (c41); the one remaining error (large-field era podium rate about 10%, actually about 12%) was corrected, and the PR-AUC no-skill-level note was added.

### N2. Sprint paragraph

- **Area / severity / status:** Text / should fix / fixed
- **Claimed or found:** 'over 99%' → 98.4% (1,107 of 1,125 races); training is 1950–2017, so every training season predates sprints; the 'secondary modern-era model' does not exist.
- **Evidence:** Counted from sprint_results.csv.
- **What was done / to do:** Corrected; the modern-era model is described as future work (c39).

### N3. Lab mentions

- **Area / severity / status:** Report / note / resolved by team
- **Claimed or found:** Report cited labs while the notebook did not.
- **Evidence:** No 'Lab N' left in main.tex (commit 2057b24).
- **What was done / to do:** None.

### N4. Bootstrap p-values were outside the notebook

- **Area / severity / status:** Report / should fix / resolved by team
- **Claimed or found:** Report p-values came from code that was not in the notebook.
- **Evidence:** The notebook now contains the paired bootstrap (commit 2057b24) and its output matches the report.
- **What was done / to do:** Bootstrap is on the out-of-scope list.

### N5. Report numbers and figures

- **Area / severity / status:** Report / should fix / fixed
- **Claimed or found:** Every report number was copied by hand from the old outputs.
- **Evidence:** Fixes change counts and scores slightly.
- **What was done / to do:** main.tex numbers and tables re-synced with the new run, figures copied to report/figures (fig_18 renamed fig_18_metrics_splits). main.pdf must be recompiled (no LaTeX on this machine).

### N6. Non-starter count in README and decisions

- **Area / severity / status:** Docs / note / fixed
- **Claimed or found:** README and decisions D4 say 1,566; the notebook said 1,563 (after de-duplication).
- **Evidence:** After the V1 fix the notebook count changes again.
- **What was done / to do:** README updated; docs/decisions.md gets a dated addendum.

### N7. '~80 mechanical failure terms'

- **Area / severity / status:** Text / note / verified
- **Claimed or found:** Comment claims about 80 terms.
- **Evidence:** 77 of 139 statuses fall into the mechanical fallback.
- **What was done / to do:** None.

### N8. Placeholders in the report

- **Area / severity / status:** Report / should fix / open
- **Claimed or found:** Team names and IDs, Kaggle link and GitHub link were placeholders.
- **Evidence:** Names and IDs were filled in by the team (commit 294ceea). The Kaggle and GitHub links in the report are still placeholders (main.tex line 38).
- **What was done / to do:** Add the Kaggle link and the GitHub link before submitting.

### X1. README promises tags the notebook no longer has

- **Area / severity / status:** Docs / note / fixed
- **Claimed or found:** README says anything beyond the labs 'is tagged'.
- **Evidence:** The notebook has no such tags since the lab mentions were removed.
- **What was done / to do:** README sentence reworded to point to OUT_OF_SCOPE.md.

### X2. Grain, missingness and cutoff are not stated in the notebook

- **Area / severity / status:** Q1/Q2/Q3 / should fix / fixed
- **Claimed or found:** The brief asks every data-engineering answer for a short justification of grain, missingness and the cutoff; the report has it, the notebook did not.
- **Evidence:** Compared the three question sections with the brief.
- **What was done / to do:** One 'Grain, missingness, cutoff' line added to each question (c44, c50, c54).

## How to reproduce

```
.venv/bin/python review/make_verify_nb.py   # builds review/verify.ipynb from review/verify_cells.py
cd review && ../.venv/bin/jupyter nbconvert --to notebook --execute verify.ipynb --output verify.ipynb   # about 3 minutes
```

Files: `review/verify_lib.py` (independent rebuild), `review/verify_cells.py` (checks), `review/verify.ipynb` (executed), `review/apply_fixes.py` and `review/apply_fixes2.py` (every edit made to the notebook, as code), `review/sweep_names.py` (out-of-scope name sweep).
