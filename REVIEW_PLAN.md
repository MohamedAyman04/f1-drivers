# Review Plan: Milestone 1 notebook (`notebooks/f1_podium_prediction.ipynb`)

Created 2026-10-14. Deadline: **18 Oct 2026, 23:59** (Kaggle link + GitHub repo + PDF report).

Cell references like `c75` are **0-based cell indices** in the notebook as of commit `1925b7f` (pulled 2026-10-14; cell 0 = the title cell, 110 cells). Commit `2057b24` added 2 cells (a bootstrap section at c92–c94), so every reference from the old c93 onward was moved by +2 on 2026-10-14. Recheck them after pulling new commits.

This plan is separate from `PLAN.md`, which was the plan for *building* the notebook. This one is for **reviewing** it before submission and the oral evaluation.

## Status (updated 2026-10-14, night)

Everything in this plan except the instructor's answers has now been done. Results are in `REVIEW_FINDINGS.md` and `OUT_OF_SCOPE.md`; the board is the team's working copy.

| Workstream | Status | Where |
|---|---|---|
| A. Out-of-scope audit | **Done.** 90 items (36 high priority, 38 required by the brief) from a name-level sweep of every call in the notebook plus a read of all cells, the README and the report | `OUT_OF_SCOPE.md`, `review/sweep_names.py`, board tab "Out of scope" |
| B. Requirements traceability | **Done.** Every brief requirement and checklist box matched to a cell | `REVIEW_FINDINGS.md` |
| C. Independent verification | **Done** except V3, which waits for office hours (question 10) | `review/verify.ipynb` (independent rebuild matches the notebook to four decimals) |
| D. Metrics | **Done on the branch.** All six metrics plotted for validation and for train/validation/test, plus discussion and an "always no podium" reference row | notebook §12–14 |
| Fixes | **Applied on branch `review-fixes`**, rebased onto the team's latest `main` (commit `294ceea`; not merged): V1, V4, V8, V9, V11, V12, N1, N2, X2 and §5. Every edit is a script, so it can be replayed on top of newer teammate changes | `review/apply_fixes.py`, `review/apply_fixes2.py` |
| Report re-sync (N5) | **Done in `main.tex` and `report/figures/`** on the branch; `main.pdf` must be recompiled (no LaTeX on the machine used) | `report/` |
| Still open | Office-hours answers to the 18 questions (board), V3 decision, merge `review-fixes`, run on Kaggle, N8 placeholders (Kaggle and GitHub links in the report; team names and IDs are done), make the repo accessible | board, `REVIEW_FINDINGS.md` |

## 0. Goals and ground rules

**Goals**
1. **Course-scope audit.** List *everything* in the notebook that the labs don't teach: concepts, tools, functions, methods and even code comments. The list goes in `OUT_OF_SCOPE.md` so the team can study each item before the evaluation.
2. **Requirements audit.** Every requirement in the brief (`csen903_project.pdf`) and every checklist box is matched to the cell that satisfies it, or flagged as a gap.
3. **Independent verification.** Don't trust the notebook. Recompute the key numbers ourselves, check each claim in the markdown against the output, and check the logic of each step.
4. **Metrics.** Precision, recall and accuracy are already computed (the `metrics()` function in c75) and appear in the result tables. They are **not plotted or discussed**. Add them to the comparison plots and the written discussion (see §5).
5. **Office-hours questions** for the instructor session on 2026-10-15 (see §1).

**Ground rules**
- **Course scope = the 4 lab PDFs (`labs/`) plus the 3 guide notebooks (`guide_notebooks/`), nothing else.** The brief does *not* make something in scope. Items the brief requires are still listed, tagged `required by brief`, so we know we'll be asked about them.
- No edits to the deliverable notebook until the review findings are written down (§3, §4). Then apply fixes in one pass, on a branch (`review-fixes`), not on `main`.
- Verification code goes in a separate scratch notebook (`review/verify.ipynb`), never in the deliverable.
- Every finding records: **cell number, what was claimed, what we found, evidence (number/plot), severity** (`blocker` / `should fix` / `note`).
- Environment: `.venv` (Python 3.13). Full run-all takes about 20 min on the local machine.
- **Repo visibility.** The GitHub link is submitted and graders can open it, and the repo has no `.gitignore`. Review files (`REVIEW_PLAN.md`, `OUT_OF_SCOPE.md`, `REVIEW_FINDINGS.md`, `review/`) are candid working notes, so keep them out of the submitted repo: add them to `.gitignore` (or keep them on a branch that is not the default one) and decide as a team before pushing. `guide_notebooks/` is untracked; same decision (the labs themselves are already tracked).
- **Naming.** "Office Qn" = office-hours question (§1). "Q1/Q2/Q3" in §3 and §4 = the three data-engineering questions.

---

## 1. Office-hours questions (2026-10-15)

The list was cut from 21 to 18 questions and reworded in plain language on 2026-10-15 (the board has the same list, with space for the answers). Starred (★) questions are the ones not to leave without an answer. The wording below is what we read out; the board also has a short "why we ask".

### Unclear in the project text
1. ★ The project text says to report “all five metrics from Section 4”, but that section only lists three: ROC-AUC, PR-AUC and F1. We report six (those three plus precision, recall and accuracy). Is that fine, or did you have five specific metrics in mind?
   *Why we ask: We already compute all six, so this only decides what we show in the results table and plots.*
2. ★ The project text doesn't say which model to explain with SHAP and LIME, or which model the prediction function should use. At the moment permutation importance covers all four models, SHAP and LIME cover the random forest, and the prediction function uses the neural network. Do you want SHAP and LIME for every model (the Lab 4 graded task does it for a neural network), or is one model enough? And should the prediction function use a particular model?
   *Why we ask: We can do SHAP and LIME on the neural network as well. It just takes more work and more runtime, so we would like to know what is expected before we add it.*
3. ★ The project text says each row is “one driver entering one race”. We only keep drivers who actually started the race, so drivers who entered but never started (didn't qualify, withdrew) are dropped. Is that okay, or should they stay in the data as “no podium”?
   *Why we ask: About 1,640 of the 26,670 rows (6%) are drivers who never started. Keeping them would lower the podium rate and count cars that never raced.*
4. ★ The project text asks for permutation importance, but our lab manuals and guide notebooks only teach SHAP and LIME. We wrote permutation importance ourselves: shuffle one input column, score the model again, and see how much worse it gets. Is that fine, and how much detail do you want when we explain it?
   *Why we ask: We checked all four lab manuals and the three guide notebooks and none of them mentions it. We wrote it by hand because the scikit-learn version doesn't work with our neural network.*
### Our choices: is this okay?
5. ★ How we choose features: we start with just the starting grid, then add one group of related features at a time (for example all the driver-standings features) and keep a group only if the validation score improves. Afterwards we run the ablation: we take each group out of the finished model and see how much worse it gets. Is that the order you expect (choose first, then use the ablation to measure each group), or should the ablation be what decides which features we keep?
   *Why we ask: The project text says features are justified before training and the ablation measures their contribution afterwards, so we read them as two separate steps. If the ablation had to do the choosing it wouldn't work well here: with all 11 features, taking out any single group barely changes the score, because the groups overlap.*
6. For the ablation, is it enough to take out the groups of our final features (starting grid and driver standings) for each model, or do you also want every candidate group removed? And should we show the results on the validation data, the test data, or both?
   *Why we ask: Right now we do all of it: the final features for all three models, and all 11 candidates for logistic regression and the forest, shown on validation and test. We only want to know whether that is more than needed. The test results are never used to make a decision.*
7. ★ The checklist asks for a “form” feature: how well a driver or team has done in its last few races. We use the share of the last 5 races that ended on the podium, using only past races. We built it and checked there is no leakage, but it didn't make the models better, so it isn't in our final features. Is that fine?
   *Why we ask: We would rather not force a feature in just to tick a box. The code, the leakage check and the test of whether it helps are all in the notebook.*
8. ★ You said to show each attempt in its own cell or section, so the before and after is visible. For choosing features we try all the options in one loop and show one table with a row per attempt (the hyperparameter search works the same way). Is a table like that enough, or do you want every attempt as a separate cell?
   *Why we ask: Separate cells would make the notebook a lot longer. The table shows every attempt with its score, so nothing is hidden, but we want to be sure that counts.*
9. The checklist asks for the “effect of the step on data distribution, range and values”. We show before and after for the cleaning steps, for filling in missing values and for scaling. Is that enough, or do you want it for every step, including the ones that build the features?
   *Why we ask: Showing it for every step would add many plots, so we want to know where to stop.*
10. ★ The project text asks for “at least two preprocessing-order experiments”. We read that as: change the order of the preprocessing steps and see whether the results change. Ours are (1) fitting the imputer and scaler on all the data before splitting, versus on the training data only, and (2) filling in missing values and then scaling, versus scaling and then filling in. Is that what you mean?
   *Why we ask: Both of ours show almost no difference in the scores. We explain why in the notebook (most rows are training rows, and scaling doesn't change which values are missing), but we want to be sure this is the kind of experiment you meant.*
11. The checklist says “plotting the effect of the features” under feature selection. We read that as plots showing how each feature relates to a podium (for example the podium rate for low versus high values of a feature), plus a plot of how the validation score changes as we add feature groups. Is that what you mean?
   *Why we ask: This step comes before the final models, so we weren't sure what “effect” refers to.*
12. Is it okay to try several different splits (for example different cut-off seasons for validation and test) and compare the results, as long as we choose the final split without looking at the test data?
   *Why we ask: The project text asks for a justified split, and we use one split by season. Trying others could show how stable our results are, but we don't want it to look like we picked the split that makes the test score look best.*
13. Our baseline is a logistic regression that only uses the starting grid. The other models score about 0.06 to 0.07 higher on our main metric (PR-AUC about 0.71 to 0.73, against 0.65 to 0.67 for the baseline). Is that a good enough baseline, and is that the size of improvement you would expect?
   *Why we ask: The starting grid already explains a lot of who finishes on the podium, so there isn't much room left to improve.*
14. For the mechanical-retirement question, we count any retirement that isn't clearly a crash, a driver problem or a rule decision as mechanical. That includes vague reasons like “Retired” or “Puncture” (about a fifth of the cases we count). Is that okay if we explain the rule, or should we leave the unclear ones out?
   *Why we ask: We checked it both ways. Leaving the unclear ones out lowers the rates (11.2% to 9.0%, and 6.9% to 5.5%), but cars still got more reliable, so the answer doesn't change.*
15. We use a few things the labs didn't teach, for example confidence intervals for the rates, joining tables with pandas, and a bootstrap test for the preprocessing experiments. Is that fine as long as we can explain each one and why we used it?
   *Why we ask: The full list is in our out-of-scope file. We would rather not drop useful methods, but we want to know what you expect.*
16. In the evaluation, how deep should our explanations go: the idea in simple words, or the maths as well (for example how SHAP values are calculated, or how a neural network learns)?
   *Why we ask: This decides how much of the extra material we need to study in depth.*
### Practical
17. Our notebook starts with a line that installs three packages (shap, lime and missingno), so the internet has to be switched on, for example on Kaggle. Is that okay, and is there a time limit for running the notebook? A full run takes about 14 minutes.
   *Why we ask: If the notebook is run somewhere without internet that first line fails, so we want to know what you expect.*
18. Do you expect the report to follow a specific structure or length? And does every plot in the notebook also have to be in the report?
   *Why we ask: The project text says the plots and answers must be in both. Our report has all 26 figures and the three answers; we just want to be sure we aren't missing a required format.*

**Dropped** (we can answer these ourselves, or the project text already answers them): qualifying position vs grid (the instructor already said grid only), qualifying features, LIME vs local SHAP (the project text says "LIME and/or local SHAP"), whether to use the same validation data for everything, whether to split by season (we justify it in the notebook), experiments with no effect, whether a hyperparameter grid counts as modeling attempts (the project text says hyperparameter variants count), home-race definition, and choosing the model on validation.

---

## 2. Workstream A: Course-scope audit (`OUT_OF_SCOPE.md`)

### A.1 Build the in-scope reference (do first; ~1 h)
From the 4 lab PDFs and 3 guide notebooks, list every function, parameter, concept and plot type taught. Starting inventory (complete it while reading):

| Lab | In scope |
|---|---|
| 1: Wrangling & viz | `read_csv`, `head`, `info`, `describe`, `isnull().sum()`, `value_counts`; numerical vs categorical; impute vs remove; `drop(columns=)`; `fillna` with **median** (robust to outliers) / **mode**; label vs one-hot encoding, `map`, `get_dummies(drop_first=True)`; `missingno.matrix`; plot types (comparison/distribution/composition/relationship); `plt.hist`, `plt.scatter`, bubble chart, `sns.heatmap` on `pivot_table` and on correlations, `pd.cut` binning, `sns.barplot`, `sns.boxplot` (outliers) |
| 2: ML foundations | supervised/unsupervised/semi-supervised; classification vs regression; `train_test_split(test_size, random_state)`; `OrdinalEncoder`; `drop_duplicates`; `LinearRegression`, `LogisticRegression(max_iter)`; sigmoid, probability + threshold; accuracy, precision, recall, F1, `classification_report`, `confusion_matrix` + heatmap; "accuracy misleads on imbalanced data"; MSE/RMSE/MAE/R²; dropping a column to avoid **leakage**; hyperparameter tuning (guide notebook markdown); baseline vs improved model + comparison plot |
| 3: Neural nets (Keras) | scaling and why; `StandardScaler` fit on train, transform test; `stratify=y`; `keras.Sequential`, `layers.Input`, `Dense(relu)`; ReLU/sigmoid/softmax; input/hidden/output layers; "deep" = more than 2 hidden layers; forward pass, loss, backpropagation, gradients; binary/categorical cross-entropy, MSE/MAE as losses; optimizers SGD vs Adam, learning rate; epochs, batches (`batch_size=32`), `validation_split`; overfitting; `EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)`; train/val loss and accuracy curves; `model.evaluate`; comparison table; `np.random.seed`, `tf.random.set_seed` |
| 4: XAI | intrinsic vs post-hoc; model-specific vs model-agnostic; global vs local; `DecisionTreeClassifier`, `plot_tree`, gini; `RandomForestClassifier`, bagging, black box; `shap.TreeExplainer`, `shap_values`, `summary_plot` (beeswarm), `force_plot`, `expected_value`, waterfall plot (graded task); LIME, surrogate model, perturbations, `LimeTabularExplainer(discretize_continuous=True)`, `explain_instance(num_features)`, `show_in_notebook`; fairness/bias/trust; `!pip install shap lime` |

### A.2 Sweep the notebook cell by cell (~3 h)
For each of the 110 cells, check every **import, function/method call, parameter, statistical concept, modeling idea, plot type, domain term and code comment** against A.1. Anything not found goes in `OUT_OF_SCOPE.md`. Also sweep `README.md` and **the report (`report/main.tex`)**, since examiners read those too. The report has content that isn't in the notebook (e.g. the paired-bootstrap p-values, see N4).

### A.3 `OUT_OF_SCOPE.md` format
One row per item, grouped by section of the notebook:

| # | Item | Where (cell) | Kind (concept / tool / method / comment / domain) | Required by brief? | What it is (1–2 lines) | Why we used it | Could a lab tool do it instead? | Study priority (H/M/L) |

**Study priority:** H = likely to be asked (it drives a result or is in the brief); M = supporting detail; L = plumbing (e.g. `glob`, `Path`).

### A.4 Starting list (to confirm and extend in A.2; *not* final)
- **Statistics:** Wilson 95% confidence interval and the idea of a confidence interval (c43); odds ratio and `exp(coef)` (c50); "controlling for" grid by stratifying, and a weighted average of the within-group gaps (c49); logistic regression used for *inference* rather than prediction (c50); `C=1e5` and regularization strength (c50, c75); Spearman rank correlation, in Q1 and for feature screening (c46, c66), where the labs only use the default `corr()`; rank stability between eras (c46).
- **Evaluation:** PR-AUC / `average_precision_score`, ROC-AUC / `roc_auc_score` (required by brief); `precision_recall_curve` and choosing the threshold that maximizes F1 (c75); `zero_division`; the prevalence-dependent baseline of PR-AUC.
- **Splitting/selection:** temporal (by-season) split instead of `train_test_split` (c56); a separate validation set instead of `validation_split`; grid search over `C`, depth, `min_samples_leaf` and `class_weight` (c75); `class_weight="balanced"` and class imbalance; forward feature selection with a `MIN_GAIN` rule (c69); ablation and preprocessing-order experiments (required by brief).
- **Preprocessing:** `SimpleImputer` (the lab used `fillna`), `make_pipeline`/`Pipeline` (c61–62), `sklearn.base.clone` (c86, c89), `inverse_transform` (c96); quantile bins with `pd.qcut` + `rank(method="first")` (c64).
- **Core pandas the labs never use (required by the brief's "multi-hop joins", but still not taught):** `merge` (c6, c30, c35, c45, c49, c53, c59), `groupby` + `agg` (in ~15 cells), `Series.map`, `np.where`, `lambda`, `sort_values`/`rank`, `assert`, f-strings, dict/set comprehensions. A grep of the lab PDFs and guide notebooks finds no `merge` or `groupby`. List them as one "required by brief" block so we can explain join keys, join type and why the row count is asserted after each join.
- **Feature engineering:** `groupby().shift()`, `.rolling(5, min_periods=1)`, `groupby().transform` (c57, c59); merging standings on the previous round (`prev_raceId`); points as a share of the leader's (c59); the nullable `Int64` type; `merge(validate="many_to_one")`; a leakage check that recomputes rows the slow way (c60).
- **Data audit:** primary/foreign keys and orphans (required by brief); sentinels; `pd.to_datetime`; parsing durations into seconds (c34); `assert`; `np.allclose`/`np.isclose`.
- **Keras:** `keras.utils.set_random_seed`; `validation_data=` instead of `validation_split`; `EarlyStopping(monitor="val_loss")` (the lab monitors `val_accuracy`); `batch_size=256`; `Adam(1e-3)` with an explicit learning rate; `model.predict` (c78).
- **XAI:** permutation importance written as a custom loop (required by brief; not in labs) (c98); the `shap.Explanation` object, `base_values` and the additivity check (c101); `shap.dependence_plot`; `summary_plot(plot_type="bar")`; LIME `as_map`, `exp.score` (R² of the surrogate), `num_samples`, `random_state`, `as_pyplot_figure` (c105).
- **Comments to explain or rewrite:** `rf_best.set_params(n_jobs=1)  # one thread = same summation order…` (floating-point summation order) (c75); `# large C = practically no regularisation` (c50); `# repr shows trailing spaces` (c22); `warnings.filterwarnings(... IProgress|sklearn.utils.parallel.delayed ...)` (c2).
- **Domain:** pit-lane start and `grid_eff` = last slot + 1; Indy 500 history; Q1/Q2/Q3 sessions; the turbo-hybrid era; shared drives; status taxonomy.
- **Added by the 2026-10-14 pull:** surrogate key vs composite key and "entity grain" (c29); majority-class baseline and class prevalence (c41); prevalence-dependence of PR-AUC (c41, and the claim there is wrong, N1); sprint races and a "secondary modern-era model" (c39); pre-qualifying era (c23); **paired bootstrap, p-values and 95% CI of a difference** (c92–c94 and the report table "Preprocessing-order experiments"); "regularisation strength $C$" and "probabilities are coarse" for the forest (report §Models).
- **Lab mentions:** the pull removed every "Lab N" reference from the notebook, so the notebook no longer shows which lab a tool comes from. The report no longer cites labs either (N3, resolved). `OUT_OF_SCOPE.md` doesn't depend on those labels, since it's built from the lab material itself.
- **Plumbing (L):** `glob`, `pathlib.Path`, Kaggle path detection, `%pip`, `display()`, `.query()`, `.xs`, `MultiIndex`/`unstack`/`pivot`, `PercentFormatter`, error bars (`xerr`/`yerr`).

**Done when:** every cell has been swept, every item has all columns filled, and each H-priority item has someone assigned to it.

---

## 3. Workstream B: Requirements and checklist traceability

Build a table in `REVIEW_FINDINGS.md`: **requirement → cell(s) → status (met / partly met / missing) → evidence → fix needed.** Cover every bullet of brief §1–§6, Deliverables 1–8, "What to expect at the evaluation" 1–5, the notebook/report requirements, and every checklist box.

Starting assessment from reading the notebook (each one to be confirmed):

| Requirement | Where | Starting status |
|---|---|---|
| EDA before and after cleaning, with plots | §3–4, §6 | Met; check that the before/after comparison is explicit for each cleaning step |
| Sentinels, semantic typing (dates, durations) | c33–34 | Met |
| One row per (raceId, driverId), rule explained | c30–31, c35 | Met |
| PK unique and non-null; FK reconcile | c27–28 | Met |
| "EDA can't be manual inspection… implementation choices + justification + visualization" | throughout | Check every cleaning decision has a plot/statistic |
| Q1 incl. **qualifying position vs grid** | c44–47 | Met: the instructor confirmed **grid only**. c44 already explains why grid and not qualifying position; add one line saying grid-only was confirmed |
| Q2 controlling for starting position | c48–51 | Met; check robustness (V10) |
| Q3 mechanical rate, two periods | c52–55 | Met; check taxonomy (V11) |
| Each DE answer: grain, missingness, cutoff justification | c44–55 | Check that missingness and grain are explicitly stated for each Q |
| Feature choice backed by evidence; trials kept in the notebook | c63–71 | Met |
| Split justified | c56 | Met |
| ≥ 2 statistical models + shallow NN, same rows and features | c74–78 | Met |
| All metrics, every model, on train/val/test | c80, c83 | Met in tables; **not all plotted or discussed** (§5) |
| Thresholds and hyperparameters chosen without the test set | c75, c80 | Met; check ablation/experiments don't feed decisions (V8) |
| Primary/secondary metric choice justified | c41 | Met |
| Feature-group ablation for each group | c86 | **Partly met: only the 2 kept groups, LR/RF only → will fix (V8)**; Office Q5–Q6 decide the scope |
| ≥ 2 preprocessing-order experiments | c89–94 | Met (both show ~0 effect) |
| Checklist: "effect of the step on data distribution/range/values" | c73 | Met for imputation+scaling; check other steps (dedup, non-starter removal) |
| Leakage-safe form feature | c59–60 | Implemented and checked, **not selected** (Office Q7) |
| Prediction model: "how it works" and "its limitations" | c74, c77, c81 | Brief; expand per model |
| Permutation importance + SHAP global; LIME/local SHAP; all interpreted | c97–106 | Met; **explains RF, final model is NN**: the brief doesn't specify the model, so ask (Office Q2, V3) |
| Non-causal statement | c95, c109 | Met; it must also be in the report |
| Inference function: complete and partly missing record | c106 | Met |
| ≥ 3 modeling attempts; baseline | c75–78 | Met |
| Run-all ready, no manual cells | — | Met locally on 2026-10-14 (0 errors, before the 2 bootstrap cells were added); **re-run locally and test on Kaggle** |
| Report (PDF) | `report/main.tex`, `report/main.pdf` | **Draft exists** (added 2026-10-14, now with a bibliography, all 26 figure numbers (29 image files; all used), all 3 DE answers, all 6 metrics, non-causal box, limitations). Missing: team names/IDs, Kaggle and GitHub links. Needs N5 and N8 (N3 resolved) |
| GitHub repo accessible; Kaggle link | — | Check repo visibility (it returned 404 publicly on 2026-10-14, so it's private) |

---

## 4. Workstream C: Independent verification ("don't trust the notebook")

Work in `review/verify.ipynb`. For every section: (1) recompute the key numbers independently (different code path, not copy-paste); (2) check each number quoted in the markdown against the actual output; (3) ask "is this step logical for this task?"

### C.1 Issues already seen (confirmed with a quick check on 2026-10-14)
- **V1 (should fix, confirmed 2026-10-14).** The notebook text (c23, c25) says drivers who "did not qualify or withdrew" are non-starters, and `PLAN.md` (line 260) meant to exclude Withdrew from race starts, but the code (c35) only catches `grid == 0 & laps == 0`. 61 rows with status `Withdrew`, `grid > 0` and `laps == 0` are therefore counted as **starters** (59 in train, 2 in test). The same holds for 11 `Did not qualify` and 7 `107% Rule` rows (1954–1997) that carry a grid slot and 0 laps; a few `Disqualified` (4), `Excluded` (1), `Not restarted` (1) and `Safety` (2) rows with 0 laps are ambiguous. Together about 87 of 24,724 modeling rows (0.35%). This affects the target denominator, `field_size`, `grid_eff` (V16) and the Q2/Q3 rates. Impact is small and the fix is cheap: add the clear cases (`Withdrew`, `Did not qualify`, `107% Rule` with `laps == 0`) to the non-starters, state the rule in c25/c35, and measure the effect. Do **not** drop all `laps == 0` rows: 353 `Collision` and 224 `Accident` rows with 0 laps are real starters who crashed at the start. Note the brief defines the unit as "one driver *entering* one race"; ask in Office Q3.
- ~~**V2.** Q1 never uses qualifying position.~~ **Closed:** the instructor said to use grid only (asked earlier).

### C.2 Things to check
| ID | Area | Check |
|---|---|---|
| V3 | XAI | Final model = NN (best val PR-AUC 0.727 vs RF 0.726, a tie) but SHAP/LIME explain the RF. **The brief doesn't say which model must be explained → question, not a fix (Office Q2).** Options once answered: make RF the final model (a tie, and it's explainable with TreeExplainer, as in Lab 4), or add SHAP/LIME for the NN. Either way, the notebook must state *why* the explained model was chosen. |
| V4 | NN | **Will fix.** Early stopping (monitor `val_loss`, patience 15) practically never fired: the three sizes ran 200, 200 and 198 of 200 epochs ("val loss still creeping down"). Re-run with more epochs to see where it stops; compare with the lab's `monitor="val_accuracy"`; add a training curve for each size; also try the lab-style variants (SGD vs Adam, with/without early stopping), which are in scope. |
| V5 | Evaluation | Validation is used for early stopping, model choice **and** threshold, so validation scores are optimistic. Say so; test is the honest number. |
| V6 | Evaluation | Claim: "train PR-AUC < val because old seasons are less predictable." Verify: compute train PR-AUC per decade; and compare podium rate per split, since PR-AUC's baseline equals the positive rate and that differs between splits (c57: train 13.3%, val 15.0%, test 15.0%). |
| V7 | Features | Round-1 rows have no standings, so they get the train medians (e.g. median `drv_pos`). What share of val/test rows is this? Is the median sensible for a reigning champion in round 1? Alternatives to *consider*: last season's final standings, or a "round 1" flag. Same question for the inference function's rookie example. |
| V8 | Ablation | **Will fix.** Only 2 groups, LR+RF only, plot shows **test** drops. Add an ablation from the all-candidates model (every group) and the NN; show val and test. Make sure no decision uses test. |
| V9 | Feature selection | The ladder depends on the order groups are added. Re-run with a different order, or with backward removal, and report whether the selected set changes. Check the 0.005 threshold is justified (noise level: bootstrap val PR-AUC). |
| V10 | Q2 | Confirm `X_q2` column 0 is `home` (the coefficient read is `coef_[0][0]`). Recompute the gap independently. Check nationality→country mapping (drivers with no home circuit; `UK`, `USA`). Consider a bootstrap CI for the gap (numpy only). |
| V11 | Q3 | List every status counted as mechanical in 2014–2024 with counts. Sensitivity: strict (clear component failures) vs broad (current). Does the team ranking change? Check team-name continuity (Alfa Romeo/Sauber, Toro Rosso/AlphaTauri/RB). |
| V12 | Q1 | Re-derive 55.5% → 76.0%. Check the `K_MIN=10` sensitivity (the printed counts for 5/10/15/20) and the era boundary. Check that the Spearman −0.03 is computed on enough circuits to mean anything. |
| V13 | Cleaning | Re-derive: 85 duplicate pairs, all pre-1979; the kept row = lowest real grid slot; counts of 1,563 non-starters (1,566 before deduplication, see N6) and 72 pit-lane starts; Indy rows. Check the rule for both-rows-no-grid. |
| V14 | Cleaning | Sentinel conversion: every converted column, no silent coercion (the "nothing lost" rule); `to_seconds` checked against `milliseconds` (already asserted). |
| V15 | Leakage | Confirm Ergast `driver_standings` at `raceId` = standings *after* that race, so joining on `prev_raceId` is safe. Check round 1 and season boundaries. Check `last5` over season boundaries and that the same race never feeds itself (c60 checks 150 rows; extend to all rows). Check `grid_rel`/`field_size`: is the number of starters known before the race? (Yes, after qualifying; but V1 affects it.) |
| V16 | Features | `grid_eff` = max grid + 1 for pit-lane starters: check races with several pit-lane starters. |
| V17 | Reproducibility | Run all twice; diff every printed number. Run on Kaggle once. |
| V18 | Text vs output | Automate: extract every number quoted in the markdown and compare it with the cell outputs (the markdown was written by hand and can go stale). Known example: c84 says test scores are "within 0.03" of validation, which holds for PR-AUC only (forest F1 0.724 → 0.667, recall 0.733 → 0.627). |
| V19 | Thresholds | Baseline threshold is also tuned on val; check F1-optimal thresholds are sensible (not extreme); show the precision/recall at that threshold. |
| V20 | Environment | **Done 2026-10-14:** `environment.txt` now lists TensorFlow + missingno (PyTorch, scipy, pyarrow, pypdf removed; none are imported) and resolves for Python 3.13; the README line is fixed. |
| V21 | Interpretation | Check the "explanations are not causal" wording is in the notebook **and** the report, and that no sentence implies causation (e.g. "grid causes…"). |

### C.3 Findings from the 2026-10-14 pull (commits `3206307`, `51e3771`, `ed25693`)
**What changed:** `3206307` (MohamedAyman04) edited only notebook *text*: no code logic changed. It fixed garbled characters (`â€“` → `–`), removed all "Lab N" mentions, added markdown on grid/entries (c23), primary key vs modeling grain (c29, c31), sprint races (c39) and the podium-rate figure (c41), and expanded the status-class comment (c36). One cell was added, so cells from old c23 on shift by +1 (the plan has been renumbered). `51e3771` (Abdelhamid Sameh) added the LaTeX report (`report/main.tex`, compiled `main.pdf`, Overleaf zip, copies of the figures (29 image files, 26 figure numbers)).

**Effect on this plan:** no V-item is resolved or made obsolete by the first pull, because the code is unchanged. The second pull (`2057b24`, `1925b7f`) resolved N3 and N4 (see the table) and added the 2 notebook cells. The report adds a new deliverable to check (§3 row, N3–N8). Some new notebook text is wrong:

| ID | Where | Finding (checked against data/outputs) | Severity |
|---|---|---|---|
| N1 | c41 | "PR-AUC evaluates ranking … **regardless of shifts in field size**" is wrong: PR-AUC's no-skill level *equals* the positive rate, so it moves with field size. ROC-AUC is the one that doesn't depend on prevalence. Also says **13.7%** podium rate; the output (c40) and report say **13.5%**. Other numbers in c41 are off too: "grids often exceeded 26–30 starters" in the 1980s–90s (actual average 24–25, maximum 27–28 starters) and "podium rate dropped to ~10–11%" (actual ~12% per decade, 11.5–14.2% per season). | should fix |
| N2 | c39 | Sprint text: "over 99% of races had no sprint" → actually **98.4%** (18 of 1,125). "~96% of training seasons (1950–2020)" → training is **1950–2017**, so 100% predate sprints. The "secondary / improved modern-era model" paragraph describes a model that doesn't exist; examiners may ask why it wasn't built. Reword as future work or move to Limitations. | should fix |
| N3 | notebook vs report | **Resolved by commit `2057b24`:** the report no longer cites labs (no "Lab N" left in `main.tex`); it now has a bibliography (`references.bib`: Wilson, Breiman, Efron, SHAP, LIME, sklearn, dataset). Still decide after Office Q15 whether to mark non-lab tools as such; `PLAN.md` required tagging them **[beyond labs]**. | resolved |
| N4 | notebook c92–c94, report table "Preprocessing-order experiments" | **Resolved by commit `2057b24`:** the paired bootstrap (2,000 redraws of validation races, `default_rng(0)`) is now in the notebook, and its output matches the report and `docs/decisions.md` D29 (LR −0.0001, p = 0.69; RF −0.0004, p = 0.29; scale→impute p = 0.99 / 1.00). Still to do: bootstrap, p-values and confidence intervals of a difference are not in the labs, so they are on the out-of-scope list; re-check these numbers after any re-run. | resolved |
| N5 | report | Every report number is hand-copied from notebook outputs. Fixes V1, V4, V8 and §5 change them (e.g. 24,724 rows, 1,563 non-starters, all metric tables, fig 17, fig 20). Re-sync the report and re-copy `notebooks/_working_local/figures/` → `report/figures/` after the final run. Extend V18 (text vs output) to `main.tex`. | after fixes |
| N6 | `README.md`, `docs/decisions.md` D4 | Say **1,566** non-starters; the notebook output (c35) and report say **1,563** (1,566 is the count before deduplication). | note |
| N7 | c36 comment | "~80 individual mechanical failure terms": **checked, fine** (77 statuses fall into the mechanical fallback out of 139). Note that "Retired" is the 2nd-largest 'mechanical' status since 2014 (50 rows, after Engine 81) although its cause is unknown (V11, Office Q14). | resolved |
| N8 | report | Placeholders: team names and IDs, Kaggle link, GitHub link. | before submission |

**Done when:** every V-item has a finding with evidence and a severity, and each blocker/should-fix item has an agreed fix.

---

## 5. Workstream D: Metrics (precision, recall, accuracy, F1)

Already done in the code: `metrics()` returns PR-AUC, ROC-AUC, F1, Precision, Recall, Accuracy, and the train/val/test tables (c80, c83) show all six. Still to do:
1. **Plots:** `fig_17` (c80, validation: now only PR-AUC, ROC-AUC, F1) and `fig_18` (c83, PR-AUC across train/val/test) show all six metrics, e.g. a grouped bar per metric with models side by side, train/val/test where it fits. `fig_19` (confusion matrices on test) stays.
2. **Discussion (markdown after c80 and c84):**
   - What each metric means **for this task**: precision = "when we say podium, how often is it right"; recall = "of the real podiums, how many we caught" (Lab 2 wording).
   - Why **accuracy** is high but useless here: the "always no podium" model gets 0.867 on train and **0.850 on validation and test** (podium rate 13.3% / 15.0% / 15.0%); c41's "~86%" is the all-seasons figure. Put the per-split number next to our accuracies (ours: 0.907–0.916 val, 0.892–0.906 test).
   - Precision and recall depend on the threshold chosen on validation; F1 balances them.
   - Which metric is primary/secondary and why (already in c41; restate next to the table).
3. **Add an "always no podium" row** to the table (accuracy 0.85 on val/test, recall 0, F1 0) as a reference point for accuracy. *Optional, decide during the fixes pass.*
4. The report's results table already has all six metrics (`main.tex`, Table "All metrics…"). Still to do there: replace `fig_17` with the six-metric version and add the same discussion points.

---

## 6. Workstream E: Evaluation readiness

- Assign each notebook section to a team member as "owner", but **every** member reads every section (anyone can be asked anything).
- For each H-priority item in `OUT_OF_SCOPE.md`: a 2–3 sentence explanation in our own words + why we chose it over the lab tool.
- Prepare a one-line defense for every decision in `docs/decisions.md` that's still in effect (M1–M6, D-items, #26), each tied to a plot or number in the notebook.
- Practice questions: "Why PR-AUC?", "Why not a random split?", "Why median imputation?", "Why is the training score lower than validation?", "What's the difference between SHAP and permutation importance?", "Why is your RF depth 6?", "How is the threshold chosen?", "What is leakage, and where could it happen here?".

---

## 7. Order of work and timeline

| When | Work |
|---|---|
| 14 Oct (today) | This plan. Finalize the office-hours questions (§1). |
| 15 Oct | **Office hours.** Then A.1 (in-scope reference) + start A.2. Start B (traceability). |
| 15–16 Oct | Finish A.2 → `OUT_OF_SCOPE.md`. C (verification) → `REVIEW_FINDINGS.md`. |
| 16 Oct | Team meeting: go through findings + office-hours answers, agree on fixes. |
| 16–17 Oct | Apply fixes on branch `review-fixes` (V1, V4, V8, N1, N2, metrics §5, plus V3 and anything else from office hours). V20 (`environment.txt`) was done on 2026-10-14. Re-run all locally **and on Kaggle**. Update markdown numbers (V18). |
| 17 Oct | Update the existing report draft (`report/main.tex`): re-sync numbers and figures with the fixed notebook (N5), fill in N8, recompile `main.pdf`. |
| 18 Oct | Final check against the checklist; make the repo accessible to the graders; submit the form before 23:59. Use the rest of the time for evaluation practice (§6). |

## 8. Output files

| File | Contents |
|---|---|
| `REVIEW_PLAN.md` | This plan |
| `OUT_OF_SCOPE.md` | Course-scope audit (§2) |
| `REVIEW_FINDINGS.md` | Requirements traceability (§3) + verification findings (§4) |
| `review/verify.ipynb` | Independent recomputation code (not a deliverable) |
| `report/main.tex` → `main.pdf` | The report (teammate's draft, 2026-10-14); reviewed under §3 and C.3 |
