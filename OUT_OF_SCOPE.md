# Out-of-scope audit: what the notebook uses that Labs 1–4 do not teach

Built on 2026-10-14. Cell numbers (0-based) are those of the **fixed notebook on branch `review-fixes`** (115 cells); the audit was done on the notebook at commit `1925b7f` (110 cells) and the numbers were remapped automatically (`review/remap.py`), with the constructs added by the fixes appended. `README.md` and `report/main.tex` were swept too.

**Scope used:** the 4 lab PDFs (`labs/`) and the 3 guide notebooks (`guide_notebooks/`), nothing else. The project brief does not make something in scope; items it requires are tagged *required by brief* so we know we will be asked about them.

**How it was built.** (1) Every name the notebook code uses (imports, methods, attributes, keyword arguments, functions) was extracted with Python's `ast` module and looked up in the lab text and guide-notebook code (`review/sweep_names.py`, result in `review/sweep_names.json`: 407 names in the fixed notebook, 229 not found). (2) All 110 cells, their markdown and their comments were then read one by one for concepts, methods and domain terms that a name search cannot find. (3) `README.md` and the report were swept for jargon. The item list lives in `review/oos_items.py`; `review/build_oos.py` regenerates this file.

**Result:** 99 items, 38 of them priority H, 42 required by the brief. Priority: **H** = likely to be asked (it drives a result or is in the brief); **M** = supporting detail; **L** = plumbing.

**Caution when reading the lab columns.** 'Not in the labs' means the lab PDFs and guide notebooks never show it. Lab 1's last exercise asks for a correlation heatmap, which implies `DataFrame.corr()`, but the code is not shown; we treat the default `corr()` as in scope and the `method="spearman"` argument as out. The Lab 3 *text* explains binary cross-entropy and sigmoid, but its guide *code* uses softmax and categorical cross-entropy.

The board (https://claude.ai/artifact/3hor4cmonHZQ4vhWBZWH8r, tab *Out of scope*) shows the same list with status, owner and note fields.

## Summary by section

| Section | Items | H |
|---|---|---|
| 1. Setup (c1–c2) | 4 | 0 |
| 2–4. Loading and EDA before cleaning (c3–c25) | 7 | 2 |
| 5. Keys and duplicates (c26–c31) | 4 | 2 |
| 6. Cleaning (c32–c38) | 10 | 4 |
| 7. Target and cutoff (c39–c41) | 4 | 1 |
| 8. Data-engineering questions (c42–c55) | 11 | 4 |
| 9. Modeling table and candidate features (c56–c60) | 8 | 5 |
| 10–11. Preprocessing and feature selection (c61–c73) | 8 | 4 |
| 12–14. Models and evaluation (c74–c84) | 19 | 8 |
| 15. Experiments (c85–c94) | 4 | 3 |
| 16. Explainability and inference (c95–c108) | 12 | 4 |
| 18. Limitations (c109) | 1 | 0 |
| Plotting and Python plumbing (L) | 4 | 1 |
| Report and README only | 3 | 0 |

## 1. Setup (c1–c2)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o001 | %pip install -q shap lime missingno | c2 | tool | no | Installs packages from inside the notebook. Needs Internet on Kaggle. | Kaggle does not ship shap, lime or missingno. | Lab 4 uses !pip install shap lime (same idea). | L |
| o002 | warnings.filterwarnings(... IProgress \| sklearn.utils.parallel.delayed ...) | c2 | comment | no | Hides two harmless warnings (a notebook progress-bar message and an sklearn parallel-call message) so the output stays clean. | Cosmetic. | None. | L |
| o003 | glob, pathlib.Path, Kaggle path detection | c2 | tool | no | Looks for results.csv under /kaggle/input; if found uses Kaggle paths, otherwise data/raw. Lets one notebook run in both places. | Brief requires run-all on Kaggle and locally. | Hard-coded path. | L |
| o004 | sns.set_theme, pd.set_option, plt.savefig(dpi, bbox_inches), save() helper | c2 | tool | yes | Plot style, table width, and writing every figure to a PNG. | The brief wants every plot in both notebook and report. | Lab 1 plots with plt/sns only. | L |

## 2–4. Loading and EDA before cleaning (c3–c25)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o005 | Chain of merge calls with suffixes= (multi-hop join) | c6, c30, c35, c45, c51, c55, c62 | method | yes | Joins results to races, circuits, drivers, constructors and status through their ID columns. suffixes= renames columns that share a name. | The brief says one table alone is not enough; every answer needs several tables. | No lab uses merge or groupby. | H |
| o006 | DataFrame.query with a string condition and ~ (not) | c6, c51, c55, c60 | method | no | Filters rows with a text expression such as "year == 2024 and round == 1" or "~is_nonstarter". | Shorter than boolean masks. | Boolean masks (df[df.x > 1]) are standard pandas. | L |
| o007 | Counting \N sentinels next to isnull() | c16 | method | yes | Counts the literal text \N in every column and compares with isnull(), showing isnull() finds none. | Brief requires sentinel handling and EDA that is not manual inspection. | Lab 1 uses isnull().sum() only. | M |
| o008 | Share missing per season (groupby + heatmap) | c20 | method | yes | Shows when each column starts to be filled (qualifying from 1994, Q2/Q3 from 2003). | Evidence for dropping qualifying times as features. | Lab 1 heatmap on pivot_table is in scope; groupby/add_prefix/reindex are not. | M |
| o009 | # repr shows trailing spaces and str.contains with a regex | c22 | comment | no | repr() prints quotes around text so an invisible trailing space ("Argentinian ") becomes visible. | Spelling clean-up evidence. | None. | L |
| o010 | grid = 0 is overloaded (non-starter or pit-lane start) | c23, c25 | concept | no | In this dataset grid 0 means either 'never started' (0 laps) or 'started from the pit lane' (laps > 0). | Drives how non-starters are removed. | None. | H |
| o011 | Field size history and the pre-qualifying era | c23, c24 | concept | no | Before the 1990s many more cars entered than could start, so entries per race were far larger than starters. | Explains why non-starters must be removed. | None. | M |

## 5. Keys and duplicates (c26–c31)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o012 | Primary-key and foreign-key (orphan) checks | c27, c28 | method | yes | Checks each key is unique and non-empty, and every foreign key value exists in its parent table (no orphans). Uses duplicated(), notna(), isin(). | Required by the brief. | None. | H |
| o013 | assert as a gate that stops the notebook | c27, c28, c33, c34, c35, c45, c51, c60, c63, c78, c108 | method | no | If a check fails the run stops, so a silent data problem cannot pass. | Proves the grain and key rules hold. | None. | M |
| o014 | Surrogate key vs composite key, entity grain | c29, c31 | concept | no | resultId is a technical key; for prediction the real identity of a row is (raceId, driverId). | Explains why we deduplicate even though resultId is unique. | None. | M |
| o015 | Shared drives and the duplicate diagnostics | c30, c31 | domain | yes | Early F1 allowed a driver to take over a teammate's car, so the same driver has two result rows. duplicated(keep=False) and groupby().ngroups count them: 85 pairs, 176 rows, all 1950–1978. | Required by the brief. | drop_duplicates is in Lab 2. | H |

## 6. Cleaning (c32–c38)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o016 | Text-to-number conversion with 'nothing lost' rule | c33 | method | yes | For each text column: strip spaces, try pd.to_numeric(errors="coerce"), and convert only if no non-missing value is lost. select_dtypes(exclude="number") picks the text columns. | Brief requires semantic typing without silent loss. | Lab 1 casts types by hand. | M |
| o017 | pd.to_datetime | c34 | tool | yes | Turns date strings into real dates. | Required by the brief. | None. | M |
| o018 | to_seconds: parse '1:27.452' into seconds | c34 | method | yes | Splits on ':' and multiplies each part by 60 per level, so 1:27.452 becomes 87.452. | Required by the brief (lap-time strings to seconds). | None. | M |
| o019 | Cross-check against the milliseconds column (np.allclose) | c34 | method | no | Dataset stores lap and pit times in ms as well, so the parsed seconds can be asserted equal to ms / 1000. | Proof the parser is right. | None. | L |
| o020 | home_country: nationality-to-country mapping | c35 | method | no | A hand-written dictionary turns 'British' into 'UK' so it can be compared with the circuit's country. | Defines a home race for Q2. | None. | M |
| o021 | Duplicate rule: keep the lowest real grid slot | c35 | method | yes | Sort so rows with a real grid slot come first (grid 0 becomes NaN and goes last, na_position="last"), then drop_duplicates on (raceId, driverId). | The brief asks us to choose and justify a rule. | drop_duplicates in Lab 2 (no ordering rule). | H |
| o022 | Non-starter flag, pit-lane flag, grid_eff | c35 | method | no | Non-starter = grid 0 and 0 laps; pit-lane start = grid 0 and laps > 0; grid_eff gives pit-lane starters the slot behind the last car (groupby().transform("max") + 1). | Needed so grid is a usable number for every starter. | None. | H |
| o023 | Indianapolis 500 exclusion (1950–1960) | c35 | domain | no | Those years' Indy 500 counted as a championship race with US oval entrants; 381 rows are flagged and removed from modeling and Q2. | Avoids rows from a different kind of race. | None. | M |
| o024 | Status taxonomy with a 'mechanical' fallback | c36, c54–c58 | domain | yes | Explicit sets for finished, accident, driver, regulatory and other; every other status (77 of 139) is mechanical. Uses a generator with next(). | Brief defines mechanical as part failure, not crash, driver or rule decision. | None. | H |
| o091 | Status-based non-starters (NEVER_RACED) and grid_eff from the last real starter | c35 | method | yes | Entries with status Withdrew, Did not qualify, Did not prequalify or 107% Rule and 0 laps are non-starters even if the data shows a grid slot (79 rows); pit-lane starters get the slot behind the last car that really started. | Brief: one row per driver who enters and starts a race; the notebook text already said withdrawn drivers are non-starters. | None. | H |

## 7. Target and cutoff (c39–c41)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o025 | Sprint races | c39 | domain | no | Short Saturday races since 2021 (18 of 1,125 races). The notebook says sprint results are excluded. | Explains why they are not a feature. Text currently has wrong numbers (see findings N2). | None. | M |
| o026 | Podium base rate equals 3 / starters | c40, c41 | concept | no | Exactly three podium places exist per race, so the chance for a random starter is 3 divided by the field size. | Explains the rate across eras and the 13–15% base rate. | None. | M |
| o027 | Class imbalance, majority-class baseline, prevalence | c41 | concept | no | About 1 in 7 starters is on the podium; a model that always says 'no' gets about 85–87% accuracy. Lab 2 only says accuracy misleads on imbalanced data. | Justifies PR-AUC as primary metric. | Partly Lab 2. | M |
| o028 | Why PR-AUC is primary (and its prevalence dependence) | c41 | concept | yes | PR-AUC ranks true podium finishers above the rest and is more informative than ROC-AUC when positives are rare. Its no-skill level equals the positive rate. | The brief asks for a justified primary/secondary choice. | Labs stop at accuracy, precision, recall, F1. | H |

## 8. Data-engineering questions (c42–c55)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o029 | Wilson 95% confidence interval | c43, c45, c48, c52, c55 | method | no | A range of plausible true rates for k out of n that behaves for small n and for rates near 0 or 1. | Honest uncertainty on small samples (circuits with few starts). | Show rate and counts only. | H |
| o030 | Front row and era split at 2014 with np.where | c44, c45 | method | yes | Front row = grid slots 1 and 2; 'era' is pre-2014 vs 2014+. | Brief asks for the 2014 hybrid-era comparison. | None. | M |
| o031 | Minimum-count rule K_MIN = 10 and sensitivity table | c48 | method | no | Only circuits with at least 10 front-row starts in both eras are ranked; the notebook prints how many circuits survive for 5, 10, 15, 20. | A circuit with 2 starts can only show 0, 50 or 100%. | None. | M |
| o032 | Spearman rank correlation between era rankings (−0.03) | c48 | method | no | Correlation of ranks: 1 means the same order, 0 means unrelated order. | Shows the circuit ranking is not stable across eras. | Lab 1 correlation heatmap (default corr). | H |
| o033 | Q1 used grid, not qualifying position | c44, c49 | concept | yes | Grid (after penalties) vs qualifying position (before penalties). The instructor confirmed grid only. | Brief says to distinguish them. | None. | M |
| o034 | Home race by nationality; Q2 stratified comparison | c50, c51 | method | yes | Podium rates compared within starting groups (pole, 2–3, 4–6, 7–10, 11+, made with pd.cut), then the home-minus-away gaps averaged weighted by home starts. | Brief asks to control for starting position. | pd.cut and groupby-free comparison; stratifying is not taught. | H |
| o035 | Logistic regression as inference: odds ratio exp(coef), C = 1e5 | c52 | method | no | A regression with home, start group and era dummies (get_dummies(drop_first=True)); exp of the home coefficient is the odds ratio (0.915). C = 1e5 means practically no regularisation. | A second, adjusted estimate of the home effect. | Lab 2 uses logistic regression only for prediction. | H |
| o036 | Overlapping 95% intervals read as 'no detectable effect' | c53, c58 | concept | no | If two intervals overlap we cannot tell the difference from zero. A rough rule, not a significance test. | Simple conclusion for Q2 and Q3. | No hypothesis test in the labs. | M |
| o037 | Q3 rate per team and period, team naming | c55, c56, c58 | method | yes | Mechanical starts / starts per constructor name for 2014–2021 and 2022–2024; teams keep the name in the data (Toro Rosso, AlphaTauri, RB are separate). | Brief question. | None. | M |
| o092 | Bootstrap range of the Spearman correlation across circuits | c48 | method | no | Redraws the 16 circuits 1,000 times and recomputes the rank correlation to show how uncertain it is (about -0.6 to +0.6). | With so few circuits a single correlation is weak evidence. | None. | M |
| o093 | Strict vs broad 'mechanical' definition (sensitivity check) | c57 | method | yes | Repeats Q3 leaving out statuses with an unclear cause (Retired, punctures, wings, wheels); the fall in the rate holds. | Brief: mechanical excludes crashes, driver and rule decisions; unclear cases are our choice. | None. | M |

## 9. Modeling table and candidate features (c56–c60)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o038 | Temporal (by-season) split with pd.cut on the year | c59, c60 | method | yes | train 1950–2017, validation 2018–2021, test 2022–2024. | The brief wants a justified train/validation/test split. | train_test_split(random_state=…) in Labs 2–3. | H |
| o039 | field_size and grid_rel via groupby().transform("size") | c60 | method | no | Number of starters in the race and grid divided by it ('10th of 20 is not 10th of 34'). | Makes grid comparable across eras. | None. | M |
| o040 | Previous round: groupby('year')['raceId'].shift(1), nullable Int64 | c62 | method | yes | Orders races by (year, round) and finds the earlier race of the same season; Int64 lets the first race be missing. | Standings must come from before the race. | None. | H |
| o041 | Standings joined on prev_raceId, merge(validate="many_to_one") | c62 | method | yes | Driver and constructor standings after the previous round are joined to each row; validate raises an error if a join would multiply rows. | Leakage-safe standings. | None. | H |
| o042 | Points as a share of the leader's points | c62 | method | no | drv_pts_share = points / max points that round (groupby.transform('max'), .where for zero). | Comparable across seasons with different scoring. | None. | M |
| o043 | Form: shift(1).rolling(5, min_periods=1).mean() inside groupby.transform(lambda) | c62 | method | yes | Podium share over the previous five races of the driver (team: share of its cars). shift(1) removes the current race. | The brief's leakage-safe form feature. | None. | H |
| o044 | Leakage check by slow recomputation of 150 random rows | c63 | method | yes | For random rows, recompute form from earlier races only and assert it matches with np.isclose(equal_nan=True). | Evidence that no future race leaks. | None. | H |
| o045 | Feature-group dictionary, sum(list, []), agg(lambda s: s.isna().mean()) | c63 | tool | no | Plain Python/pandas bookkeeping for groups and missing shares per split. | None. | None. | L |

## 10–11. Preprocessing and feature selection (c61–c73)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o046 | SimpleImputer(strategy="median") and make_pipeline | c64, c65 | tool | no | Fits the median on training rows and chains it with StandardScaler so both are fitted on train only. | Same result as fillna(median) in Lab 1. | fillna(median) in Lab 1; StandardScaler in Lab 3. | M |
| o047 | Univariate 'fifths' bins with pd.qcut(rank(method="first")) | c67 | method | no | Cuts a column into five equal-sized groups after ranking (ties broken by order) and plots the podium rate of each ('binned conversion rate analysis'). | Exploratory evidence for each feature. | pd.cut in Lab 1 (equal-width, not equal-size). | M |
| o048 | Spearman correlation with the target: corrwith(method="spearman") | c69 | method | no | Rank correlation of every candidate with podium, and a heatmap between candidates. | Relations are monotonic but not linear. | Lab 1 correlation heatmap (default method). | H |
| o049 | Forward group selection with MIN_GAIN = 0.005 | c72, c73, c76 | method | yes | Add one feature group at a time; keep it only if validation PR-AUC rises at least 0.005 for both LR and RF. Result can depend on the order. | The brief requires trials kept in the notebook as evidence. | None. | H |
| o050 | average_precision_score (PR-AUC) | c72, c80, c93, c96, c100, c105 | metric | yes | Area under the precision-recall curve. | Primary metric required by the brief. | Labs stop at accuracy, precision, recall, F1. | H |
| o051 | Train medians read from SimpleImputer.statistics_ | c78 | tool | no | Prints the fill values (e.g. a typical mid-table position) used for missing standings. | Shows the effect of imputation. | None. | L |
| o094 | All 120 group orders with itertools.permutations and a score cache | c74 | method | yes | Re-runs the forward selection for every order of the five groups (only 32 distinct feature sets exist, so scores are cached) to see if the selected set depends on the order: it does (8 sets), but driver standings plus grid information is always in. | Our selection rule is order-dependent; brief wants evidence for feature choices. | None. | H |
| o095 | Bootstrap range of a PR-AUC difference (noise of the 0.005 rule) | c75 | method | no | Redraws whole validation races 500 times to give a 95% range for the gain from adding a group (driver standings +0.045, range 0.008 to 0.080). | Shows which gains are bigger than noise. | None. | M |

## 12–14. Models and evaluation (c74–c84)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o052 | Hyperparameter grid with nested loops over C, depth, leaf size, class_weight | c80 | method | no | Fits every combination and keeps the best validation PR-AUC; n_estimators fixed at 300. | Choosing settings without the test set. | Lab 2 mentions tuning in text only. | H |
| o053 | class_weight="balanced" | c80 | parameter | no | Gives the rare podium class more weight. It never helped here. | Part of the tuning grid. | None. | M |
| o054 | precision_recall_curve and the F1-optimal threshold | c80, c85 | method | yes | Tries every cut-off on validation and keeps the one with the highest F1 (1e-12 avoids division by zero). | Brief: thresholds chosen on train/validation only. | Fixed 0.5 threshold in Lab 2. | H |
| o055 | ROC-AUC (roc_auc_score) | c80 | metric | yes | Area under the ROC curve: chance a random podium finisher is scored above a random non-podium one. | Secondary metric required by the brief. | Labs mention neither ROC nor PR curves. | H |
| o056 | metrics(): PR-AUC and ROC-AUC from probabilities, F1/precision/recall/accuracy at a threshold, zero_division=0 | c80 | tool | yes | One function returning all six metrics for any model and split. | Brief asks all metrics for every model. | accuracy, precision, recall, F1 are in Lab 2. | M |
| o057 | rf_best.set_params(n_jobs=1)  # same summation order… | c80 | comment | no | Predicting with one thread makes floating-point sums run in the same order, so a score sitting exactly on the threshold cannot flip. | Reproducibility of the threshold decision. | Delete the comment or simplify. | M |
| o058 | Threshold dictionary THR applied unchanged to test | c85, c90 | method | yes | Each model has one validation-chosen cut-off that is reused on test. | No test information in decisions. | None. | H |
| o059 | Keras: sigmoid output + binary_crossentropy | c82, c83 | concept | no | One output neuron giving a probability; loss for yes/no problems. Described in Lab 3 text but the guide notebook uses softmax with categorical cross-entropy. | Binary podium target. | Partly in Lab 3. | M |
| o060 | EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True) | c83 | parameter | no | Stops when validation loss has not improved for 15 epochs and goes back to the best epoch. The lab monitors val_accuracy. | Prevents overfitting. | val_accuracy in Lab 3. | H |
| o061 | validation_data=, batch_size=256, epochs=200, verbose=0 | c83 | parameter | no | A fixed validation set (not a random 30% split), bigger batches than the lab's 32, and an epoch cap. | Season-based validation. | validation_split and batch_size 32 in Lab 3. | M |
| o062 | Adam(1e-3), keras.utils.set_random_seed, model.predict(...).ravel(), to_numpy('float32') | c83 | tool | no | Explicit learning rate, one-call seeding, probabilities back as a flat array. | Reproducible network. | optimizer='adam'; np.random.seed + tf.random.set_seed in Lab 3. | L |
| o063 | Network size chosen by validation PR-AUC; 'epochs run' logged | c83 | method | no | Three sizes are fitted and the best on validation kept; the number of epochs shows whether early stopping fired. | Model selection. | None. | M |
| o064 | Multi-index results table, DataFrame.xs, unstack | c85, c90, c115 | tool | no | Tables with two-level row labels (model, split) and selection of one level. | Display only. | None. | L |
| o065 | Reading 'train score lower than validation' | c88 | concept | no | We say old seasons (bigger, less predictable fields) make the training set harder, so a lower train score is not underfitting. | Will be asked at the evaluation. | Lab 3 teaches overfitting only. | H |
| o066 | 'Tie within noise' with only 243 validation podiums | c88, c91 | concept | no | Differences of about 0.01 PR-AUC are within sampling noise, so models are called equal. | Honest ranking. | None. | H |
| o067 | Choosing the final model on validation, not test | c91 | concept | yes | The final model is picked on validation PR-AUC even though the forest scores higher on test. | Brief: no test set for decisions. | None. | H |
| o096 | 'Always no podium' reference row and six-metric plots | c85, c90 | concept | no | A constant model has PR-AUC = podium rate, ROC-AUC 0.5, F1 0 and accuracy 1 - podium rate (about 0.85); it is the yardstick for accuracy. | Shows why accuracy is not used to choose anything. | Lab 2 only states that accuracy misleads on imbalanced data. | M |
| o097 | Training PR-AUC by decade (groupby + apply returning a Series) | c87 | method | no | Scores the forest on the training seasons decade by decade to show that old seasons are less predictable. | Backs the claim 'train score is lower than validation'. | None. | M |
| o098 | Confusion counts with confusion_matrix(...).ravel() | c90 | tool | no | Unpacks the 2x2 matrix into TN, FP, FN, TP to print podiums found, missed and false alarms. | Numbers quoted in the report. | confusion_matrix and its heatmap are in Lab 2. | L |

## 15. Experiments (c85–c94)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o068 | Feature-group ablation with sklearn.base.clone | c93 | method | yes | Remove one group, retrain a fresh copy (clone) with the same settings and measure the drop in PR-AUC. | Required by the brief. | None. | H |
| o069 | Preprocessing-order experiments (leaky fit on all seasons; scale-then-impute) | c96, c98 | method | yes | Two controlled experiments on the order of imputation and scaling. | Required by the brief. | Lab 3 teaches fit on train, transform test. | H |
| o070 | Paired bootstrap by race, percentile range, two-sided p-value | c99, c100 | method | no | Redraw whole validation races 2,000 times, rescore both versions on the same rows, take the middle 95% of the differences; p-value = how often the difference lands on the other side of zero. | Shows the experiment differences are not more than chance. | No resampling or hypothesis test in the labs. | H |
| o099 | Ablation that retrains the network for a fixed number of epochs (refit_scores) | c93 | method | yes | The network is retrained without validation data for as many epochs as the chosen network used, so the ablation does not look at validation while training. | Brief: report results for each feature group. | None. | M |

## 16. Explainability and inference (c95–c108)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o071 | Permutation importance written as a custom loop | c105 | method | yes | Shuffle one column (default_rng.permutation), rescore, record the drop in validation PR-AUC, 10 repeats. | Required by the brief; sklearn's version needs an sklearn model. | Not in the labs. | H |
| o072 | shap.TreeExplainer called on data to get a shap.Explanation; class slice [..., 1]; additivity assert | c108 | tool | no | The Explanation object holds values and base_values; base + sum of contributions must equal the forest's probability. | Proof the explanation is exact. | shap_values() and expected_value in Lab 4. | M |
| o073 | summary_plot(plot_type="bar") and dependence_plot(interaction_index=None) | c108 | method | yes | Mean absolute SHAP per feature, and the effect of one feature on its own. | Global SHAP plots. | Beeswarm summary_plot in Lab 4. | M |
| o074 | shap.plots.waterfall on a hand-built shap.Explanation | c111 | tool | yes | Waterfall for one row, with values back in original units. | Local SHAP. | Waterfall is named in Lab 4. | M |
| o075 | LIME extras: as_map, exp.score (R² of the local model), num_samples, random_state, as_pyplot_figure | c112 | tool | no | Weights of the local linear model, how well it fits the forest, how many perturbed copies were used. | Local explanation. | explain_instance + show_in_notebook in Lab 4. | M |
| o076 | Hit/miss row selection rule (argmax / argmin on masked probabilities) | c109, c111 | method | no | Hit = true podium finisher the forest was most sure about; miss = the one it was least sure about. | Rows chosen by rule, not by hand. | None. | M |
| o077 | to_raw / to_scaled with inverse_transform | c103 | method | no | Undo imputation and scaling for display, redo them before calling the model. | Plots in real units (grid 1 = pole). | None. | M |
| o078 | SHAP vs permutation importance; SHAP vs LIME | c106, c109, c113 | concept | yes | Permutation measures lost ranking quality; SHAP measures how far an input moves the probability; SHAP is exact, LIME is a local linear approximation. | Interpretation required by the brief. | Lab 4 describes SHAP and LIME separately. | H |
| o079 | Explaining the random forest while the final model is the network | c102, c103, c115 | concept | no | The brief does not name the model to explain. | See office-hours question 2. | None. | H |
| o080 | Explanations are not causal | c102, c116 | concept | yes | Importance means the model relies on a feature, not that the feature causes podiums. | Required by the brief. | Lab 4 touches fairness and trust only. | H |
| o081 | Outside knowledge in the text (2019 German GP was wet; Hamilton's 2018 US GP pole) | c113 | domain | no | The miss row is explained by rain, which is not in the data. | The notebook says it is outside knowledge. | Be ready to say where it comes from. | M |
| o082 | predict_podium: dict to DataFrame with NaN, median fill, FINAL_MODEL by idxmax, its validation threshold | c115 | method | yes | Complete and partial record demos. | Required by the brief. | None. | M |

## 18. Limitations (c109)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o083 | Qualifying coverage drift, era sensitivity, dominant teams, race-day events | c116 | concept | yes | Known limits of the data and the split. | Required by the brief. | None. | M |

## Plotting and Python plumbing (L)

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o084 | Plot styling arguments: xerr/yerr/capsize, gridspec_kw, bbox_to_anchor, PercentFormatter, axhline/axvline, annot_kws, vmin/vmax, suptitle, tick_params, missingno sparkline=False | many | tool | no | Layout and labelling details. | Readability. | Lab 1 covers the basic plots. | L |
| o085 | Python constructs: f-strings, dict/set comprehensions, lambda, zip, enumerate, closures, default arguments | many | tool | no | General Python used throughout. | None. | None. | L |
| o086 | groupby and agg | c20, c30, c35, c38, c40, c45, c48, c51, c55 | method | yes | Group rows and count or average them: every rate, field size, Q1–Q3. | Brief requires multi-hop joins and rates. | pivot_table in Lab 1. | H |
| o087 | Index helpers: loc, set_index, reset_index, reindex, rename, sort_values, sort_index, dropna, copy, round, isin, unstack, to_dict, to_numpy, tolist | many | tool | no | Day-to-day pandas. | None. | None of these are shown in the labs. | L |

## Report and README only

| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |
|---|---|---|---|---|---|---|---|---|
| o088 | Bibliography: Wilson 1927, Breiman 2001, Efron & Tibshirani 1993, Lundberg & Lee 2017, Ribeiro et al. 2016, scikit-learn, the dataset | report references.bib | concept | no | Papers behind Wilson intervals, random forests, the bootstrap, SHAP and LIME. | Report cites them. | Know one sentence on each. | M |
| o089 | 'Probabilities are coarse' for a random forest; 'regularisation strength C' | report, Models | concept | no | A forest of depth 6 can only output a limited set of probability values; C is the inverse of regularisation strength. | Report wording. | Lab 4 explains forests; C is not in the labs. | M |
| o090 | README: 'Anything beyond the labs is tagged' | README.md line 90 | comment | no | The README promises tags for tools beyond the labs, but the notebook no longer has them. | Fix or remove the sentence. | N/A | L |

## Appendix A: what the labs do teach (the in-scope reference)

| Lab | In scope |
|---|---|
| 1: Wrangling and visualisation | `read_csv`, `head`, `info`, `describe`, `isnull().sum()`, `value_counts`; numerical vs categorical; impute vs remove; `drop(columns=)`; `fillna` with median / mode; label vs one-hot encoding, `map`, `get_dummies(drop_first=True)`; `missingno.matrix`; plot types (comparison, distribution, composition, relationship); `plt.hist`, `plt.scatter`, bubble chart, `sns.heatmap` on `pivot_table` and on correlations, `pd.cut`, `sns.barplot`, `sns.boxplot` |
| 2: ML foundations | supervised / unsupervised; classification vs regression; `train_test_split`; `OrdinalEncoder`; `drop_duplicates`; `LinearRegression`, `LogisticRegression(max_iter)`; sigmoid, probability and threshold; accuracy, precision, recall, F1, `classification_report`, `confusion_matrix`; accuracy misleads on imbalanced data; MSE/RMSE/MAE/R²; leakage by dropping a column; baseline vs improved model |
| 3: Neural networks | scaling, `StandardScaler` fit on train; `stratify=y`; `keras.Sequential`, `Input`, `Dense(relu)`; sigmoid, softmax; depth; forward pass, loss, backpropagation; binary and categorical cross-entropy; SGD vs Adam, learning rate; epochs, batches, `validation_split`; overfitting; `EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)`; loss/accuracy curves; `np.random.seed`, `tf.random.set_seed` |
| 4: XAI | intrinsic vs post-hoc, model-specific vs agnostic, global vs local; `DecisionTreeClassifier`, `plot_tree`; `RandomForestClassifier` (bagging); `shap.TreeExplainer`, `shap_values`, `summary_plot`, `force_plot`, waterfall, `expected_value`; LIME, surrogate model, `LimeTabularExplainer(discretize_continuous=True)`, `explain_instance`, `show_in_notebook`; fairness, bias, trust; `!pip install shap lime` |

## Appendix B: code names not found in the lab material

Automatic list from `review/sweep_names.json` (imports, methods, attributes, keyword arguments, functions). A name that appears in the lab text can still be used differently; this list only says which names the labs never mention. Names defined in the notebook itself (for example `wilson`, `prepare`) are not included in the count of library names.

215 library names. By kind:

- **import** (10): `glob`, `itertools`, `pathlib.Path`, `sklearn.base.clone`, `sklearn.impute.SimpleImputer`, `sklearn.metrics.average_precision_score`, `sklearn.metrics.precision_recall_curve`, `sklearn.metrics.roc_auc_score`, `sklearn.pipeline.make_pipeline`, `warnings`
- **func** (26): `Path`, `SimpleImputer`, `_scores`, `always_no`, `average_precision_score`, `bool`, `clone`, `confusion_counts`, `dict`, `display`, `enumerate`, `float`, `frozenset`, `ladder_for`, `list`, `make_pipeline`, `min`, `precision_recall_curve`, `refit_scores`, `roc_auc_score`, `round`, `sorted`, `str`, `top3`, `tuple`, `zip`
- **attr** (108): `Explanation`, `PercentFormatter`, `Rectangle`, `Series`, `T`, `abs`, `add_prefix`, `agg`, `allclose`, `api`, `append`, `arange`, `argmax`, `argmin`, `as_map`, `as_pyplot_figure`, `asarray`, `axhline`, `axvline`, `barh`, `base_values`, `clip`, `close`, `coef_`, `concat`, `concatenate`, `copy`, `corr`, `corrwith`, `cwd`, `default_rng`, `dependence_plot`, `div`, `dropna`, `dtype`, `dtypes`, `duplicated`, `eq`, `filterwarnings`, `gcf`, `glob`, `groupby`, `hlines`, `idxmax`, `indices`, `integers`, `inverse_transform`, `is_numeric_dtype`, `isclose`, `isin`, `isna`, `join`, `loc`, `max`, `merge`, `min`, `mkdir`, `nanpercentile`, `ndim`, `ngroups`, `notna`, `nunique`, `parent`, `percentile`, `permutation`, `permutations`, `qcut`, `query`, `ravel`, `reindex`, `rename`, `reset_index`, `rolling`, `round`, `savefig`, `select_dtypes`, `set_index`, `set_major_formatter`, `set_option`, `set_params`, `set_random_seed`, `set_theme`, `set_xticks`, `set_yticks`, `setdefault`, `shift`, `sort_index`, `sort_values`, `startswith`, `statistics_`, `str`, `strip`, `suptitle`, `tail`, `tick_params`, `ticker`, `to_datetime`, `to_dict`, `to_frame`, `to_numeric`, `to_numpy`, `to_string`, `tolist`, `unstack`, `upper`, `utils`, `xaxis`, `xs`
- **kwarg** (71): `C`, `alpha`, `annot_kws`, `ascending`, `atol`, `ax`, `base_values`, `bbox_inches`, `bbox_to_anchor`, `capsize`, `cbar`, `cbar_kws`, `class_weight`, `decade`, `dpi`, `edgecolors`, `equal_nan`, `era`, `exclude`, `exist_ok`, `facecolors`, `fill_value`, `fit_on`, `gridspec_kw`, `ha`, `interaction_index`, `key`, `labelsize`, `loc`, `ls`, `lw`, `max_display`, `message`, `min_periods`, `min_samples_leaf`, `n_jobs`, `na_position`, `ncol`, `normalize`, `num_samples`, `observed`, `p`, `parents`, `plot_type`, `podium`, `podium_rate`, `podiums`, `rect`, `recursive`, `rot`, `rotation`, `sharex`, `sharey`, `slot`, `sparkline`, `stacked`, `style`, `suffixes`, `upper`, `validate`, `validation_data`, `vmax`, `vmin`, `width`, `xerr`, `xticklabels`, `year`, `yerr`, `ylim`, `yticklabels`, `zero_division`
