"""Refreshes the markdown that quotes numbers from outputs, using the numbers of the final run (2026-10-14).
Run after copying the executed notebook over notebooks/f1_podium_prediction.ipynb."""
import nbformat
P = "notebooks/f1_podium_prediction.ipynb"
nb = nbformat.read(P, as_version=4); C = nb.cells
def find(sub, kind="markdown"):
    hits = [i for i, c in enumerate(C) if sub in c.source and c.cell_type == kind]
    assert len(hits) == 1, (sub, hits); return hits[0]
def sub(i, old, new):
    assert C[i].source.count(old) == 1, (i, old[:70], C[i].source.count(old)); C[i].source = C[i].source.replace(old, new)
def setmd(i, text): C[i].source = text.strip("\n")

# podium paragraph
# Q1
i = find("**Answer Q1:**")
setmd(i, """**Answer Q1:** A front-row start converts much more often in the hybrid era: **55.9% before 2014 → 76.0% from 2014** (95% intervals 54–58% and 72–80%; they do not overlap).
The best circuits since 2014 are Suzuka, Yas Marina and Monaco (heatmap). There is **no detectable agreement** between the circuit rankings of the two eras: the rank correlation is −0.03 over 16 circuits, and with so few circuits it could plausibly lie anywhere between about −0.6 and +0.6 (redrawing the circuits). So the data do not show that a track that rewards a front-row start in one era does so in the other.""")
# Q2
i = find("**Answer Q2:**")
sub(i, "0.70 percentage points", "0.69 percentage points"); sub(i, "The odds ratio is 0.915", "The odds ratio is 0.921")
# Q3
i = find("**Answer Q3:**")
sub(i, "(95% intervals 10.1–12.3% and 5.7–8.4%)", "(95% intervals 10.2–12.3% and 5.7–8.4%)")
C[i].source += "\n**Sensitivity to the definition:** if statuses with an unclear cause (plain \"Retired\", punctures, wing and wheel problems; about 20% of the starts counted as mechanical) are left out, the rates are 9.0% and 5.5%, so the fall holds. The first and second least reliable teams stay the same in both periods; the third in 2022–2024 changes from RB (only 48 starts) to AlphaTauri."
# feature selection reading
i = find("**Reading the table and the plot**")
sub(i, "goes from 0.668 to 0.714 (LR) and 0.726 (RF)", "goes from 0.668 to 0.714 (LR) and 0.725 (RF)")
sub(i, "**Field size** adds only +0.001 / +0.003", "**Field size** adds only +0.003 / +0.003")
sub(i, "(LR 0.702, RF 0.720) than the selected set (0.714, 0.726)", "(LR 0.702, RF 0.722) than the selected set (0.714, 0.725)")
j = C[i].source.index("*Limit of this method:*")
C[i].source = C[i].source[:j] + """*Limit of this method and how much it matters:* the groups are added in a fixed order and a dropped group is not tried again, so the result can depend on the order. The two cells above check this. **All 120 possible orders** lead to 8 different selected sets, so the exact list is not fixed. But every one of the 8 sets contains the driver standings and the grid information (`grid`, or `grid_rel`, which carries it), so the part that matters is stable: **driver standings on top of the grid**. The other groups come and go because they repeat each other. The bootstrap table shows why the 0.005 bar cannot separate small gains from noise: redrawing validation races, the gain from driver standings is +0.045 (95% range 0.008 to 0.080) and the gain from field size is +0.002 (0.000 to 0.008). The "all candidates" row and the ablation in 15.1 are further checks."""
# NN section
i = find("## 13. Shallow neural network")
# model comparison reading
i = find("**Reading the comparison**")
sub(i, "0.714 (LR), 0.726 (RF) and 0.727 (NN), against 0.668. The forest and the network are tied; validation has only 243 podiums, so a 0.001 gap is noise.",
       "0.714 (LR), 0.725 (RF) and 0.728 (NN), against 0.668. The forest and the network are tied; validation has only 243 podiums, so a 0.003 gap is noise.")
sub(i, "The three network sizes score within 0.003 of each other, so the smallest one (16 units) is kept: with 4 inputs, a bigger network gains nothing. It trained for the full 200 epochs (the validation loss was still creeping down).",
       "The three network sizes score within 0.004 of each other, so the smallest one (16 units) is kept: with 4 inputs, a bigger network gains nothing. It stopped at epoch 266 (best epoch 251), so the early-stopping rule ended training, not the epoch cap.")
sub(i, "This is not underfitting: the training seasons include the 1950s–80s, with bigger fields and far less predictable races. There is also no sign of overfitting (train is not far above validation).",
       "This is not underfitting: the training seasons include the 1950s–80s, with bigger fields and far less predictable races. The table below shows the forest's training PR-AUC by decade: 0.46–0.50 in 1950–1989, 0.56 in the 1990s, 0.68 in the 2000s and 0.73 in 2010–2017, which is as high as on validation (0.725). There is also no sign of overfitting (train is not far above validation).")
# metrics discussion
i = find("<<METRICS_DISCUSSION>>")
setmd(i, """
**The six metrics for this task**
- **Precision** = of the drivers the model calls podium finishers, how many really finish on the podium. **Recall** = of the real podium finishers, how many the model finds. **F1** is the harmonic mean of the two, so it is low if either is low. These three depend on the threshold, which was chosen on validation to maximise F1.
- **PR-AUC** and **ROC-AUC** use the probabilities and no threshold. PR-AUC is the primary metric (section 7): it looks only at the podium class, and its no-skill level is the podium rate (0.134 on train, 0.150 on validation).
- **Accuracy is misleading here.** A model that always says "no podium" scores 0.866 on train and 0.850 on validation, with zero precision and recall (the dashed line in the plot). Our models reach 0.91, only 0.06 above that reference, so accuracy is reported but not used to choose anything.
- **Threshold dependence:** the cut-offs chosen on validation are 0.34–0.44. Precision and recall are close for the logistic regression and the forest (0.70–0.73 each); the network trades precision (0.67) for recall (0.78).
- **Validation scores are slightly optimistic**, because validation chose the features, the settings, the stopping epoch, the final model and the thresholds. Test (next section) is the honest number.
""")
# test reading
i = find("**Reading the test results**")
setmd(i, """
**Reading the test results**
- Every model beats the baseline on test: random forest 0.721, logistic regression 0.708, neural network 0.704, baseline 0.647 (PR-AUC).
- PR-AUC on test is within 0.03 of validation for every model (largest drop: the network, 0.728 → 0.704). The threshold-based metrics move more: for the forest, F1 falls from 0.721 to 0.672 and recall from 0.728 to 0.637. Settings and thresholds were chosen on validation, and 2022–2024 is a new rules era, so a drop is expected.
- F1 on test is 0.67–0.68 for the three models. Accuracy is 0.89–0.91, against 0.849 for a model that always says "no podium" (reference row).
- **Final model: the neural network**, because it has the best *validation* PR-AUC (the primary metric). On test the forest scores higher, but the choice must not be made on test. The three models are within about 0.02 of each other on both sets, so the honest summary is that they perform about the same.
""")
# ablation reading
i = find("<<ABLATION_TEXT>>")
setmd(i, """
**Reading the ablation**
- **Selected model.** `grid` matters most: removing it costs 0.14 test PR-AUC for logistic regression and the network and 0.16 for the forest (0.08–0.10 on validation). The driver standings cost 0.06–0.07 on test (0.05–0.06 on validation). Both kept groups help all three models on validation and on test, so the selection in section 11 holds up after training.
- **All 11 candidates.** With every candidate in the model, removing any single group costs almost nothing: at most 0.02 on validation and 0.01 on test, and a few drops are slightly negative (the model is no worse, or better, without the group). The groups repeat each other: removing `grid` leaves `grid_rel` and `field_size`, which together contain the grid slot, and removing the driver standings leaves the team standings and recent form. So **no single group is essential once the others are present**. What the model needs is *some* grid information and *some* standings information, which agrees with the order check in section 11.2.
- **Dropped groups.** Team standings and recent form change test PR-AUC by at most 0.01 either way, which supports leaving them out: they add nothing the selected features lack.
- The test column is for reading only; nothing was chosen from it.
""")
# experiments reading (bootstrap)
i = find("**Reading the experiments**")
setmd(i, """
**Reading the experiments**
- **Neither difference is significant:** every bootstrap 95% range includes 0, and the p-values are far above 0.05 (experiment 1: 0.85 for LR and 0.17 for RF; experiment 2: 0.40 and 1.00).
- **Experiment 1:** the leaky version changes validation PR-AUC by at most 0.0006. Little leaks here because 88% of all rows are training rows, so the medians, means and standard deviations hardly move, and shifting or stretching a column barely changes logistic regression or a forest. Fitting on train only is still the correct order: it is the only one that is possible when predicting a real future race.
- **Experiment 2:** the order makes no visible difference (at most 0.0001). The filled value is the same point either way; only the scaler's mean and standard deviation differ slightly. Imputing first is kept, so the scaler sees exactly the complete data the model gets.
""")
# XAI
i = find("## 16. Explainability (XAI)"); sub(i, "(0.726 vs 0.727)", "(0.725 vs 0.728)")
i = find("**Reading the permutation importance**")
sub(i, "by 0.37 (RF), 0.41 (LR) and 0.36 (NN)", "by 0.37 (RF), 0.41 (LR) and 0.39 (NN)")
sub(i, "The forest leans on `drv_pos` (0.025), logistic regression on `drv_pts_share` (0.029), and the network on `drv_wins` (0.053)", "The forest leans on `drv_pos` (0.026), logistic regression on `drv_pts_share` (0.029), and the network on `drv_wins` (0.076)")
i = find("**Reading the SHAP plots**"); sub(i, "**Base value 0.133:**", "**Base value 0.134:**")
i = find("**Reading the local explanations**")
sub(i, "Starting from the average 0.133", "Starting from the average 0.134"); sub(i, "`drv_wins = 9` (+0.10)", "`drv_wins = 9` (+0.09)"); sub(i, "(R² 0.82)", "(R² 0.81)")
# the smallest Spearman correlation among the strong candidates is 0.385, which rounds to 0.39
i = find("Spearman correlations with podium are 0.38–0.45"); sub(i, "0.38–0.45", "0.39–0.45")
nbformat.write(nb, P); print("text refreshed")

