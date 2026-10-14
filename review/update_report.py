"""Re-syncs report/main.tex with the final notebook run (2026-10-14). Usage: python review/update_report.py"""
import re, json
P = "report/main.tex"; t = open(P).read()
nb = json.load(open("notebooks/f1_podium_prediction.ipynb"))["cells"]
def rep(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:80]); t = t.replace(old, new)
# --- confusion counts from the notebook output
conf = None
for c in nb:
    if c["cell_type"] == "code" and "confusion_counts" in "".join(c["source"]):
        for o in c["outputs"]:
            if "data" in o and "text/plain" in o["data"]: conf = "".join(o["data"]["text/plain"])
print(conf)
def cnt(model, col):
    """The printed table wraps over two blocks; read the 4 numbers of the row in each block."""
    idx = ["TP", "FN", "FP", "TN"].index(col); block = 0 if idx < 2 else 1
    blocks = conf.split("\n\n") if "\n\n" in conf else [conf]
    for line in blocks[block].splitlines():
        if line.strip().startswith(model):
            nums = [int(x) for x in line.split()[-2:]]
            return nums[idx % 2]
NN_TP, NN_FP, RF_TP, RF_FP = cnt("Neural network","TP"), cnt("Neural network","FP"), cnt("Random forest","TP"), cnt("Random forest","FP")
print("NN TP/FP", NN_TP, NN_FP, "RF TP/FP", RF_TP, RF_FP)

# --- abstract, conclusion
rep("a leakage-safe modeling table of 24{,}724 starts is built", "a leakage-safe modeling table of 24{,}645 starts is built")
rep("test PR-AUC 0.708, 0.718 and 0.699 against 0.647", "test PR-AUC 0.708, 0.721 and 0.704 against 0.647")
rep("a leakage-safe table of 24{,}724 starts was built", "a leakage-safe table of 24{,}645 starts was built")
rep("(55.5\\% $\\rightarrow$ 76.0\\%) with a completely reshuffled circuit ranking", "(55.9\\% $\\rightarrow$ 76.0\\%) with no detectable agreement between the circuit rankings of the two eras")
# --- cleaning
rep("6. Flags & 1{,}563 non-starters (grid 0 and 0 laps; excluded from modeling and Q3);", "6. Flags & 1{,}642 non-starters (1{,}563 with grid 0 and 0 laps, plus 79 entries that still show a grid slot but were withdrawn or failed to qualify: status ``Withdrew'', ``Did not qualify'', ``Did not prequalify'' or ``107\\% Rule'' with 0 laps; excluded from modeling and from Q1--Q3);")
rep("excluded from Q2 and modeling). \\\\", "excluded from Q1, Q2 and modeling). \\\\")
# --- metric choice
rep("a model that always predicts ``no podium'' is about 86\\% accurate and useless, so accuracy is reported for reference only.", "a model that always predicts ``no podium'' is about 85\\% accurate (0.866 on train, 0.850 on validation, 0.849 on test) and useless, so accuracy is reported for reference only and an ``always no podium'' row is shown next to the results.")
# --- Q1
rep("We only rank a circuit if it has hosted", "Non-starters and the Indianapolis~500 are excluded, as in the modeling table. We only rank a circuit if it has hosted")
rep("($\\approx$10\\%) than today", "($\\approx$12\\%) than today")
rep("\\textbf{55.5\\% before 2014} (1{,}002 of 1{,}805; 95\\% CI 53.2--57.8\\%)", "\\textbf{55.9\\% before 2014} (994 of 1{,}778; 95\\% CI 53.6--58.2\\%)")
rep("The ranking \\textbf{changed completely}: the Spearman rank correlation between the two eras is $-0.03$. Monaco, for example, moved from 15th of 16 before 2014 (49\\%) to 3rd (85\\%). Which track rewards a front-row start depends on the era, not on the track alone.",
    "The two eras' rankings show \\textbf{no detectable agreement}: the Spearman rank correlation is $-0.03$ over 16 circuits, and when the circuits are redrawn it could lie anywhere between about $-0.64$ and $+0.55$. Monaco, for example, moved from 15th of 16 before 2014 (49\\%) to 3rd (85\\%). The data do not show that a track that rewards a front-row start in one era does so in the other.")
# --- Q2
rep("24{,}724 starts, 1{,}944 of them at home (7.9\\%)", "24{,}645 starts, 1{,}934 of them at home (7.8\\%)")
for a, b in [("Pole & 64.2\\% & 58.3\\% & $-5.9$ pp & 84 & 1{,}040", "Pole & 64.3\\% & 58.3\\% & $-6.0$ pp & 84 & 1{,}039"),
             ("2--3 & 50.9\\% & 53.7\\% & $+2.9$ pp & 147 & 2{,}078", "2--3 & 50.9\\% & 53.7\\% & $+2.8$ pp & 147 & 2{,}075"),
             ("4--6 & 26.0\\% & 26.4\\% & $+0.4$ pp & 227 & 3{,}114", "4--6 & 26.0\\% & 26.5\\% & $+0.5$ pp & 226 & 3{,}111"),
             ("7--10 & 9.2\\% & 7.1\\% & $-2.1$ pp & 336 & 4{,}117", "7--10 & 9.2\\% & 7.2\\% & $-2.1$ pp & 334 & 4{,}101"),
             ("11+ & 1.6\\% & 1.0\\% & $-0.6$ pp & 1{,}150 & 12{,}431", "11+ & 1.6\\% & 1.0\\% & $-0.6$ pp & 1{,}143 & 12{,}385")]: rep(a, b)
rep("home drivers are 0.70 percentage points", "home drivers are 0.69 percentage points"); rep("odds ratio of \\textbf{0.915}", "odds ratio of \\textbf{0.921}")
# --- Q3
rep("(4{,}619 starts)", "(4{,}611 starts)")
rep("2014--2021 & Lotus F1 & 76 & 18 & 23.7\\% \\\\", "2014--2021 & Lotus F1 & 75 & 18 & 24.0\\% \\\\")
rep("CI 10.1--12.3\\%", "CI 10.2--12.3\\%")
rep("Caterham's rate rests on only 34 starts, so its interval is wide.", "Caterham's rate rests on only 34 starts, so its interval is wide. \\textbf{Sensitivity:} if statuses with an unclear cause (a plain ``Retired'', punctures, wing and wheel problems; about 20\\% of the starts counted as mechanical) are left out, the rates are 9.0\\% and 5.5\\%, so the fall holds; the third place in 2022--2024 changes from RB (48 starts) to AlphaTauri.")
# --- split
rep("The modeling table has one row per driver who \\emph{started} a race in 1950--2024, without the Indianapolis~500: \\textbf{24{,}724 rows}", "The modeling table has one row per driver who \\emph{started} a race in 1950--2024, without the Indianapolis~500: \\textbf{24{,}645 rows}")
rep("Train & 1950--2017 & 21{,}749 & 2{,}895 & 13.3\\%", "Train & 1950--2017 & 21{,}672 & 2{,}895 & 13.4\\%")
rep("Test & 2022--2024 & 1{,}356 & 204 & 15.0\\%", "Test & 2022--2024 & 1{,}354 & 204 & 15.1\\%")
# --- ladder
rep("+ field size & 0.669 & 0.671 & dropped (gain $<0.005$)", "+ field size & 0.671 & 0.671 & dropped (gain $<0.005$)")
rep("+ driver standings & 0.714 & 0.726 & \\textbf{kept}", "+ driver standings & 0.714 & 0.725 & \\textbf{kept}")
rep("+ team standings & 0.705 & 0.703 & dropped (worse)", "+ team standings & 0.705 & 0.702 & dropped (worse)")
rep("all 11 candidates together & 0.702 & 0.720 & for comparison", "all 11 candidates together & 0.702 & 0.722 & for comparison")
rep("A limitation of this method is that the result can depend on the order of the groups, and a dropped group is not retried; the all-candidates comparison and the ablation (Section~\\ref{sec:ablation}) are checks on this.",
    "\\paragraph{Does the order of the groups matter?} The method adds groups in a fixed order and never retries a dropped one, so the result could depend on the order. The notebook tries \\textbf{all 120 orders}: they lead to 8 different selected sets, so the exact list is not fixed. But every one of the 8 sets contains the driver standings and the grid information (\\code{grid}, or \\code{grid\\_rel}, which carries it): what is stable is \\emph{driver standings on top of the grid}; the other groups come and go because they repeat each other. A bootstrap over validation races shows why the 0.005 bar cannot separate small gains from noise: the gain from driver standings is $+0.045$ (95\\% range 0.008 to 0.080), the gain from field size is $+0.002$ (0.000 to 0.008). The all-candidates comparison and the ablation (Section~\\ref{sec:ablation}) are further checks.")
# --- models: thresholds, tuning, NN
rep("baseline 0.427, logistic regression 0.399, random forest 0.431, network 0.329", "baseline 0.428, logistic regression 0.400, random forest 0.437, network 0.337")
rep("RF & depth 4, min leaf 20 / 50 & 0.723 / 0.715 & 0.717 / 0.717", "RF & depth 4, min leaf 20 / 50 & 0.722 / 0.714 & 0.717 / 0.716")
rep("RF & depth 6, min leaf 20 / 50 & \\textbf{0.726} / 0.719 & 0.723 / 0.723", "RF & depth 6, min leaf 20 / 50 & \\textbf{0.725} / 0.719 & 0.724 / 0.724")
rep("RF & depth 8, min leaf 20 / 50 & 0.720 / 0.719 & 0.718 / 0.722", "RF & depth 8, min leaf 20 / 50 & 0.722 / 0.718 & 0.723 / 0.723")
rep("NN & hidden (16) / (32) / (32$\\to$16) & \\textbf{0.727} / 0.725 / 0.726 & -- \\\\", "NN & hidden (16) / (32) / (32$\\to$16) & \\textbf{0.728} / 0.725 / 0.726 & -- \\\\")
rep("The three network sizes score within 0.003 of each other, so the smallest one (one hidden layer of 16 units) is kept: with four inputs a bigger network gains nothing. It trained for the full 200 epochs because the validation loss was still decreasing slowly (\\fig{fig:nnloss}).",
    "The three network sizes score within 0.004 of each other, so the smallest one (one hidden layer of 16 units) is kept: with four inputs a bigger network gains nothing. Up to 1{,}000 epochs are allowed, so that the early-stopping rule ends training: the chosen network stopped at epoch 266 (best epoch 251), the other two at 282 and 220 (\\fig{fig:nnloss}).")
rep("\\includegraphics[width=0.6\\linewidth]{fig_16_nn_loss.png}\n\\caption{Neural network: training and validation loss per epoch.}", "\\includegraphics[width=\\linewidth]{fig_16_nn_loss.png}\n\\caption{Neural network: training and validation loss per epoch for each size; the dashed line marks the epoch whose weights are kept.}")
rep("Fill values: grid 12, \\code{drv\\_pos} 12, \\code{drv\\_pts\\_share} 0.077, \\code{drv\\_wins} 0.", "Fill values: grid 12, \\code{drv\\_pos} 12, \\code{drv\\_pts\\_share} 0.077, \\code{drv\\_wins} 0.") if False else None
# --- results table
new_rows = """Baseline (LR on grid) & train & 0.488 & 0.881 & 0.534 & 0.534 & 0.535 & 0.876 \\\\
 & val & 0.668 & 0.918 & 0.712 & 0.712 & 0.712 & 0.914 \\\\
 & test & 0.647 & 0.914 & 0.649 & 0.650 & 0.647 & 0.894 \\\\
\\midrule
Logistic regression & train & 0.546 & 0.891 & 0.535 & 0.566 & 0.507 & 0.882 \\\\
 & val & 0.714 & 0.928 & 0.709 & 0.702 & 0.716 & 0.912 \\\\
 & test & 0.708 & 0.930 & 0.678 & 0.690 & 0.667 & 0.905 \\\\
\\midrule
Random forest & train & 0.568 & 0.900 & 0.542 & 0.586 & 0.504 & 0.886 \\\\
 & val & 0.725 & \\textbf{0.939} & \\textbf{0.721} & 0.714 & 0.728 & 0.915 \\\\
 & test & \\textbf{0.721} & \\textbf{0.940} & 0.672 & 0.710 & 0.637 & 0.906 \\\\
\\midrule
Neural network & train & 0.550 & 0.895 & 0.565 & 0.525 & 0.612 & 0.874 \\\\
 & val & \\textbf{0.728} & 0.937 & \\textbf{0.722} & 0.671 & 0.782 & 0.910 \\\\
 & test & 0.704 & 0.938 & \\textbf{0.683} & 0.620 & 0.760 & 0.894 \\\\
\\midrule
Always ``no podium'' & train & 0.134 & 0.500 & 0.000 & 0.000 & 0.000 & 0.866 \\\\
 & val & 0.150 & 0.500 & 0.000 & 0.000 & 0.000 & 0.850 \\\\
 & test & 0.151 & 0.500 & 0.000 & 0.000 & 0.000 & 0.849 \\\\
"""
a = t.index("Baseline (LR on grid) & train & 0.487"); b = t.index("\\bottomrule", a)
t = t[:a] + new_rows + t[b:]
rep("\\includegraphics[width=0.49\\linewidth]{fig_17_models_val.png}\\hfill\n\\includegraphics[width=0.49\\linewidth]{fig_18_pr_auc_splits.png}\n\\caption{Left: validation PR-AUC, ROC-AUC and F1. Right: PR-AUC on train, validation and test.}",
    "\\includegraphics[width=0.9\\linewidth]{fig_17_models_val.png}\\\\[6pt]\n\\includegraphics[width=\\linewidth]{fig_18_metrics_splits.png}\n\\caption{Top: all six metrics on validation. Bottom: all six metrics on train, validation and test. The dashed line is the accuracy of a model that always predicts ``no podium''.}")
rep("(204 podiums among 1{,}356 starts)", "(204 podiums among 1{,}354 starts)")
rep("PR-AUC rises from 0.647 to 0.699--0.718 and ROC-AUC from 0.914 to 0.930--0.940.", "PR-AUC rises from 0.647 to 0.704--0.721 and ROC-AUC from 0.914 to 0.930--0.940.")
rep("the network and the forest are tied (0.727 vs.\\ 0.726); on test the forest is best (0.718).", "the network and the forest are tied (0.728 vs.\\ 0.725); on test the forest is best (0.721).")
a = t.index("The network also has the highest test F1 (0.683)"); b = t.index("\\item \\textbf{No overfitting.}")
t = t[:a] + f"The network also has the highest test F1 (0.683) and recall (finds {NN_TP} of 204 podiums), at the cost of more false alarms ({NN_FP}), while the forest is more conservative ({RF_TP} found, {RF_FP} false alarms).\n" + t[b:]
rep("Test scores are within 0.03 of validation.", "PR-AUC on test is within 0.03 of validation; threshold-based metrics move more (forest F1 0.721 $\\rightarrow$ 0.672, recall 0.728 $\\rightarrow$ 0.637).")
rep("\\end{itemize}\n\n%==============================================================================\n\\section{Experiments}",
    "\\item \\textbf{Accuracy is misleading.} All models reach 0.89--0.92 accuracy, but a model that always predicts ``no podium'' already scores 0.85. Precision, recall and F1 at the validation-chosen threshold are the informative threshold-based metrics; they are also optimistic on validation, because validation chose the features, settings, stopping epoch, final model and thresholds.\n\\end{itemize}\n\n%==============================================================================\n\\section{Experiments}")
# --- ablation
a = t.index("\\subsection{Feature-group ablation}"); b = t.index("\\subsection{Preprocessing-order experiments}")
t = t[:a] + """\\subsection{Feature-group ablation}\\label{sec:ablation}
Each group is removed in turn and the model is retrained with its chosen settings. First, each kept group is removed from the final feature set for all three models (the network is retrained for the same number of epochs as the chosen network). Second, each of the five groups is removed from the model that uses all 11 candidates, for logistic regression and the forest, so that the groups dropped in feature selection are covered too. Drops are shown on validation and on test; nothing is chosen from the test column.

\\begin{table}[H]
\\centering\\small
\\caption{Drop in PR-AUC when a feature group is removed from the selected model.}
\\label{tab:ablation}
\\begin{tabular}{llrr}
\\toprule
Removed group & Model & Validation drop & Test drop \\\\
\\midrule
grid & Logistic regression & 0.078 & 0.142 \\\\
grid & Random forest & 0.095 & 0.161 \\\\
grid & Neural network & 0.099 & 0.143 \\\\
driver standings & Logistic regression & 0.046 & 0.060 \\\\
driver standings & Random forest & 0.056 & 0.074 \\\\
driver standings & Neural network & 0.060 & 0.056 \\\\
\\bottomrule
\\end{tabular}
\\end{table}

\\begin{table}[H]
\\centering\\small
\\caption{Drop in PR-AUC when a group is removed from the model with all 11 candidates (negative = the model is better without the group).}
\\label{tab:ablation11}
\\begin{tabular}{lrrrr}
\\toprule
 & \\multicolumn{2}{c}{Logistic regression} & \\multicolumn{2}{c}{Random forest} \\\\
\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}
Removed group & Validation & Test & Validation & Test \\\\
\\midrule
grid & $-0.001$ & $-0.001$ & 0.003 & 0.011 \\\\
field size & 0.000 & 0.000 & 0.008 & 0.008 \\\\
driver standings & 0.007 & 0.007 & 0.020 & 0.000 \\\\
team standings & $-0.001$ & $-0.002$ & $-0.003$ & 0.000 \\\\
recent form & $-0.004$ & 0.010 & 0.006 & 0.001 \\\\
\\bottomrule
\\end{tabular}
\\end{table}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\linewidth]{fig_20_ablation.png}
\\caption{Feature-group ablation: drop in PR-AUC on validation (left) and test (right), for the selected model (top) and for the model with all 11 candidates (bottom).}
\\label{fig:ablation}
\\end{figure}

\\textbf{Selected model.} The grid is the most important group (a drop of 0.14--0.16 on test). The driver standings add 0.06--0.07 on test and 0.05--0.06 on validation. Both kept groups help all three models on validation \\emph{and} on test, so the selection made before training holds up after training.
\\textbf{All 11 candidates.} With every candidate present, removing a single group costs almost nothing (at most 0.02 on validation, 0.01 on test): the groups repeat each other. Removing \\code{grid} leaves \\code{grid\\_rel} and \\code{field\\_size}, which together contain the grid slot; removing the driver standings leaves the team standings and form. So no single group is essential once the others are present; the model needs \\emph{some} grid information and \\emph{some} standings information. The dropped groups (team standings, recent form) change test PR-AUC by at most 0.01, which supports leaving them out.

""" + t[b:]
# --- experiments table
rep("1. Fit imputer + scaler on train only & LR & 0.7138 & 0.7137 & $-0.0001$ & 0.69 \\\\", "1. Fit imputer + scaler on train only & LR & 0.7139 & 0.7139 & $0.0000$ & 0.85 \\\\")
rep("\\phantom{1. }vs.\\ on all seasons (leaky) & RF & 0.7256 & 0.7252 & $-0.0004$ & 0.29 \\\\", "\\phantom{1. }vs.\\ on all seasons (leaky) & RF & 0.7245 & 0.7239 & $-0.0006$ & 0.17 \\\\")
rep("2. Impute $\\rightarrow$ scale & LR & 0.7138 & 0.7138 & $+0.0000$ & 0.99 \\\\", "2. Impute $\\rightarrow$ scale & LR & 0.7139 & 0.7139 & $+0.0001$ & 0.40 \\\\")
rep("\\phantom{2. }vs.\\ scale $\\rightarrow$ impute & RF & 0.7256 & 0.7256 & $0.0000$ & 1.00 \\\\", "\\phantom{2. }vs.\\ scale $\\rightarrow$ impute & RF & 0.7245 & 0.7245 & $0.0000$ & 1.00 \\\\")
# --- XAI numbers
rep("\\code{grid} & 0.516 & 0.414 & 0.373 & 0.362 \\\\", "\\code{grid} & 0.516 & 0.414 & 0.372 & 0.385 \\\\")
rep("\\code{drv\\_pos} & -- & 0.003 & 0.025 & 0.015 \\\\", "\\code{drv\\_pos} & -- & 0.003 & 0.026 & 0.015 \\\\")
rep("\\code{drv\\_pts\\_share} & -- & 0.029 & 0.018 & 0.014 \\\\", "\\code{drv\\_pts\\_share} & -- & 0.029 & 0.018 & 0.008 \\\\")
rep("\\code{drv\\_wins} & -- & 0.010 & 0.006 & 0.053 \\\\", "\\code{drv\\_wins} & -- & 0.010 & 0.008 & 0.076 \\\\")
rep("is 0.133.", "is 0.134.")
rep("$+0.350$ & $+0.263$ && 14 & $-0.069$ & $-0.107$", "$+0.349$ & $+0.262$ && 14 & $-0.068$ & $-0.107$")
rep("$+0.146$ & $+0.076$ && 14", "$+0.146$ & $+0.075$ && 14")
rep("$+0.116$ & $+0.057$ && 0.054 & $-0.020$ & $-0.022$", "$+0.119$ & $+0.059$ && 0.054 & $-0.021$ & $-0.022$")
rep("$+0.097$ & $+0.023$ && 0 & $-0.003$ & $-0.020$", "$+0.094$ & $+0.024$ && 0 & $-0.003$ & $-0.022$")
rep("\\multicolumn{3}{c}{0.82} &&", "\\multicolumn{3}{c}{0.81} &&")
rep("from the 0.133 base", "from the 0.134 base"); rep("$R^2 = 0.82$", "$R^2 = 0.81$")
rep("& 0.688 & podium (correct)", "& 0.666 & podium (correct)"); rep("& 0.259 & no podium", "& 0.262 & no podium")
# --- limitations
rep("\\item \\textbf{Small validation set.}", "\\item \\textbf{Round~1 and rookies use training medians.} About 5\\% of validation and test rows have no standings and receive a typical mid-table value. In a side check, using last season's final standings instead helped validation (+0.007 to +0.011 PR-AUC) but not clearly test, so the simple median was kept.\n\\item \\textbf{Feature selection depends on the order of the groups.} The part that is stable is the driver standings on top of the grid; the exact list of extra features is not (Section on feature selection).\n\\item \\textbf{Small validation set.}")
open(P, "w").write(t); print("report updated")
