"""Applies the review fixes to notebooks/f1_podium_prediction.ipynb (branch review-fixes). Markdown numbers that depend on
the re-run are written afterwards by review/update_text.py. Idempotent only on a fresh checkout of the 1925b7f notebook."""
import nbformat, re
P = "notebooks/f1_podium_prediction.ipynb"
nb = nbformat.read(P, as_version=4)
C = nb.cells
def find(sub, kind=None):
    hits = [i for i, c in enumerate(C) if sub in c.source and (kind is None or c.cell_type == kind)]
    assert len(hits) == 1, (sub, hits)
    return hits[0]
def sub(i, old, new, count=1):
    assert C[i].source.count(old) == count, (i, old[:60], C[i].source.count(old))
    C[i].source = C[i].source.replace(old, new)
def code(src): return nbformat.v4.new_code_cell(src.strip("\n"))
def md(src): return nbformat.v4.new_markdown_cell(src.strip("\n"))
def insert_after(i, *cells):
    for k, c in enumerate(cells): C.insert(i + 1 + k, c)

# ---- V1: entries that never raced are non-starters --------------------------------------------------------------
i = find("`grid = 0` has two meanings")
sub(i, "Cleaning separates them.", "Cleaning separates them. It also treats entries that have a grid slot in the data but never raced (status `Withdrew`, `Did not qualify`, `107% Rule`, with 0 laps) as non-starters.")
i = find("# 6. Flags. A pit-lane starter")
sub(i, '''# 6. Flags. A pit-lane starter gets the slot behind the last car on the grid
r = r.merge(clean["races"][["raceId", "year", "name"]], on="raceId")
r["is_nonstarter"] = (r["grid"] == 0) & (r["laps"] == 0)
r["pitlane_start"] = (r["grid"] == 0) & (r["laps"] > 0)
r["grid_eff"] = r["grid"].where(r["grid"] > 0, r.groupby("raceId")["grid"].transform("max") + 1)''',
'''# 6. Flags. A non-starter never raced: no grid slot and 0 laps, or an entry that was withdrawn or failed to qualify
# (status "Withdrew", "Did not qualify", ...) with 0 laps, even if the dataset still shows a grid slot for it.
# A pit-lane starter gets the slot behind the last car that really started.
r = r.merge(clean["races"][["raceId", "year", "name"]], on="raceId")
r = r.merge(clean["status"], on="statusId", how="left")
NEVER_RACED = {"Withdrew", "Did not qualify", "Did not prequalify", "107% Rule"}
r["is_nonstarter"] = ((r["grid"] == 0) & (r["laps"] == 0)) | (r["status"].isin(NEVER_RACED) & (r["laps"] == 0))
r["pitlane_start"] = (r["grid"] == 0) & (r["laps"] > 0)
last_slot = r["grid"].where(~r["is_nonstarter"]).groupby(r["raceId"]).transform("max")
r["grid_eff"] = r["grid"].where(r["grid"] > 0, last_slot + 1)''')
sub(i, '''print(f"non-starters: {r['is_nonstarter'].sum():,} | pit-lane starts''', '''print(f"non-starters: {r['is_nonstarter'].sum():,} (of which {int((r['is_nonstarter'] & (r['grid'] > 0)).sum())} still show a grid slot) | pit-lane starts''')
i = find("# 7. Why each race ended")
sub(i, '''r = r.merge(clean["status"], on="statusId", how="left")\nr["status_class"]''', '''r["status_class"]''')

# ---- Q1 consistency: same rows as the rest of the analysis -----------------------------------------------------
i = find("### 8.1 Q1")
sub(i, "- **Conversion rate**", "- Non-starters and the Indy 500 are excluded, as in the modeling table. Grid-only was confirmed with the instructor.\n- **Conversion rate**")
i = find("q1 = (clean[\"results\"]")
sub(i, 'q1 = (clean["results"]\n      .merge(clean["races"][["raceId", "circuitId"]], on="raceId")', 'q1_rows = clean["results"].query("~is_nonstarter and ~is_indy500")\nq1 = (q1_rows\n      .merge(clean["races"][["raceId", "circuitId"]], on="raceId")')
sub(i, "assert len(q1) == len(clean[\"results\"])", "assert len(q1) == len(q1_rows)")

i = find('races_per_era = q1.groupby(["circuit", "era"])["raceId"].nunique().unstack(fill_value=0)', "code")
sub(i, 'races_per_era = q1.groupby(["circuit", "era"])["raceId"].nunique().unstack(fill_value=0)',
    'races_all = clean["races"].merge(clean["circuits"][["circuitId", "name"]].rename(columns={"name": "circuit"}), on="circuitId")\nraces_all["era"] = np.where(races_all["year"] >= 2014, "2014+", "pre-2014")\nraces_per_era = races_all.groupby(["circuit", "era"])["raceId"].nunique().unstack(fill_value=0)   # races hosted, whatever the starters')

# ---- N1/N2: wrong text -------------------------------------------------------------------------------------------
i = find("**Handling of Sprint Races:**")
sub(i, "Across the entire 1950–2024 history, over 99% of races had no sprint, and ~96% of training seasons (1950–2020) entirely predate the concept.",
    "Across the entire 1950–2024 history, 98.4% of races (1,107 of 1,125) had no sprint, and every training season (1950–2017) predates them.")
j = C[i].source.index("- **Secondary / improved modern-era model:**")
C[i].source = C[i].source[:j] + "- **Future work, not built here:** for a model specialised on the modern era (2021+), sprint finishing positions and lap pace could be a strong pre-race feature, because the sprint is held before the Grand Prix. Our single model covers 1950–2024, so sprints are not used."
i = find("- Only drivers who started are counted: 13.5% of starts end on the podium")
sub(i, "than today (≈15%)", "than today (≈15%)")
sub(i, "(≈10%)", "(≈12%)")
sub(i, "(secondary: ROC-AUC and F1 at a threshold chosen on validation).", "(secondary: ROC-AUC and F1 at a threshold chosen on validation). Note that the no-skill level of PR-AUC *equals* the podium rate (13–15% here), so PR-AUC values should only be compared between sets with similar podium rates; ROC-AUC does not depend on the rate.")

# ---- V9: order sensitivity of the selection ----------------------------------------------------------------------
i = find("fig_14_feature_ladder")
insert_after(i, code('''
# How much does the result depend on the order of the groups? Try all 120 orders. Scores are cached: only 32 different feature sets exist.
import itertools
_cache = {}
def _scores(cols):
    key = tuple(sorted(cols))
    if key not in _cache:
        _cache[key] = val_pr_auc(list(key))
    return _cache[key]

def ladder_for(order):
    kept_, ks = [], None
    for g in order:
        s = _scores(kept_ + GROUPS[g])
        if ks is None or all(s[m] - ks[m] >= MIN_GAIN for m in s):
            kept_, ks = kept_ + GROUPS[g], s
    return frozenset(kept_)

by_set = {}
for order in itertools.permutations(GROUPS):
    by_set.setdefault(ladder_for(order), []).append(order)
order_check = pd.DataFrame([{"orders": len(v), "selected features": ", ".join(sorted(k)),
                             "has driver standings": "drv_pos" in k, "has grid information": bool({"grid", "grid_rel"} & k)}
                            for k, v in sorted(by_set.items(), key=lambda kv: -len(kv[1]))])
print(f"{sum(len(v) for v in by_set.values())} possible group orders lead to {len(by_set)} different selected sets")
order_check
'''), code('''
# How large is the noise in a validation PR-AUC difference? Redraw whole validation races (same idea as section 15.2).
(Xa, Xb, _), _ = prepare(["grid"])
(Xc, Xd, _), _ = prepare(SELECTED)
(Xe, Xf, _), _ = prepare(["grid", "field_size", "grid_rel"])
p_grid = LogisticRegression(max_iter=2000).fit(Xa, y_train).predict_proba(Xb)[:, 1]
p_stand = LogisticRegression(max_iter=2000).fit(Xc, y_train).predict_proba(Xd)[:, 1]
p_field = LogisticRegression(max_iter=2000).fit(Xe, y_train).predict_proba(Xf)[:, 1]
races_va = list(va.reset_index(drop=True).groupby("raceId").indices.values())
rng_ = np.random.default_rng(1)
redraw = [np.concatenate([races_va[i] for i in rng_.integers(0, len(races_va), len(races_va))]) for _ in range(500)]
yv_ = y_val.to_numpy()
noise = []
for name, p_new in [("+ field size", p_field), ("+ driver standings", p_stand)]:
    d = np.array([average_precision_score(yv_[r], p_new[r]) - average_precision_score(yv_[r], p_grid[r]) for r in redraw])
    noise.append({"step (logistic regression)": name, "gain": average_precision_score(yv_, p_new) - average_precision_score(yv_, p_grid),
                  "95% low": np.percentile(d, 2.5), "95% high": np.percentile(d, 97.5)})
pd.DataFrame(noise).round(3)
'''))

# ---- V4: neural network trains until early stopping fires ---------------------------------------------------------
i = find("def build_nn(units):")
sub(i, "def build_nn(units):\n    keras.utils.set_random_seed(SEED)\n    model = keras.Sequential([keras.Input(shape=(len(FEATURES),))]",
        "def build_nn(units, n_inputs=None):\n    keras.utils.set_random_seed(SEED)\n    model = keras.Sequential([keras.Input(shape=(n_inputs or len(FEATURES),))]")
sub(i, "for units in [(16,), (32,), (32, 16)]:", "SIZES = [(16,), (32,), (32, 16)]\nfor units in SIZES:")
sub(i, "epochs=200, batch_size=256", "epochs=1000, batch_size=256")
sub(i, '''"epochs run": len(h.history["loss"]),''', '''"epochs run": len(h.history["loss"]), "best epoch": int(np.argmin(h.history["val_loss"])) + 1,''')
sub(i, '''nn_units = nn_tuning.loc[nn_tuning["val PR-AUC"].idxmax(), "hidden layers"]\nnn_best = nn_models[nn_units]''',
        '''nn_pick = nn_tuning["val PR-AUC"].idxmax()\nnn_units, nn_size, nn_epochs = nn_tuning.loc[nn_pick, "hidden layers"], SIZES[nn_pick], int(nn_tuning.loc[nn_pick, "best epoch"])\nnn_best = nn_models[nn_units]''')
k = C[i].source.index("fig, ax = plt.subplots(figsize=(7, 3.5))")
C[i].source = C[i].source[:k] + '''fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for ax, units in zip(axes, SIZES):
    hist = nn_history[str(units)]
    ax.plot(hist["loss"], label="train loss")
    ax.plot(hist["val_loss"], label="validation loss")
    ax.axvline(int(np.argmin(hist["val_loss"])), color="black", ls="--", lw=0.8, label="best epoch (kept)")
    ax.set(xlabel="epoch", title=f"hidden layers {units}" + ("  (chosen)" if str(units) == nn_units else ""))
axes[0].set_ylabel("binary cross-entropy")
axes[0].legend()
plt.tight_layout()
save("fig_16_nn_loss")
plt.show()'''
i = find("## 13. Shallow neural network")
sub(i, "- **Early stopping:** training stops when the validation loss has not improved for 15 epochs, and the weights roll back to the best epoch. This prevents overfitting.",
       "- **Early stopping:** training stops when the validation loss has not improved for 15 epochs, and the weights roll back to the best epoch. This prevents overfitting. Up to 1,000 epochs are allowed, so that the stopping rule, not the cap, ends training.")

# ---- Section 5: all six metrics plotted and discussed ---------------------------------------------------------------
i = find("THR = {m: best_f1_threshold(y_val, f(X_val))")
c = C[i]
c.source = c.source.replace('''ax = results.xs("val", level=1)[["PR-AUC", "ROC-AUC", "F1"]].plot.bar(rot=0, figsize=(10, 3.8), ylim=(0, 1))
ax.set_title("Validation scores (thresholds chosen on validation)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))''', '''# Reference: a model that always says "no podium". Its accuracy looks good only because podiums are rare.
def always_no(y):
    prevalence = float(np.mean(y))
    return {"PR-AUC": prevalence, "ROC-AUC": 0.5, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, "Accuracy": 1 - prevalence}
reference = pd.DataFrame({s: always_no(y) for s, (X, y) in SETS.items()}).T
print('Reference, a model that always predicts "no podium":')
display(reference.round(3))

METRICS = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]
ax = results.xs("val", level=1)[METRICS].T.plot.bar(rot=0, figsize=(11, 4), ylim=(0, 1), width=0.8)
ax.hlines(reference.loc["val", "Accuracy"], 4.55, 5.45, color="black", ls="--", lw=1.2, label="always ‘no podium’ (accuracy)")
ax.set_title("All six metrics on validation (thresholds chosen on validation)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))''')
assert "always_no" in c.source
insert_after(i, md("<<METRICS_DISCUSSION>>"), code('''
# Why is the training score lower than the validation score? Random forest, training seasons only, decade by decade.
dec = tr.assign(p=score["Random forest"](X_train), decade=(tr["year"] // 10) * 10)
by_decade = dec.groupby("decade")[["podium", "field_size", "p"]].apply(
    lambda d: pd.Series({"rows": len(d), "podium rate": d["podium"].mean(), "avg starters": d["field_size"].mean(),
                         "PR-AUC": average_precision_score(d["podium"], d["p"])}))
by_decade["rows"] = by_decade["rows"].astype(int)
by_decade.round(3)
'''))
i = find('SETS["test"] = (X_test, y_test)')
c = C[i]
c.source = c.source.replace('''display(final.round(3))

ax = final["PR-AUC"].unstack()[["train", "val", "test"]].plot.bar(rot=0, figsize=(10, 3.8), ylim=(0, 1))
ax.set_title("PR-AUC on train, validation and test")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
save("fig_18_pr_auc_splits")
plt.show()''', '''reference = pd.DataFrame({s: always_no(y) for s, (X, y) in SETS.items()}).T
display(final.round(3))
print('Reference, a model that always predicts "no podium":')
display(reference.round(3))

fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharey=True)
for ax, met in zip(axes.ravel(), METRICS):
    final[met].unstack()[["train", "val", "test"]].plot.bar(rot=15, ax=ax, ylim=(0, 1), legend=False, width=0.8)
    ax.set(title=met, xlabel="")
    if met == "Accuracy":
        ax.axhline(reference.loc["test", "Accuracy"], color="black", ls="--", lw=1)
fig.suptitle("All six metrics on train, validation and test (thresholds chosen on validation; dashed = always ‘no podium’)")
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig.legend([plt.Rectangle((0, 0), 1, 1, color=f"C{k}") for k in range(3)], ["train", "val", "test"], loc="lower center", ncol=3)
save("fig_18_metrics_splits")
plt.show()''')
assert "fig_18_metrics_splits" in c.source

# ---- V8: ablation for every model, kept groups and all candidates, validation and test ---------------------------------
i = find("### 15.1 Feature-group ablation")
C[i].source = '''## 15. Experiments

### 15.1 Feature-group ablation

Remove one feature group at a time, retrain with the chosen settings, and measure how much PR-AUC drops. Section 11 justified each group *before* training; this measures what each one contributes *after* training.
- **Selected model:** each kept group is removed from the final feature set, for all three models (the network is retrained for the same number of epochs as the chosen network, without looking at validation).
- **All 11 candidates:** each of the five groups is removed from the model that uses every candidate, for logistic regression and the forest, so that the groups that were dropped in section 11 are covered too.
- Drops are shown on **validation and test**. They are for reading only: nothing is chosen from the test column.'''
i = find("ABLATE = {g: cols for g, cols in GROUPS.items()")
C[i].source = '''MODEL = {"Logistic regression": lr_best, "Random forest": rf_best}

def refit_scores(model_name, cols):
    """Retrain `model_name` on `cols` with its chosen settings. Returns (validation PR-AUC, test PR-AUC)."""
    (Xtr, Xva, Xte), _ = prepare(cols)
    if model_name == "Neural network":
        net = build_nn(nn_size, len(cols))
        net.fit(Xtr.to_numpy("float32"), y_train.to_numpy(), epochs=nn_epochs, batch_size=256, verbose=0)
        pv, pt = (net.predict(X.to_numpy("float32"), verbose=0).ravel() for X in (Xva, Xte))
    else:
        fit = clone(MODEL[model_name]).fit(Xtr, y_train)
        pv, pt = fit.predict_proba(Xva)[:, 1], fit.predict_proba(Xte)[:, 1]
    return average_precision_score(y_val, pv), average_precision_score(y_test, pt)

rows = []
for setting, base_cols, models in [("selected model", SELECTED, list(MODEL) + ["Neural network"]),
                                   ("all 11 candidates", CANDIDATES, list(MODEL))]:
    for m in models:
        full_val, full_test = refit_scores(m, base_cols)
        for group, cols in GROUPS.items():
            if not set(cols) <= set(base_cols):
                continue
            v, t = refit_scores(m, [c for c in base_cols if c not in cols])
            rows.append({"setting": setting, "without": group, "model": m, "val drop": full_val - v, "test drop": full_test - t})
ablation = pd.DataFrame(rows)

MODEL_COLOR = {"Logistic regression": "C0", "Random forest": "C1", "Neural network": "C2"}
fig, axes = plt.subplots(2, 2, figsize=(13, 6.8), sharex="col")
for r_, setting in enumerate(["selected model", "all 11 candidates"]):
    d = ablation[ablation["setting"] == setting]
    for c_, col in enumerate(["val drop", "test drop"]):
        piv = d.pivot(index="without", columns="model", values=col)
        piv.plot.barh(ax=axes[r_, c_], legend=False, color=[MODEL_COLOR[m] for m in piv.columns])
        axes[r_, c_].axvline(0, color="black", lw=0.8)
        axes[r_, c_].set(title=f"{setting}: {col.replace('drop', 'PR-AUC drop')}", ylabel="")
fig.suptitle("Feature-group ablation: drop in PR-AUC when the group is removed (the network is only run on the selected features)")
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COLOR.values()], list(MODEL_COLOR), loc="lower center", ncol=3)
save("fig_20_ablation")
plt.show()
ablation.round(3)'''
i = find("**Reading the ablation**")
C[i].source = "<<ABLATION_TEXT>>"

# ---- V7 note in limitations -------------------------------------------------------------------------------------
i = find("## 18. Limitations")
C[i].source += "\n- **Round 1 and rookies are filled with training medians:** the first race of a season has no standings, so about 5% of validation and test rows get a typical mid-table value. In a side check (`review/verify.ipynb`), using last season's final standings instead helped validation (+0.007 to +0.011 PR-AUC) but not clearly test, so the simple median was kept."

nbformat.write(nb, P)
print("cells:", len(C))
