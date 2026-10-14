"""Second pass of review fixes (run after apply_fixes.py)."""
import nbformat
P = "notebooks/f1_podium_prediction.ipynb"
nb = nbformat.read(P, as_version=4); C = nb.cells
def find(sub, kind=None):
    hits = [i for i, c in enumerate(C) if sub in c.source and (kind is None or c.cell_type == kind)]
    assert len(hits) == 1, (sub, hits); return hits[0]
def sub(i, old, new):
    assert C[i].source.count(old) == 1, (i, old[:60]); C[i].source = C[i].source.replace(old, new)
def code(src): return nbformat.v4.new_code_cell(src.strip("\n"))
# X2: grain, missingness and cutoff for every data-engineering answer (brief section 2)
i = find("### 8.1 Q1")
sub(i, "- **Conversion rate**", "- **Grain, missingness, cutoff:** one front-row starter in one race (every join is checked to keep the row count); `grid` and `positionOrder` have no missing values; the question looks back at results, and `grid` is known after qualifying, before the race.\n- **Conversion rate**")
i = find("### 8.2 Q2")
C[i].source += "\n- **Grain, missingness, cutoff:** one starter in one race; `home_country` is filled for every driver (asserted) and the starting group comes from `grid`, known after qualifying, so no race result is used as an input."
i = find("### 8.3 Q3")
C[i].source += "\n- **Grain, missingness, cutoff:** one race start per team in 2014–2024; `status` is never missing. This question looks back at what happened in the race, so it has no pre-race cutoff."
# V12: how stable is the Spearman?
i = find('rho = rates["pre-2014"].corr(rates["2014+"], method="spearman")')
C[i].source += '''
rng_q1 = np.random.default_rng(SEED)
boot = [rates.iloc[rng_q1.integers(0, len(rates), len(rates))].corr(method="spearman").iloc[0, 1] for _ in range(1000)]
print(f"95% range of that correlation when the circuits are redrawn: {np.nanpercentile(boot, 2.5):.2f} to {np.nanpercentile(boot, 97.5):.2f} (only {len(rates)} circuits)")'''
# V11: Q3 sensitivity to the mechanical definition
i = find('fig_11_q3_outcomes')
C.insert(i + 1, code('''
# Sensitivity to the definition: "strict" leaves out statuses whose cause is not clearly a failed part
UNCLEAR = {"Retired", "Puncture", "Tyre", "Tyre puncture", "Wheel", "Wheel nut", "Wheel rim", "Out of fuel", "Fuel", "Fire", "Safety belt",
           "Front wing", "Rear wing", "Broken wing", "Undertray", "Brake duct", "Handling", "Vibrations", "Chassis", "Technical", "Axle"}
q3["mechanical_strict"] = q3["mechanical"] & ~q3["status"].isin(UNCLEAR)
sens = q3.groupby("period")[["mechanical", "mechanical_strict"]].mean().rename(columns={"mechanical": "broad (used)", "mechanical_strict": "strict"})
sens["unclear share of 'mechanical'"] = [(q3.loc[q3["period"] == p, "mechanical"] & q3.loc[q3["period"] == p, "status"].isin(UNCLEAR)).sum()
                                         / q3.loc[q3["period"] == p, "mechanical"].sum() for p in sens.index]
display(sens.round(3))
top3 = lambda col: (q3.groupby(["period", "team"])[col].agg(rate="mean", starts="size").reset_index()
                      .sort_values("rate", ascending=False).groupby("period").head(3))
pd.concat({"broad (used)": top3("mechanical"), "strict": top3("mechanical_strict")}).round(3)
'''))
# confusion counts as a printed table after the confusion-matrix figure (numbers quoted in the report)
i = find('save("fig_19_confusion_test")', "code")
sub(i, 'save("fig_19_confusion_test")\nplt.show()', '''save("fig_19_confusion_test")
plt.show()

def confusion_counts(f, thr):
    tn, fp, fn, tp = confusion_matrix(y_test, f(X_test) >= thr).ravel()
    return {"podiums found (TP)": tp, "podiums missed (FN)": fn, "false alarms (FP)": fp, "correct 'no podium' (TN)": tn}
pd.DataFrame({m: confusion_counts(f, THR[m]) for m, f in score.items()}).T''')
nbformat.write(nb, P); print("second pass ok:", len(C), "cells")
