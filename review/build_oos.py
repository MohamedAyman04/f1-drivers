import json, collections
import re, sys
from oos_items import ITEMS
import cellmap
_cells = cellmap.load(); _m = {int(k): v for k, v in json.load(open('cell_remap.json')).items()}
_m[86] = cellmap.index(_cells, 'ablation_code')
_m[41] = cellmap.index(_cells, 'podium_text')
def _remap(s):
    f = lambda x: str(_m.get(int(x), int(x)))
    return re.sub(r'c(\d+)(?:(–)c?(\d+))?', lambda m: 'c' + f(m.group(1)) + (m.group(2) + 'c' + f(m.group(3)) if m.group(2) else ''), s)
for it in ITEMS:
    it['where'] = _remap(it['where']) if it.get('legacy', True) else cellmap.resolve(it['where'], _cells)
    if it.get('legacy', True): it['what'] = it['what']

secs=collections.OrderedDict()
for it in ITEMS: secs.setdefault(it["sec"],[]).append(it)
n=len(ITEMS); H=sum(i["pri"]=="H" for i in ITEMS); R=sum(i["req"] for i in ITEMS)
out=[]
out.append("# Out-of-scope audit: what the notebook uses that Labs 1–4 do not teach\n")
out.append("Built on 2026-10-14. Cell numbers (0-based) are those of the **fixed notebook on branch `review-fixes`** (115 cells); the audit was done on the notebook at commit `1925b7f` (110 cells) and the numbers were remapped automatically (`review/remap.py`), with the constructs added by the fixes appended. `README.md` and `report/main.tex` were swept too.\n")
out.append("**Scope used:** the 4 lab PDFs (`labs/`) and the 3 guide notebooks (`guide_notebooks/`), nothing else. The project brief does not make something in scope; items it requires are tagged *required by brief* so we know we will be asked about them.\n")
out.append("**How it was built.** (1) Every name the notebook code uses (imports, methods, attributes, keyword arguments, functions) was extracted with Python's `ast` module and looked up in the lab text and guide-notebook code (`review/sweep_names.py`, result in `review/sweep_names.json`: "+str(len(json.load(open('sweep_names.json'))))+" names in the fixed notebook, "+str(sum(not s['in_labs'] for s in json.load(open('sweep_names.json'))))+" not found). (2) All 110 cells, their markdown and their comments were then read one by one for concepts, methods and domain terms that a name search cannot find. (3) `README.md` and the report were swept for jargon. The item list lives in `review/oos_items.py`; `review/build_oos.py` regenerates this file.\n")
out.append(f"**Result:** {n} items, {H} of them priority H, {R} required by the brief. Priority: **H** = likely to be asked (it drives a result or is in the brief); **M** = supporting detail; **L** = plumbing.\n")
out.append("**Caution when reading the lab columns.** 'Not in the labs' means the lab PDFs and guide notebooks never show it. Lab 1's last exercise asks for a correlation heatmap, which implies `DataFrame.corr()`, but the code is not shown; we treat the default `corr()` as in scope and the `method=\"spearman\"` argument as out. The Lab 3 *text* explains binary cross-entropy and sigmoid, but its guide *code* uses softmax and categorical cross-entropy.\n")
out.append("The board (https://claude.ai/artifact/3hor4cmonHZQ4vhWBZWH8r, tab *Out of scope*) shows the same list with status, owner and note fields.\n")
out.append("## Summary by section\n")
out.append("| Section | Items | H |\n|---|---|---|")
for s,its in secs.items(): out.append(f"| {s} | {len(its)} | {sum(i['pri']=='H' for i in its)} |")
out.append("")
for s,its in secs.items():
    out.append(f"## {s}\n")
    out.append("| # | Item | Where (cell) | Kind | Required by brief? | What it is | Why we used it | Lab alternative | Priority |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    esc=lambda x: str(x).replace("|","\\|").replace("\n"," ")
    for i in its:
        out.append(f"| {i['id']} | {esc(i['item'])} | {esc(i['where'])} | {i['kind']} | {'yes' if i['req'] else 'no'} | {esc(i['what'])} | {esc(i['why'])} | {esc(i['alt'])} | {i['pri']} |")
    out.append("")
out.append("## Appendix A: what the labs do teach (the in-scope reference)\n")
out.append("""| Lab | In scope |
|---|---|
| 1: Wrangling and visualisation | `read_csv`, `head`, `info`, `describe`, `isnull().sum()`, `value_counts`; numerical vs categorical; impute vs remove; `drop(columns=)`; `fillna` with median / mode; label vs one-hot encoding, `map`, `get_dummies(drop_first=True)`; `missingno.matrix`; plot types (comparison, distribution, composition, relationship); `plt.hist`, `plt.scatter`, bubble chart, `sns.heatmap` on `pivot_table` and on correlations, `pd.cut`, `sns.barplot`, `sns.boxplot` |
| 2: ML foundations | supervised / unsupervised; classification vs regression; `train_test_split`; `OrdinalEncoder`; `drop_duplicates`; `LinearRegression`, `LogisticRegression(max_iter)`; sigmoid, probability and threshold; accuracy, precision, recall, F1, `classification_report`, `confusion_matrix`; accuracy misleads on imbalanced data; MSE/RMSE/MAE/R²; leakage by dropping a column; baseline vs improved model |
| 3: Neural networks | scaling, `StandardScaler` fit on train; `stratify=y`; `keras.Sequential`, `Input`, `Dense(relu)`; sigmoid, softmax; depth; forward pass, loss, backpropagation; binary and categorical cross-entropy; SGD vs Adam, learning rate; epochs, batches, `validation_split`; overfitting; `EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)`; loss/accuracy curves; `np.random.seed`, `tf.random.set_seed` |
| 4: XAI | intrinsic vs post-hoc, model-specific vs agnostic, global vs local; `DecisionTreeClassifier`, `plot_tree`; `RandomForestClassifier` (bagging); `shap.TreeExplainer`, `shap_values`, `summary_plot`, `force_plot`, waterfall, `expected_value`; LIME, surrogate model, `LimeTabularExplainer(discretize_continuous=True)`, `explain_instance`, `show_in_notebook`; fairness, bias, trust; `!pip install shap lime` |
""")
out.append("## Appendix B: code names not found in the lab material\n")
out.append("Automatic list from `review/sweep_names.json` (imports, methods, attributes, keyword arguments, functions). A name that appears in the lab text can still be used differently; this list only says which names the labs never mention. Names defined in the notebook itself (for example `wilson`, `prepare`) are not included in the count of library names.\n")
sw=json.load(open("sweep_names.json"))
miss=[s for s in sw if not s["in_labs"]]
lib=[s for s in miss if s["name"] not in {"best_f1_threshold","build_nn","is_missing","last5_rate","permutation_drop","predict_podium","save","standings_before","to_raw","to_scaled","val_pr_auc","val_preds","val_scores","wilson"}]
out.append(f"{len(lib)} library names. By kind:\n")
for kind in ["import","func","attr","kwarg"]:
    names=sorted({s['name'] for s in lib if s['kind']==kind})
    out.append(f"- **{kind}** ({len(names)}): " + ", ".join(f"`{x}`" for x in names))
out.append("")
open("../OUT_OF_SCOPE.md","w").write("\n".join(out))
json.dump(ITEMS,open("oos_items.json","w"),indent=1,ensure_ascii=False)
print(n,"items;",H,"H;",R,"required;",len(secs),"sections")
