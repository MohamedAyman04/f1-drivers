"""Maps short names to a substring that identifies a notebook cell, so docs can quote current cell numbers."""
import json, re
MARK = {
 "eda_before": "## 3. First look at the raw data", "eda_after": "### The data after cleaning",
 "sentinel": 'rows = [(t, c, int((df[c] == r"\\N")', "typing": "# 2. Dates and durations",
 "pk": "PRIMARY_KEYS = {", "fk": "FOREIGN_KEYS = [", "dups": 'KEY = ["raceId", "driverId"]',
 "dup_rule": "# 5. One row per (race, driver)", "flags": "# 6. Flags.", "grid0_text": "`grid = 0` has two meanings",
 "sprint_text": "**Handling of Sprint Races:**", "podium_text": "- Only drivers who started are counted: 13.5% of starts end on the podium",
 "q1_def": "### 8.1 Q1", "q2_def": "### 8.2 Q2", "q3_def": "### 8.3 Q3", "q1_answer": "**Answer Q1:**", "q3_sens": "UNCLEAR = {",
 "leak_check": "# Leakage check:", "form": "def last5_rate", "order_code": "def ladder_for(", "tuning": "MODELS = {",
 "nn_code": "def build_nn(", "val_table": "THR = {m: best_f1_threshold", "metrics_text": "**The six metrics for this task**",
 "decade_code": "by_decade = dec.groupby", "test_table": 'SETS["test"] = (X_test, y_test)', "ablation_code": "def refit_scores(",
 "noise_code": "noise = []", "q1_boot": "boot = [rates.iloc", "xai_intro": "## 16. Explainability (XAI)", "xai_model": "XAI_MODEL = rf_best",
}
def load(path="../notebooks/f1_podium_prediction.ipynb"):
    return json.load(open(path))["cells"]
def index(cells, key):
    sub = MARK[key]
    hits = [i for i, c in enumerate(cells) if sub in "".join(c["source"])]
    return hits[0] if hits else None
def resolve(text, cells):
    def f(m):
        k = m.group(1); i = index(cells, k)
        return f"c{i}" if i is not None else f"[{k}?]"
    return re.sub(r"\{c:(\w+)\}", f, text)
