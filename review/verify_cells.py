# %% [markdown]
# # Independent verification of the F1 podium notebook
# Scratch notebook, **not** part of the deliverable. It rebuilds the modeling table from the raw CSVs with different code
# (`verify_lib.py`) and checks the notebook's numbers and claims. Item ids (V1, N1, ...) refer to `REVIEW_PLAN.md`.
# Run from the repo root or from `review/`; takes about 5 minutes.

# %%
import sys, json, re, itertools, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd() if Path("verify_lib.py").exists() else Path.cwd() / "review"))
import numpy as np, pandas as pd
from verify_lib import build, read, GROUPS, SELECTED, RAW
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score as AP, precision_recall_curve
SEED = 42
mt = build()
tr, va, te = (mt[mt.split == s] for s in ("train", "val", "test"))
def prep(cols, fit=None):
    fit = tr if fit is None else fit
    p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(fit[cols].astype(float))
    return [p.transform(d[cols].astype(float)) for d in (tr, va, te)], p
def LR(): return LogisticRegression(max_iter=2000, C=10)
def RF(): return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
(Xtr, Xva, Xte), p_sel = prep(SELECTED)
lr, rf = LR().fit(Xtr, tr.podium), RF().fit(Xtr, tr.podium)
print("rows", len(mt), mt.groupby("split").size().to_dict())
print("LR val/test", round(AP(va.podium, lr.predict_proba(Xva)[:, 1]), 4), round(AP(te.podium, lr.predict_proba(Xte)[:, 1]), 4), "| notebook 0.7138 / 0.708")
print("RF val/test", round(AP(va.podium, rf.predict_proba(Xva)[:, 1]), 4), round(AP(te.podium, rf.predict_proba(Xte)[:, 1]), 4), "| notebook 0.7256 / 0.718")

# %% [markdown]
# ## V13/V14: cleaning counts and sentinels

# %%
raw_res = pd.read_csv(RAW / "results.csv", dtype=str, keep_default_na=False)
res = read("results"); races = read("races"); status = read("status")
dup = res[res.duplicated(["raceId", "driverId"], keep=False)]
g = dup.groupby(["raceId", "driverId"])
print("duplicate pairs", g.ngroups, "rows", len(dup), "| sizes of groups:", g.size().value_counts().to_dict())
print("years of duplicates", races.set_index("raceId").loc[dup.raceId, "year"].agg(["min", "max"]).tolist())
both0 = (g["grid"].apply(lambda s: (s == 0).all())).sum()
print("pairs where every row has grid 0 (tie-break by positionOrder):", int(both0))
res2 = res.merge(races[["raceId", "year", "name"]], on="raceId")
print("raw non-starters (grid 0, laps 0):", int(((res.grid == 0) & (res.laps == 0)).sum()), "| pit-lane starts:", int(((res.grid == 0) & (res.laps > 0)).sum()))
print("Indy 500 rows 1950-60 (raw):", int((res2.name.str.contains("Indianapolis") & (res2.year <= 1960)).sum()))
sent = {}
for f in sorted(RAW.glob("*.csv")):
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    for c in d.columns:
        n = int((d[c] == r"\N").sum())
        if n: sent[(f.stem, c)] = n
print("sentinel cells:", sum(sent.values()), "in", len(sent), "columns (notebook c16 prints this too)")

# %% [markdown]
# ## V1: counted-as-starter rows that look like non-starters, and what fixing them does

# %%
res3 = res.merge(status, on="statusId")
cand = res3[(res3.grid > 0) & (res3.laps == 0)]
print(cand.status.value_counts().head(12).to_string())
mt2 = build("plus_clear_cases")
print("modeling rows:", len(mt), "->", len(mt2), "| removed by split:", (mt.groupby("split").size() - mt2.groupby("split").size()).to_dict())
def fit_eval(m):
    t, v, e = (m[m.split == s] for s in ("train", "val", "test"))
    pp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(t[SELECTED].astype(float))
    X = [pp.transform(d[SELECTED].astype(float)) for d in (t, v, e)]
    a, b = LR().fit(X[0], t.podium), RF().fit(X[0], t.podium)
    return {"LR val": AP(v.podium, a.predict_proba(X[1])[:, 1]), "LR test": AP(e.podium, a.predict_proba(X[2])[:, 1]),
            "RF val": AP(v.podium, b.predict_proba(X[1])[:, 1]), "RF test": AP(e.podium, b.predict_proba(X[2])[:, 1]),
            "podium rate": m.podium.mean()}
eff = pd.DataFrame({"current rule": fit_eval(mt), "plus Withdrew/DNQ/107% with 0 laps": fit_eval(mt2)})
eff["change"] = eff.iloc[:, 1] - eff.iloc[:, 0]
print(eff.round(4).to_string())
fr = res3[(res3.grid.isin([1, 2])) & (res3.laps == 0) & res3.status.isin(["Withdrew", "Did not qualify", "107% Rule"])]
print("front-row (grid 1-2) rows that are really non-starters, used in Q1:", len(fr))

# %% [markdown]
# ## V5/V6/N1: why train PR-AUC is lower than validation; podium rate and field size by decade

# %%
def split_scores(model, X, d): return AP(d.podium, model.predict_proba(X)[:, 1])
mt["decade"] = (mt.year // 10) * 10
mt_X = p_sel.transform(mt[SELECTED].astype(float))
dec = mt.assign(p_lr=lr.predict_proba(mt_X)[:, 1], p_rf=rf.predict_proba(mt_X)[:, 1]).groupby("decade").apply(
    lambda d: pd.Series({"rows": len(d), "podium rate": d.podium.mean(), "avg starters": d.field_size.mean(), "max starters": d.field_size.max(),
                         "LR PR-AUC": AP(d.podium, d.p_lr), "RF PR-AUC": AP(d.podium, d.p_rf)}))
print(dec.round(3).to_string())
print("\nPR-AUC lift over no-skill (PR-AUC / podium rate):")
for s, d in [("train", tr), ("val", va), ("test", te)]:
    X = p_sel.transform(d[SELECTED].astype(float)); a = AP(d.podium, rf.predict_proba(X)[:, 1])
    print(f"  {s:5} RF PR-AUC {a:.3f}  podium rate {d.podium.mean():.3f}  lift {a / d.podium.mean():.2f}")
modern = tr[tr.year >= 1990]
print("RF PR-AUC on train seasons 1990-2017:", round(AP(modern.podium, rf.predict_proba(p_sel.transform(modern[SELECTED].astype(float)))[:, 1]), 3))
print("always-'no podium' accuracy:", {s: round(1 - d.podium.mean(), 3) for s, d in [("train", tr), ("val", va), ("test", te)]})
print("c41 claims: 1980s-90s starters 26-30 and podium rate 10-11%  ->  see decade table above")

# %% [markdown]
# ## V7: round-1 and rookie imputation

# %%
for s, d in [("train", tr), ("val", va), ("test", te)]:
    miss = d.drv_pos.isna()
    print(f"{s:5} missing standings {miss.mean():.3f}  of which round 1 (no previous race): {d.loc[miss, 'prev'].isna().mean():.2f}")
print("train medians used:", dict(zip(SELECTED, p_sel[0].statistics_.round(2))))
r1 = va[va.prev.isna()]
print("val round-1 rows:", len(r1), "| podium rate in round 1:", round(r1.podium.mean(), 3), "vs other val rows", round(va[va.prev.notna()].podium.mean(), 3))
# alternative: for round 1 use the final standings of the previous season
last_race = races.sort_values(["year", "round"]).groupby("year").tail(1).set_index("year")["raceId"]
mt3 = mt.copy()
r1m = mt3.prev.isna() & (mt3.year > 1950)
mt3.loc[r1m, "prev"] = (mt3.loc[r1m, "year"] - 1).map(last_race)
ds = read("driver_standings")
alt = mt3.loc[r1m, ["prev", "driverId"]].merge(ds[["raceId", "driverId", "position", "points", "wins"]].rename(columns={"raceId": "prev"}), on=["prev", "driverId"], how="left")
alt.index = mt3.loc[r1m].index
lead = ds.groupby("raceId").points.max()
mt3.loc[r1m, "drv_pos"] = alt["position"]; mt3.loc[r1m, "drv_wins"] = alt["wins"]
mt3.loc[r1m, "drv_pts_share"] = alt["points"] / alt["prev"].map(lead).replace(0, np.nan)
def score_table(m):
    t, v, e = (m[m.split == s] for s in ("train", "val", "test"))
    pp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(t[SELECTED].astype(float))
    X = [pp.transform(d[SELECTED].astype(float)) for d in (t, v, e)]
    a, b = LR().fit(X[0], t.podium), RF().fit(X[0], t.podium)
    return {"LR val": AP(v.podium, a.predict_proba(X[1])[:, 1]), "LR test": AP(e.podium, a.predict_proba(X[2])[:, 1]),
            "RF val": AP(v.podium, b.predict_proba(X[1])[:, 1]), "RF test": AP(e.podium, b.predict_proba(X[2])[:, 1])}
alt_cmp = pd.DataFrame({"median fill (used)": score_table(mt), "round 1 = last season's final standings": score_table(mt3)})
alt_cmp["change"] = alt_cmp.iloc[:, 1] - alt_cmp.iloc[:, 0]
print(alt_cmp.round(4).to_string())

# %% [markdown]
# ## V9: does the selected feature set depend on the order of the groups? (all 120 orders, plus backward removal, plus noise level)

# %%
cache = {}
def val_scores_for(cols):
    key = tuple(sorted(cols))
    if key not in cache:
        (a, b, _), _ = prep(list(cols))
        cache[key] = (AP(va.podium, LR().fit(a, tr.podium).predict_proba(b)[:, 1]),
                      AP(va.podium, RF().fit(a, tr.podium).predict_proba(b)[:, 1]))
    return cache[key]
def ladder(order, gain=0.005):
    kept, ks = [], None
    for g in order:
        s = val_scores_for(kept + GROUPS[g])
        if ks is None or all(s[i] - ks[i] >= gain for i in range(2)):
            kept, ks = kept + GROUPS[g], s
            kept_names = None
    return kept
names = list(GROUPS)
ordering = {}
for perm in itertools.permutations(names):
    k = tuple(sorted(ladder(list(perm))))
    ordering.setdefault(k, []).append(perm)
print("distinct selected sets over", sum(len(v) for v in ordering.values()), "orders:", len(ordering))
for k, v in sorted(ordering.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):3} orders -> {list(k)}   e.g. {v[0]}")
orig = tuple(sorted(ladder(names)))
print("notebook order selects:", list(orig))
# backward removal from all candidates
cur = [c for g in GROUPS.values() for c in g]; groups_in = list(GROUPS)
base = val_scores_for(cur)
changed = True
while changed and len(groups_in) > 1:
    changed = False
    drops = []
    for g in groups_in:
        rest = [c for gg in groups_in if gg != g for c in GROUPS[gg]]
        s = val_scores_for(rest)
        drops.append((g, s, all(s[i] >= base[i] - 0.005 for i in range(2))))
    ok = [d for d in drops if d[2]]
    if ok:
        g, s, _ = max(ok, key=lambda d: sum(d[1]))
        groups_in.remove(g); base = s; changed = True
        print("backward: removed", g, "-> LR %.4f RF %.4f" % s)
print("backward elimination keeps groups:", groups_in)
print("all-11 vs selected val PR-AUC:", np.round(val_scores_for([c for g in GROUPS.values() for c in g]), 4), np.round(val_scores_for(SELECTED), 4))
# noise level for MIN_GAIN: bootstrap whole validation races
(Xa, Xb, _), _ = prep(["grid"]); (Xc, Xd, _), _ = prep(SELECTED)
pA = LR().fit(Xa, tr.podium).predict_proba(Xb)[:, 1]; pB = lr.predict_proba(Xva)[:, 1]
rng = np.random.default_rng(0); idx = [g.index.to_numpy() for _, g in va.reset_index(drop=True).groupby("raceId")]
yv = va.podium.to_numpy(); d = []
for _ in range(500):
    r = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]); d.append(AP(yv[r], pB[r]) - AP(yv[r], pA[r]))
print("bootstrap SD of a PR-AUC difference between two LR feature sets on validation: %.4f" % np.std(d), "(MIN_GAIN is 0.005)")
d2 = []
for _ in range(500):
    r = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]); d2.append(AP(yv[r], pA[r]))
print("bootstrap SD of one PR-AUC itself: %.4f" % np.std(d2))

# %% [markdown]
# ## V10: Q2 home effect

# %%
drv, circ = read("drivers"), read("circuits")
print("driver nationalities:", drv.nationality.nunique(), "| circuit countries:", circ.country.nunique())
nb = json.load(open(Path("../notebooks/f1_podium_prediction.ipynb") if Path("../notebooks").exists() else Path("notebooks/f1_podium_prediction.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
m = re.search(r"NATIONALITY_TO_COUNTRY = (\{.*?\n\})", src, re.S)
N2C = eval(m.group(1)); FIX = eval(re.search(r"NATIONALITY_FIX = (\{.*?\})", src, re.S).group(1))
nat = drv.nationality.str.strip().replace(FIX)
unmapped = sorted(set(nat) - set(N2C)); print("nationalities with no mapping:", unmapped)
cc = set(circ.country.str.strip()); no_circuit = sorted({v for v in N2C.values() if v not in cc})
print("mapped countries that never host a race (no home race possible):", no_circuit)
print("circuit countries nobody maps to (before the United States -> USA fix):", sorted(cc - set(N2C.values())))
s = mt.merge(drv[["driverId", "nationality"]], on="driverId").merge(races[["raceId", "circuitId"]], on="raceId").merge(circ[["circuitId", "country"]], on="circuitId")
s["country"] = s.country.str.strip().replace({"United States": "USA"})   # same spelling fix as notebook c35
s["home_country"] = s.nationality.str.strip().replace(FIX).map(N2C)
s["home"] = (s.home_country == s.country.str.strip()).astype(int)
s["start"] = pd.cut(s.grid, [0, 1, 3, 6, 10, 100], labels=["pole", "2-3", "4-6", "7-10", "11+"])
def gap(d):
    t = d.groupby(["start", "home"], observed=True).podium.agg(["sum", "count"]).unstack("home")
    w = t["count"][1]; diff = t["sum"][1] / t["count"][1] - t["sum"][0] / t["count"][0]
    return (diff * w).sum() / w.sum()
print("rows", len(s), "home share", round(s.home.mean(), 3), "| weighted gap (pp):", round(gap(s) * 100, 2), "| notebook -0.70")
idx = [g.index.to_numpy() for _, g in s.groupby("raceId")]; rng = np.random.default_rng(1)
bs = [gap(s.loc[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])]) for _ in range(500)]
print("race-bootstrap 95%% range of the gap (pp): %.2f to %.2f" % tuple(np.percentile(bs, [2.5, 97.5]) * 100))
s["era"] = pd.cut(s.year, [1949, 1979, 2013, 2024], labels=["pre-1980", "1980-2013", "2014+"])
X = pd.get_dummies(s[["home", "start", "era"]], drop_first=True)
print("column 0 is", X.columns[0], "| odds ratio", round(float(np.exp(LogisticRegression(C=1e5, max_iter=1000).fit(X, s.podium).coef_[0][0])), 3), "| notebook 0.915")
print(s[s.home == 1].groupby("home_country").size().sort_values(ascending=False).head(8).to_dict())

# %% [markdown]
# ## V11: Q3 mechanical-retirement taxonomy and sensitivity

# %%
G = {"accident": {"Accident", "Collision", "Collision damage", "Spun off", "Fatal accident", "Damage", "Debris"},
     "driver": {"Driver unwell", "Illness", "Injured", "Injury", "Eye injury", "Physical", "Stalled"},
     "regulatory": {"Disqualified", "Excluded", "Underweight", "107% Rule", "Did not qualify", "Did not prequalify", "Withdrew", "Not restarted", "Safety concerns", "Safety", "Seat", "Driver Seat", "Not classified"},
     "other": {"Refuelling"}}
def cls(x):
    if x == "Finished" or x.startswith("+"): return "finished"
    return next((k for k, v in G.items() if x in v), "mechanical")
AMBIG = {"Retired", "Puncture", "Tyre", "Tyre puncture", "Wheel", "Wheel nut", "Wheel rim", "Out of fuel", "Fuel", "Fire", "Safety belt", "Front wing", "Rear wing", "Broken wing", "Undertray", "Brake duct", "Handling", "Vibrations", "Chassis", "Technical", "Mechanical", "Axle"}
q = res3.merge(races[["raceId", "year"]], on="raceId")
q = q.merge(read("constructors")[["constructorId", "name"]].rename(columns={"name": "team"}), on="constructorId")
q = q[(q.year.between(2014, 2024)) & ~((q.grid == 0) & (q.laps == 0))].copy()
q["cls"] = q.status.map(cls); q["period"] = np.where(q.year <= 2021, "2014-2021", "2022-2024")
q["mech_broad"] = q.cls == "mechanical"; q["mech_strict"] = q.mech_broad & ~q.status.isin(AMBIG)
print("mechanical statuses 2014-2024:"); print(q[q.mech_broad].status.value_counts().head(25).to_string())
print("share of 'mechanical' starts whose status is ambiguous:", round((q.mech_broad & q.status.isin(AMBIG)).sum() / q.mech_broad.sum(), 3))
for col in ["mech_broad", "mech_strict"]:
    print(col, q.groupby("period")[col].mean().round(4).to_dict())
tt = q.groupby(["period", "team"]).agg(n=("mech_broad", "size"), broad=("mech_broad", "mean"), strict=("mech_strict", "mean")).reset_index()
for per in ["2014-2021", "2022-2024"]:
    d = tt[tt.period == per]
    print(per, "top-3 broad:", d.nlargest(3, "broad")[["team", "n", "broad"]].round(3).values.tolist())
    print(per, "top-3 strict:", d.nlargest(3, "strict")[["team", "n", "strict"]].round(3).values.tolist())
    print(per, "rank correlation broad vs strict:", round(d.broad.corr(d.strict, method="spearman"), 3))
print("teams by name in 2022-2024:", sorted(tt[tt.period == "2022-2024"].team))

# %% [markdown]
# ## V12: Q1 recomputation, sensitivity and the Indy rows

# %%
def q1_table(data, boundary=2014):
    d = data[data.grid.isin([1, 2])].copy()
    d["era"] = np.where(d.year >= boundary, "late", "early")
    return d
base_rows = res3.merge(races[["raceId", "year", "circuitId"]], on="raceId")
base_rows = base_rows.merge(read("circuits")[["circuitId", "name"]].rename(columns={"name": "circuit"}), on="circuitId")
dd = base_rows.sort_values(["raceId", "driverId"]).copy()
dd["_slot"] = dd.grid.where(dd.grid > 0, 10**6)
dd = dd.sort_values(["raceId", "driverId", "_slot", "positionOrder"]).groupby(["raceId", "driverId"]).head(1)
dd["podium"] = dd.positionOrder <= 3
dd["indy"] = dd.circuit.str.contains("Indianapolis") & (dd.year <= 1960)
for label, data in [("as in the notebook (Indy rows kept)", dd), ("Indy rows removed", dd[~dd.indy])]:
    t = q1_table(data); o = t.groupby("era").podium.agg(["sum", "count"]); o["rate"] = o["sum"] / o["count"]
    print(label, o.round(3).to_dict("index"))
t = q1_table(dd[~dd.indy])
cnt = t.groupby(["circuit", "era"]).podium.agg(["sum", "count"]).unstack("era").fillna(0)
worse = cnt["count"].min(axis=1)
print("circuits ranked for K = 5,10,15,20:", {k: int((worse >= k).sum()) for k in (5, 10, 15, 20)}, "| notebook 21/16/13/7")
def rho_for(k, boundary=2014, d=None):
    tt = q1_table(dd[~dd.indy], boundary); c = tt.groupby(["circuit", "era"]).podium.agg(["sum", "count"]).unstack("era").fillna(0)
    ok = c[c["count"].min(axis=1) >= k]; r = ok["sum"] / ok["count"]
    return len(ok), r["early"].corr(r["late"], method="spearman")
for k in (5, 10, 15):
    n, r = rho_for(k); print(f"K_MIN={k}: {n} circuits, Spearman {r:.2f}")
for b in (2012, 2014, 2016, 2018):
    tt = q1_table(dd[~dd.indy], b).groupby("era").podium.agg(["sum", "count"]); print("boundary", b, (tt["sum"] / tt["count"]).round(3).to_dict())
n, r = rho_for(10); rng = np.random.default_rng(3)
ok = cnt[cnt["count"].min(axis=1) >= 10]; rates = pd.DataFrame({"e": ok["sum"]["early"] / ok["count"]["early"], "l": ok["sum"]["late"] / ok["count"]["late"]})
bs = [rates.iloc[rng.integers(0, len(rates), len(rates))].corr(method="spearman").iloc[0, 1] for _ in range(1000)]
print(f"bootstrap 95% range of the Spearman over circuits (n={n}): {np.nanpercentile(bs, 2.5):.2f} to {np.nanpercentile(bs, 97.5):.2f}")

# %% [markdown]
# ## V15/V16: leakage on every row, standings semantics, pit-lane ties

# %%
# (a) Perturbation test: change the podium outcome of ONE race; features of that race must not move. Done for every race.
def features_only(m):
    out = m.copy()
    for key, col in [("driverId", "drv_podium_last5"), ("constructorId", "con_podium_last5")]:
        per = out.groupby([key, "race_idx"])["podium"].mean()
        vals = {}
        for k, s_ in per.groupby(level=0):
            s_ = s_.droplevel(0); v = s_.to_numpy()
            for j, r in enumerate(s_.index): vals[(k, r)] = v[max(0, j - 5):j].mean() if j else np.nan
        out[col] = [vals[(k, r)] for k, r in zip(out[key], out.race_idx)]
    return out
cols = ["drv_podium_last5", "con_podium_last5"]
rng = np.random.default_rng(7); bad = 0; tested = 0
races_in = mt.race_idx.unique(); sample = rng.choice(races_in, 120, replace=False)
for r in sample:
    m2 = mt.copy(); m2.loc[m2.race_idx == r, "podium"] = 1 - m2.loc[m2.race_idx == r, "podium"]
    f2 = features_only(m2)
    same = np.allclose(f2.loc[f2.race_idx == r, cols].to_numpy(float), mt.loc[mt.race_idx == r, cols].to_numpy(float), equal_nan=True)
    tested += 1; bad += (not same)
print(f"races whose own outcome changed their own form features: {bad} of {tested} tested (0 expected)")
later = [r for r in sample[:20]]
m2 = mt.copy(); r = later[0]; m2.loc[m2.race_idx == r, "podium"] = 1 - m2.loc[m2.race_idx == r, "podium"]
f2 = features_only(m2); chg = (~np.isclose(f2[cols].to_numpy(float), mt[cols].to_numpy(float), equal_nan=True)).any(axis=1)
print("sanity: the same change DOES alter later races' features:", int(chg.sum()), "rows, all later:", bool((mt.race_idx[chg] > r).all()))
# (b) What do the standings contain? Compare standings at a race with points summed from results up to and including that race.
ds = read("driver_standings"); rs = read("results").merge(races[["raceId", "year", "round"]], on="raceId")
spr = read("sprint_results").merge(races[["raceId", "year", "round"]], on="raceId")
chk = []
for y in (1995, 2005, 2015, 2019, 2023):
    ry = rs[rs.year == y].sort_values("round"); rounds = sorted(ry["round"].unique())
    for rd in ([rounds[0], rounds[len(rounds) // 2], rounds[-1]] if len(rounds) > 2 else rounds):
        rid = races[(races.year == y) & (races["round"] == rd)].raceId.iloc[0]
        cum_incl = ry[ry["round"] <= rd].groupby("driverId").points.sum()
        cum_prev = ry[ry["round"] < rd].groupby("driverId").points.sum()
        if y >= 2021:   # standings also count sprint points
            sp_ = spr[spr.year == y]
            cum_incl = cum_incl.add(sp_[sp_["round"] <= rd].groupby("driverId").points.sum(), fill_value=0)
            cum_prev = cum_prev.add(sp_[sp_["round"] < rd].groupby("driverId").points.sum(), fill_value=0)
        st = ds[ds.raceId == rid].set_index("driverId").points
        chk.append((y, rd, float(np.isclose(st.reindex(cum_incl.index).fillna(0), cum_incl).mean()), float(np.isclose(st.reindex(cum_prev.index).fillna(0), cum_prev).mean()) if len(cum_prev) else np.nan))
print(pd.DataFrame(chk, columns=["year", "round", "share equal to points INCLUDING this race", "share equal to points BEFORE this race"]).round(2).to_string(index=False))
print("-> standings at raceId r already include race r, so joining on the previous round (prev) is the correct, leakage-safe choice")
print("duplicates in driver_standings (raceId, driverId):", int(ds.duplicated(["raceId", "driverId"]).sum()))
# (c) pit-lane starters per race
pit = res.merge(races[["raceId", "year"]], on="raceId"); pit = pit[(pit.grid == 0) & (pit.laps > 0)]
c = pit.groupby("raceId").size()
print("races with a pit-lane starter:", len(c), "| with 2 or more:", int((c > 1).sum()), "-> those starters all get the same grid_eff (ties)")

# %% [markdown]
# ## V19: thresholds

# %%
for name, model, X, d in [("LR", lr, Xva, va), ("RF", rf, Xva, va)]:
    p = model.predict_proba(X)[:, 1]; pr, rc, th = precision_recall_curve(d.podium, p)
    f = 2 * pr * rc / (pr + rc + 1e-12); i = f[:-1].argmax()
    print(f"{name}: threshold {th[i]:.3f}  precision {pr[i]:.3f}  recall {rc[i]:.3f}  F1 {f[i]:.3f}  predicted-positive share {np.mean(p >= th[i]):.3f}  (true rate {d.podium.mean():.3f})")
    for lo, hi in [(th[i] - 0.1, th[i] + 0.1)]:
        j = [k for k in range(len(th)) if lo <= th[k] <= hi]; print("   F1 range for thresholds within +-0.1:", round(f[j].min(), 3), "to", round(f[j].max(), 3))

# %% [markdown]
# ## V21: causal wording

# %%
md = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
pat = re.compile(r"[^.\n]*\b(causes?|caused|because|leads? to|drives?|makes? a driver|produces?|due to|result(s|ed)? in)\b[^.\n]*", re.I)
for i, t in md:
    if i < 93: continue
    for m_ in pat.finditer(t): print(f"c{i}:", m_.group(0).strip()[:200])
tex = Path("../report/main.tex" if Path("../report").exists() else "report/main.tex").read_text()
print("--- report")
for m_ in pat.finditer(tex[tex.find("Explainab"):]): print(m_.group(0).strip()[:200])
print("non-causal statement in notebook:", "not causes" in "".join(t for _, t in md).lower() or "not causal" in "".join(t for _, t in md).lower(), "| in report:", "not causal" in tex.lower())

# %% [markdown]
# ## V18: numbers quoted in the markdown vs the cell outputs

# %%
def outputs_text(c):
    out = []
    for o in c.get("outputs", []):
        if "text" in o: out.append("".join(o["text"]))
        elif "data" in o and "text/plain" in o["data"]: out.append("".join(o["data"]["text/plain"]))
    return "\n".join(out)
pool_txt = "\n".join(outputs_text(c) for c in nb["cells"] if c["cell_type"] == "code")
pool = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", pool_txt.replace(",", ""))]
pool = np.abs(np.array(pool))
def in_pool(v, decimals):
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return bool((np.abs(pool - v) <= tol).any() or (np.abs(pool * 100 - v) <= tol).any() or (np.abs(pool / 100 - v) <= tol).any())
flag = []
for i, t in md:
    t0 = re.sub(r"`[^`]*`", " ", t)                         # skip code spans
    for m_ in re.finditer(r"(?<![\w.#-])(-?\d[\d,]*(?:\.\d+)?)(%?)", t0):
        s_ = m_.group(1).replace(",", ""); v = abs(float(s_))
        if v in (0, 1, 2, 3, 4, 5, 10, 14, 16, 18, 15): continue
        if 1949 < v < 2030 and "." not in s_: continue
        dec = len(s_.split(".")[1]) if "." in s_ else 0
        if not in_pool(v, dec): flag.append((i, m_.group(0), t0[max(0, m_.start() - 60):m_.end() + 30].replace("\n", " ")))
print(len(flag), "numbers in markdown not found in any output (manual review needed):")
for i, tok, ctx in flag: print(f"  c{i}: {tok:>8}  ...{ctx}")

# %% [markdown]
# ## Claim checks: Q1 'best circuits', c84 'within 0.03'

# %%
t1 = q1_table(dd[~dd.indy]); cnt = t1.groupby(["circuit", "era"]).podium.agg(["sum", "count"]).unstack("era").fillna(0)
ok = cnt[cnt["count"].min(axis=1) >= 10]; late = (ok["sum"]["late"] / ok["count"]["late"]).sort_values(ascending=False)
print("best circuits 2014+ (notebook says Suzuka, Yas Marina, Monaco):", late.head(4).round(3).to_dict())
print("worst circuits 2014+:", late.tail(3).round(3).to_dict())
rows = []
for name, model in [("LR", lr), ("RF", rf)]:
    pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    pr, rc, th = precision_recall_curve(va.podium, pv); f = 2 * pr * rc / (pr + rc + 1e-12); thr = th[f[:-1].argmax()]
    from sklearn.metrics import f1_score, recall_score, precision_score
    for met, fn in [("PR-AUC", lambda y, p: AP(y, p)), ("F1", lambda y, p: f1_score(y, p >= thr)), ("Recall", lambda y, p: recall_score(y, p >= thr)), ("Precision", lambda y, p: precision_score(y, p >= thr))]:
        rows.append((name, met, fn(va.podium, pv), fn(te.podium, pt)))
cmp_ = pd.DataFrame(rows, columns=["model", "metric", "val", "test"]); cmp_["val - test"] = cmp_.val - cmp_.test
print(cmp_.round(3).to_string(index=False))
print("c84 says test scores are within 0.03 of validation: true for PR-AUC only; F1/recall drops are larger for the forest")
