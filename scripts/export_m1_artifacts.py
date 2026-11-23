"""Export the trained Milestone 1 podium prediction models, preprocessing pipeline,
SHAP/LIME background data, and leakage-free pre-race feature lookup table to disk.

Why this script exists:
Milestone 3 requires:
- "Do not retrain the podium model inside the application. Load the saved model and its
  preprocessing from file."
- "If your pre-race features need data that is not in that schema (for example standings),
  do not extend the graph for it: compute those features from the Milestone 1 CSVs with
  the same leakage rules as Milestone 1."

This script runs the exact Milestone 1 pipeline (from `notebooks/f1_podium_prediction.ipynb`)
once and writes:
1. `artifacts/m1_podium_bundle.joblib`
2. `artifacts/pre_race_features.csv`
"""

from __future__ import annotations

import html
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["grid", "drv_pos", "drv_pts_share", "drv_wins"]

FEATURE_SPECS = {
    "grid": {
        "name": "grid",
        "label": "Starting Grid Position (grid)",
        "input_type": "integer",
        "unit": "grid slot (1 = pole position)",
        "min": 1,
        "max": 34,
        "default": 12,
        "help_text": (
            "Starting slot on the Grand Prix grid after qualifying and penalties (1 = pole position; "
            "pit-lane starts are placed behind the last car on the grid). Data range in 1950–2024: 1 to 34."
        ),
        "justification": "Minimum starting slot is 1 (pole); maximum field size in championship F1 races (excluding Indy 500) is 34 cars.",
    },
    "drv_pos": {
        "name": "drv_pos",
        "label": "Driver Championship Position Before Race (drv_pos)",
        "input_type": "integer",
        "unit": "championship rank (1 = leader)",
        "min": 1,
        "max": 108,
        "default": 12,
        "help_text": (
            "Driver's World Championship standing after the previous round (round - 1). "
            "For Round 1 or season debuts, the training median (12) is used in chat mode."
        ),
        "justification": "Championship standings positions after round - 1 span 1 (leader) to 108 across 1950–2024 (1 to 102 in train 1950–2017).",
    },
    "drv_pts_share": {
        "name": "drv_pts_share",
        "label": "Share of Championship Leader's Points Before Race (drv_pts_share)",
        "input_type": "float",
        "unit": "ratio (0.0 to 1.0)",
        "min": 0.0,
        "max": 1.0,
        "default": 0.076,
        "help_text": (
            "Driver's championship points after round - 1 divided by the championship leader's points "
            "(1.0 = championship leader, 0.0 = zero points). Training median is 0.076."
        ),
        "justification": "Normalized ratio points / max(points) at round - 1 is mathematically bounded in [0.0, 1.0] across all scoring eras.",
    },
    "drv_wins": {
        "name": "drv_wins",
        "label": "Season Grand Prix Wins Before Race (drv_wins)",
        "input_type": "integer",
        "unit": "wins in current season prior to race",
        "min": 0,
        "max": 24,
        "default": 0,
        "help_text": (
            "Number of Grand Prix wins accumulated by the driver in the current season up to round - 1 "
            "(0 at Round 1; training median is 0)."
        ),
        "justification": "A driver cannot have negative wins, and a season has at most 24 rounds (max observed prior wins is 18 in 2023; train max was 13).",
    },
}


def best_f1_threshold(y: pd.Series | np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
    return float(thr[f1s.argmax()])


def build_m1_modeling_table(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Reproduce the exact Milestone 1 cleaning and leakage-free pre-race feature table."""
    tables = [
        "circuits", "constructors", " drivers".strip(), "driver_standings",
        "constructor_standings", "races", "results", "status",
    ]
    data = {t: pd.read_csv(raw_dir / f"{t}.csv") for t in tables}

    clean: dict[str, pd.DataFrame] = {}
    for t, df in data.items():
        df = df.replace(r"\N", np.nan)
        for c in df.select_dtypes(exclude="number").columns:
            df[c] = df[c].str.strip().map(html.unescape, na_action="ignore")
            as_number = pd.to_numeric(df[c], errors="coerce")
            if as_number.notna().sum() == df[c].notna().sum():
                df[c] = as_number
        clean[t] = df

    clean["races"]["date_str"] = clean["races"]["date"].astype(str)
    clean["races"]["date"] = pd.to_datetime(clean["races"]["date"])

    # Dedup shared drives: keep lowest real grid slot first (grid 0 sorted last), then best positionOrder
    key = ["raceId", "driverId"]
    r = clean["results"]
    r = r.assign(slot=r["grid"].replace(0, np.nan)).sort_values(["slot", "positionOrder"], na_position="last")
    r = r.drop_duplicates(key).drop(columns="slot").sort_index()

    r = r.merge(clean["races"][["raceId", "year", "round", "name", "circuitId", "date_str"]], on="raceId")
    status_name = r["statusId"].map(clean["status"].set_index("statusId")["status"])
    r["status_text"] = status_name
    r["not_qualified"] = status_name.isin(["Did not qualify", "Did not prequalify"])
    r["pitlane_start"] = (r["grid"] == 0) & (r["laps"] > 0)
    last_slot = r["grid"].where(~r["not_qualified"]).groupby(r["raceId"]).transform("max")
    r["grid_eff"] = r["grid"].where(r["grid"] > 0, last_slot + 1)
    r["is_indy500"] = r["name"].eq("Indianapolis 500")

    mt = r.query("~not_qualified and ~is_indy500").copy()
    mt["podium"] = (mt["positionOrder"] <= 3).astype(int)
    mt["grid"] = mt["grid_eff"]
    mt["field_size"] = mt.groupby("raceId")["raceId"].transform("size")
    mt["grid_rel"] = mt["grid"] / mt["field_size"]
    mt["split"] = pd.cut(mt["year"], [1949, 2017, 2021, 2024], labels=["train", "val", "test"])

    rc = clean["races"].sort_values(["year", "round"]).reset_index(drop=True)
    rc["race_idx"] = rc.index
    rc["prev_raceId"] = rc.groupby("year")["raceId"].shift(1).astype("Int64")
    mt = mt.merge(rc[["raceId", "race_idx", "prev_raceId"]], on="raceId", validate="many_to_one")

    def standings_before(table: str, entity_key: str, prefix: str) -> pd.DataFrame:
        s = clean[table].rename(
            columns={"raceId": "prev_raceId", "position": f"{prefix}_pos", "wins": f"{prefix}_wins"}
        )
        leader = s.groupby("prev_raceId")["points"].transform("max")
        s[f"{prefix}_pts_share"] = (s["points"] / leader).where(leader > 0, 0.0)
        return s[["prev_raceId", entity_key, f"{prefix}_pos", f"{prefix}_pts_share", f"{prefix}_wins"]]

    mt = mt.merge(
        standings_before("driver_standings", "driverId", "drv"),
        on=["prev_raceId", "driverId"],
        how="left",
        validate="many_to_one",
    )

    # Add human-readable entity names for lookup
    drv_names = (
        clean["drivers"]
        .assign(driver_name=clean["drivers"]["forename"] + " " + clean["drivers"]["surname"])
        [["driverId", "driver_name", "surname", "code"]]
    )
    con_names = clean["constructors"][["constructorId", "name"]].rename(columns={"name": "constructor_name"})
    cir_names = clean["circuits"][["circuitId", "name"]].rename(columns={"name": "circuit_name"})

    mt = (
        mt.merge(drv_names, on="driverId", how="left")
        .merge(con_names, on="constructorId", how="left")
        .merge(cir_names, on="circuitId", how="left")
    )
    return mt, clean


def main() -> None:
    np.random.seed(SEED)
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    mt, _ = build_m1_modeling_table(raw_dir)
    tr, va, te = (mt[mt["split"] == s] for s in ["train", "val", "test"])
    y_train, y_val = tr["podium"], va["podium"]

    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    prep.fit(tr[FEATURES].astype(float))
    names = prep.get_feature_names_out()

    X_train = pd.DataFrame(prep.transform(tr[FEATURES].astype(float)), columns=names, index=tr.index)
    X_val = pd.DataFrame(prep.transform(va[FEATURES].astype(float)), columns=names, index=va.index)
    X_test = pd.DataFrame(prep.transform(te[FEATURES].astype(float)), columns=names, index=te.index)

    # Fit exact Milestone 1 models & select best hyperparameters on validation PR-AUC
    lr_best = LogisticRegression(max_iter=2000, C=10, class_weight=None, random_state=SEED).fit(X_train, y_train)
    rf_best = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, class_weight=None, random_state=SEED, n_jobs=1
    ).fit(X_train, y_train)
    base_lr = LogisticRegression(max_iter=2000).fit(X_train[["grid"]], y_train)

    thresholds = {
        "Random forest": best_f1_threshold(y_val, rf_best.predict_proba(X_val)[:, 1]),
        "Logistic regression": best_f1_threshold(y_val, lr_best.predict_proba(X_val)[:, 1]),
        "Baseline (LR on grid)": best_f1_threshold(y_val, base_lr.predict_proba(X_val[["grid"]])[:, 1]),
    }
    medians = dict(zip(FEATURES, [float(v) for v in prep[0].statistics_]))

    # Compute global SHAP importance on the 300 validation rows (exact Milestone 1 setup)
    background = X_train.sample(100, random_state=SEED)
    X_shap = X_val.sample(300, random_state=SEED)
    tree_exp = shap.TreeExplainer(rf_best, data=background)
    sv_tree = tree_exp(X_shap)
    sv_rf_vals = sv_tree[..., 1].values if sv_tree.values.ndim == 3 else sv_tree.values
    rf_base_val = float(np.mean(sv_tree[..., 1].base_values if sv_tree.values.ndim == 3 else sv_tree.base_values))

    lr_exp = shap.explainers.Permutation(
        lambda A: lr_best.predict_proba(pd.DataFrame(A, columns=FEATURES))[:, 1],
        background,
        seed=SEED,
    )
    sv_lr = lr_exp(X_shap, max_evals=100, silent=True)

    global_shap = {
        "Random forest": {
            "mean_abs_shap": dict(zip(FEATURES, [round(float(x), 4) for x in np.abs(sv_rf_vals).mean(axis=0)])),
            "base_value": round(rf_base_val, 4),
            "summary": (
                "Across 300 validation entries in Milestone 1, starting grid slot (`grid`) is the dominant "
                "feature for Random Forest (mean |SHAP| = 0.1055, or ~10.6 percentage points of podium probability), "
                "followed by prior driver championship rank (`drv_pos`, 0.0316), share of the leader's points "
                "(`drv_pts_share`, 0.0252), and season wins before the race (`drv_wins`, 0.0068), relative to an "
                "average base podium probability of 0.1418."
            ),
        },
        "Logistic regression": {
            "mean_abs_shap": dict(zip(FEATURES, [round(float(x), 4) for x in np.abs(sv_lr.values).mean(axis=0)])),
            "base_value": round(float(np.mean(sv_lr.base_values)), 4),
            "summary": (
                "For Logistic Regression, `grid` has the largest global effect (mean |SHAP| = 0.1174), "
                "followed by `drv_pts_share` (0.0297), `drv_pos` (0.0097), and `drv_wins` (0.0074), relative "
                "to a base podium probability of 0.1398."
            ),
        },
    }

    # Raw (imputed, unscaled) training array for LimeTabularExplainer
    X_train_raw = pd.DataFrame(prep[-1].inverse_transform(X_train), columns=FEATURES, index=X_train.index)

    bundle = {
        "seed": SEED,
        "features": FEATURES,
        "feature_specs": FEATURE_SPECS,
        "preprocessing": prep,
        "medians": medians,
        "models": {
            "Random forest": rf_best,
            "Logistic regression": lr_best,
            "Baseline (LR on grid)": base_lr,
        },
        "default_model": "Random forest",
        "thresholds": thresholds,
        "background_scaled": background,
        "X_train_raw_values": X_train_raw.to_numpy(dtype=float),
        "global_shap_importance": global_shap,
    }

    bundle_path = out_dir / "m1_podium_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    export_cols = [
        "raceId", "year", "round", "name", "date_str",
        "driverId", "driver_name", "surname", "code",
        "constructorId", "constructor_name",
        "circuitId", "circuit_name",
        "grid", "drv_pos", "drv_pts_share", "drv_wins",
        "positionOrder", "podium", "status_text", "split",
    ]
    pre_race_path = out_dir / "pre_race_features.csv"
    mt[export_cols].sort_values(["year", "round", "positionOrder"]).to_csv(pre_race_path, index=False)

    print(f"Saved model bundle to: {bundle_path}")
    print(f"Saved pre-race feature table ({len(mt):,} rows) to: {pre_race_path}")
    print("Thresholds:", {k: round(v, 3) for k, v in thresholds.items()})
    print("Training medians:", {k: round(v, 3) for k, v in medians.items()})
    print("Global SHAP (RF):", global_shap["Random forest"]["mean_abs_shap"])


if __name__ == "__main__":
    main()
