"""Independent rebuild of the modeling table from the raw CSVs. Deliberately NOT copied from the notebook:
different code paths (groupby-idxmin instead of sort+drop_duplicates, explicit loops for form, pandas na_values for sentinels)."""
import numpy as np, pandas as pd
from pathlib import Path
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
def read(t): return pd.read_csv(RAW / f"{t}.csv", na_values=r"\N", keep_default_na=False)

def build(nonstarter_rule="grid0_laps0", with_form=True):
    res, races, status = read("results"), read("races"), read("status")
    res = res.merge(races[["raceId", "year", "round", "name"]], on="raceId").merge(status, on="statusId")
    res = res.reset_index(drop=True)
    # --- one row per (raceId, driverId): the row with the lowest real grid slot, ties by best positionOrder
    res["_slot"] = res["grid"].where(res["grid"] > 0, 10**6)
    keep = res.sort_values(["raceId", "driverId", "_slot", "positionOrder"]).groupby(["raceId", "driverId"]).head(1).index
    res = res.loc[sorted(keep)].copy()                    # sorted index == original file order
    # --- flags
    if nonstarter_rule == "grid0_laps0":
        res["nonstarter"] = (res["grid"] == 0) & (res["laps"] == 0)
    elif nonstarter_rule == "plus_clear_cases":
        clear = res["status"].isin(["Withdrew", "Did not qualify", "Did not prequalify", "107% Rule"]) & (res["laps"] == 0)
        res["nonstarter"] = ((res["grid"] == 0) & (res["laps"] == 0)) | clear
    res["pit"] = (res["grid"] == 0) & (res["laps"] > 0)
    res["indy"] = res["name"].str.contains("Indianapolis") & (res["year"] <= 1960)
    mt = res[~res["nonstarter"] & ~res["indy"]].copy()
    mt["podium"] = (mt["positionOrder"] <= 3).astype(int)
    max_grid = res.groupby("raceId")["grid"].max()          # over all rows of the race, like the notebook
    mt["grid_eff"] = np.where(mt["grid"] > 0, mt["grid"], mt["raceId"].map(max_grid) + 1)
    mt["field_size"] = mt.groupby("raceId")["raceId"].transform("count")
    mt["grid_rel"] = mt["grid_eff"] / mt["field_size"]
    mt["split"] = np.select([mt.year <= 2017, mt.year <= 2021], ["train", "val"], "test")
    # --- previous round of the same season
    rc = races.sort_values(["year", "round"]).reset_index(drop=True)
    rc["race_idx"] = rc.index
    prev = {}
    for y, g in rc.groupby("year"):
        ids = g["raceId"].tolist()
        for a, b in zip(ids[:-1], ids[1:]): prev[b] = a
    mt["prev"] = mt["raceId"].map(prev)
    mt["race_idx"] = mt["raceId"].map(rc.set_index("raceId")["race_idx"])
    def standings(tbl, key, p):
        s = read(tbl)
        lead = s.groupby("raceId")["points"].transform("max")
        s[p + "_pts_share"] = np.where(lead > 0, s["points"] / lead.replace(0, np.nan), 0.0)
        s = s.rename(columns={"position": p + "_pos", "wins": p + "_wins", "raceId": "prev"})
        return s[["prev", key, p + "_pos", p + "_pts_share", p + "_wins"]]
    mt = mt.merge(standings("driver_standings", "driverId", "drv"), on=["prev", "driverId"], how="left")
    mt = mt.merge(standings("constructor_standings", "constructorId", "con"), on=["prev", "constructorId"], how="left")
    if with_form:
        for key, col in [("driverId", "drv_podium_last5"), ("constructorId", "con_podium_last5")]:
            per = mt.groupby([key, "race_idx"])["podium"].mean()      # one value per entity and race
            out = {}
            for k, s in per.groupby(level=0):
                vals = s.droplevel(0)
                idx, v = vals.index.to_numpy(), vals.to_numpy()
                for j, r in enumerate(idx):
                    out[(k, r)] = v[max(0, j - 5):j].mean() if j > 0 else np.nan
            mt[col] = [out[(k, r)] for k, r in zip(mt[key], mt["race_idx"])]
    mt["grid"] = mt["grid_eff"]
    return mt.reset_index(drop=True)

GROUPS = {"grid": ["grid"], "field size": ["field_size", "grid_rel"],
          "driver standings": ["drv_pos", "drv_pts_share", "drv_wins"],
          "team standings": ["con_pos", "con_pts_share", "con_wins"],
          "recent form": ["drv_podium_last5", "con_podium_last5"]}
SELECTED = ["grid", "drv_pos", "drv_pts_share", "drv_wins"]
