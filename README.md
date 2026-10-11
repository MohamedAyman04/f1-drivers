# F1 Podium Prediction Companion — CSEN903 Milestone 1

Predict whether a driver finishes on the podium (`positionOrder ≤ 3`) using **only
information available after qualifying and before the race start**.

## Task framing

| Item | Value |
|---|---|
| Unit of analysis | One driver entering one race — `(raceId, driverId)` |
| Target | `podium = positionOrder ≤ 3` (binary, ~12.7% positive) |
| Prediction cutoff | After qualifying, before lights out (no race outcomes as features) |
| Modeling window | 1950–2024, `grid` as the only position feature, Indy 500 removed (decision M1) |
| Split (temporal, by season) | Train 1950–2017 / Validation 2018–2021 / Test 2022–2024 (decision M2) |
| Primary metric | PR-AUC (ROC-AUC + F1@val-threshold secondary) |
| Deliverable | One standalone notebook that runs end-to-end with Run All (Kaggle + local) |

## Key decisions (evidence in notebook, full log in `docs/decisions.md`)

| Decision | Outcome |
|---|---|
| Dedup (85 shared-drive pairs, all pre-1979) | Keep the car the driver **started** in; fallback best `positionOrder` |
| `grid = 0` (1,638 rows) | 72 pit-lane starts kept with flag; 1,566 non-starters excluded |
| Status taxonomy | 6-class (`finished/mechanical/accident/driver/regulatory/ambiguous`), strict mechanical primary |
| Qualifying coverage | Not used as features (absent pre-1994); `grid` + `field_size` + `grid_rel` instead. No era/year feature. 2003+ with quali is a late experiment |
| Indianapolis 500 (1950–60) | Flagged; excluded from home-race analysis **and removed from the modeling table** (M1) |
| Features | Chosen by adding one group at a time from 15 candidates (#26): `grid`, driver standings after the previous round (`drv_pos`, `drv_pts_share`, `drv_wins`), past-only career podium rates `driver_te` / `constructor_te`, + 3 `no_history` flags. No sprint, no era/year |

## Data

The **only input** is the 14 raw CSVs (Formula 1 World Championship 1950–2024) in
`data/raw/` — read-only, checksummed at load and re-verified by a guard cell at the end
of the notebook. `data/f1.db` is a legacy local mirror and is **not used**.

## Repository structure

```
├── README.md                        # this file
├── PLAN.md                          # living project plan with decision gates
├── docs/                            # decisions, leakage spec, method + Phase-3 notes (moved out of the notebook)
├── environment.txt                  # pip requirements (Python 3.13, CPU torch)
├── notebooks/
│   ├── f1_podium_prediction.ipynb   # THE deliverable: code + analysis + decisions
│   └── _working_local/              # regenerated figures (local run cache)
├── data/raw/                        # 14 immutable raw CSVs (sole input)
├── scripts/build_db.py              # legacy DB builder (not used by the notebook)
├── labs/                            # course lab manuals (toolkit reference)
├── f1.pdf                           # F1 domain primer (eras, formats, glossary)
└── csen903_project.pdf              # milestone brief
```

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment.txt
jupyter lab notebooks/f1_podium_prediction.ipynb   # then: Restart & Run All
```

On Kaggle, attach the 14 CSVs as a dataset under `/kaggle/input` — the notebook detects
the path automatically. The first cell installs `shap lime missingno` (Lab 4 pattern),
so Internet must be **On**. Figures regenerate into `/kaggle/working/figures/`.

## Notebook map

| § | Contents | Status |
|---|---|---|
| 1–4 | Setup, CSV loading + checksums, raw audit, Gates 1.A/1.B | Done |
| 5–10 | PK/FK checks, grain analysis, typing inventory, coverage EDA, target, leakage rules | Done |
| 11 | **Cleaning**: sentinels→NaN, typing, one spelling per value + driver home country, R2 dedup, flags, status classes, before/after EDA | Done |
| 12 | Decision log | Done |
| 13 | **Data-Engineering Questions (Q1, Q2, Q3)** on the cleaned data: multi-hop joins, Wilson CIs, stratified & logistic regression, constructor reliability, Figures 8–15 | Done |
| 14 | Modeling table + split, 15 candidate features (past-only, brute-force checked) | Done |
| 15 | Fill-and-flag + scaling helpers (`make_xy`, train-only) | Done |
| 16 | **Feature selection**: per-feature plots + correlations, then add one group at a time (kept only if both LR and RF improve in ≥80% of validation redraws) | Done |
| 17–19 | Chosen features prepared; baseline (LR on grid) / LR / RF / NN on validation | Done |
| 20 | Roadmap; raw-integrity guard | — |
| — | Test evaluation, ablation, XAI, inference, report | Next |

## Toolkit

Everything comes from course Labs 1–4 (`pandas/numpy/matplotlib/seaborn/missingno`,
`scikit-learn`, `TensorFlow/Keras`, `shap`/`lime`). Anything beyond the labs is tagged
**[beyond labs]** with a one-line justification. See `PLAN.md` §0.3 for the full mapping.

Deadline: **18 Oct 2026, 23:59** — Kaggle link + repo + PDF report.
