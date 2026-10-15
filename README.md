# F1 Podium Prediction Companion — CSEN903 Milestone 1

Predict whether a driver finishes on the podium (`positionOrder ≤ 3`) using **only
information available after qualifying and before the race start**.

## Team resources

- **Office-hours board** (questions for the instructor, out-of-scope list, shared comments and ticks; team members only): https://claude.ai/artifact/3hor4cmonHZQ4vhWBZWH8r
- **Review plan:** `REVIEW_PLAN.md`

## Task framing

| Item | Value |
|---|---|
| Unit of analysis | One driver entering one race — `(raceId, driverId)` |
| Target | `podium = positionOrder ≤ 3` (binary, ~13.5% of starts) |
| Prediction cutoff | After qualifying, before lights out (no race outcomes as features) |
| Modeling window | 1950–2024, `grid` as the only position feature, Indy 500 removed (decision M1) |
| Split (temporal, by season) | Train 1950–2017 / Validation 2018–2021 / Test 2022–2024 (decision M2) |
| Primary metric | PR-AUC (ROC-AUC + F1@val-threshold secondary) |
| Deliverable | One standalone notebook that runs end-to-end with Run All (Kaggle + local) |

## Key decisions (evidence in notebook, full log in `docs/decisions.md`)

| Decision | Outcome |
|---|---|
| Dedup (85 shared-drive pairs, all pre-1979) | Keep the car the driver **started** in; fallback best `positionOrder` |
| `grid = 0` (1,638 rows) | 72 pit-lane starts kept with flag; 1,342 entries that did not qualify excluded, 221 non-starters that qualified or withdrew kept |
| Status taxonomy | `finished/accident/driver/regulatory/other`; everything else (incl. unclear reasons) counts as mechanical (D24) |
| Qualifying coverage | Qualifying times not used as features (absent pre-1994); `grid` is the qualifying result after penalties. No era/year feature |
| Indianapolis 500 (1950–60) | Flagged; excluded from home-race analysis **and removed from the modeling table** (M1) |
| Features | Chosen by adding one group at a time from 11 candidates (kept if validation PR-AUC rises ≥ 0.005 for LR and RF, D28/D29): `grid` + driver standings after the previous round (`drv_pos`, `drv_pts_share`, `drv_wins`) |
| Preprocessing | `SimpleImputer(median)` + `StandardScaler`, fit on train only |
| Final model | Shallow NN (best validation PR-AUC 0.727, tied with RF 0.726). Test PR-AUC: RF 0.718, LR 0.708, NN 0.699, grid-only baseline 0.647 |

## Data

The **only input** is the 14 raw CSVs (Formula 1 World Championship 1950–2024) in
`data/raw/`: read-only; all cleaning happens on in-memory copies. `data/f1.db` is a legacy local mirror and is **not used**.

## Repository structure

```
├── README.md                        # this file
├── PLAN.md                          # living project plan with decision gates
├── docs/                            # decisions, leakage spec, method + Phase-3 notes (moved out of the notebook)
├── environment.txt                  # pip requirements (Python 3.13, TensorFlow)
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
the path automatically. The first cell runs `%pip install shap lime missingno` (Lab 4 pattern),
so Internet must be **On**. Figures regenerate into `/kaggle/working/figures/`.

## Notebook map

| § | Contents |
|---|---|
| 1–5 | Setup, loading, first look, missing values, key checks, duplicates (EDA before cleaning) |
| 6–7 | Cleaning on copies + after-cleaning EDA; target and pre-race cutoff |
| 8 | Data-engineering questions Q1–Q3 (multi-hop joins, Wilson intervals, logistic regression) |
| 9–11 | Modeling table + split, 11 past-only candidates (leakage-checked), imputer/scaler, feature selection |
| 12–14 | Baseline / LR / RF / shallow NN, tuned on validation; test scored once, confusion matrices |
| 15 | Feature-group ablation; two preprocessing-order experiments |
| 16 | XAI: permutation importance (all models), SHAP and LIME for LR, RF and NN |
| 17–18 | One inference function per model (complete, partial and real test rows); limitations |

Still to do: the PDF report.

## Toolkit

Everything comes from course Labs 1–4 (`pandas/numpy/matplotlib/seaborn/missingno`,
`scikit-learn`, `TensorFlow/Keras`, `shap`/`lime`). Anything beyond the labs is tagged
**[beyond labs]** with a one-line justification. See `PLAN.md` §0.3 for the full mapping.

Deadline: **18 Oct 2026, 23:59** — Kaggle link + repo + PDF report.
