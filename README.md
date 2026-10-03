# F1 Podium Prediction Companion — CSEN903 Milestone 1

Predict whether a driver finishes on the podium (`positionOrder ≤ 3`) using **only
information available after qualifying and before the race start**.

## Task framing

| Item | Value |
|---|---|
| Unit of analysis | One driver entering one race — `(raceId, driverId)` |
| Target | `podium = positionOrder ≤ 3` (binary, ~12.7% positive) |
| Prediction cutoff | After qualifying, before lights out (no race outcomes as features) |
| Modeling window | 2003–2024 (full qualifying coverage; ~8,970 rows) |
| Split (temporal, by season) | Train 2003–2019 / Validation 2020–2021 / Test 2022–2024 |
| Primary metric | PR-AUC (ROC-AUC + F1@val-threshold secondary) |
| Deliverable | One standalone notebook that runs end-to-end with Run All (Kaggle + local) |

## Key decisions (evidence in notebook, full log in §13)

| Decision | Outcome |
|---|---|
| Dedup (85 shared-drive pairs, all pre-1979) | Keep the car the driver **started** in; fallback best `positionOrder` |
| `grid = 0` (1,638 rows) | 72 pit-lane starts kept with flag; 1,563 non-starters excluded |
| Status taxonomy | 6-class (`finished/mechanical/accident/driver/regulatory/ambiguous`), strict mechanical primary |
| Qualifying coverage | q2/q3 structurally missing pre-2003 → window starts 2003 |
| Indianapolis 500 (1950–60) | Flagged, excluded from home-race analysis |

## Data

The **only input** is the 14 raw CSVs (Formula 1 World Championship 1950–2024) in
`data/raw/` — read-only, checksummed at load and re-verified by a guard cell at the end
of the notebook. `data/f1.db` is a legacy local mirror and is **not used**.

## Repository structure

```
├── README.md                        # this file
├── PLAN.md                          # living project plan with decision gates
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
| 1–4 | Setup, CSV loading + checksums, Lab-1 raw audit, Gates 1.A/1.B | Done |
| 5–11 | PK/FK checks, grain analysis, typing inventory, coverage EDA, target, Q1–Q3 inputs, leakage rules | Done |
| 12 | **Cleaning**: sentinels→NaN, semantic typing, R2 dedup, flags, status taxonomy, before/after EDA | Done |
| 13–14 | Decision log, roadmap, raw-integrity guard | Done |
| — | Q1–Q3 answers, modeling table, splits, models, XAI, inference, report | Next |

## Toolkit

Everything comes from course Labs 1–4 (`pandas/numpy/matplotlib/seaborn/missingno`,
`scikit-learn`, `TensorFlow/Keras`, `shap`/`lime`). Anything beyond the labs is tagged
**[beyond labs]** with a one-line justification. See `PLAN.md` §0.3 for the full mapping.

Deadline: **18 Oct 2026, 23:59** — Kaggle link + repo + PDF report.
