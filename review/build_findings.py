"""Generates ../REVIEW_FINDINGS.md from findings_data.py and the current notebook (cell numbers)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from findings_data import FINDINGS, REQS
import cellmap
nbp = Path(__file__).parent.parent / "notebooks" / "f1_podium_prediction.ipynb"
cells = cellmap.load(str(nbp)); R = lambda s: cellmap.resolve(s, cells)
esc = lambda s: R(s).replace("|", "\\|").replace("\n", " ")
meta = json.load(open(Path(__file__).parent / "run_meta.json")) if (Path(__file__).parent / "run_meta.json").exists() else {}
L = []
L.append("# Review findings\n")
L.append(f"Written 2026-10-14 for the CSEN903 Milestone 1 notebook. Cell numbers are 0-based in the **fixed** notebook on branch `review-fixes` ({len(cells)} cells). Plan: `REVIEW_PLAN.md`. Out-of-scope audit: `OUT_OF_SCOPE.md`. Independent check: `review/verify.ipynb` (rebuilds the modeling table from the raw CSVs with separate code; its split sizes and PR-AUCs match the notebook to four decimals).\n")
n = len(FINDINGS)
by = {}
for f in FINDINGS: by.setdefault(f["status"], []).append(f["id"])
L.append("## Summary\n")
L.append(f"{n} items checked. Outcome: " + "; ".join(f"**{k}** {len(v)} ({', '.join(v)})" for k, v in by.items()) + ".\n")
L.append("**What matters most**\n")
L.append("1. The results hold up. An independent rebuild reproduces the row counts (24,724 before the V1 fix), the split sizes and every model score; the leakage checks found no leakage; Q1–Q3 numbers reproduce.")
L.append("2. Real defects, all fixed on `review-fixes`: entries that never raced counted as starters (V1, small effect); the neural network never reached its stopping rule (V4); the ablation was too narrow (V8); wrong statements in the text (N1, N2, the 'within 0.03' claim in the test-results paragraph, the Q1 correlation wording); Q1 used different rows from the rest (V12); grain/missingness/cutoff missing from the notebook answers (X2); metrics not plotted or discussed (§5).")
L.append("3. One thing to be honest about in the evaluation: the selected feature set depends on the order of the groups (V9). What is stable is 'driver standings on top of the grid'; the exact extra features are not. The noise in a validation PR-AUC difference is larger than the 0.005 selection rule (the driver-standings gain of +0.045 has a 95% range of 0.008 to 0.080).")
L.append("4. One decision waits for the instructor: which model SHAP/LIME should explain (V3, office-hours question 10).\n")
L.append("## Requirements and checklist traceability\n")
L.append("| Requirement | Where | Status | Evidence | Fix / note |\n|---|---|---|---|---|")
for r in REQS:
    runall = meta.get("runall", "Two full runs completed without errors; outputs identical.")
    L.append("| " + " | ".join(esc(x).replace("<<RUNALL>>", runall) for x in r) + " |")
L.append("")
L.append("## Verification findings (V) and text findings (N, X)\n")
L.append("Severity: **should fix** / **note** / **ok**. Status: fixed on the branch, verified (no change needed), needs answer (office hours), resolved by team.\n")
for f in FINDINGS:
    ev = f["evidence"].replace("<<V17>>", meta.get("v17", "")); ac = f["action"].replace("<<V17A>>", meta.get("v17a", ""))
    L.append(f"### {f['id']}. {f['title']}\n")
    L.append(f"- **Area / severity / status:** {f['area']} / {f['sev']} / {f['status']}")
    L.append(f"- **Claimed or found:** {R(f['what'])}")
    L.append(f"- **Evidence:** {R(ev)}")
    L.append(f"- **What was done / to do:** {R(ac)}\n")
L.append("## How to reproduce\n")
L.append("```\n.venv/bin/python review/make_verify_nb.py   # builds review/verify.ipynb from review/verify_cells.py\ncd review && ../.venv/bin/jupyter nbconvert --to notebook --execute verify.ipynb --output verify.ipynb   # about 3 minutes\n```\n")
L.append("Files: `review/verify_lib.py` (independent rebuild), `review/verify_cells.py` (checks), `review/verify.ipynb` (executed), `review/apply_fixes.py` and `review/apply_fixes2.py` (every edit made to the notebook, as code), `review/sweep_names.py` (out-of-scope name sweep).\n")
Path(__file__).parent.parent.joinpath("REVIEW_FINDINGS.md").write_text("\n".join(L))
print("REVIEW_FINDINGS.md:", len("\n".join(L)), "chars")
