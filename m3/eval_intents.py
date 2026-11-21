"""Evaluate both intent classifiers on m3/eval/intent_testset.csv.

    python m3/eval_intents.py

Writes m3/eval/results/: intent_report.md (accuracy, per-intent recall, errors,
threshold sweep) and confusion_<name>.png. A "clarify" prediction (the classifier
asked the user to rephrase) counts as wrong and gets its own confusion-matrix column.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intents import INTENTS, LABELS, EmbeddingClassifier, HybridClassifier, RuleClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "eval" / "results"
COLS = LABELS + ["clarify"]


def load(name="intent_testset.csv"):
    with (HERE / "eval" / name).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    examples = {x.lower() for d in INTENTS.values() for x in d["examples"]}
    assert not [r for r in rows if r["question"].lower() in examples], "test item copies an example"
    counts = Counter(r["intent"] for r in rows)
    assert name != "intent_testset.csv" or (len(rows) >= 50 and all(counts[lab] >= 4 for lab in LABELS)), counts
    return rows


def run(clf, rows):
    return [clf.classify(r["question"], prev=r["prev_intent"] or None) for r in rows]


def confusion(rows, preds):
    m = np.zeros((len(LABELS), len(COLS)), dtype=int)
    for r, p in zip(rows, preds):
        m[LABELS.index(r["intent"]), COLS.index(p.label or "clarify")] += 1
    return m


def plot(m, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(COLS)), COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def md_matrix(m):
    head = "| true \\ pred | " + " | ".join(c[:10] for c in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    body = [f"| {LABELS[i]} | " + " | ".join(str(v) if v else "." for v in m[i]) + " |" for i in range(len(LABELS))]
    return "\n".join([head, sep, *body])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    gold = [r["intent"] for r in rows]
    emb = EmbeddingClassifier()
    systems = {
        "rule-based": (RuleClassifier(), None),
        "embedding (masked, deployed thresholds)": (emb, None),
        "hybrid (rules, embedding fallback)": (HybridClassifier(emb), None),
    }
    report = [f"# Intent classification: rule-based vs embedding similarity\n",
              f"Test set: `m3/eval/intent_testset.csv`, {len(rows)} questions, "
              f"{dict(Counter(gold))}. Phrased differently from the classifier examples "
              f"(checked: no test question equals an example). Rows with `prev_intent` are follow-ups "
              f"evaluated with that cached intent.\n",
              f"Embedding model: `{EmbeddingClassifier.MODEL}`, min score {emb.min_score}, margin {emb.margin}.\n"]

    summary = []
    for name, (clf, _) in systems.items():
        preds = run(clf, rows)
        acc = np.mean([p.label == g for p, g in zip(preds, gold)])
        abst = np.mean([p.label is None for p in preds])
        m = confusion(rows, preds)
        slug = name.split()[0]
        plot(m, f"{name}: accuracy {acc:.1%}", OUT / f"confusion_{slug}.png")
        summary.append((name, acc, abst))
        report += [f"\n## {name}\n", f"Accuracy **{acc:.1%}** ({sum(p.label == g for p, g in zip(preds, gold))}/{len(rows)}), "
                   f"asked to rephrase on {abst:.1%}.\n", f"![confusion](confusion_{slug}.png)\n", md_matrix(m), "\n",
                   "\nPer-intent recall: " + ", ".join(
                       f"{lab} {m[i, i] / m[i].sum():.0%}" for i, lab in enumerate(LABELS)) + "\n",
                   "\nErrors:\n"]
        report += [f"- `{r['question']}` true **{g}**, got **{p.label or 'clarify'}** ({p.reason or 'top score'})"
                   for r, g, p in zip(rows, gold, preds) if p.label != g] or ["- none"]

    # Ablation: embedding without masking names/years.
    raw = EmbeddingClassifier(use_mask=False)
    acc_raw = np.mean([p.label == g for p, g in zip(run(raw, rows), gold)])
    report += ["\n## Ablation: masking entities before embedding\n",
               f"Without masking (raw text embedded): accuracy {acc_raw:.1%}; with masking: {summary[1][1]:.1%}.\n"]

    # Sensitivity of the two thresholds (fixed a priori; this table is for the report, not tuning).
    report += ["\n## Threshold sensitivity (embedding)\n", "| min score | margin | accuracy | rephrase rate |", "|---|---|---|---|"]
    for ms in (0.35, 0.45, 0.55):
        for mg in (0.0, 0.03, 0.06):
            emb.min_score, emb.margin = ms, mg
            ps = run(emb, rows)
            report.append(f"| {ms} | {mg} | {np.mean([p.label == g for p, g in zip(ps, gold)]):.1%} | "
                          f"{np.mean([p.label is None for p in ps]):.1%} |")

    # Held-out set: written before the improvement round, never used to pick patterns or examples.
    held = load("intent_heldout.csv")
    emb.min_score, emb.margin = 0.45, 0.03
    held_acc = []
    for name, (clf, _) in systems.items():
        ps = run(clf, held)
        held_acc.append(np.mean([p.label == r["intent"] for p, r in zip(ps, held)]))
        report += [f"\n## Held-out set ({len(held)} questions): {name}\n", f"Accuracy {held_acc[-1]:.1%}. Errors:\n"]
        report += [f"- `{r['question']}` true **{r['intent']}**, got **{p.label or 'clarify'}** ({p.reason or 'top score'})"
                   for r, p in zip(held, ps) if p.label != r["intent"]] or ["- none"]

    report += ["\n## Summary\n", "| classifier | test accuracy | rephrase rate | held-out accuracy |", "|---|---|---|---|"]
    report += [f"| {n} | {a:.1%} | {b:.1%} | {h:.1%} |" for (n, a, b), h in zip(summary, held_acc)]
    summary = [(n, a, b) for n, a, b in summary]
    print("held-out:", ", ".join(f"{n.split()[0]} {h:.1%}" for (n, _, _), h in zip(summary, held_acc)))
    (OUT / "intent_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for n, a, b in summary:
        print(f"{n:45s} accuracy {a:.1%}  rephrase {b:.1%}")
    print(f"embedding without masking               accuracy {acc_raw:.1%}")
    print(f"report: {OUT / 'intent_report.md'}")


if __name__ == "__main__":
    main()
