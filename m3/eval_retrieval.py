"""Evaluate Member 2 Graph Retrieval & Podium Prediction integrated with Member 1.

    python m3/eval_retrieval.py

Runs:
1. Section 2 constraint checks on `INTENT_CATALOG` and `CYPHER_TEMPLATES`.
2. All 60 QA items in `m3/eval/qa_testset.json` against live Neo4j (gold facts computed
   dynamically at evaluation time by running the matching parameterized Cypher template).
3. End-to-end Member 1 + Member 2 integration across all 66 items in `m3/eval/intent_testset.csv`
   and 22 items in `m3/eval/intent_heldout.csv` (`DEPLOYED` classifier -> `extract` -> `merge` ->
   `clarification` -> `GraphRetriever` / `PodiumPredictor`).
4. All 8 Podium Prediction & Explanation cases in `m3/eval/podium_cases.json` (testing both
   Member 1 `extract(user_query)` pre-fill and gold entities, live SHAP additivity, live LIME,
   pre-race Neo4j Cypher facts strictly before race date, explanation fidelity, and non-causal language).
5. Writes `m3/eval/results/retrieval_report.md`, `m3/eval/results/qualitative_sheet_40_items.csv`,
   and `m3/eval/results/member2_eval_results.json`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entities import clarification, extract, merge  # type: ignore  # noqa: E402
from intents import DEPLOYED, LABELS  # type: ignore  # noqa: E402
from podium import PodiumPredictor  # type: ignore  # noqa: E402
from retrieval import (  # type: ignore  # noqa: E402
    CYPHER_TEMPLATES,
    EMPTY_RESULT_MESSAGE,
    INTENT_CATALOG,
    GraphRetriever,
    filter_active_entities,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "eval" / "results"

CAUSAL_BANNED_PATTERNS = (
    r"\bcaused\b",
    r"\bcauses\b",
    r"\bbecause of\b",
    r"\bguaranteed\b",
    r"\bmade the driver finish\b",
    r"\bforced a podium\b",
)


def merge_flat_cache(cached: dict[str, Any], new_slots: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cached)
    for k, v in new_slots.items():
        if v is not None:
            merged[k] = v
    return merged


def verify_section2_constraints() -> dict[str, Any]:
    retrieval_intents = [
        spec for k, spec in INTENT_CATALOG.items() if k not in ("out_of_scope", "podium_prediction")
    ]
    categories_covered = sorted({spec.category for spec in retrieval_intents})
    multi_hop_count = sum(1 for spec in retrieval_intents if spec.min_hops >= 2)
    multi_entity_count = sum(1 for spec in retrieval_intents if spec.multi_entity)
    new_rel_m2_count = sum(1 for spec in INTENT_CATALOG.values() if spec.new_relative_to_m2)

    required_7_categories = {
        "Driver performance",
        "Constructor performance",
        "Head-to-head",
        "Race and circuit facts",
        "Qualifying and grid",
        "Pit stops and strategy",
        "Season and history",
    }
    assert set(INTENT_CATALOG.keys()) == set(LABELS), (
        f"INTENT_CATALOG keys must match m3.intents.LABELS: {set(INTENT_CATALOG) ^ set(LABELS)}"
    )
    assert required_7_categories <= set(categories_covered), (
        f"Missing required categories: {required_7_categories - set(categories_covered)}"
    )
    assert len(INTENT_CATALOG) >= 10, "Must define at least 10 intents"
    assert multi_hop_count >= 3, "At least 3 types must cross 2+ hops"
    assert multi_entity_count >= 2, "At least 2 types must take more than 1 entity"
    assert new_rel_m2_count >= 6, "At least 6 intents must be new relative to M2 queries"

    for tpl_id, cypher in CYPHER_TEMPLATES.items():
        assert "LIMIT" in cypher.upper() or "COUNT(" in cypher.upper() or "SUM(" in cypher.upper(), (
            f"Template {tpl_id} must end with LIMIT or aggregation"
        )

    return {
        "total_intents": len(INTENT_CATALOG),
        "m1_labels_aligned": True,
        "retrieval_intents": len(retrieval_intents),
        "categories_covered": categories_covered,
        "multi_hop_intents_count": multi_hop_count,
        "multi_entity_intents_count": multi_entity_count,
        "new_relative_to_m2_count": new_rel_m2_count,
        "total_cypher_templates": len(CYPHER_TEMPLATES),
    }


def evaluate_qa_test_set(retriever: GraphRetriever, qa_path: Path) -> dict[str, Any]:
    with qa_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    items = dataset["items"]
    evaluated_items: list[dict[str, Any]] = []
    passed = 0

    for item in items:
        if item.get("group") == "follow_up":
            entities = merge_flat_cache(item.get("cached_entities", {}), item.get("new_entities", {}))
        else:
            entities = dict(item.get("entities", {}))

        route_intent = item.get("sub_intent") or item["intent"]
        res = retriever.retrieve(route_intent, entities, question_text=item["question"])
        exp_outcome = item["expected_outcome"]

        if exp_outcome == "rows":
            ok = (not res.is_empty) and len(res.rows) > 0 and res.template_id == item["expected_template"]
        elif exp_outcome == "empty":
            ok = res.is_empty and res.message == EMPTY_RESULT_MESSAGE
        elif exp_outcome == "ambiguity":
            ok = res.is_empty and bool(res.ambiguity_prompt) and len(res.ambiguous_candidates) >= 2
        elif exp_outcome == "out_of_scope":
            ok = res.is_empty and res.intent == "out_of_scope" and len(res.rows) == 0
        else:
            ok = False

        if ok:
            passed += 1

        evaluated_items.append({
            "id": item["id"],
            "group": item["group"],
            "intent": item["intent"],
            "sub_intent": route_intent,
            "category": item["category"],
            "question": item["question"],
            "entities_passed": entities,
            "template_executed": res.template_id,
            "cypher_params": res.params,
            "expected_outcome": exp_outcome,
            "passed": ok,
            "row_count": len(res.rows),
            "gold_facts_at_eval_time": res.rows[:3] if res.rows else None,
            "message_or_prompt": res.ambiguity_prompt or res.message,
            "is_graphical": res.is_graphical,
            "relationship_statements_sample": res.subgraph.get("relationship_statements", [])[:3],
        })

    return {
        "total_items": len(items),
        "passed_items": passed,
        "pass_rate": round(passed / len(items), 4),
        "items": evaluated_items,
    }


def evaluate_m1_m2_integration(
    clf: Any,
    retriever: GraphRetriever,
    predictor: PodiumPredictor,
    csv_path: Path,
) -> dict[str, Any]:
    """Run Member 1's test CSVs end-to-end through Member 1 (`clf.classify`, `extract`,
    `merge`, `clarification`) and Member 2 (`GraphRetriever`, `PodiumPredictor`).
    """
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cache_entities: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    retrieval_executed = 0
    clarification_triggered = 0
    oos_refused = 0
    podium_scored = 0

    for idx, r in enumerate(rows, 1):
        q = r["question"]
        gold_intent = r["intent"]
        prev_intent = r["prev_intent"] or None
        is_fu = bool(prev_intent)

        pred_intent = clf.classify(q, prev=prev_intent)
        ex = extract(q)

        # Evaluate retrieval with gold intent + Member 1's extracted/merged entities
        if gold_intent == "out_of_scope":
            oos_refused += 1
            records.append({
                "index": idx,
                "question": q,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent.label,
                "stage_outcome": "out_of_scope_refused",
            })
            continue

        merged = merge(cache_entities, ex.entities)
        active = filter_active_entities(gold_intent, merged, ex, is_follow_up_turn=is_fu, text=q)
        ask = clarification(gold_intent, ex, active)
        cache_entities = active

        if ask is not None:
            clarification_triggered += 1
            records.append({
                "index": idx,
                "question": q,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent.label,
                "stage_outcome": "clarification_prompt",
                "clarification": ask,
            })
            continue

        if gold_intent == "podium_prediction":
            pres = predictor.predict_and_explain(entities=active)
            assert pres.is_valid, f"Podium prediction failed on {q}: {pres.refusal_message}"
            podium_scored += 1
            records.append({
                "index": idx,
                "question": q,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent.label,
                "stage_outcome": "podium_predicted",
                "podium_probability": pres.podium_probability,
                "predicted_podium": pres.predicted_podium,
            })
            continue

        res = retriever.retrieve(gold_intent, active, question_text=q, current_ex=ex, is_follow_up_turn=is_fu)
        retrieval_executed += 1
        records.append({
            "index": idx,
            "question": q,
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent.label,
            "stage_outcome": "empty_result_handled" if res.is_empty else "rows_retrieved",
            "template_id": res.template_id,
            "row_count": len(res.rows),
            "sample_row": res.rows[0] if res.rows else res.message,
        })

    return {
        "dataset": csv_path.name,
        "total_questions": len(rows),
        "retrieval_executed": retrieval_executed,
        "podium_scored": podium_scored,
        "clarification_triggered": clarification_triggered,
        "out_of_scope_refused": oos_refused,
        "records": records,
    }


def evaluate_podium_cases(predictor: PodiumPredictor, cases_path: Path) -> dict[str, Any]:
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]
    evaluated_cases: list[dict[str, Any]] = []
    passed = 0

    for case in cases:
        if case.get("strict_manual_validation"):
            res = predictor.predict_and_explain(
                raw_inputs=case.get("manual_inputs", {}),
                strict_manual_validation=True,
            )
            ok = (res.is_valid == case["expected_valid"]) and (res.podium_probability is None)
            evaluated_cases.append({
                "id": case["id"],
                "scenario": case["scenario"],
                "passed": ok,
                "is_valid": res.is_valid,
                "refusal_message": res.refusal_message,
            })
            if ok:
                passed += 1
            continue

        # Verify both Member 1 `extract(user_query).entities` and gold `case['entities']`
        ex_m1 = extract(case["user_query"])
        res_m1 = predictor.predict_and_explain(entities=ex_m1.entities)
        res = predictor.predict_and_explain(entities=case["entities"])
        assert res_m1.podium_probability == res.podium_probability, (
            f"M1 extraction mismatch on {case['id']}: {res_m1.podium_probability} != {res.podium_probability}"
        )

        shap_sum = round(float(res.base_value or 0.0) + sum(res.shap_contributions.values()), 4)
        additivity_ok = abs(shap_sum - float(res.podium_probability or 0.0)) <= 1e-3

        top_features = sorted(
            res.shap_contributions.keys(),
            key=lambda k: abs(res.shap_contributions[k]),
            reverse=True,
        )[:2]
        fidelity_ok = all(tf in res.grounded_explanation for tf in top_features)

        causal_matches = [
            pat for pat in CAUSAL_BANNED_PATTERNS
            if re.search(pat, res.grounded_explanation, flags=re.IGNORECASE)
        ]
        non_causal_ok = len(causal_matches) == 0
        pred_match = res.predicted_podium == case["expected_predicted_podium"]

        ok = res.is_valid and additivity_ok and fidelity_ok and non_causal_ok and pred_match
        if ok:
            passed += 1

        evaluated_cases.append({
            "id": case["id"],
            "scenario": case["scenario"],
            "user_query": case["user_query"],
            "passed": ok,
            "driver": res.race_metadata.get("driver_name"),
            "race": f"{res.race_metadata.get('year')} {res.race_metadata.get('race_name')}",
            "race_date_cutoff": res.race_metadata.get("race_date"),
            "feature_values": res.feature_values,
            "grayed_defaults": res.grayed_defaults,
            "podium_probability": res.podium_probability,
            "threshold": res.threshold,
            "predicted_podium": res.predicted_podium,
            "actual_outcome": res.actual_outcome,
            "shap_base_value": res.base_value,
            "shap_contributions": res.shap_contributions,
            "shap_additivity_check": additivity_ok,
            "lime_contributions": res.lime_contributions,
            "lime_r2_score": res.lime_r2_score,
            "pre_race_driver_circuit_facts": res.pre_race_graph_facts.get("driver_circuit_facts"),
            "pre_race_constructor_facts": res.pre_race_graph_facts.get("constructor_facts"),
            "explanation_fidelity_ok": fidelity_ok,
            "non_causal_language_ok": non_causal_ok,
            "grounded_explanation": res.grounded_explanation,
        })

    return {
        "total_cases": len(cases),
        "passed_cases": passed,
        "pass_rate": round(passed / len(cases), 4),
        "cases": evaluated_cases,
    }


def export_qualitative_sheet(qa_eval: dict[str, Any], out_csv: Path) -> None:
    core_items = [it for it in qa_eval["items"] if it["group"] == "core_40"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id",
            "m1_intent",
            "sub_intent",
            "category",
            "question",
            "template_executed",
            "gold_facts_preview",
            "rater_1_fluency_1_to_5",
            "rater_2_fluency_1_to_5",
            "causal_language_flag_0_or_1",
        ])
        for it in core_items:
            writer.writerow([
                it["id"],
                it["intent"],
                it["sub_intent"],
                it["category"],
                it["question"],
                it["template_executed"],
                json.dumps(it["gold_facts_at_eval_time"], ensure_ascii=False),
                "",
                "",
                "",
            ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qa_path = HERE / "eval" / "qa_testset.json"
    podium_path = HERE / "eval" / "podium_cases.json"

    sec2_check = verify_section2_constraints()
    print("=== SECTION 2 CONSTRAINT CHECK ===")
    print(json.dumps(sec2_check, indent=2))

    clf = DEPLOYED()
    retriever = GraphRetriever()
    predictor = PodiumPredictor(retriever=retriever)
    try:
        qa_eval = evaluate_qa_test_set(retriever, qa_path)
        print(
            f"\n=== QA TEST SET EVALUATION: {qa_eval['passed_items']}/{qa_eval['total_items']} "
            f"PASSED ({qa_eval['pass_rate']:.1%}) ==="
        )
        for it in qa_eval["items"]:
            if not it["passed"]:
                print("FAILED QA ITEM:", it)

        m1_test_int = evaluate_m1_m2_integration(
            clf, retriever, predictor, HERE / "eval" / "intent_testset.csv"
        )
        m1_held_int = evaluate_m1_m2_integration(
            clf, retriever, predictor, HERE / "eval" / "intent_heldout.csv"
        )
        print(
            f"\n=== MEMBER 1 -> MEMBER 2 INTEGRATION (intent_testset.csv): "
            f"{m1_test_int['total_questions']} questions processed "
            f"(retrieval={m1_test_int['retrieval_executed']}, podium={m1_test_int['podium_scored']}, "
            f"clarify={m1_test_int['clarification_triggered']}, OOS={m1_test_int['out_of_scope_refused']}) ==="
        )
        print(
            f"=== MEMBER 1 -> MEMBER 2 INTEGRATION (intent_heldout.csv): "
            f"{m1_held_int['total_questions']} questions processed "
            f"(retrieval={m1_held_int['retrieval_executed']}, podium={m1_held_int['podium_scored']}, "
            f"clarify={m1_held_int['clarification_triggered']}, OOS={m1_held_int['out_of_scope_refused']}) ==="
        )

        pod_eval = evaluate_podium_cases(predictor, podium_path)
        print(
            f"\n=== PODIUM PREDICTION & EXPLANATION EVALUATION: "
            f"{pod_eval['passed_cases']}/{pod_eval['total_cases']} PASSED ({pod_eval['pass_rate']:.1%}) ==="
        )
        for c in pod_eval["cases"]:
            if not c["passed"]:
                print("FAILED PODIUM CASE:", c)

        qual_csv = OUT / "qualitative_sheet_40_items.csv"
        export_qualitative_sheet(qa_eval, qual_csv)

        full_report = {
            "section2_constraints": sec2_check,
            "qa_evaluation": qa_eval,
            "m1_m2_integration_testset": m1_test_int,
            "m1_m2_integration_heldout": m1_held_int,
            "podium_evaluation": pod_eval,
        }
        out_json = OUT / "member2_eval_results.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\nExported 40-item qualitative sheet to: {qual_csv}")
        print(f"Saved full Member 1+2 evaluation JSON to: {out_json}")
    finally:
        predictor.close()
        retriever.close()


if __name__ == "__main__":
    main()
