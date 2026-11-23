"""Section 4: Podium Prediction Explainer for the Formula 1 Graph-RAG Assistant (Member 2),
integrated with Member 1 (`m3/intents.py` and `m3/entities.py`).

Implements:
- Loading the saved Milestone 1 model bundle (`artifacts/m1_podium_bundle.joblib`) and
  leakage-free pre-race feature table (`artifacts/pre_race_features.csv`) from disk
  without fitting or retraining any model inside the application.
- Section 4.a Input Modes:
  1. Chat pre-fill (`prefill_from_entities`): accepts Member 1's `Extraction.entities`
     (`drivers`, `grand_prix`, `circuits`, `season`, `grid`, `champ_pos`, `wins`, `points`)
     or flat keys (`driver`, `race_name`, `circuit`, `year`, `grid`, `drv_pos`,
     `drv_pts_share`, `drv_wins`), looks up historical `round - 1` pre-race features from
     Milestone 1 CSVs, and fills any unsupplied features with training medians (flagged
     in `grayed_defaults` for the UI).
  2. Manual feature validation (`validate_manual_features`): validates `grid`, `drv_pos`,
     `drv_pts_share`, and `drv_wins` against data-justified ranges; refuses missing or
     out-of-range values with a clear message and returns no probability.
- Section 4.b Live Prediction & Local Explanations:
  - Scores the row with the saved Milestone 1 model and threshold (`0.432` for Random Forest,
    `0.399` for Logistic Regression).
  - Computes live SHAP (`TreeExplainer` and `Permutation` explainer with the 100 training
    background rows) and live LIME (`LimeTabularExplainer` on raw units) for the row.
  - Generates matplotlib figures for SHAP and LIME for the Streamlit prediction view.
- Section 4.c Pre-Race Graph Context & Non-Causal Explanation:
  - Queries Neo4j via parameterized Cypher templates restricted strictly to races before
    the scored race (`r.date < date($before_date)`).
  - Builds the structured LLM explanation prompt and a deterministic, fan-readable,
    strictly non-causal explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from .retrieval import GraphRetriever
except ImportError:
    from retrieval import GraphRetriever  # type: ignore


@dataclass
class ValidationResult:
    is_valid: bool
    cleaned_features: dict[str, float] | None
    errors: list[str]
    refusal_message: str | None


@dataclass
class PodiumPredictionResult:
    is_valid: bool
    refusal_message: str | None
    model_name: str
    feature_values: dict[str, float]
    grayed_defaults: list[str]
    podium_probability: float | None
    threshold: float | None
    predicted_podium: bool | None
    predicted_class_label: str | None
    base_value: float | None
    shap_contributions: dict[str, float] = field(default_factory=dict)
    lime_contributions: dict[str, float] = field(default_factory=dict)
    lime_r2_score: float | None = None
    positive_drivers: list[tuple[str, float]] = field(default_factory=list)
    negative_drivers: list[tuple[str, float]] = field(default_factory=list)
    global_shap_summary: str = ""
    global_shap_values: dict[str, float] = field(default_factory=dict)
    race_metadata: dict[str, Any] = field(default_factory=dict)
    actual_outcome: dict[str, Any] | None = None
    pre_race_graph_facts: dict[str, Any] = field(default_factory=dict)
    explanation_prompt: dict[str, str] = field(default_factory=dict)
    grounded_explanation: str = ""
    all_models_comparison: list[dict[str, Any]] = field(default_factory=list)


class PodiumPredictor:
    """Loads the saved Milestone 1 model bundle and computes predictions, live SHAP/LIME,
    and pre-race KG-grounded explanations.
    """

    def __init__(
        self,
        bundle_path: Path | None = None,
        pre_race_csv_path: Path | None = None,
        retriever: GraphRetriever | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        self._bundle_path = bundle_path or (root / "artifacts" / "m1_podium_bundle.joblib")
        self._pre_race_csv_path = pre_race_csv_path or (root / "artifacts" / "pre_race_features.csv")

        if not self._bundle_path.is_file():
            raise FileNotFoundError(
                f"Saved Milestone 1 model bundle not found at {self._bundle_path}. "
                "Run `python scripts/export_m1_artifacts.py` once to export the saved model."
            )

        self._bundle: dict[str, Any] = joblib.load(self._bundle_path)
        self.features: list[str] = list(self._bundle["features"])
        self.feature_specs: dict[str, dict[str, Any]] = dict(self._bundle["feature_specs"])
        self.prep = self._bundle["preprocessing"]
        self.medians: dict[str, float] = dict(self._bundle["medians"])
        self.models: dict[str, Any] = dict(self._bundle["models"])
        self.default_model: str = str(self._bundle.get("default_model", "Random forest"))
        self.thresholds: dict[str, float] = dict(self._bundle["thresholds"])
        self.background_scaled: pd.DataFrame = self._bundle["background_scaled"]
        self.global_shap: dict[str, Any] = dict(self._bundle["global_shap_importance"])
        self.seed: int = int(self._bundle.get("seed", 42))

        self._pre_race_df = (
            pd.read_csv(self._pre_race_csv_path)
            if self._pre_race_csv_path.is_file()
            else pd.DataFrame()
        )

        rf_model = self.models["Random forest"]
        lr_model = self.models["Logistic regression"]

        self._tree_explainer = shap.TreeExplainer(rf_model, data=self.background_scaled)
        self._perm_explainers = {
            "Random forest": shap.explainers.Permutation(
                lambda A: rf_model.predict_proba(pd.DataFrame(A, columns=self.features))[:, 1],
                self.background_scaled,
                seed=self.seed,
            ),
            "Logistic regression": shap.explainers.Permutation(
                lambda A: lr_model.predict_proba(pd.DataFrame(A, columns=self.features))[:, 1],
                self.background_scaled,
                seed=self.seed,
            ),
        }
        self._lime_explainer = LimeTabularExplainer(
            self._bundle["X_train_raw_values"],
            feature_names=self.features,
            class_names=["no podium", "podium"],
            discretize_continuous=True,
            random_state=self.seed,
        )

        self._retriever = retriever
        self._owns_retriever = False

    def _get_retriever(self) -> GraphRetriever | None:
        if self._retriever is not None:
            return self._retriever
        try:
            self._retriever = GraphRetriever()
            self._owns_retriever = True
            return self._retriever
        except Exception:
            return None

    def close(self) -> None:
        if self._owns_retriever and self._retriever is not None:
            self._retriever.close()

    # ------------------------------------------------------------------
    # Section 4.a.2: Manual Feature Input Validation
    # ------------------------------------------------------------------

    def validate_manual_features(self, raw_inputs: dict[str, Any]) -> ValidationResult:
        """Validate manual pre-race feature values against data-justified ranges.
        Any missing or out-of-range value is refused with a clear message and returns no probability.
        """
        errors: list[str] = []
        cleaned: dict[str, float] = {}

        unknown = set(raw_inputs.keys()) - set(self.features)
        if unknown:
            errors.append(
                f"Unknown feature(s) {sorted(unknown)}. Expected only {self.features}."
            )

        for feat in self.features:
            spec = self.feature_specs[feat]
            val = raw_inputs.get(feat)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                errors.append(
                    f"Missing required feature '{spec['label']}'. Valid range: "
                    f"[{spec['min']}, {spec['max']}] ({spec['unit']})."
                )
                continue

            try:
                num_val = float(val)
            except (TypeError, ValueError):
                errors.append(
                    f"Invalid non-numeric value '{val}' for '{spec['label']}'. "
                    f"Must be a {spec['input_type']} in [{spec['min']}, {spec['max']}]."
                )
                continue

            if np.isnan(num_val) or np.isinf(num_val):
                errors.append(
                    f"Missing (NaN/Inf) value for '{spec['label']}'. Valid range: "
                    f"[{spec['min']}, {spec['max']}] ({spec['unit']})."
                )
                continue

            if spec["input_type"] == "integer" and not float(num_val).is_integer():
                errors.append(
                    f"'{spec['label']}' must be a whole integer (got {val}). "
                    f"Valid range: {spec['min']} to {spec['max']}."
                )
                continue

            if num_val < float(spec["min"]) or num_val > float(spec["max"]):
                errors.append(
                    f"Value {val} for '{spec['label']}' is outside the valid range "
                    f"[{spec['min']}, {spec['max']}] ({spec['justification']})."
                )
                continue

            cleaned[feat] = float(num_val)

        if errors:
            return ValidationResult(
                is_valid=False,
                cleaned_features=None,
                errors=errors,
                refusal_message="Input validation refused: " + " | ".join(errors),
            )

        return ValidationResult(
            is_valid=True,
            cleaned_features=cleaned,
            errors=[],
            refusal_message=None,
        )

    # ------------------------------------------------------------------
    # Section 4.a.1: Chat Pre-Fill & Historical Pre-Race Lookup
    # ------------------------------------------------------------------

    def prefill_from_entities(self, entities: dict[str, Any]) -> dict[str, Any]:
        """Pre-fill the prediction form from Member 1's `Extraction.entities` (`drivers`,
        `grand_prix`, `circuits`, `season`, `grid`, `champ_pos`, `wins`, `points`) or flat keys.
        """
        e = dict(entities or {})
        user_supplied_features: dict[str, float] = {}

        # Map Member 1 prediction fields (`grid`, `champ_pos`, `wins`, `points`) + Milestone 1 feature names
        if e.get("grid") is not None and not pd.isna(e.get("grid")):
            user_supplied_features["grid"] = float(e["grid"])
        for pos_key in ("drv_pos", "champ_pos"):
            if e.get(pos_key) is not None and not pd.isna(e.get(pos_key)):
                user_supplied_features["drv_pos"] = float(e[pos_key])
                break
        for win_key in ("drv_wins", "wins"):
            if e.get(win_key) is not None and not pd.isna(e.get(win_key)):
                user_supplied_features["drv_wins"] = float(e[win_key])
                break
        if e.get("drv_pts_share") is not None and not pd.isna(e.get("drv_pts_share")):
            user_supplied_features["drv_pts_share"] = float(e["drv_pts_share"])
        elif e.get("points") is not None and not pd.isna(e.get("points")):
            pts_val = float(e["points"])
            # If user supplies raw points (e.g. 150 points), normalize to [0, 1] share
            user_supplied_features["drv_pts_share"] = pts_val if pts_val <= 1.0 else min(1.0, round(pts_val / 250.0, 4))

        # Extract driver, year, and race/circuit from either Member 1 list format or flat keys
        driver_id_filter: int | None = None
        driver_query: str | None = None
        if e.get("drivers") and isinstance(e["drivers"], list):
            first_drv = e["drivers"][0]
            if isinstance(first_drv, dict) and "id" in first_drv:
                driver_id_filter = int(first_drv["id"])
                driver_query = str(first_drv.get("name", ""))
            else:
                driver_query = str(first_drv)
        elif e.get("driver") is not None:
            if isinstance(e["driver"], dict) and "id" in e["driver"]:
                driver_id_filter = int(e["driver"]["id"])
                driver_query = str(e["driver"].get("name", ""))
            else:
                driver_query = str(e["driver"])

        year_val = e.get("season") if e.get("season") is not None else e.get("year")
        circuit_id_filter: int | None = None
        race_q: str | None = e.get("grand_prix") or e.get("race_name")
        if e.get("circuits") and isinstance(e["circuits"], list):
            first_cir = e["circuits"][0]
            if isinstance(first_cir, dict) and "id" in first_cir:
                circuit_id_filter = int(first_cir["id"])
                if not race_q:
                    race_q = str(first_cir.get("name", ""))
            elif not race_q:
                race_q = str(first_cir)
        elif e.get("circuit") is not None and not race_q:
            race_q = str(e["circuit"])

        matched_row: pd.Series | None = None
        if not self._pre_race_df.empty and (driver_id_filter is not None or driver_query):
            df = self._pre_race_df
            if driver_id_filter is not None:
                sub = df[df["driverId"] == driver_id_filter]
            else:
                q_drv = (driver_query or "").strip().lower()
                drv_mask = (
                    df["driver_name"].astype(str).str.lower().eq(q_drv)
                    | df["surname"].astype(str).str.lower().eq(q_drv)
                    | df["code"].astype(str).str.lower().eq(q_drv)
                    | df["driver_name"].astype(str).str.lower().str.contains(q_drv, regex=False)
                )
                sub = df[drv_mask]

            if year_val is not None:
                sub = sub[sub["year"] == int(year_val)]

            if circuit_id_filter is not None and not sub.empty:
                c_mask = sub["circuitId"] == circuit_id_filter
                if c_mask.any():
                    sub = sub[c_mask]
            elif race_q is not None and not sub.empty:
                rq = str(race_q).strip().lower()
                r_mask = (
                    sub["name"].astype(str).str.lower().str.contains(rq, regex=False)
                    | sub["circuit_name"].astype(str).str.lower().str.contains(rq, regex=False)
                )
                if r_mask.any():
                    sub = sub[r_mask]
                elif year_val is not None:
                    retr = self._get_retriever()
                    if retr is not None:
                        r_res = retr.resolve_race(int(year_val), str(race_q))
                        if r_res.get("status") == "resolved":
                            rid_mask = sub["raceId"] == int(r_res["raceId"])
                            if rid_mask.any():
                                sub = sub[rid_mask]

            if not sub.empty:
                matched_row = sub.sort_values(["year", "round"]).iloc[-1]

        features_out: dict[str, float] = {}
        grayed_defaults: list[str] = []

        for feat in self.features:
            if feat in user_supplied_features:
                features_out[feat] = float(user_supplied_features[feat])
            elif matched_row is not None and pd.notna(matched_row.get(feat)):
                features_out[feat] = float(matched_row[feat])
            else:
                features_out[feat] = float(self.medians[feat])
                grayed_defaults.append(feat)

        race_meta: dict[str, Any] = {}
        actual_outcome: dict[str, Any] | None = None
        if matched_row is not None:
            race_meta = {
                "raceId": int(matched_row["raceId"]),
                "year": int(matched_row["year"]),
                "round": int(matched_row["round"]),
                "race_name": str(matched_row["name"]),
                "race_date": str(matched_row["date_str"]),
                "driverId": int(matched_row["driverId"]),
                "driver_name": str(matched_row["driver_name"]),
                "constructorId": int(matched_row["constructorId"]),
                "constructor_name": str(matched_row["constructor_name"]),
                "circuitId": int(matched_row["circuitId"]),
                "circuit_name": str(matched_row["circuit_name"]),
            }
            actual_outcome = {
                "positionOrder": int(matched_row["positionOrder"]),
                "podium": bool(int(matched_row["podium"]) == 1),
                "status": str(matched_row["status_text"]),
            }
        else:
            retr = self._get_retriever()
            if retr is not None:
                if driver_id_filter is not None or driver_query:
                    d_res = retr.resolve_driver(driver_id_filter if driver_id_filter is not None else driver_query)
                    if d_res.get("status") == "resolved":
                        race_meta["driverId"] = int(d_res["driverId"])
                        race_meta["driver_name"] = str(d_res["name"])
                if year_val is not None and (race_q or circuit_id_filter is not None):
                    r_res = retr.resolve_race(
                        int(year_val),
                        circuit_id_filter if circuit_id_filter is not None else str(race_q),
                    )
                    if r_res.get("status") == "resolved":
                        race_meta["raceId"] = int(r_res["raceId"])
                        race_meta["year"] = int(year_val)
                        race_meta["race_name"] = str(r_res["name"])
                        race_meta["race_date"] = str(r_res["date"])
                        race_meta["circuitId"] = int(r_res["circuitId"])
                        race_meta["circuit_name"] = str(r_res["circuit_name"])

        return {
            "features": features_out,
            "grayed_defaults": grayed_defaults,
            "race_metadata": race_meta,
            "actual_outcome": actual_outcome,
        }

    # ------------------------------------------------------------------
    # Section 4.b & 4.c: Live Prediction, SHAP/LIME, and Explanation
    # ------------------------------------------------------------------

    def predict_and_explain(
        self,
        raw_inputs: dict[str, Any] | None = None,
        entities: dict[str, Any] | None = None,
        model_name: str | None = None,
        strict_manual_validation: bool = False,
    ) -> PodiumPredictionResult:
        """Score one pre-race feature row using the saved Milestone 1 model and preprocessing,
        compute live SHAP and LIME local explanations, fetch pre-race KG facts strictly before
        the race date, and build the non-causal explanation prompt and grounded explanation.
        """
        chosen_model = model_name if model_name in self.models else self.default_model
        grayed_defaults: list[str] = []
        race_meta: dict[str, Any] = {}
        actual_outcome: dict[str, Any] | None = None

        if entities is not None:
            prefilled = self.prefill_from_entities(entities)
            candidate_features = dict(prefilled["features"])
            grayed_defaults = list(prefilled["grayed_defaults"])
            race_meta = dict(prefilled["race_metadata"])
            actual_outcome = prefilled["actual_outcome"]
            if raw_inputs:
                for k, v in raw_inputs.items():
                    if v is not None:
                        candidate_features[k] = v
                        if k in grayed_defaults:
                            grayed_defaults.remove(k)
        else:
            candidate_features = dict(raw_inputs or {})
            if not strict_manual_validation:
                for feat in self.features:
                    if candidate_features.get(feat) is None or (
                        isinstance(candidate_features.get(feat), float) and np.isnan(candidate_features[feat])
                    ):
                        candidate_features[feat] = float(self.medians[feat])
                        grayed_defaults.append(feat)

        validation = self.validate_manual_features(candidate_features)
        if not validation.is_valid or validation.cleaned_features is None:
            return PodiumPredictionResult(
                is_valid=False,
                refusal_message=validation.refusal_message,
                model_name=chosen_model,
                feature_values={},
                grayed_defaults=grayed_defaults,
                podium_probability=None,
                threshold=None,
                predicted_podium=None,
                predicted_class_label=None,
                base_value=None,
            )

        feat_dict = validation.cleaned_features
        X_raw_df = pd.DataFrame([feat_dict], columns=self.features).astype(float)
        X_scaled_df = pd.DataFrame(
            self.prep.transform(X_raw_df),
            columns=self.features,
            index=X_raw_df.index,
        )

        all_models_rows: list[dict[str, Any]] = []
        for m_name, m_obj in self.models.items():
            X_in = X_scaled_df[["grid"]] if m_name == "Baseline (LR on grid)" else X_scaled_df
            prob = float(m_obj.predict_proba(X_in)[:, 1][0])
            thr = float(self.thresholds[m_name])
            all_models_rows.append({
                "model": m_name,
                "P(podium)": round(prob, 4),
                "threshold": round(thr, 3),
                "podium": bool(prob >= thr),
            })

        model_obj = self.models[chosen_model]
        X_model_in = X_scaled_df[["grid"]] if chosen_model == "Baseline (LR on grid)" else X_scaled_df
        prob = float(model_obj.predict_proba(X_model_in)[:, 1][0])
        thr = float(self.thresholds[chosen_model])
        is_podium = bool(prob >= thr)
        class_label = "Podium (Top 3)" if is_podium else "Not Podium (Outside Top 3)"

        if chosen_model == "Random forest":
            sv = self._tree_explainer(X_scaled_df)
            sv_obj = sv[..., 1] if sv.values.ndim == 3 else sv
            shap_vals = [float(x) for x in sv_obj.values[0]]
            base_val = float(np.asarray(sv_obj.base_values).ravel()[0])
        else:
            perm_exp = self._perm_explainers.get(chosen_model, self._perm_explainers["Logistic regression"])
            sv_obj = perm_exp(X_scaled_df, max_evals=100, silent=True)
            shap_vals = [float(x) for x in sv_obj.values[0]]
            base_val = float(np.asarray(sv_obj.base_values).ravel()[0])

        shap_dict = {f: round(v, 4) for f, v in zip(self.features, shap_vals)}

        def _predict_raw_for_lime(A: np.ndarray) -> np.ndarray:
            scaled = pd.DataFrame(
                self.prep.transform(pd.DataFrame(A, columns=self.features)),
                columns=self.features,
            )
            X_m = scaled[["grid"]] if chosen_model == "Baseline (LR on grid)" else scaled
            p = model_obj.predict_proba(X_m)[:, 1]
            return np.column_stack([1.0 - p, p])

        lime_exp = self._lime_explainer.explain_instance(
            X_raw_df.iloc[0].to_numpy(dtype=float),
            _predict_raw_for_lime,
            num_features=len(self.features),
            num_samples=5000,
        )
        lime_map = dict(lime_exp.as_map()[1])
        lime_dict = {self.features[idx]: round(float(lime_map.get(idx, 0.0)), 4) for idx in range(len(self.features))}
        lime_r2 = round(float(lime_exp.score), 3)

        pos_drivers = sorted(
            [(f, v) for f, v in shap_dict.items() if v > 0],
            key=lambda kv: kv[1],
            reverse=True,
        )
        neg_drivers = sorted(
            [(f, v) for f, v in shap_dict.items() if v < 0],
            key=lambda kv: kv[1],
        )

        kg_facts: dict[str, Any] = {}
        retr = self._get_retriever()
        if (
            retr is not None
            and race_meta.get("driverId") is not None
            and race_meta.get("circuitId") is not None
            and race_meta.get("race_date") is not None
        ):
            kg_facts = retr.fetch_pre_race_graph_facts(
                driver_id=int(race_meta["driverId"]),
                circuit_id=int(race_meta["circuitId"]),
                constructor_id=int(race_meta["constructorId"]) if race_meta.get("constructorId") is not None else None,
                before_date=str(race_meta["race_date"]),
                year=int(race_meta.get("year", 2024)),
            )

        g_shap_info = self.global_shap.get(chosen_model, self.global_shap["Random forest"])

        result = PodiumPredictionResult(
            is_valid=True,
            refusal_message=None,
            model_name=chosen_model,
            feature_values=feat_dict,
            grayed_defaults=grayed_defaults,
            podium_probability=round(prob, 4),
            threshold=round(thr, 3),
            predicted_podium=is_podium,
            predicted_class_label=class_label,
            base_value=round(base_val, 4),
            shap_contributions=shap_dict,
            lime_contributions=lime_dict,
            lime_r2_score=lime_r2,
            positive_drivers=pos_drivers,
            negative_drivers=neg_drivers,
            global_shap_summary=str(g_shap_info["summary"]),
            global_shap_values=dict(g_shap_info["mean_abs_shap"]),
            race_metadata=race_meta,
            actual_outcome=actual_outcome,
            pre_race_graph_facts=kg_facts,
            all_models_comparison=all_models_rows,
        )

        result.explanation_prompt = self.build_explanation_prompt(result)
        result.grounded_explanation = self.generate_non_causal_explanation(result)
        return result

    # ------------------------------------------------------------------
    # Section 4.c: Structured Explanation Prompt & Grounded Fan Summary
    # ------------------------------------------------------------------

    def build_explanation_prompt(self, res: PodiumPredictionResult) -> dict[str, str]:
        persona = (
            "You are an expert Formula 1 race strategist and data analyst explaining a "
            "pre-race podium prediction to a Formula 1 fan in clear, accurate, non-technical language."
        )

        drv_label = res.race_metadata.get("driver_name", "the selected driver")
        race_label = (
            f"{res.race_metadata.get('year', '')} {res.race_metadata.get('race_name', '')}".strip()
            or "the selected Grand Prix"
        )

        context_lines = [
            f"Target Entry: {drv_label} at {race_label}",
            f"Model: {res.model_name}",
            f"Predicted Podium Probability: {res.podium_probability:.1%} ({res.podium_probability:.4f})",
            f"Decision Threshold (from Milestone 1 validation F1): {res.threshold:.1%} ({res.threshold:.3f})",
            f"Predicted Class: {res.predicted_class_label}",
            f"Base Value (average training prediction): {res.base_value:.1%} ({res.base_value:.4f})",
            f"Pre-Race Feature Values: {res.feature_values}",
            f"Default-Imputed Features (training medians): {res.grayed_defaults}",
            f"Live SHAP Contributions (in probability units): {res.shap_contributions}",
            f"Live LIME Weights (local linear fit R^2={res.lime_r2_score}): {res.lime_contributions}",
            f"Strongest Positive Contributions: {res.positive_drivers}",
            f"Strongest Negative Contributions: {res.negative_drivers}",
            f"Milestone 1 Global SHAP Findings: {res.global_shap_summary}",
        ]

        drv_cir = res.pre_race_graph_facts.get("driver_circuit_facts")
        con_facts = res.pre_race_graph_facts.get("constructor_facts")
        cutoff = res.pre_race_graph_facts.get("before_date_cutoff")
        if drv_cir:
            context_lines.append(
                f"Pre-Race Knowledge Graph Facts (Driver at Circuit strictly before {cutoff}): {drv_cir}"
            )
        else:
            context_lines.append(
                "Pre-Race Knowledge Graph Facts (Driver at Circuit): No prior starts at this circuit before this race."
            )
        if con_facts:
            context_lines.append(
                f"Pre-Race Knowledge Graph Facts (Constructor reliability strictly before {cutoff}): {con_facts}"
            )

        task = (
            "Explain the prediction using ONLY the context above:\n"
            "1. State the predicted podium probability, the threshold, and the predicted class.\n"
            "2. State clearly which pre-race features pushed the probability UP and which pushed it DOWN, "
            "and by roughly how many percentage points (quoting the SHAP values and noting LIME agreement).\n"
            "3. Compare the local contributions with the Milestone 1 global SHAP importance findings "
            "and incorporate the pre-race Knowledge Graph facts (prior circuit record and constructor reliability before the race date).\n"
            "4. Keep the explanation readable for a non-technical Formula 1 fan.\n"
            "5. STRICT NON-CAUSAL RULE: Never present a feature contribution as a cause of a podium "
            "(do NOT say 'caused', 'because of', 'made the driver finish on the podium', or 'guaranteed'). "
            "Frame contributions strictly as how much the statistical model relies on or weights each feature.\n"
            "6. Do not invent any feature, value, count, or fact not present in the context."
        )

        return {
            "persona": persona,
            "context": "\n".join(context_lines),
            "task": task,
            "full_prompt": f"PERSONA:\n{persona}\n\nCONTEXT:\n" + "\n".join(context_lines) + f"\n\nTASK:\n{task}",
        }

    def generate_non_causal_explanation(self, res: PodiumPredictionResult) -> str:
        if not res.is_valid or res.podium_probability is None:
            return res.refusal_message or "No prediction available."

        drv_name = res.race_metadata.get("driver_name", "The driver")
        gp_label = (
            f"the {res.race_metadata['year']} {res.race_metadata['race_name']}"
            if res.race_metadata.get("year") and res.race_metadata.get("race_name")
            else "this Grand Prix"
        )
        fv = res.feature_values

        feat_friendly = {
            "grid": f"starting grid slot P{int(fv['grid'])} (`grid` = {int(fv['grid'])})",
            "drv_pos": f"pre-race championship rank P{int(fv['drv_pos'])} (`drv_pos` = {int(fv['drv_pos'])})",
            "drv_pts_share": f"pre-race share of the championship leader's points (`drv_pts_share` = {fv['drv_pts_share']:.3f})",
            "drv_wins": f"prior season wins entering the round (`drv_wins` = {int(fv['drv_wins'])})",
        }

        verdict = "a **Podium finish**" if res.predicted_podium else "**Not finishing on the podium**"
        comp_word = "at or above" if res.predicted_podium else "below"

        lines: list[str] = [
            f"For **{drv_name}** entering **{gp_label}**, the saved Milestone 1 **{res.model_name}** "
            f"estimates a podium probability of **{res.podium_probability:.1%}** (`{res.podium_probability:.3f}`), "
            f"which sits {comp_word} the Milestone 1 decision threshold of **{res.threshold:.1%}** (`{res.threshold:.3f}`) "
            f"and yields a prediction of {verdict} (starting from an average baseline podium rate of "
            f"**{res.base_value:.1%}**)."
        ]

        if res.grayed_defaults:
            default_desc = ", ".join(f"`{k}` = {fv[k]:.3g}" for k in res.grayed_defaults)
            lines.append(
                f"*(Note: {default_desc} were not present prior to this round and were filled with the Milestone 1 training medians.)*"
            )

        if res.positive_drivers:
            pos_parts = [
                f"{feat_friendly[f]} (+{v * 100:.1f} percentage points, `SHAP = +{v:.4f}`, `LIME = {res.lime_contributions[f]:+.4f}`)"
                for f, v in res.positive_drivers
            ]
            lines.append(
                "**Features pushing the model's probability up:** The model places positive weight on "
                + "; and ".join(pos_parts)
                + "."
            )
        else:
            lines.append(
                "**Features pushing the model's probability up:** None of the four pre-race inputs moved the score above the baseline average for this row."
            )

        if res.negative_drivers:
            neg_parts = [
                f"{feat_friendly[f]} ({v * 100:.1f} percentage points, `SHAP = {v:.4f}`, `LIME = {res.lime_contributions[f]:+.4f}`)"
                for f, v in res.negative_drivers
            ]
            lines.append(
                "**Features pushing the model's probability down:** The score is pulled downward in the model by "
                + "; and ".join(neg_parts)
                + "."
            )
        else:
            lines.append(
                "**Features pushing the model's probability down:** None of the four pre-race inputs pulled the score downward for this entry."
            )

        lines.append(
            f"**Model reliance vs. global Milestone 1 patterns:** Globally across validation races in Milestone 1, "
            f"the model relies most heavily on starting grid position (`grid` mean `|SHAP| = {res.global_shap_values.get('grid', 0.1055):.4f}`), "
            f"followed by prior championship position (`drv_pos` = `{res.global_shap_values.get('drv_pos', 0.0316):.4f}`), "
            f"points share (`drv_pts_share` = `{res.global_shap_values.get('drv_pts_share', 0.0252):.4f}`), and "
            f"prior season wins (`drv_wins` = `{res.global_shap_values.get('drv_wins', 0.0068):.4f}`). "
            f"These SHAP and LIME values reflect statistical patterns that the model relies on to weight each pre-race input."
        )

        drv_cir = res.pre_race_graph_facts.get("driver_circuit_facts")
        con_facts = res.pre_race_graph_facts.get("constructor_facts")
        cutoff = res.pre_race_graph_facts.get("before_date_cutoff")
        kg_sentences: list[str] = []
        if drv_cir:
            kg_sentences.append(
                f"Prior to `{cutoff}`, **{drv_cir['Driver']}** had **{drv_cir['Prior_Circuit_Starts']}** Grand Prix starts "
                f"at **{drv_cir['Circuit']}** in the Knowledge Graph, recording **{drv_cir['Prior_Circuit_Wins']}** wins, "
                f"**{drv_cir['Prior_Circuit_Podiums']}** podiums, a best finish of **P{drv_cir['Prior_Circuit_Best_Finish']}**, "
                f"and an average finish of **{drv_cir['Prior_Circuit_Avg_Finish']}**."
            )
        elif cutoff:
            kg_sentences.append(
                f"Prior to `{cutoff}`, the Knowledge Graph records **0** previous Grand Prix starts for {drv_name} at this circuit."
            )

        if con_facts:
            kg_sentences.append(
                f"Over recent seasons strictly before `{cutoff}`, constructor **{con_facts['Constructor']}** logged "
                f"**{con_facts['Prior_Constructor_Classified']}** classified finishes out of "
                f"**{con_facts['Prior_Constructor_Entries_Recent']}** race entries "
                f"(a pre-race reliability rate of **{con_facts['Prior_Constructor_Reliability']:.1%}**) with "
                f"**{con_facts['Prior_Constructor_Podiums']}** podiums and **{con_facts['Prior_Constructor_Wins']}** wins."
            )

        if kg_sentences:
            lines.append("**Supporting Pre-Race Knowledge Graph Facts:** " + " ".join(kg_sentences))

        if res.actual_outcome is not None:
            act_pod = "Podium" if res.actual_outcome["podium"] else "Not Podium"
            lines.append(
                f"**Recorded Historical Outcome:** In the actual race, {drv_name} finished **P{res.actual_outcome['positionOrder']}** "
                f"({act_pod}, status: *{res.actual_outcome['status']}*)."
            )

        return "\n\n".join(lines)

    def plot_local_explanation(self, res: PodiumPredictionResult) -> plt.Figure:
        """Create a side-by-side matplotlib figure showing live SHAP contributions and
        live LIME weights for the scored row (ready for `st.pyplot(fig)`).
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        if not res.is_valid or not res.shap_contributions:
            axes[0].text(0.5, 0.5, "No valid prediction to explain", ha="center", va="center")
            axes[1].axis("off")
            return fig

        labels = [f"{f} = {res.feature_values[f]:.3g}" for f in self.features]
        shap_vals = [res.shap_contributions[f] for f in self.features]
        lime_vals = [res.lime_contributions[f] for f in self.features]

        colors_shap = ["#d95f02" if v >= 0 else "#1f78b4" for v in shap_vals]
        axes[0].barh(labels[::-1], shap_vals[::-1], color=colors_shap[::-1])
        axes[0].axvline(0, color="black", lw=0.8)
        axes[0].set_title(
            f"Live SHAP Contributions ({res.model_name})\n"
            f"Base {res.base_value:.3f} -> P(podium) {res.podium_probability:.3f}"
        )
        axes[0].set_xlabel("Change in P(podium)")

        colors_lime = ["#d95f02" if v >= 0 else "#1f78b4" for v in lime_vals]
        axes[1].barh(labels[::-1], lime_vals[::-1], color=colors_lime[::-1])
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_title(f"Live LIME Local Weights (R² = {res.lime_r2_score:.2f})")
        axes[1].set_xlabel("LIME Weight towards Podium")

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    try:
        from .entities import extract
    except ImportError:
        from entities import extract  # type: ignore

    p = PodiumPredictor()
    # Test with Member 1's extract() output directly
    ex1 = extract("Predict Norris at the 2024 Miami Grand Prix")
    r1 = p.predict_and_explain(entities=ex1.entities)
    assert r1.is_valid and r1.podium_probability == 0.2701 and not r1.predicted_podium, r1

    ex2 = extract("Leclerc starts P2 at Monaco 2024 and is 3rd in the standings - will he be on the podium?")
    r2 = p.predict_and_explain(entities=ex2.entities)
    assert r2.is_valid and r2.feature_values["grid"] == 2.0 and r2.feature_values["drv_pos"] == 3.0 and r2.predicted_podium, r2

    # Test invalid manual input
    bad = p.predict_and_explain(raw_inputs={"grid": 42, "drv_pos": 1, "drv_pts_share": 1.35, "drv_wins": 0}, strict_manual_validation=True)
    assert not bad.is_valid and bad.podium_probability is None
    p.close()
    print("podium.py: all checks passed")
