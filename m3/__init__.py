"""Milestone 3: Graph-RAG Formula 1 Assistant package (Members 1 & 2 integrated).

Exports:
- Member 1 (`m3.intents`, `m3.entities`):
  `INTENTS`, `LABELS`, `RuleClassifier`, `EmbeddingClassifier`, `HybridClassifier`, `DEPLOYED`,
  `extract`, `clarification`, `resolve_choice`, `merge`, `mask`, `norm`
- Member 2 (`m3.retrieval`, `m3.podium`):
  `INTENT_CATALOG`, `CYPHER_TEMPLATES`, `GraphRetriever`, `RetrievalResult`, `PipelineSession`,
  `EMPTY_RESULT_MESSAGE`, `OUT_OF_SCOPE_MESSAGE`, `PodiumPredictor`, `PodiumPredictionResult`
"""

from .entities import Extraction, clarification, extract, mask, merge, norm, resolve_choice
from .intents import (
    DEPLOYED,
    INTENTS,
    LABELS,
    EmbeddingClassifier,
    HybridClassifier,
    Intent,
    RuleClassifier,
    is_follow_up,
)
from .podium import PodiumPredictionResult, PodiumPredictor, ValidationResult
from .retrieval import (
    CYPHER_TEMPLATES,
    EMPTY_RESULT_MESSAGE,
    INTENT_CATALOG,
    OUT_OF_SCOPE_MESSAGE,
    REPHRASE_MESSAGE,
    GraphRetriever,
    IntentSpec,
    PipelineSession,
    RetrievalResult,
    filter_active_entities,
)

__all__ = [
    "INTENTS",
    "LABELS",
    "Intent",
    "RuleClassifier",
    "EmbeddingClassifier",
    "HybridClassifier",
    "DEPLOYED",
    "is_follow_up",
    "Extraction",
    "extract",
    "clarification",
    "resolve_choice",
    "merge",
    "mask",
    "norm",
    "INTENT_CATALOG",
    "IntentSpec",
    "CYPHER_TEMPLATES",
    "EMPTY_RESULT_MESSAGE",
    "OUT_OF_SCOPE_MESSAGE",
    "REPHRASE_MESSAGE",
    "GraphRetriever",
    "RetrievalResult",
    "PipelineSession",
    "filter_active_entities",
    "PodiumPredictor",
    "PodiumPredictionResult",
    "ValidationResult",
]
