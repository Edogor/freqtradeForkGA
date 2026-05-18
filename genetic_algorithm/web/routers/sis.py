"""
SIS API — Strategy Intelligence System health and diagnostics.

Provides endpoints for:
- On-demand SIS health reports
- Predictor model status
- Archetype cluster info
- Corpus statistics
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sis", tags=["sis"])


def _get_evaluator() -> Any:
    """Lazy-load SIS evaluator with current corpus/predictor state."""
    try:
        from genetic_algorithm.intelligence.sis_evaluator import SISEvaluator
        from genetic_algorithm.intelligence.corpus import CORPUS_PATH
        from genetic_algorithm.intelligence.predictors import MultiTargetPredictor

        import pandas as pd
        from pathlib import Path

        corpus_df = None
        if CORPUS_PATH.exists():
            corpus_df = pd.read_parquet(CORPUS_PATH)

        predictor = None
        predictor_path = Path("genetic_algorithm/data/sis_predictor.pkl")
        if predictor_path.exists():
            predictor = MultiTargetPredictor.load(predictor_path)

        classifier = None
        classifier_path = Path("genetic_algorithm/data/sis_classifier.pkl")
        if classifier_path.exists():
            from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
            classifier = ArchetypeClassifier.load(classifier_path)

        return SISEvaluator(corpus_df, predictor, classifier)
    except Exception as e:
        logger.warning(f"[SIS API] Failed to load evaluator: {e}")
        raise HTTPException(status_code=503, detail=f"SIS not available: {e}")


@router.get("/health")
def sis_health() -> Dict[str, Any]:
    """Full SIS health report: corpus, predictors, archetypes."""
    evaluator = _get_evaluator()
    return evaluator.health_report()


@router.get("/predictors")
def sis_predictors() -> Dict[str, Any]:
    """Detailed predictor model health."""
    evaluator = _get_evaluator()
    return evaluator.evaluate_predictor_health()


@router.get("/archetypes")
def sis_archetypes() -> Dict[str, Any]:
    """Archetype cluster health."""
    evaluator = _get_evaluator()
    return evaluator.evaluate_archetype_health()


@router.get("/corpus")
def sis_corpus() -> Dict[str, Any]:
    """Corpus statistics and staleness."""
    evaluator = _get_evaluator()
    return evaluator.evaluate_corpus_health()
