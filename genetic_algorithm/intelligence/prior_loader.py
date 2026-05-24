"""
Prior Loader (T4.1)
====================

Loads a data-driven enrichment YAML (produced by :mod:`prior_rebuilder`)
and exposes it to the rest of the SIS stack.

Design notes
------------
* The hardcoded ``SIS_INDICATOR_ENRICHMENT`` / ``SIS_OPERATOR_ENRICHMENT``
  / ``SYNERGY_GRAPH`` constants in :mod:`sis_integrator` remain the
  **default fallback** — when no YAML is provided, behavior is exactly
  as before this commit.
* When a YAML path is provided, indicators present in the YAML override
  the hardcoded value; indicators absent from the YAML keep their
  hardcoded weight (graceful partial override).
* The loader never throws on a malformed file — it logs and returns the
  original priors so a bad config never breaks a running GA.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_load(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        logger.info(f"[T4.1] No prior file at {path} — keeping hardcoded enrichment")
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"[T4.1] Could not read {path}: {e}")
        return None

    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(text)
        except ImportError:
            logger.warning("[T4.1] PyYAML missing, trying JSON parser")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[T4.1] YAML parse failed for {path}: {e}")
            return None

    try:
        return json.loads(text)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[T4.1] JSON parse failed for {path}: {e}")
        return None


def _merge_weights(
    base: Dict[str, float], override: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    merged = dict(base)
    if not isinstance(override, dict):
        return merged
    for key, val in override.items():
        try:
            merged[str(key)] = float(val)
        except (TypeError, ValueError):
            continue
    return merged


def _merge_graph(
    base: Dict[str, Dict[str, float]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Override nodes wholesale: if YAML lists ``CCI``, its entire
    neighbour map replaces the hardcoded one for that node.  Nodes not
    in the YAML keep the hardcoded edges.
    """
    merged = {k: dict(v) for k, v in base.items()}
    if not isinstance(override, dict):
        return merged
    for src, neighbours in override.items():
        if not isinstance(neighbours, dict):
            continue
        cleaned: Dict[str, float] = {}
        for tgt, weight in neighbours.items():
            try:
                cleaned[str(tgt)] = float(weight)
            except (TypeError, ValueError):
                continue
        merged[str(src)] = cleaned
    return merged


def load_data_driven_priors(
    path: Optional[str],
    *,
    base_indicator: Dict[str, float],
    base_operator: Dict[str, float],
    base_synergy: Dict[str, Dict[str, float]],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Return ``(indicator, operator, synergy, anti_patterns)`` priors.

    Args:
        path: Optional YAML/JSON path.  ``None`` or missing file →
            returns the hardcoded baselines unchanged (and empty
            anti-patterns).
        base_indicator: Hardcoded indicator enrichment fallback.
        base_operator: Hardcoded operator enrichment fallback.
        base_synergy: Hardcoded synergy graph fallback.
    """
    if not path:
        return dict(base_indicator), dict(base_operator), {
            k: dict(v) for k, v in base_synergy.items()
        }, {}

    payload = _safe_load(path)
    if not isinstance(payload, dict):
        return dict(base_indicator), dict(base_operator), {
            k: dict(v) for k, v in base_synergy.items()
        }, {}

    indicator = _merge_weights(base_indicator, payload.get("indicator_enrichment"))
    operator = _merge_weights(base_operator, payload.get("operator_enrichment"))
    synergy = _merge_graph(base_synergy, payload.get("synergy_graph"))

    anti_raw = payload.get("anti_patterns") or {}
    anti: Dict[str, Dict[str, float]] = {}
    if isinstance(anti_raw, dict):
        for src, neighbours in anti_raw.items():
            if not isinstance(neighbours, dict):
                continue
            cleaned: Dict[str, float] = {}
            for tgt, val in neighbours.items():
                try:
                    cleaned[str(tgt)] = float(val)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                anti[str(src)] = cleaned

    meta = payload.get("metadata", {})
    logger.info(
        "[T4.1] Loaded data-driven priors from %s "
        "(n_total=%s, n_t1=%s, base_rate=%s, indicators=%d, anti_patterns=%d)",
        path,
        meta.get("n_total"),
        meta.get("n_t1"),
        meta.get("base_t1_rate"),
        len(indicator),
        sum(len(v) for v in anti.values()),
    )
    return indicator, operator, synergy, anti
