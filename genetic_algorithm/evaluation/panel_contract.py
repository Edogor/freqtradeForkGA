"""Deterministic evaluation-panel identity and common replay helpers.

Fitness values are comparable only when they were produced by the same data,
cost, evaluator, and scoring policy.  This module gives that context a stable
identity and provides the only supported way to rank candidates originating
from different search panels.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


PANEL_ROLE_OPTIMIZATION = "OPTIMIZATION"
PANEL_ROLE_COMMON_REPLAY = "COMMON_REPLAY"
PANEL_ROLE_HOLDOUT = "HOLDOUT"
VALID_PANEL_ROLES = {
    PANEL_ROLE_OPTIMIZATION,
    PANEL_ROLE_COMMON_REPLAY,
    PANEL_ROLE_HOLDOUT,
}
_PANEL_ID_PATTERN = re.compile(r"^panel_v1_[0-9a-f]{24}$")


def validate_panel_identity(
    panel_id: Optional[str], panel_role: Optional[str]
) -> None:
    """Reject partial or malformed persisted panel provenance."""
    if panel_id is None and panel_role is None:
        return
    if panel_id is None or panel_role is None:
        raise ValueError("Evaluation panel id and role must be persisted together")
    if not isinstance(panel_id, str) or not _PANEL_ID_PATTERN.fullmatch(panel_id):
        raise ValueError(f"Invalid evaluation panel id: {panel_id!r}")
    if panel_role not in VALID_PANEL_ROLES:
        raise ValueError(f"Unknown evaluation panel role: {panel_role!r}")

_SEMANTIC_CONFIG_SECTIONS = (
    "backtesting",
    "strategy_constraints",
    "fitness_weights",
    "fitness_penalties",
    "fitness_bounds",
    "fitness_policy_v3",
    "trade_frequency_thresholds",
    "walk_forward",
    "pair_validation",
    "regime_aware",
    "monte_carlo",
    "deflated_sharpe",
    "evaluation_v2",
    "short_selling",
    "multi_timeframe",
    "exchange",
    "data",
    "data_manifest",
)


def _canonicalize(value: Any) -> Any:
    """Convert supported runtime objects to deterministic JSON values."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        canonical = [_canonicalize(item) for item in value]
        return sorted(
            canonical,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Evaluation panel cannot contain NaN or infinite values")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _segment_descriptor(segment: Any) -> dict[str, Any]:
    if hasattr(segment, "to_dict"):
        raw = segment.to_dict()
    elif isinstance(segment, Mapping):
        raw = dict(segment)
    else:
        raw = {
            "segment_id": getattr(segment, "segment_id", ""),
            "timerange": getattr(segment, "timerange", ""),
            "regime": getattr(segment, "regime", ""),
            "confidence": getattr(segment, "confidence", 0.0),
            "role": getattr(segment, "role", "optimization"),
        }
    # These fields determine which observations are evaluated and how regime
    # aggregation weights them.  Incidental diagnostic metadata is excluded.
    return _canonicalize(
        {
            "segment_id": raw.get("segment_id", ""),
            "timerange": raw.get("timerange", ""),
            "regime": raw.get("regime", ""),
            "confidence": raw.get("confidence", 0.0),
            "role": raw.get("role", "optimization"),
        }
    )


@dataclass(frozen=True)
class EvaluationPanel:
    """Stable identity for one exactly comparable fitness context."""

    panel_id: str
    role: str
    evaluator_kind: str
    descriptor: Mapping[str, Any]
    data_identity_verified: bool = False

    def __post_init__(self) -> None:
        if self.role not in VALID_PANEL_ROLES:
            raise ValueError(f"Unknown evaluation panel role: {self.role!r}")
        validate_panel_identity(self.panel_id, self.role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "role": self.role,
            "evaluator_kind": self.evaluator_kind,
            "descriptor": copy.deepcopy(dict(self.descriptor)),
            "data_identity_verified": self.data_identity_verified,
        }


def build_evaluation_panel(
    config: Mapping[str, Any],
    *,
    evaluator: Any = None,
    segments: Optional[Sequence[Any]] = None,
    role: str = PANEL_ROLE_OPTIMIZATION,
    data_manifest_hash: Optional[str] = None,
) -> EvaluationPanel:
    """Build a panel identity from data/cost/scoring semantics and segments."""
    if role not in VALID_PANEL_ROLES:
        raise ValueError(f"Unknown evaluation panel role: {role!r}")

    evaluator_kind = "regime_aware" if hasattr(evaluator, "base_evaluator") else "standard"
    if segments is None and evaluator_kind == "regime_aware":
        if role == PANEL_ROLE_HOLDOUT:
            segments = list(getattr(evaluator, "_holdout_segments", []))
        else:
            segments = list(getattr(evaluator, "_optimization_segments", []))

    semantic_config = {
        section: copy.deepcopy(config[section])
        for section in _SEMANTIC_CONFIG_SECTIONS
        if section in config
    }
    if data_manifest_hash is not None and not (
        isinstance(data_manifest_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", data_manifest_hash)
    ):
        raise ValueError("data_manifest_hash must be a lowercase SHA-256 digest")
    data_identity_verified = data_manifest_hash is not None
    canonical_segments = sorted(
        (_segment_descriptor(segment) for segment in (segments or [])),
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
    )
    descriptor = _canonicalize(
        {
            "contract_version": 1,
            "role": role,
            "evaluator_kind": evaluator_kind,
            "semantic_config": semantic_config,
            "data_manifest_hash": (
                data_manifest_hash if data_identity_verified else None
            ),
            "segments": canonical_segments,
        }
    )
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    panel_id = f"panel_v1_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"
    return EvaluationPanel(
        panel_id=panel_id,
        role=role,
        evaluator_kind=evaluator_kind,
        descriptor=descriptor,
        data_identity_verified=data_identity_verified,
    )


def phenotype_fingerprint(strategy_gene: Any) -> str:
    """Hash executable genome semantics while ignoring transient identifiers."""
    payload = copy.deepcopy(strategy_gene.to_dict())
    payload.pop("generation", None)
    payload.pop("individual_id", None)
    canonical = json.dumps(
        _canonicalize(payload), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def replay_on_common_panel(
    candidates: Iterable[Any],
    *,
    evaluator: Any,
    panel: EvaluationPanel,
    logger: Any,
    fingerprint: Callable[[Any], str] = phenotype_fingerprint,
    parallel_evaluator: Any = None,
) -> list[Any]:
    """Deduplicate and re-evaluate candidates before cross-panel ranking.

    The supplied objects are updated in place.  Failed replays retain explicit
    failed-backtest evidence and are excluded from the returned ranking pool.
    When supplied, ``parallel_evaluator`` evaluates the stable, deduplicated
    candidate list as one batch.  The scalar evaluator remains the fallback so
    legacy callers and environments without a worker pool keep identical
    behaviour.
    """
    if panel.role != PANEL_ROLE_COMMON_REPLAY:
        raise ValueError("Cross-panel ranking requires a COMMON_REPLAY panel")

    unique: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = fingerprint(candidate.strategy_gene)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    source_context = [
        {
            "source_fitness": candidate.raw_fitness,
            "source_panel_id": getattr(candidate, "fitness_panel_id", None),
            "source_panel_role": getattr(candidate, "fitness_panel_role", None),
            "source_island": candidate.metrics.get(
                "evaluation_island", candidate.metrics.get("island_name")
            ),
        }
        for candidate in unique
    ]

    def evaluate_serially() -> None:
        for index, candidate in enumerate(unique):
            try:
                fitness, metrics = evaluator.evaluate(
                    candidate.strategy_gene,
                    strategy_name=f"CommonReplay_{index}_{candidate.id}",
                )
                metrics = dict(metrics or {})
            except Exception as exc:
                fitness = 0.0
                metrics = {
                    "error": f"common replay failed: {exc}",
                    "profit": 0.0,
                    "num_trades": 0,
                    "max_drawdown": 1.0,
                }
            candidate.set_fitness(fitness, metrics)

    if parallel_evaluator is None:
        evaluate_serially()
    else:
        # Finalists already carry their local-search evidence.  Clear it
        # before batching so a cancelled/missing worker result cannot be
        # mistaken for a successful common-panel replay by evaluate_batch's
        # unevaluated-candidate cleanup.
        for candidate in unique:
            candidate.adopt_unevaluated_genome(candidate)
        try:
            parallel_evaluator.evaluate_batch(unique)
        except Exception as exc:
            logger.warning(
                "[COMMON REPLAY] Parallel batch failed (%s); falling back "
                "to serial evaluation",
                exc,
            )
            # A failed batch may have updated only a prefix.  Re-run the full
            # stable list serially so the resulting ranking is deterministic.
            for candidate in unique:
                candidate.adopt_unevaluated_genome(candidate)
            evaluate_serially()

    successful: list[Any] = []
    for candidate, source in zip(unique, source_context):
        if not candidate.evaluated:
            candidate.set_fitness(
                0.0,
                {
                    "error": "common replay produced no worker result",
                    "profit": 0.0,
                    "num_trades": 0,
                    "max_drawdown": 1.0,
                },
            )
        candidate.metrics.update(
            {
                "fitness_panel_id": panel.panel_id,
                "fitness_panel_role": panel.role,
                "common_replay": True,
                **source,
            }
        )
        candidate.assign_fitness_panel(panel.panel_id, panel.role)
        if candidate.has_comparable_fitness(panel.panel_id):
            successful.append(candidate)
        else:
            logger.warning(
                "[COMMON REPLAY] Rejected %s: %s",
                candidate.id,
                candidate.metrics.get("error", "invalid measured fitness"),
            )

    return sorted(successful, key=lambda item: float(item.raw_fitness), reverse=True)
