"""Diversity-preserving V5 archive selection.

The archive is intentionally not a second fitness function.  Its niche scores
are values already emitted by the raw multi-pair contract; diversity merely
decides which otherwise useful parents are kept for future recombination.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class ArchiveNicheV5(StrEnum):
    BALANCED = "BALANCED"
    PRODUCTIVE = "PRODUCTIVE"
    EDGE = "EDGE"
    ACTIVITY = "ACTIVITY"
    NOVELTY = "NOVELTY"


ARCHIVE_QUOTAS_V5: dict[ArchiveNicheV5, int] = {
    ArchiveNicheV5.BALANCED: 3,
    ArchiveNicheV5.PRODUCTIVE: 3,
    ArchiveNicheV5.EDGE: 2,
    ArchiveNicheV5.ACTIVITY: 2,
    ArchiveNicheV5.NOVELTY: 2,
}


@dataclass(frozen=True)
class ArchiveCandidateV5:
    candidate_id: str
    evolutionary_phenotype_hash: str
    balanced_score: float
    edge_score: float
    activity_score: float
    productive_frequency_score: float
    # Fixed pair order, six values per pair: Q, A, F, R, D and U.
    behavior_vector: tuple[float, ...]
    # Structural indicator/entry/exit tokens, never generated trading rules.
    logic_tokens: frozenset[str]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.evolutionary_phenotype_hash:
            raise ValueError("archive candidates need stable identities")
        if len(self.behavior_vector) != 36 or any(
            not math.isfinite(value) for value in self.behavior_vector
        ):
            raise ValueError("behavior_vector must contain 36 finite components")
        if any(
            not math.isfinite(value)
            for value in (
                self.balanced_score,
                self.edge_score,
                self.activity_score,
                self.productive_frequency_score,
            )
        ):
            raise ValueError("archive scores must be finite")

    @property
    def logic_signature(self) -> str:
        return "|".join(sorted(self.logic_tokens))


@dataclass(frozen=True)
class ArchiveAssignmentV5:
    niche: ArchiveNicheV5
    candidate: ArchiveCandidateV5
    niche_value: float
    diversity_distance: float


def behavior_distance(first: ArchiveCandidateV5, second: ArchiveCandidateV5) -> float:
    """Normalised mean L1 distance for six-pair economic behaviour."""

    # Q/F/R are signed (-1..1); A/D/U occupy (0..1).  This ordering is
    # repeated once per pair, so each coordinate has a known finite scale.
    signed_positions = {0, 2, 3}
    distances = []
    for index, (left, right) in enumerate(
        zip(first.behavior_vector, second.behavior_vector, strict=True)
    ):
        scale = 2.0 if index % 6 in signed_positions else 1.0
        distances.append(min(1.0, abs(left - right) / scale))
    return sum(distances) / len(distances)


def logic_distance(first: ArchiveCandidateV5, second: ArchiveCandidateV5) -> float:
    union = first.logic_tokens | second.logic_tokens
    if not union:
        return 0.0
    return 1.0 - len(first.logic_tokens & second.logic_tokens) / len(union)


def combined_distance(first: ArchiveCandidateV5, second: ArchiveCandidateV5) -> float:
    """Equal-weight behaviour and executable-logic distance."""

    return 0.5 * behavior_distance(first, second) + 0.5 * logic_distance(first, second)


def select_archive_v5(
    candidates: Iterable[ArchiveCandidateV5],
    *,
    quotas: dict[ArchiveNicheV5, int] | None = None,
) -> list[ArchiveAssignmentV5]:
    """Select at most twelve unique, niche-labelled archive parents.

    A phenotype can occupy one niche only.  Within a niche, max-marginal
    relevance uses 70% existing niche merit and 30% distance to the already
    selected archive.  An exact logic signature is capped at two entries while
    alternatives remain, preventing ATR-like clones from consuming the vault.
    """

    quotas = quotas or ARCHIVE_QUOTAS_V5
    unique: dict[str, ArchiveCandidateV5] = {}
    for candidate in candidates:
        incumbent = unique.get(candidate.evolutionary_phenotype_hash)
        if incumbent is None or candidate.balanced_score > incumbent.balanced_score:
            unique[candidate.evolutionary_phenotype_hash] = candidate
    pool = list(unique.values())
    selected: list[ArchiveAssignmentV5] = []
    selected_hashes: set[str] = set()
    logic_counts: dict[str, int] = {}
    order = (
        ArchiveNicheV5.BALANCED,
        ArchiveNicheV5.PRODUCTIVE,
        ArchiveNicheV5.EDGE,
        ArchiveNicheV5.ACTIVITY,
        ArchiveNicheV5.NOVELTY,
    )
    allocated = {niche: 0 for niche in order}
    while True:
        progressed = False
        for niche in order:
            if allocated[niche] >= int(quotas.get(niche, 0)):
                continue
            available = [
                item for item in pool if item.evolutionary_phenotype_hash not in selected_hashes
            ]
            if not available:
                continue
            under_logic_cap = [
                item for item in available if logic_counts.get(item.logic_signature, 0) < 2
            ]
            if under_logic_cap:
                available = under_logic_cap
            choice, merit, distance = _choose(niche, available, selected)
            selected.append(ArchiveAssignmentV5(niche, choice, merit, distance))
            selected_hashes.add(choice.evolutionary_phenotype_hash)
            logic_counts[choice.logic_signature] = logic_counts.get(choice.logic_signature, 0) + 1
            allocated[niche] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _choose(
    niche: ArchiveNicheV5,
    available: list[ArchiveCandidateV5],
    selected: list[ArchiveAssignmentV5],
) -> tuple[ArchiveCandidateV5, float, float]:
    merit = {
        candidate.evolutionary_phenotype_hash: _niche_value(niche, candidate)
        for candidate in available
    }
    low, high = min(merit.values()), max(merit.values())

    def key(candidate: ArchiveCandidateV5) -> tuple[float, float, str]:
        raw = merit[candidate.evolutionary_phenotype_hash]
        quality = 1.0 if high == low else (raw - low) / (high - low)
        distance = (
            min(combined_distance(candidate, item.candidate) for item in selected)
            if selected
            else 1.0
        )
        if niche is ArchiveNicheV5.NOVELTY:
            return (distance, quality, candidate.evolutionary_phenotype_hash)
        return (0.7 * quality + 0.3 * distance, raw, candidate.evolutionary_phenotype_hash)

    chosen = max(available, key=key)
    distance = (
        min(combined_distance(chosen, item.candidate) for item in selected) if selected else 1.0
    )
    return chosen, merit[chosen.evolutionary_phenotype_hash], distance


def _niche_value(niche: ArchiveNicheV5, candidate: ArchiveCandidateV5) -> float:
    return {
        ArchiveNicheV5.BALANCED: candidate.balanced_score,
        ArchiveNicheV5.PRODUCTIVE: candidate.productive_frequency_score,
        ArchiveNicheV5.EDGE: candidate.edge_score,
        ArchiveNicheV5.ACTIVITY: candidate.activity_score,
        # Novelty uses distance as its primary selector; score only breaks ties.
        ArchiveNicheV5.NOVELTY: candidate.balanced_score,
    }[niche]
