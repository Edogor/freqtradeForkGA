from genetic_algorithm.orchestration.hardcore_archive_v5 import (
    ARCHIVE_QUOTAS_V5,
    ArchiveCandidateV5,
    ArchiveNicheV5,
    combined_distance,
    select_archive_v5,
)


def _candidate(name: str, *, score: float, q: float, a: float, f: float, logic: str, shift: float = 0.0):
    # Q/A/F/R/D/U repeated for six pairs.
    vector = tuple(value for _ in range(6) for value in (q + shift, a, f + shift, q, .1, .1))
    return ArchiveCandidateV5(
        candidate_id=name,
        evolutionary_phenotype_hash=f"{name:0<64}",
        balanced_score=score,
        edge_score=q,
        activity_score=a,
        productive_frequency_score=f,
        behavior_vector=vector,
        logic_tokens=frozenset(logic.split(",")),
    )


def test_archive_keeps_productive_candidate_that_is_not_balanced_champion():
    candidates = [
        _candidate("balanced", score=31.7, q=.53, a=.14, f=.08, logic="ATR"),
        _candidate("productive", score=30.3, q=.49, a=.20, f=.106, logic="ATR,BBANDS"),
        _candidate("activity", score=-20, q=-.2, a=1.0, f=-.2, logic="CMF"),
        _candidate("novel", score=10, q=.2, a=.1, f=.02, logic="ROC,DONCHIAN", shift=.3),
    ]
    selected = select_archive_v5(candidates)
    first_by_niche = {}
    for item in selected:
        first_by_niche.setdefault(item.niche, item.candidate.candidate_id)
    assert first_by_niche[ArchiveNicheV5.BALANCED] == "balanced"
    assert first_by_niche[ArchiveNicheV5.PRODUCTIVE] == "productive"
    assert len(selected) == len(candidates)


def test_archive_deduplicates_phenotypes_and_limits_logic_clones_when_possible():
    candidates = [
        _candidate(f"atr{i}", score=30-i, q=.5, a=.15, f=.08, logic="ATR")
        for i in range(5)
    ] + [
        _candidate(f"other{i}", score=20-i, q=.4-i*.01, a=.16, f=.07, logic=f"I{i}")
        for i in range(10)
    ]
    selected = select_archive_v5(candidates)
    assert sum(item.candidate.logic_signature == "ATR" for item in selected) <= 2
    assert len({item.candidate.evolutionary_phenotype_hash for item in selected}) == len(selected)
    assert combined_distance(selected[0].candidate, selected[-1].candidate) >= 0.0
    assert sum(ARCHIVE_QUOTAS_V5.values()) == 12
