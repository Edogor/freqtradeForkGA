"""T4.8 — Strategy embedding space + mode-collapse diagnostics.

Lightweight embedding pipeline that turns a population of strategies
(or HoF entries) into a numerical vector space where neighbouring
strategies are structurally similar.  Two consumers:

1. **Mode-collapse detection** — measure pairwise distance dispersion
   in the embedding space; a sharp drop signals the population is
   converging to one genome family.

2. **2D visualisation hook** — produce (x, y) coordinates per
   strategy suitable for plotting in the web dashboard.  Uses PCA
   from scikit-learn (already a project dep via T3.8 surrogate).
   We deliberately do not pull UMAP/sentence-transformers — both
   are heavy and the embedding signal is dominated by the gene
   token bag, not by semantic code distance.

The featuriser is intentionally simple and pure-stdlib for the token
extraction step so it can run on the phone:

  tokens(gene) = {
      f"ind:{indicator.type}" for each indicator,
      f"op:{condition.operator}" for each entry/exit condition,
      f"logic:{condition.logic}" for each condition,
      f"tf:{tf}" for each informative timeframe,
      "can_short" if can_short,
  }

A document-frequency / inverse-document-frequency vectoriser is
applied across the population so common tokens (e.g. ``ind:RSI``)
contribute less than rare ones.  We rely on sklearn's
``TfidfVectorizer`` with a custom analyser so the token set survives
unmodified.

When scikit-learn is unavailable, ``embed_population`` falls back to
a raw token-frequency dict per strategy and the 2D projection is
disabled (returns ``None``).  All numeric diagnostics still work on
the raw token sets via Jaccard distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def _safe_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
        elif isinstance(obj, dict) and n in obj:
            return obj[n]
    return default


def extract_tokens(gene: Any) -> List[str]:
    """Extract a stable, sorted list of structural tokens from a gene.

    Works on either real ``Gene`` objects (duck-typed) or plain dicts
    with the same field names.  Order is stable so equal gene
    structures produce equal token lists.
    """
    if gene is None:
        return []

    tokens: List[str] = []

    indicators = _safe_attr(gene, "indicators", default=[]) or []
    for ind in indicators:
        t = _safe_attr(ind, "type", "name", default=None)
        if t:
            tokens.append(f"ind:{str(t).upper()}")

    for cond_attr in ("entry_conditions", "exit_conditions",
                      "short_entry_conditions", "short_exit_conditions"):
        conds = _safe_attr(gene, cond_attr, default=[]) or []
        for c in conds:
            op = _safe_attr(c, "operator", "op", default=None)
            if op:
                tokens.append(f"op:{op}")
            logic = _safe_attr(c, "logic", default=None)
            if logic:
                tokens.append(f"logic:{logic}")

    tfs = _safe_attr(gene, "informative_timeframes", default=[]) or []
    for tf in tfs:
        if tf:
            tokens.append(f"tf:{tf}")

    if _safe_attr(gene, "can_short", default=False):
        tokens.append("can_short")

    return sorted(set(tokens))


# ---------------------------------------------------------------------------
# Distance metrics that work without sklearn
# ---------------------------------------------------------------------------


def jaccard_distance(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Embedding result
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    tokens: List[List[str]]
    vectors: Optional[Any] = None            # numpy ndarray when sklearn
    coords_2d: Optional[List[Tuple[float, float]]] = None
    vocab: List[str] = field(default_factory=list)
    n_features: int = 0
    backend: str = "tfidf"                    # or "fallback_tokens"


# ---------------------------------------------------------------------------
# Embedding entry-point
# ---------------------------------------------------------------------------


def embed_population(
    genes: Sequence[Any],
    *,
    project_2d: bool = True,
) -> EmbeddingResult:
    """Embed a population of genes into a token-frequency space.

    Falls back gracefully when sklearn is unavailable.  Always returns
    the raw token sets so downstream callers (diversity metrics, etc.)
    can work without numeric vectors.
    """
    token_sets = [extract_tokens(g) for g in genes]

    try:
        import numpy as np  # noqa: F401  (sklearn drags it in anyway)
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return EmbeddingResult(
            tokens=token_sets,
            vectors=None,
            coords_2d=None,
            vocab=[],
            n_features=0,
            backend="fallback_tokens",
        )

    docs = [" ".join(toks) for toks in token_sets]
    # Drop empties so the vectoriser doesn't complain; keep a mapping.
    keep_idx = [i for i, d in enumerate(docs) if d]
    if not keep_idx:
        return EmbeddingResult(
            tokens=token_sets,
            backend="fallback_tokens",
        )

    vectoriser = TfidfVectorizer(
        token_pattern=r"\S+",
        lowercase=False,
        min_df=1,
    )
    mat = vectoriser.fit_transform([docs[i] for i in keep_idx])
    vocab = list(vectoriser.get_feature_names_out())

    # Re-expand into the full population shape (zero rows for empty docs).
    import numpy as np
    full = np.zeros((len(docs), mat.shape[1]), dtype=float)
    full[keep_idx] = mat.toarray()

    coords: Optional[List[Tuple[float, float]]] = None
    if project_2d and full.shape[0] >= 2 and full.shape[1] >= 2:
        try:
            from sklearn.decomposition import PCA

            n_comp = 2
            pca = PCA(n_components=n_comp, random_state=0)
            xy = pca.fit_transform(full)
            coords = [(float(row[0]), float(row[1])) for row in xy]
        except Exception:
            coords = None

    return EmbeddingResult(
        tokens=token_sets,
        vectors=full,
        coords_2d=coords,
        vocab=vocab,
        n_features=full.shape[1],
        backend="tfidf",
    )


# ---------------------------------------------------------------------------
# Diversity / mode-collapse diagnostics
# ---------------------------------------------------------------------------


def population_diversity(result: EmbeddingResult) -> Dict[str, float]:
    """Compute mode-collapse diagnostics from an embedding result.

    Returns:
      - mean_pairwise_distance: 0 = identical, 1 = orthogonal
      - min_pairwise_distance:  0 = at least two duplicates
      - effective_unique_ratio: |unique token sets| / |population|
      - n_clusters_estimate:    rough cluster count via per-pair
                                jaccard < 0.2 grouping (cheap proxy).
    """
    tokens = result.tokens
    n = len(tokens)
    out = {
        "mean_pairwise_distance": 0.0,
        "min_pairwise_distance": 0.0,
        "effective_unique_ratio": 0.0,
        "n_clusters_estimate": 0,
    }
    if n == 0:
        return out

    unique_keys = {tuple(t) for t in tokens}
    out["effective_unique_ratio"] = round(len(unique_keys) / n, 4)

    if n < 2:
        return out

    total = 0.0
    pair_count = 0
    minimum = 1.0

    # If we have numeric vectors, use cosine distance (cheap with numpy).
    if result.vectors is not None and result.n_features > 0:
        try:
            import numpy as np
            v = result.vectors
            # Normalise rows
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vn = v / norms
            sim = vn @ vn.T
            # Convert to distance, mask diagonal
            dist = 1.0 - sim
            np.fill_diagonal(dist, np.nan)
            with np.errstate(invalid="ignore"):
                total = float(np.nanmean(dist))
                minimum = float(np.nanmin(dist))
            out["mean_pairwise_distance"] = round(max(0.0, total), 4)
            out["min_pairwise_distance"] = round(max(0.0, minimum), 4)
        except Exception:
            # Fall through to jaccard
            pass

    if out["mean_pairwise_distance"] == 0.0:
        for i in range(n):
            for j in range(i + 1, n):
                d = jaccard_distance(tokens[i], tokens[j])
                total += d
                if d < minimum:
                    minimum = d
                pair_count += 1
        if pair_count:
            out["mean_pairwise_distance"] = round(total / pair_count, 4)
            out["min_pairwise_distance"] = round(minimum, 4)

    # Cheap cluster estimate: union-find with edges where jaccard < 0.2
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard_distance(tokens[i], tokens[j]) < 0.2:
                union(i, j)
    out["n_clusters_estimate"] = len({find(k) for k in range(n)})

    return out


__all__ = [
    "EmbeddingResult",
    "embed_population",
    "extract_tokens",
    "jaccard_distance",
    "population_diversity",
]
