"""Hybrid rank fusion and duplicate removal."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import replace

from .models import RankedCandidate, SearchCandidate


def reciprocal_rank_fusion(
    dense_candidates: Sequence[SearchCandidate],
    lexical_candidates: Sequence[SearchCandidate],
    *,
    dense_weight: float = 0.65,
    lexical_weight: float = 0.35,
    rank_constant: float = 60.0,
    limit: int | None = 30,
) -> list[RankedCandidate]:
    """Fuse incompatible dense/lexical score scales using their ranks.

    Duplicate chunk IDs within a result list count only once.  The raw scores
    are retained for diagnostics, but they are never incorrectly averaged.
    """

    if dense_weight < 0 or lexical_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if dense_weight + lexical_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    by_id: dict[str, SearchCandidate] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}

    for rank, candidate in _unique_by_chunk(dense_candidates):
        by_id[candidate.chunk_id] = candidate
        dense_ranks[candidate.chunk_id] = rank
        dense_scores[candidate.chunk_id] = candidate.score

    for rank, candidate in _unique_by_chunk(lexical_candidates):
        by_id.setdefault(candidate.chunk_id, candidate)
        lexical_ranks[candidate.chunk_id] = rank
        lexical_scores[candidate.chunk_id] = candidate.score

    fused: list[RankedCandidate] = []
    for chunk_id, candidate in by_id.items():
        score = 0.0
        dense_rank = dense_ranks.get(chunk_id)
        lexical_rank = lexical_ranks.get(chunk_id)
        if dense_rank is not None:
            score += dense_weight / (rank_constant + dense_rank)
        if lexical_rank is not None:
            score += lexical_weight / (rank_constant + lexical_rank)
        fused.append(
            RankedCandidate(
                candidate=candidate,
                fusion_score=score,
                final_score=score,
                dense_rank=dense_rank,
                lexical_rank=lexical_rank,
                dense_score=dense_scores.get(chunk_id),
                lexical_score=lexical_scores.get(chunk_id),
            )
        )

    # Stable secondary keys make local/evaluation runs reproducible.
    fused.sort(
        key=lambda item: (
            -item.fusion_score,
            item.dense_rank or 10**9,
            item.lexical_rank or 10**9,
            item.chunk_id,
        )
    )
    return fused if limit is None else fused[:limit]


def deduplicate_ranked_candidates(
    candidates: Sequence[RankedCandidate],
    *,
    near_duplicate_threshold: float = 0.92,
) -> list[RankedCandidate]:
    """Remove identical and near-identical passages, preserving best rank."""

    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    selected: list[RankedCandidate] = []
    seen_hashes: set[str] = set()
    seen_texts: set[str] = set()
    token_sets: list[set[str]] = []

    for ranked in candidates:
        candidate = ranked.candidate
        normalized = _normalize_for_dedupe(candidate.text)
        content_hash = candidate.content_hash
        if content_hash and content_hash in seen_hashes:
            continue
        if normalized in seen_texts:
            continue

        tokens = set(_WORD_RE.findall(normalized))
        if tokens and any(
            _token_similarity(tokens, prior) >= near_duplicate_threshold
            for prior in token_sets
            if prior
        ):
            continue

        selected.append(ranked)
        if content_hash:
            seen_hashes.add(content_hash)
        seen_texts.add(normalized)
        token_sets.append(tokens)

    return selected


def attach_rerank_scores(
    candidates: Sequence[RankedCandidate],
    rerank_scores: Sequence[float],
    *,
    rerank_weight: float = 0.70,
) -> list[RankedCandidate]:
    """Combine normalized fused and reranker scores without mixing raw scales."""

    if len(candidates) != len(rerank_scores):
        raise ValueError("reranker returned a different number of scores")
    if not 0 <= rerank_weight <= 1:
        raise ValueError("rerank_weight must be between 0 and 1")
    if not candidates:
        return []

    normalized_fusion = _minmax([item.fusion_score for item in candidates])
    normalized_rerank = _minmax([float(score) for score in rerank_scores])
    rescored = [
        replace(
            item,
            rerank_score=float(raw_rerank),
            final_score=(1 - rerank_weight) * fused + rerank_weight * reranked,
        )
        for item, raw_rerank, fused, reranked in zip(
            candidates,
            rerank_scores,
            normalized_fusion,
            normalized_rerank,
            strict=True,
        )
    ]
    rescored.sort(key=lambda item: (-item.final_score, item.chunk_id))
    return rescored


def _unique_by_chunk(
    candidates: Iterable[SearchCandidate],
) -> Iterable[tuple[int, SearchCandidate]]:
    seen: set[str] = set()
    effective_rank = 0
    for candidate in candidates:
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)
        effective_rank += 1
        yield effective_rank, candidate


_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _normalize_for_dedupe(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _token_similarity(left: set[str], right: set[str]) -> float:
    """Sørensen-Dice overlap is tolerant of a few added boundary words."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]
