"""Maximal marginal relevance (MMR) evidence selection."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .models import RankedCandidate, Vector


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have matching dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def select_diverse_candidates(
    candidates: Sequence[RankedCandidate],
    *,
    limit: int = 8,
    lambda_mult: float = 0.75,
    max_per_document: int = 3,
    max_per_section: int = 2,
) -> tuple[list[RankedCandidate], bool]:
    """Select strong but non-redundant evidence.

    Returns ``(selected, diversity_used)``.  If embeddings are unavailable,
    this gracefully becomes rank-order selection while retaining document and
    section caps.
    """

    if limit < 1:
        return [], False
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    if max_per_document < 1 or max_per_section < 1:
        raise ValueError("source caps must be positive")

    remaining = list(candidates)
    selected: list[RankedCandidate] = []
    document_counts: Counter[str] = Counter()
    section_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    dimensions = {
        len(item.candidate.embedding) for item in remaining if item.candidate.embedding is not None
    }
    diversity_available = (
        len(dimensions) == 1
        and sum(item.candidate.embedding is not None for item in remaining) >= 2
    )

    while remaining and len(selected) < limit:
        eligible = [
            item
            for item in remaining
            if document_counts[item.candidate.document_id] < max_per_document
            and section_counts[item.candidate.section_key] < max_per_section
        ]
        if not eligible:
            break

        if not selected or not diversity_available:
            chosen = max(eligible, key=lambda item: (item.final_score, item.chunk_id))
        else:
            chosen = max(
                eligible,
                key=lambda item: (
                    _mmr_score(item, selected, lambda_mult),
                    item.final_score,
                    item.chunk_id,
                ),
            )

        selected.append(chosen)
        remaining.remove(chosen)
        document_counts[chosen.candidate.document_id] += 1
        section_counts[chosen.candidate.section_key] += 1

    return selected, diversity_available and len(selected) > 1


def _mmr_score(
    candidate: RankedCandidate,
    selected: Sequence[RankedCandidate],
    lambda_mult: float,
) -> float:
    embedding = candidate.candidate.embedding
    if embedding is None:
        redundancy = 0.0
    else:
        similarities = []
        for prior in selected:
            prior_embedding = prior.candidate.embedding
            if prior_embedding is None or len(prior_embedding) != len(embedding):
                continue
            similarities.append(cosine_similarity(embedding, prior_embedding))
        redundancy = max(similarities, default=0.0)
    return lambda_mult * candidate.final_score - (1 - lambda_mult) * redundancy
