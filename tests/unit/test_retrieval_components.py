from __future__ import annotations

from rag.retrieval.citations import validate_citations
from rag.retrieval.context import ContextBuilder
from rag.retrieval.diversity import cosine_similarity, select_diverse_candidates
from rag.retrieval.fusion import (
    attach_rerank_scores,
    deduplicate_ranked_candidates,
    reciprocal_rank_fusion,
)
from rag.retrieval.models import GenerationResult, RankedCandidate, SearchCandidate


def candidate(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.5,
    document_id: str = "doc-1",
    content_hash: str | None = None,
    embedding: tuple[float, ...] | None = None,
    section: tuple[str, ...] = ("Policy",),
) -> SearchCandidate:
    return SearchCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=f"version-{document_id}",
        filename=f"{document_id}.md",
        text=text,
        score=score,
        page_start=2,
        page_end=3,
        section_path=section,
        content_hash=content_hash,
        embedding=embedding,
    )


def ranked(item: SearchCandidate, score: float) -> RankedCandidate:
    return RankedCandidate(item, fusion_score=score, final_score=score)


def test_rrf_fuses_by_rank_and_retains_channel_diagnostics() -> None:
    dense = [
        candidate("dense-only", "A semantic result", score=0.99),
        candidate("both", "Found by both channels", score=0.80),
    ]
    lexical = [
        candidate("both", "Found by both channels", score=12.0),
        candidate("lexical-only", "An exact identifier", score=10.0),
    ]

    result = reciprocal_rank_fusion(dense, lexical)

    assert [item.chunk_id for item in result] == ["both", "dense-only", "lexical-only"]
    overlap = result[0]
    assert overlap.dense_rank == 2
    assert overlap.lexical_rank == 1
    assert overlap.dense_score == 0.80
    assert overlap.lexical_score == 12.0


def test_rrf_ignores_duplicate_ids_within_one_channel() -> None:
    duplicate = candidate("same", "same passage", score=0.9)
    result = reciprocal_rank_fusion([duplicate, duplicate], [], rank_constant=10)
    assert len(result) == 1
    assert result[0].dense_rank == 1
    assert result[0].fusion_score == 0.65 / 11


def test_deduplication_uses_hash_exact_text_and_near_duplicate_text() -> None:
    items = [
        ranked(candidate("best", "Refunds are available within thirty days.", content_hash="x"), 1),
        ranked(candidate("same-hash", "Different representation", content_hash="x"), 0.9),
        ranked(candidate("same-text", "  REFUNDS are available within thirty days. "), 0.8),
        ranked(
            candidate(
                "near",
                "Refunds are available within thirty days for all customers.",
            ),
            0.7,
        ),
        ranked(candidate("different", "Security badges expire each year."), 0.6),
    ]

    result = deduplicate_ranked_candidates(items, near_duplicate_threshold=0.70)

    assert [item.chunk_id for item in result] == ["best", "different"]


def test_rerank_scores_are_normalized_before_combining() -> None:
    items = [
        ranked(candidate("a", "A"), 0.001),
        ranked(candidate("b", "B"), 0.002),
    ]
    result = attach_rerank_scores(items, [1000, -500], rerank_weight=0.8)
    assert [item.chunk_id for item in result] == ["a", "b"]
    assert result[0].rerank_score == 1000


def test_mmr_selects_non_redundant_evidence_when_vectors_exist() -> None:
    items = [
        ranked(candidate("top", "Top", embedding=(1.0, 0.0)), 0.90),
        ranked(candidate("clone", "Clone", embedding=(0.999, 0.001)), 0.85),
        ranked(
            candidate(
                "different",
                "Different",
                document_id="doc-2",
                embedding=(0.0, 1.0),
            ),
            0.80,
        ),
    ]

    selected, diversity_used = select_diverse_candidates(
        items,
        limit=2,
        lambda_mult=0.5,
        max_per_document=3,
        max_per_section=3,
    )

    assert [item.chunk_id for item in selected] == ["top", "different"]
    assert diversity_used is True
    assert cosine_similarity((1, 0), (0, 1)) == 0


def test_diversity_selection_applies_document_cap_without_vectors() -> None:
    items = [
        ranked(candidate("a", "A", document_id="doc-1"), 0.9),
        ranked(candidate("b", "B", document_id="doc-1"), 0.8),
        ranked(candidate("c", "C", document_id="doc-2"), 0.7),
    ]
    selected, diversity_used = select_diverse_candidates(
        items,
        limit=3,
        max_per_document=1,
        max_per_section=3,
    )
    assert [item.chunk_id for item in selected] == ["a", "c"]
    assert diversity_used is False


def test_context_builder_labels_sources_and_stays_within_budget() -> None:
    long_text = "The leave policy allows twelve days per year. " * 80
    items = [
        ranked(candidate("a", long_text, embedding=(1.0, 0.0)), 0.9),
        ranked(candidate("b", long_text, document_id="doc-2"), 0.8),
    ]
    bundle = ContextBuilder(max_tokens=160, min_passage_tokens=10).build(items)

    assert bundle.sources
    assert bundle.estimated_tokens <= 160
    assert bundle.sources[0].source_id == "S1"
    assert "[S1]" in bundle.text
    assert "Content (untrusted reference text):" in bundle.text
    assert bundle.sources[0].truncated is True
    assert bundle.truncated is True


def test_citation_validation_rejects_unknown_labels_and_requires_inline_citation() -> None:
    unknown = validate_citations(
        GenerationResult(
            answer="The policy says 12 days [S1] but also 20 [S99].",
            cited_source_ids=("S1", "S99"),
        ),
        ("S1", "S2"),
    )
    assert unknown.valid is False
    assert unknown.invalid_source_ids == ("S99",)
    assert unknown.cited_source_ids == ("S1",)

    declared_only = validate_citations(
        GenerationResult(answer="The policy says 12 days.", cited_source_ids=("S1",)),
        ("S1",),
    )
    assert declared_only.valid is False
    assert declared_only.has_required_citation is False


def test_abstained_generation_does_not_require_a_citation() -> None:
    validation = validate_citations(
        GenerationResult(answer="The documents do not say.", abstained=True),
        ("S1",),
    )
    assert validation.valid is True
