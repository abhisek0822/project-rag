"""Deterministic, keyless providers for development and smoke tests."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence

from rag.retrieval.models import GenerationRequest, GenerationResult, SearchCandidate

_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class HashEmbeddingProvider:
    """Feature-hashed bag-of-words embeddings with no model download.

    This is intended for local demos and deterministic tests, not production
    semantic quality.  It still exercises genuine vector storage and cosine
    retrieval using the exact same vector for ingestion and query paths.
    """

    provider_name = "local"
    model_name = "feature-hash-v1"

    def __init__(self, *, dimensions: int = 1536) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Alias used by the ingestion pipeline's embedding port."""
        return await self.embed_documents(texts)

    def _embed(self, text: str) -> list[float]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens = _TOKEN_RE.findall(normalized)
        vector = [0.0] * self._dimensions
        features = tokens + [
            f"{left}::{right}" for left, right in zip(tokens, tokens[1:], strict=False)
        ]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class TokenOverlapReranker:
    """Small deterministic lexical reranker used by the keyless profile."""

    name = "local-token-overlap-v1"

    async def rerank(self, query: str, candidates: Sequence[SearchCandidate]) -> Sequence[float]:
        query_terms = _important_terms(query)
        scores: list[float] = []
        for candidate in candidates:
            passage_terms = _important_terms(candidate.text)
            if not query_terms or not passage_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & passage_terms)
            scores.append(overlap / math.sqrt(len(query_terms) * len(passage_terms)))
        return scores


class ExtractiveGenerator:
    """Keyless generator that quotes the most query-relevant source sentences."""

    provider_name = "local"
    model_name = "extractive-v1"

    def __init__(
        self,
        *,
        max_sources: int = 3,
        max_sentence_characters: int = 360,
        minimum_query_coverage: float = 0.5,
    ) -> None:
        if max_sources < 1 or max_sentence_characters < 40:
            raise ValueError("local generator limits are too small")
        if not 0 <= minimum_query_coverage <= 1:
            raise ValueError("minimum_query_coverage must be between 0 and 1")
        self._max_sources = max_sources
        self._max_sentence_characters = max_sentence_characters
        self._minimum_query_coverage = minimum_query_coverage

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not request.sources:
            return GenerationResult(
                answer="The uploaded documents do not contain enough information to answer.",
                confidence=0.0,
                missing_information=("No source passages were supplied.",),
                abstained=True,
            )

        query_terms = _important_terms(request.question)
        ranked_sentences: list[tuple[float, int, str, str]] = []
        for source_index, source in enumerate(request.sources):
            sentences = [
                sentence.strip() for sentence in _SENTENCE_RE.split(source.text) if sentence.strip()
            ] or [source.text.strip()]
            best_sentence = max(
                sentences,
                key=lambda sentence: (
                    _sentence_score(query_terms, sentence),
                    -len(sentence),
                    sentence,
                ),
            )
            ranked_sentences.append(
                (
                    _sentence_score(query_terms, best_sentence),
                    -source_index,
                    best_sentence,
                    source.source_id,
                )
            )

        ranked_sentences.sort(reverse=True)
        best_score = ranked_sentences[0][0] if ranked_sentences else 0.0
        if best_score < self._minimum_query_coverage:
            return GenerationResult(
                answer="The uploaded documents do not contain enough information to answer.",
                confidence=0.0,
                missing_information=("No source sentence matched enough of the question terms.",),
                abstained=True,
            )

        relevance_floor = max(0.15, best_score * 0.9)
        chosen: list[tuple[float, str, str]] = []
        seen: set[str] = set()
        for score, _, sentence, source_id in ranked_sentences:
            if score < relevance_floor:
                continue
            normalized = " ".join(sentence.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            chosen.append(
                (score, _trim_sentence(sentence, self._max_sentence_characters), source_id)
            )
            if len(chosen) >= self._max_sources:
                break

        if not chosen:
            return GenerationResult(
                answer="The uploaded documents do not contain enough information to answer.",
                confidence=0.0,
                missing_information=("No readable source passage was available.",),
                abstained=True,
            )

        statements = [f"{sentence} [{source_id}]" for _, sentence, source_id in chosen]
        answer = "The documents state: " + " ".join(statements)
        cited = tuple(source_id for _, _, source_id in chosen)
        confidence = min(0.9, 0.45 + best_score * 0.45)
        return GenerationResult(
            answer=answer,
            cited_source_ids=cited,
            confidence=confidence,
            missing_information=(),
            abstained=False,
            provider_metadata={"provider": self.provider_name, "model": self.model_name},
        )


def _important_terms(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def _sentence_score(query_terms: set[str], sentence: str) -> float:
    sentence_terms = _important_terms(sentence)
    if not query_terms or not sentence_terms:
        return 0.0
    return len(query_terms & sentence_terms) / len(query_terms)


def _trim_sentence(sentence: str, limit: int) -> str:
    # A source may itself contain strings that look like our immutable labels.
    # Strip those before appending the controlled label for this source.
    compact = re.sub(r"\[S\d+\]", "", " ".join(sentence.split())).strip()
    if len(compact) <= limit:
        return compact
    trimmed = compact[: limit - 1].rsplit(" ", 1)[0]
    return trimmed.rstrip(" ,;:") + "…"
