"""Provider-neutral retrieval and grounded generation package."""

from .context import ContextBuilder, ContextBundle, estimate_tokens
from .exact_count import (
    CountableDocument,
    ExactCountEvidence,
    ExactCountRequest,
    ExactCountResult,
    count_exact_occurrences,
    format_exact_count_answer,
    parse_exact_count_request,
)
from .interfaces import EmbeddingProvider, Generator, Reranker, RetrievalRepository
from .models import (
    AnswerResult,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    RankedCandidate,
    RetrievalDiagnostics,
    RetrievalFilters,
    RetrievalResult,
    SearchCandidate,
    SourceExcerpt,
)
from .service import RetrievalConfig, RetrievalError, RetrievalService, normalize_query

__all__ = [
    "AnswerResult",
    "ChatMessage",
    "ContextBuilder",
    "ContextBundle",
    "CountableDocument",
    "EmbeddingProvider",
    "ExactCountEvidence",
    "ExactCountRequest",
    "ExactCountResult",
    "GenerationRequest",
    "GenerationResult",
    "Generator",
    "RankedCandidate",
    "Reranker",
    "RetrievalConfig",
    "RetrievalDiagnostics",
    "RetrievalError",
    "RetrievalFilters",
    "RetrievalRepository",
    "RetrievalResult",
    "RetrievalService",
    "SearchCandidate",
    "SourceExcerpt",
    "count_exact_occurrences",
    "estimate_tokens",
    "format_exact_count_answer",
    "normalize_query",
    "parse_exact_count_request",
]
