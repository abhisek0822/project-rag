"""Factories for the replaceable embedding and answer-generation providers."""

from __future__ import annotations

from rag.adapters import (
    ExtractiveGenerator,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenAIResponsesGenerator,
    TokenOverlapReranker,
)
from rag.config import Settings, get_settings
from rag.retrieval import RetrievalConfig, RetrievalService


class ProviderConfigurationError(RuntimeError):
    """Raised when a selected model provider is not configured completely."""


def create_embedding_provider(settings: Settings | None = None):
    configuration = settings or get_settings()
    provider = configuration.embedding_provider.strip().casefold()
    if provider in {"hash", "local"}:
        return HashEmbeddingProvider(dimensions=configuration.embedding_dimensions)
    if provider == "openai":
        if not configuration.openai_api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        return OpenAIEmbeddingProvider(
            api_key=configuration.openai_api_key,
            model=configuration.embedding_model,
            dimensions=configuration.embedding_dimensions,
        )
    raise ProviderConfigurationError(
        f"Unsupported embedding provider {configuration.embedding_provider!r}"
    )


def create_generator(settings: Settings | None = None):
    configuration = settings or get_settings()
    provider = getattr(configuration, "generation_provider", "openai").strip().casefold()
    if provider in {"extractive", "local"}:
        return ExtractiveGenerator()
    if provider == "openai":
        if not configuration.openai_api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is required when GENERATION_PROVIDER=openai"
            )
        return OpenAIResponsesGenerator(
            api_key=configuration.openai_api_key,
            model=configuration.generation_model,
            reasoning_effort=configuration.generation_reasoning_effort,
        )
    raise ProviderConfigurationError(f"Unsupported generation provider {provider!r}")


def create_retrieval_service(repository, settings: Settings | None = None) -> RetrievalService:
    configuration = settings or get_settings()
    return RetrievalService(
        repository=repository,
        embedder=create_embedding_provider(configuration),
        generator=create_generator(configuration),
        reranker=TokenOverlapReranker(),
        config=RetrievalConfig(
            dense_limit=configuration.dense_candidate_limit,
            lexical_limit=configuration.lexical_candidate_limit,
            fusion_limit=configuration.rerank_candidate_limit,
            source_limit=configuration.context_chunk_limit,
            max_context_tokens=configuration.context_token_budget,
        ),
    )
