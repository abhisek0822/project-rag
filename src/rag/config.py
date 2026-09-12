"""Application settings loaded from environment variables or a local ``.env`` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration shared by the API and background worker.

    Defaults make running Python directly convenient. Docker Compose overrides the
    service hostnames (Postgres, Redis, and MinIO) for its private network.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Custom RAG"
    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    database_echo: bool = False
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=20, ge=0)

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False

    blob_store: Literal["local", "minio", "s3"] = "local"
    local_blob_path: Path = Path("data/blobs")
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_access_key: str | None = "ragminio"
    s3_secret_key: str | None = "ragminiosecret"
    s3_bucket: str = "rag-documents"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    openai_api_key: str | None = None
    embedding_provider: str = "hash"
    embedding_model: str = "feature-hash-v1"
    embedding_dimensions: int = Field(default=1536, ge=1)
    generation_provider: str = "extractive"
    generation_model: str = "extractive-v1"
    generation_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = (
        "none"
    )

    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    chunk_target_tokens: int = Field(default=500, ge=1)
    chunk_max_tokens: int = Field(default=750, ge=1)
    chunk_min_tokens: int = Field(default=100, ge=1)
    chunk_overlap_tokens: int = Field(default=75, ge=0)

    dense_candidate_limit: int = Field(default=40, ge=1)
    lexical_candidate_limit: int = Field(default=40, ge=1)
    rerank_candidate_limit: int = Field(default=30, ge=1)
    context_chunk_limit: int = Field(default=8, ge=1)
    context_token_budget: int = Field(default=6000, ge=1)

    @field_validator("database_url")
    @classmethod
    def require_sync_database_driver(cls, value: str) -> str:
        if value.startswith("postgresql+asyncpg"):
            raise ValueError("This application uses sync SQLAlchemy; use postgresql+psycopg")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def require_schema_vector_size(cls, value: int) -> int:
        # The first schema generation deliberately has one fixed vector size so an
        # incompatible embedding model can never be mixed into the active index.
        if value != 1536:
            raise ValueError("The current database schema requires 1536-dimensional embeddings")
        return value

    @property
    def effective_celery_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_celery_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-in-practice settings object per process."""

    return Settings()
