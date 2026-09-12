"""Celery application and reliable document-ingestion task."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from celery import Celery
from sqlalchemy.orm import Session

from rag.blob_store import create_blob_store
from rag.config import get_settings
from rag.db import SessionLocal
from rag.ingestion import Chunker, IngestionError, IngestionService
from rag.models import VersionStatus
from rag.providers import create_embedding_provider
from rag.repositories import IngestionRepository

settings = get_settings()
celery_app = Celery(
    "rag",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_always_eager=getattr(settings, "celery_task_always_eager", False),
    task_store_eager_result=getattr(settings, "celery_task_always_eager", False),
)


class TransactionalIngestionRepository:
    """Commit each resumable stage while keeping publish itself atomic.

    Partially written chunks are never searchable because retrieval only selects
    the current READY version. A failed SQL statement is rolled back before the
    ingestion service records its user-facing failure state.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.inner = IngestionRepository(session)

    def get_version(self, version_id: str):
        return self.inner.get_version(version_id)

    def _write(self, operation: Callable[..., Any], *args: Any, **kwargs: Any):
        try:
            result = operation(*args, **kwargs)
            self.session.commit()
            # The worker session is long-lived for one task. Expiring its identity
            # map makes a concurrent API deletion visible before the next stage.
            self.session.expire_all()
            return result
        except Exception:
            self.session.rollback()
            raise

    def set_status(self, version_id: str, status: str, **kwargs: Any):
        version = self.inner.get_version(version_id)
        if version.deleted_at is not None or version.status in {
            VersionStatus.DELETING,
            VersionStatus.DELETED,
        }:
            raise RuntimeError("Document was deleted while ingestion was running")
        return self._write(self.inner.set_status, version_id, status, **kwargs)

    def replace_chunks(self, version_id: str, chunks):
        return self._write(self.inner.replace_chunks, version_id, chunks)

    def save_embeddings(self, version_id: str, embeddings, dimensions: int):
        return self._write(
            self.inner.save_embeddings,
            version_id,
            embeddings,
            dimensions,
        )

    def publish_version(self, version_id: str, expected_chunk_count: int):
        return self._write(
            self.inner.publish_version,
            version_id,
            expected_chunk_count,
        )


@celery_app.task(name="rag.ingest_document")
def ingest_document(version_id: str) -> dict[str, Any]:
    """Parse, chunk, embed, index, and atomically publish one immutable version."""

    session = SessionLocal()
    try:
        repository = TransactionalIngestionRepository(session)
        version = repository.get_version(version_id)
        if version.deleted_at is not None or version.status in {
            VersionStatus.DELETING,
            VersionStatus.DELETED,
        }:
            return {"version_id": version_id, "status": "CANCELLED"}

        service = IngestionService(
            blob_store=create_blob_store(settings),
            repository=repository,
            embedding_provider=create_embedding_provider(settings),
            chunker=Chunker(
                target_tokens=settings.chunk_target_tokens,
                max_tokens=settings.chunk_max_tokens,
                min_tokens=settings.chunk_min_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            ),
        )
        result = service.ingest_sync(version_id)
        return {
            "version_id": result.version_id,
            "status": result.status.value,
            "chunk_count": result.chunk_count,
            "embedding_dimensions": result.embedding_dimensions,
        }
    except IngestionError:
        # The orchestration layer has already persisted a precise failure state.
        raise
    except Exception:
        session.rollback()
        try:
            repository = IngestionRepository(session)
            version = repository.get_version(version_id)
            if version.document.deleted_at is None:
                repository.set_status(
                    version_id,
                    VersionStatus.FAILED_INDEX,
                    error_code="worker_failure",
                    error_message="The background ingestion worker failed unexpectedly.",
                )
                session.commit()
        except Exception:
            session.rollback()
        raise
    finally:
        session.close()
