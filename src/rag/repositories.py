"""Persistence repositories used by upload, ingestion, and retrieval services.

Methods flush but do not commit. The API or worker owns the transaction, normally
through :func:`rag.db.session_scope`, so multi-step transitions stay atomic.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.orm import Session, selectinload

from rag.config import Settings, get_settings
from rag.models import (
    EMBEDDING_DIMENSIONS,
    Chunk,
    Document,
    DocumentStatus,
    DocumentVersion,
    IngestionJob,
    JobStage,
    JobStatus,
    VersionStatus,
)
from rag.retrieval.models import RetrievalFilters, SearchCandidate


class RepositoryError(RuntimeError):
    """Base class for persistence failures with a useful domain meaning."""


class NotFoundError(RepositoryError):
    pass


class PublishValidationError(RepositoryError):
    pass


@dataclass(frozen=True)
class UploadRegistration:
    document: Document
    version: DocumentVersion
    job: IngestionJob | None
    created: bool


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).upper()


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_uuid(value: uuid.UUID | str, *, namespace_seed: str | None = None) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as error:
        if namespace_seed is None:
            raise ValueError(f"Expected a UUID, got {value!r}") from error
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace_seed}:{value}")


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class DocumentRepository:
    """CRUD and immutable-version operations for the document catalogue."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    def get(
        self,
        document_id: uuid.UUID | str,
        *,
        tenant_id: str | None = None,
        include_deleted: bool = False,
    ) -> Document:
        statement = (
            select(Document)
            .where(Document.id == _as_uuid(document_id))
            .options(selectinload(Document.versions))
        )
        if tenant_id is not None:
            statement = statement.where(Document.tenant_id == tenant_id)
        if not include_deleted:
            statement = statement.where(Document.deleted_at.is_(None))
        document = self.session.scalar(statement)
        if document is None:
            raise NotFoundError(f"Document {document_id} was not found")
        return document

    get_document = get

    def get_version(self, version_id: uuid.UUID | str) -> DocumentVersion:
        statement = (
            select(DocumentVersion)
            .where(DocumentVersion.id == _as_uuid(version_id))
            .options(selectinload(DocumentVersion.document))
        )
        version = self.session.scalar(statement)
        if version is None:
            raise NotFoundError(f"Document version {version_id} was not found")
        return version

    def list(
        self,
        *,
        tenant_id: str = "default",
        collection_id: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.tenant_id == tenant_id)
            .order_by(desc(Document.created_at))
            .limit(min(max(limit, 1), 500))
            .offset(max(offset, 0))
        )
        if collection_id is not None:
            statement = statement.where(Document.collection_id == collection_id)
        if not include_deleted:
            statement = statement.where(Document.deleted_at.is_(None))
        return list(self.session.scalars(statement).all())

    list_documents = list

    def find_by_content_hash(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        content_hash: str,
        include_deleted: bool = False,
    ) -> DocumentVersion | None:
        statement = (
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.tenant_id == tenant_id,
                Document.collection_id == collection_id,
                DocumentVersion.content_hash == content_hash.lower(),
            )
            .options(selectinload(DocumentVersion.document))
            .order_by(desc(DocumentVersion.created_at))
        )
        if not include_deleted:
            statement = statement.where(
                Document.deleted_at.is_(None),
                DocumentVersion.deleted_at.is_(None),
                DocumentVersion.status != VersionStatus.DELETED,
            )
        return self.session.scalars(statement).first()

    def create_document(
        self,
        *,
        filename: str,
        tenant_id: str = "default",
        collection_id: str = "default",
        title: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Document:
        document = Document(
            tenant_id=tenant_id,
            collection_id=collection_id,
            filename=filename,
            title=title,
            tags=list(tags),
            metadata_=dict(metadata or {}),
        )
        self.session.add(document)
        self.session.flush()
        return document

    def list_ready_versions(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        document_ids: Sequence[str] = (),
    ) -> list[DocumentVersion]:
        """Return current READY versions with scope enforced in the query."""

        statement = (
            select(DocumentVersion)
            .join(Document, Document.current_version_id == DocumentVersion.id)
            .where(
                Document.tenant_id == tenant_id,
                Document.collection_id == collection_id,
                Document.status == DocumentStatus.READY,
                Document.deleted_at.is_(None),
                DocumentVersion.status == VersionStatus.READY,
                DocumentVersion.deleted_at.is_(None),
            )
            .options(selectinload(DocumentVersion.document))
            .order_by(Document.created_at, Document.id)
        )
        if document_ids:
            statement = statement.where(
                Document.id.in_([_as_uuid(document_id) for document_id in document_ids])
            )
        return list(self.session.scalars(statement).all())

    create = create_document

    def create_version(
        self,
        *,
        document: Document | None = None,
        document_id: uuid.UUID | str | None = None,
        content_hash: str,
        blob_uri: str,
        filename: str,
        mime_type: str,
        byte_size: int,
        index_generation: int = 1,
        provenance: Mapping[str, Any] | None = None,
    ) -> DocumentVersion:
        if document is None:
            if document_id is None:
                raise ValueError("document or document_id is required")
            document = self.get(document_id, include_deleted=True)
        digest = content_hash.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("content_hash must be a lowercase or uppercase SHA-256 hex digest")
        if byte_size < 0:
            raise ValueError("byte_size cannot be negative")

        # Serialize additions to the same logical document so version_number is stable.
        self.session.execute(
            select(Document.id).where(Document.id == document.id).with_for_update()
        )
        last_number = self.session.scalar(
            select(func.max(DocumentVersion.version_number)).where(
                DocumentVersion.document_id == document.id
            )
        )
        version = DocumentVersion(
            document_id=document.id,
            tenant_id=document.tenant_id,
            collection_id=document.collection_id,
            version_number=(last_number or 0) + 1,
            content_hash=digest,
            blob_uri=blob_uri,
            original_filename=filename,
            mime_type=mime_type,
            byte_size=byte_size,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.embedding_model,
            embedding_dimensions=self.settings.embedding_dimensions,
            index_generation=index_generation,
            provenance=dict(provenance or {}),
        )
        self.session.add(version)
        document.status = DocumentStatus.UPLOADED
        self.session.flush()
        return version

    def register_upload(
        self,
        *,
        filename: str,
        content_hash: str,
        blob_uri: str,
        mime_type: str,
        byte_size: int,
        tenant_id: str = "default",
        collection_id: str = "default",
        title: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> UploadRegistration:
        """Register an upload or return its existing record for idempotency."""

        existing = self.find_by_content_hash(
            tenant_id=tenant_id,
            collection_id=collection_id,
            content_hash=content_hash,
        )
        if existing is not None:
            job = self.session.scalars(
                select(IngestionJob)
                .where(IngestionJob.version_id == existing.id)
                .order_by(desc(IngestionJob.created_at))
            ).first()
            return UploadRegistration(existing.document, existing, job, False)

        document = self.create_document(
            filename=filename,
            tenant_id=tenant_id,
            collection_id=collection_id,
            title=title,
            tags=tags,
            metadata=metadata,
        )
        version = self.create_version(
            document=document,
            content_hash=content_hash,
            blob_uri=blob_uri,
            filename=filename,
            mime_type=mime_type,
            byte_size=byte_size,
            provenance=provenance,
        )
        job = IngestionJob(version_id=version.id)
        self.session.add(job)
        self.session.flush()
        return UploadRegistration(document, version, job, True)

    def update_status(self, document_id: uuid.UUID | str, status: DocumentStatus | str) -> Document:
        document = self.get(document_id, include_deleted=True)
        document.status = DocumentStatus(_enum_value(status))
        self.session.flush()
        return document

    set_status = update_status

    def mark_deleting(self, document_id: uuid.UUID | str) -> Document:
        """Immediately make a document invisible to all retrieval queries."""

        document = self.get(document_id, include_deleted=True)
        now = datetime.now(timezone.utc)
        document.status = DocumentStatus.DELETING
        document.current_version_id = None
        document.deleted_at = now
        self.session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .values(status=VersionStatus.DELETING, deleted_at=now)
        )
        self.session.flush()
        return document

    delete = mark_deleting

    def mark_deleted(self, document_id: uuid.UUID | str) -> Document:
        document = self.get(document_id, include_deleted=True)
        now = document.deleted_at or datetime.now(timezone.utc)
        document.status = DocumentStatus.DELETED
        document.current_version_id = None
        document.deleted_at = now
        self.session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .values(status=VersionStatus.DELETED, deleted_at=now)
        )
        self.session.flush()
        return document


class ChunkRepository:
    """Chunk persistence plus pre-filtered dense and lexical retrieval."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def list_for_version(
        self, version_id: uuid.UUID | str, *, include_embeddings: bool = True
    ) -> list[Chunk]:
        # SQLAlchemy does not defer pgvector by default; the flag is retained as a
        # stable API point for a future large-vector optimization.
        del include_embeddings
        statement = (
            select(Chunk).where(Chunk.version_id == _as_uuid(version_id)).order_by(Chunk.ordinal)
        )
        return list(self.session.scalars(statement).all())

    def delete_for_version(self, version_id: uuid.UUID | str) -> int:
        result = self.session.execute(delete(Chunk).where(Chunk.version_id == _as_uuid(version_id)))
        return int(result.rowcount or 0)

    def replace_for_version(self, version: DocumentVersion, chunks: Iterable[Any]) -> list[Chunk]:
        self.delete_for_version(version.id)
        persisted: list[Chunk] = []
        for fallback_ordinal, source in enumerate(chunks):
            ordinal = int(_read(source, "ordinal", fallback_ordinal))
            canonical_text = str(_read(source, "text", "")).strip()
            if not canonical_text:
                raise ValueError(f"Chunk {ordinal} has no text")
            content_hash = str(
                _read(source, "content_hash", hashlib.sha256(canonical_text.encode()).hexdigest())
            ).lower()
            source_id = _read(source, "id", None) or _read(source, "chunk_id", None)
            if source_id is None:
                chunk_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"rag-chunk:{version.id}:{ordinal}:{content_hash}",
                )
            else:
                chunk_id = _as_uuid(source_id, namespace_seed=f"rag-chunk:{version.id}")

            section_path = _read(source, "section_path", ()) or ()
            if isinstance(section_path, str):
                section_path = (section_path,)
            page = _read(source, "page", None)
            block_type = _read(source, "block_type", _read(source, "kind", "paragraph"))
            block_type = str(getattr(block_type, "value", block_type))
            persisted.append(
                Chunk(
                    id=chunk_id,
                    version_id=version.id,
                    document_id=version.document_id,
                    tenant_id=version.tenant_id,
                    collection_id=version.collection_id,
                    ordinal=ordinal,
                    text=canonical_text,
                    embedding_text=str(_read(source, "embedding_text", canonical_text)).strip(),
                    token_count=int(_read(source, "token_count", 0)),
                    content_hash=content_hash,
                    block_type=block_type,
                    page_start=_read(source, "page_start", page),
                    page_end=_read(source, "page_end", page),
                    section_path=list(section_path),
                    source_locator=dict(_read(source, "source_locator", {}) or {}),
                    provenance=dict(_read(source, "provenance", {}) or {}),
                    access_control=dict(
                        _read(source, "access_control", {"public": True}) or {"public": True}
                    ),
                )
            )
        self.session.add_all(persisted)
        self.session.flush()
        version.expected_chunk_count = len(persisted)
        return persisted

    insert_many = replace_for_version

    def save_embeddings(
        self,
        version_id: uuid.UUID | str,
        embeddings: Iterable[tuple[uuid.UUID | str, Sequence[float]]],
        dimensions: int,
    ) -> int:
        if dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding dimension {dimensions} does not match schema dimension "
                f"{EMBEDDING_DIMENSIONS}"
            )
        version_uuid = _as_uuid(version_id)
        updated = 0
        seen: set[uuid.UUID] = set()
        for source_id, vector in embeddings:
            chunk_id = _as_uuid(source_id, namespace_seed=f"rag-chunk:{version_uuid}")
            if chunk_id in seen:
                raise ValueError(f"Duplicate embedding supplied for chunk {source_id}")
            seen.add(chunk_id)
            numeric_vector = [float(value) for value in vector]
            if len(numeric_vector) != dimensions:
                raise ValueError(
                    f"Chunk {source_id} has {len(numeric_vector)} dimensions; expected {dimensions}"
                )
            if not all(math.isfinite(value) for value in numeric_vector):
                raise ValueError(f"Chunk {source_id} embedding contains a non-finite value")
            result = self.session.execute(
                update(Chunk)
                .where(Chunk.id == chunk_id, Chunk.version_id == version_uuid)
                .values(embedding=numeric_vector)
            )
            if not result.rowcount:
                raise NotFoundError(f"Chunk {source_id} does not belong to version {version_id}")
            updated += int(result.rowcount)
        self.session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version_uuid)
            .values(indexed_chunk_count=updated)
        )
        self.session.flush()
        return updated

    update_embeddings = save_embeddings

    def _retrieval_statement(self, filters: RetrievalFilters):
        if not filters.tenant_id.strip():
            raise ValueError("tenant_id is required for retrieval")
        statement = (
            select(Chunk, Document.filename)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.tenant_id == filters.tenant_id,
                Document.tenant_id == filters.tenant_id,
                Document.deleted_at.is_(None),
                Document.status == DocumentStatus.READY,
                Document.current_version_id == Chunk.version_id,
                DocumentVersion.status == VersionStatus.READY,
                Chunk.embedding.is_not(None),
            )
        )
        if filters.collection_id is not None:
            statement = statement.where(Chunk.collection_id == filters.collection_id)
        if filters.document_ids:
            statement = statement.where(
                Chunk.document_id.in_([_as_uuid(value) for value in filters.document_ids])
            )
        if filters.tags:
            statement = statement.where(Document.tags.contains(list(filters.tags)))
        created_after = _parse_iso_datetime(filters.created_after)
        created_before = _parse_iso_datetime(filters.created_before)
        if created_after is not None:
            statement = statement.where(Document.created_at >= created_after)
        if created_before is not None:
            statement = statement.where(Document.created_at <= created_before)
        return statement

    @staticmethod
    def _candidate(chunk: Chunk, filename: str, score: float) -> SearchCandidate:
        raw_vector = chunk.embedding
        embedding = tuple(float(value) for value in raw_vector) if raw_vector is not None else None
        return SearchCandidate(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            version_id=str(chunk.version_id),
            filename=filename,
            text=chunk.text,
            score=float(score),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=tuple(str(part) for part in chunk.section_path),
            content_hash=chunk.content_hash,
            embedding=embedding,
            metadata={
                "ordinal": chunk.ordinal,
                "block_type": chunk.block_type,
                "source_locator": dict(chunk.source_locator),
                "provenance": dict(chunk.provenance),
            },
        )

    def dense_candidates(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int,
        filters: RetrievalFilters,
    ) -> list[SearchCandidate]:
        vector = [float(value) for value in query_embedding]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Query has {len(vector)} dimensions; expected {EMBEDDING_DIMENSIONS}")
        distance = Chunk.embedding.cosine_distance(vector)
        statement = (
            self._retrieval_statement(filters)
            .add_columns(distance.label("distance"))
            .order_by(distance)
            .limit(min(max(limit, 1), 500))
        )
        rows = self.session.execute(statement).all()
        return [
            self._candidate(chunk, filename, 1.0 - float(cosine_distance))
            for chunk, filename, cosine_distance in rows
        ]

    async def dense_search(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int,
        filters: RetrievalFilters,
    ) -> Sequence[SearchCandidate]:
        # The application deliberately uses sync SQLAlchemy. FastAPI executes sync
        # endpoints in its worker pool; the async retrieval port remains convenient
        # for model/network adapters. This short DB call is therefore synchronous.
        return self.dense_candidates(query_embedding=query_embedding, limit=limit, filters=filters)

    def lexical_candidates(
        self,
        *,
        query: str,
        limit: int,
        filters: RetrievalFilters,
    ) -> list[SearchCandidate]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        # Natural-language questions often contain terms such as "many" that do
        # not appear verbatim in the answer. OR keeps lexical recall broad; the
        # fusion/reranking stages decide which candidates are actually relevant.
        query_terms = re.findall(r"[\w'-]+", normalized_query, flags=re.UNICODE)[:32]
        if not query_terms:
            return []
        tsquery = func.websearch_to_tsquery("english", " OR ".join(query_terms))
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery)
        statement = (
            self._retrieval_statement(filters)
            .where(Chunk.search_vector.op("@@")(tsquery))
            .add_columns(rank.label("rank"))
            .order_by(desc(rank))
            .limit(min(max(limit, 1), 500))
        )
        rows = self.session.execute(statement).all()
        return [self._candidate(chunk, filename, float(score)) for chunk, filename, score in rows]

    async def lexical_search(
        self,
        *,
        query: str,
        limit: int,
        filters: RetrievalFilters,
    ) -> Sequence[SearchCandidate]:
        return self.lexical_candidates(query=query, limit=limit, filters=filters)


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, version_id: uuid.UUID | str) -> IngestionJob:
        job = IngestionJob(version_id=_as_uuid(version_id))
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: uuid.UUID | str) -> IngestionJob:
        job = self.session.get(IngestionJob, _as_uuid(job_id))
        if job is None:
            raise NotFoundError(f"Ingestion job {job_id} was not found")
        return job

    def latest_for_version(self, version_id: uuid.UUID | str) -> IngestionJob | None:
        return self.session.scalars(
            select(IngestionJob)
            .where(IngestionJob.version_id == _as_uuid(version_id))
            .order_by(desc(IngestionJob.created_at))
        ).first()

    def update(
        self,
        job_id: uuid.UUID | str,
        *,
        stage: JobStage | str | None = None,
        status: JobStatus | str | None = None,
        checkpoint: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        lease_owner: str | None = None,
    ) -> IngestionJob:
        job = self.get(job_id)
        now = datetime.now(timezone.utc)
        if stage is not None:
            job.stage = JobStage(_enum_value(stage))
        if status is not None:
            next_status = JobStatus(_enum_value(status))
            job.status = next_status
            if next_status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = now
            if next_status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                job.finished_at = now
        if checkpoint is not None:
            job.checkpoint = dict(checkpoint)
        job.error_code = error_code
        job.error_message = error_message
        if lease_owner is not None:
            job.lease_owner = lease_owner
        job.heartbeat_at = now
        self.session.flush()
        return job


class IngestionRepository:
    """Adapter implementing the persistence port consumed by ingestion orchestration."""

    _STAGE_BY_STATUS = {
        VersionStatus.UPLOADED: JobStage.ACCEPT,
        VersionStatus.QUEUED: JobStage.ACCEPT,
        VersionStatus.PARSING: JobStage.PARSE,
        VersionStatus.CHUNKING: JobStage.CHUNK,
        VersionStatus.EMBEDDING: JobStage.EMBED,
        VersionStatus.INDEXING: JobStage.INDEX,
        VersionStatus.READY: JobStage.PUBLISH,
        VersionStatus.DELETING: JobStage.CLEANUP,
        VersionStatus.DELETED: JobStage.CLEANUP,
    }

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.documents = DocumentRepository(session, settings=settings)
        self.chunks = ChunkRepository(session)
        self.jobs = IngestionJobRepository(session)

    def get_version(self, version_id: uuid.UUID | str) -> DocumentVersion:
        return self.documents.get_version(version_id)

    def set_status(
        self,
        version_id: uuid.UUID | str,
        status: VersionStatus | str,
        *,
        stage: JobStage | str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> DocumentVersion:
        version = self.get_version(version_id)
        next_status = VersionStatus(_enum_value(status))
        version.status = next_status
        version.error_code = error_code
        version.error_message = error_message
        document = version.document

        if next_status == VersionStatus.READY:
            document.status = DocumentStatus.READY
        elif next_status == VersionStatus.NEEDS_OCR:
            document.status = DocumentStatus.NEEDS_OCR
        elif next_status in {
            VersionStatus.REJECTED,
            VersionStatus.FAILED_PARSE,
            VersionStatus.FAILED_EMBED,
            VersionStatus.FAILED_INDEX,
        }:
            document.status = DocumentStatus.FAILED
        elif next_status == VersionStatus.DELETING:
            document.status = DocumentStatus.DELETING
        elif next_status == VersionStatus.DELETED:
            document.status = DocumentStatus.DELETED
        elif next_status != VersionStatus.CANCELLED:
            document.status = DocumentStatus.PROCESSING

        job = self.jobs.latest_for_version(version.id)
        if job is not None:
            failure = next_status in {
                VersionStatus.REJECTED,
                VersionStatus.NEEDS_OCR,
                VersionStatus.FAILED_PARSE,
                VersionStatus.FAILED_EMBED,
                VersionStatus.FAILED_INDEX,
            }
            if next_status == VersionStatus.READY:
                job_status = JobStatus.SUCCEEDED
            elif next_status == VersionStatus.CANCELLED:
                job_status = JobStatus.CANCELLED
            elif failure:
                job_status = JobStatus.FAILED
            else:
                job_status = JobStatus.RUNNING
            inferred_stage = self._STAGE_BY_STATUS.get(next_status, job.stage)
            self.jobs.update(
                job.id,
                stage=stage or inferred_stage,
                status=job_status,
                error_code=error_code,
                error_message=error_message,
            )
        self.session.flush()
        return version

    def replace_chunks(self, version_id: uuid.UUID | str, chunks: Iterable[Any]) -> list[Chunk]:
        version = self.get_version(version_id)
        return self.chunks.replace_for_version(version, chunks)

    def save_embeddings(
        self,
        version_id: uuid.UUID | str,
        embeddings: Iterable[tuple[uuid.UUID | str, Sequence[float]]],
        dimensions: int,
    ) -> int:
        return self.chunks.save_embeddings(version_id, embeddings, dimensions)

    def publish_version(
        self, version_id: uuid.UUID | str, expected_chunk_count: int
    ) -> DocumentVersion:
        """Atomically validate an index and expose it as the document's current version."""

        version = self.get_version(version_id)
        version_uuid = version.id
        actual_count, embedded_count = self.session.execute(
            select(
                func.count(Chunk.id),
                func.count(Chunk.embedding),
            ).where(Chunk.version_id == version_uuid)
        ).one()
        if actual_count != expected_chunk_count:
            message = (
                f"Version {version_id} expected {expected_chunk_count} chunks "
                f"but has {actual_count}"
            )
            raise PublishValidationError(message)
        if embedded_count != actual_count:
            raise PublishValidationError(
                f"Version {version_id} has vectors for {embedded_count} of {actual_count} chunks"
            )
        version.expected_chunk_count = expected_chunk_count
        version.indexed_chunk_count = embedded_count
        version.status = VersionStatus.READY
        version.error_code = None
        version.error_message = None
        version.ready_at = datetime.now(timezone.utc)
        version.document.current_version_id = version.id
        version.document.status = DocumentStatus.READY
        job = self.jobs.latest_for_version(version.id)
        if job is not None:
            self.jobs.update(job.id, stage=JobStage.PUBLISH, status=JobStatus.SUCCEEDED)
        self.session.flush()
        return version
