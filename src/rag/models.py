"""SQLAlchemy entities for documents, immutable versions, chunks, and jobs.

The migration is the production schema source of truth. These mappings mirror it
and deliberately denormalize tenant/collection identifiers onto chunks so every
retrieval query can apply security filters *before* ranking.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from rag.db import Base

EMBEDDING_DIMENSIONS = 1536

JsonObject = dict[str, Any]
json_object_type = MutableDict.as_mutable(JSON().with_variant(JSONB, "postgresql"))
json_list_type = MutableList.as_mutable(JSON().with_variant(JSONB, "postgresql"))


class DocumentStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    NEEDS_OCR = "NEEDS_OCR"
    FAILED = "FAILED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class VersionStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    REJECTED = "REJECTED"
    NEEDS_OCR = "NEEDS_OCR"
    FAILED_PARSE = "FAILED_PARSE"
    FAILED_EMBED = "FAILED_EMBED"
    FAILED_INDEX = "FAILED_INDEX"
    CANCELLED = "CANCELLED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStage(str, enum.Enum):
    ACCEPT = "ACCEPT"
    PARSE = "PARSE"
    CHUNK = "CHUNK"
    EMBED = "EMBED"
    INDEX = "INDEX"
    PUBLISH = "PUBLISH"
    CLEANUP = "CLEANUP"


def _string_enum(enum_type: type[enum.Enum], name: str) -> SqlEnum:
    """Use readable strings and portable CHECK constraints, not PG enum types."""

    return SqlEnum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_scope_status", "tenant_id", "collection_id", "status"),
        Index("ix_documents_deleted_at", "deleted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, default="default")
    collection_id: Mapped[str] = mapped_column(String(100), nullable=False, default="default")
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[DocumentStatus] = mapped_column(
        _string_enum(DocumentStatus, "document_status"),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        server_default=DocumentStatus.UPLOADED.value,
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_versions.id",
            name="fk_documents_current_version",
            ondelete="SET NULL",
            use_alter=True,
        ),
    )
    tags: Mapped[list[Any]] = mapped_column(json_list_type, default=list, nullable=False)
    metadata_: Mapped[JsonObject] = mapped_column(
        "metadata", json_object_type, default=dict, nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="DocumentVersion.document_id",
        order_by="DocumentVersion.version_number",
    )
    current_version: Mapped[DocumentVersion | None] = relationship(
        foreign_keys=[current_version_id], post_update=True
    )

    @property
    def display_name(self) -> str:
        return self.title or self.filename


class DocumentVersion(TimestampMixin, Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_version_document_number"),
        UniqueConstraint(
            "document_id", "content_hash", "index_generation", name="uq_version_pipeline_identity"
        ),
        Index("ix_versions_scope_hash", "tenant_id", "collection_id", "content_hash"),
        Index("ix_versions_status", "status"),
        CheckConstraint("byte_size >= 0", name="ck_versions_byte_size_nonnegative"),
        CheckConstraint("embedding_dimensions > 0", name="ck_versions_dimensions_positive"),
        CheckConstraint("expected_chunk_count >= 0", name="ck_versions_expected_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    collection_id: Mapped[str] = mapped_column(String(100), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    blob_uri: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)

    parser_version: Mapped[str] = mapped_column(String(100), default="1", nullable=False)
    normalizer_version: Mapped[str] = mapped_column(String(100), default="1", nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(100), default="1", nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(
        Integer, default=EMBEDDING_DIMENSIONS, nullable=False
    )
    distance_metric: Mapped[str] = mapped_column(String(30), default="cosine", nullable=False)
    embedding_normalized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    index_generation: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    expected_chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    indexed_chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    status: Mapped[VersionStatus] = mapped_column(
        _string_enum(VersionStatus, "version_status"),
        nullable=False,
        default=VersionStatus.UPLOADED,
        server_default=VersionStatus.UPLOADED.value,
    )
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    provenance: Mapped[JsonObject] = mapped_column(json_object_type, default=dict, nullable=False)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="versions", foreign_keys=[document_id])
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="Chunk.ordinal",
    )
    jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )

    @property
    def filename(self) -> str:
        return self.original_filename

    @property
    def title(self) -> str:
        if self.document is not None and self.document.title:
            return self.document.title
        return self.original_filename


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "ordinal", name="uq_chunks_version_ordinal"),
        Index("ix_chunks_scope", "tenant_id", "collection_id"),
        Index("ix_chunks_version", "version_id"),
        Index("ix_chunks_document", "document_id"),
        Index("ix_chunks_content_hash", "content_hash"),
        CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        CheckConstraint("token_count >= 0", name="ck_chunks_tokens_nonnegative"),
        CheckConstraint(
            "page_start IS NULL OR page_end IS NULL OR page_end >= page_start",
            name="ck_chunks_page_order",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    collection_id: Mapped[str] = mapped_column(String(100), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    block_type: Mapped[str] = mapped_column(String(50), default="paragraph", nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[list[Any]] = mapped_column(json_list_type, default=list, nullable=False)
    source_locator: Mapped[JsonObject] = mapped_column(
        json_object_type, default=dict, nullable=False
    )
    provenance: Mapped[JsonObject] = mapped_column(json_object_type, default=dict, nullable=False)
    access_control: Mapped[JsonObject] = mapped_column(
        json_object_type, default=lambda: {"public": True}, nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english'::regconfig, coalesce(text, ''))", persisted=True),
    )

    version: Mapped[DocumentVersion] = relationship(back_populates="chunks")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        Index("ix_jobs_status_created", "status", "created_at"),
        Index("ix_jobs_version", "version_id"),
        CheckConstraint("attempt_count >= 0", name="ck_jobs_attempt_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[JobStage] = mapped_column(
        _string_enum(JobStage, "job_stage"),
        default=JobStage.ACCEPT,
        server_default=JobStage.ACCEPT.value,
        nullable=False,
    )
    status: Mapped[JobStatus] = mapped_column(
        _string_enum(JobStatus, "job_status"),
        default=JobStatus.PENDING,
        server_default=JobStatus.PENDING.value,
        nullable=False,
    )
    checkpoint: Mapped[JsonObject] = mapped_column(json_object_type, default=dict, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(255))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    version: Mapped[DocumentVersion] = relationship(back_populates="jobs")
