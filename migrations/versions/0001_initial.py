"""Create the initial self-managed RAG schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


document_status = sa.Enum(
    "UPLOADED",
    "PROCESSING",
    "READY",
    "NEEDS_OCR",
    "FAILED",
    "DELETING",
    "DELETED",
    name="document_status",
    native_enum=False,
    create_constraint=True,
)
version_status = sa.Enum(
    "UPLOADED",
    "QUEUED",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "READY",
    "REJECTED",
    "NEEDS_OCR",
    "FAILED_PARSE",
    "FAILED_EMBED",
    "FAILED_INDEX",
    "CANCELLED",
    "DELETING",
    "DELETED",
    name="version_status",
    native_enum=False,
    create_constraint=True,
)
job_stage = sa.Enum(
    "ACCEPT",
    "PARSE",
    "CHUNK",
    "EMBED",
    "INDEX",
    "PUBLISH",
    "CLEANUP",
    name="job_stage",
    native_enum=False,
    create_constraint=True,
)
job_status = sa.Enum(
    "PENDING",
    "RUNNING",
    "RETRYING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    name="job_status",
    native_enum=False,
    create_constraint=True,
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=100), nullable=False),
        sa.Column("collection_id", sa.String(length=100), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("status", document_status, server_default="UPLOADED", nullable=False),
        # The FK is added after document_versions to avoid a creation cycle.
        sa.Column("current_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_documents_scope_status",
        "documents",
        ["tenant_id", "collection_id", "status"],
    )
    op.create_index("ix_documents_deleted_at", "documents", ["deleted_at"])

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=100), nullable=False),
        sa.Column("collection_id", sa.String(length=100), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("blob_uri", sa.Text(), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("parser_version", sa.String(length=100), server_default="1", nullable=False),
        sa.Column("normalizer_version", sa.String(length=100), server_default="1", nullable=False),
        sa.Column("chunker_version", sa.String(length=100), server_default="1", nullable=False),
        sa.Column("embedding_provider", sa.String(length=100), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), server_default="1536", nullable=False),
        sa.Column("distance_metric", sa.String(length=30), server_default="cosine", nullable=False),
        sa.Column("embedding_normalized", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("index_generation", sa.Integer(), server_default="1", nullable=False),
        sa.Column("expected_chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("indexed_chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", version_status, server_default="UPLOADED", nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "provenance",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("byte_size >= 0", name="ck_versions_byte_size_nonnegative"),
        sa.CheckConstraint("embedding_dimensions > 0", name="ck_versions_dimensions_positive"),
        sa.CheckConstraint("expected_chunk_count >= 0", name="ck_versions_expected_nonnegative"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "content_hash", "index_generation", name="uq_version_pipeline_identity"
        ),
        sa.UniqueConstraint("document_id", "version_number", name="uq_version_document_number"),
    )
    op.create_index(
        "ix_versions_scope_hash",
        "document_versions",
        ["tenant_id", "collection_id", "content_hash"],
    )
    op.create_index("ix_versions_status", "document_versions", ["status"])
    op.create_foreign_key(
        "fk_documents_current_version",
        "documents",
        "document_versions",
        ["current_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=100), nullable=False),
        sa.Column("collection_id", sa.String(length=100), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("block_type", sa.String(length=50), server_default="paragraph", nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column(
            "section_path",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "source_locator",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "provenance",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "access_control",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{\"public\": true}'::jsonb"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(dim=1536), nullable=True),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english'::regconfig, coalesce(text, ''))", persisted=True),
            nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        sa.CheckConstraint("token_count >= 0", name="ck_chunks_tokens_nonnegative"),
        sa.CheckConstraint(
            "page_start IS NULL OR page_end IS NULL OR page_end >= page_start",
            name="ck_chunks_page_order",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id", "ordinal", name="uq_chunks_version_ordinal"),
    )
    op.create_index("ix_chunks_scope", "chunks", ["tenant_id", "collection_id"])
    op.create_index("ix_chunks_version", "chunks", ["version_id"])
    op.create_index("ix_chunks_document", "chunks", ["document_id"])
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])
    op.create_index(
        "ix_chunks_search_vector_gin",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"m": 16, "ef_construction": 64},
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", job_stage, server_default="ACCEPT", nullable=False),
        sa.Column("status", job_status, server_default="PENDING", nullable=False),
        sa.Column(
            "checkpoint",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_owner", sa.String(length=255), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_jobs_attempt_nonnegative"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_status_created", "ingestion_jobs", ["status", "created_at"])
    op.create_index("ix_jobs_version", "ingestion_jobs", ["version_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_version", table_name="ingestion_jobs")
    op.drop_index("ix_jobs_status_created", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_search_vector_gin", table_name="chunks")
    op.drop_index("ix_chunks_content_hash", table_name="chunks")
    op.drop_index("ix_chunks_document", table_name="chunks")
    op.drop_index("ix_chunks_version", table_name="chunks")
    op.drop_index("ix_chunks_scope", table_name="chunks")
    op.drop_table("chunks")

    op.drop_constraint("fk_documents_current_version", "documents", type_="foreignkey")
    op.drop_index("ix_versions_status", table_name="document_versions")
    op.drop_index("ix_versions_scope_hash", table_name="document_versions")
    op.drop_table("document_versions")

    op.drop_index("ix_documents_deleted_at", table_name="documents")
    op.drop_index("ix_documents_scope_status", table_name="documents")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
