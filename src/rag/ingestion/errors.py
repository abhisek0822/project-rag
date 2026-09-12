"""Errors and lifecycle states raised by the ingestion pipeline.

The exceptions deliberately carry stable, machine-readable codes.  API and
worker layers can map them to HTTP responses or job states without parsing an
exception message.
"""

from __future__ import annotations

from enum import Enum


class IngestionStatus(str, Enum):
    """Document-version states understood by the ingestion service."""

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


class IngestionError(Exception):
    """Base class for expected, user-facing ingestion failures."""

    code = "ingestion_error"
    status = IngestionStatus.FAILED_PARSE


class UnsupportedFileTypeError(IngestionError):
    code = "unsupported_file_type"
    status = IngestionStatus.REJECTED

    def __init__(self, filename: str) -> None:
        super().__init__(f"Unsupported file type: {filename}")
        self.filename = filename


class ParserDependencyError(IngestionError):
    code = "parser_dependency_missing"
    status = IngestionStatus.FAILED_PARSE


class DocumentParseError(IngestionError):
    code = "document_parse_failed"
    status = IngestionStatus.FAILED_PARSE


class EmptyDocumentError(DocumentParseError):
    code = "empty_document"

    def __init__(self, filename: str = "document") -> None:
        super().__init__(f"No readable text was found in {filename}")
        self.filename = filename


class NeedsOCRError(DocumentParseError):
    code = "needs_ocr"
    status = IngestionStatus.NEEDS_OCR

    def __init__(self, filename: str = "PDF") -> None:
        super().__init__(f"No usable text layer was found in {filename}; OCR is required")
        self.filename = filename


class ChunkingError(IngestionError):
    code = "chunking_failed"
    status = IngestionStatus.FAILED_PARSE


class EmbeddingError(IngestionError):
    code = "embedding_failed"
    status = IngestionStatus.FAILED_EMBED


class EmbeddingValidationError(EmbeddingError):
    code = "invalid_embedding_response"


class IndexingError(IngestionError):
    code = "indexing_failed"
    status = IngestionStatus.FAILED_INDEX
