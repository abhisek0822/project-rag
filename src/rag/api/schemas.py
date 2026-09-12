"""Public HTTP request and response contracts.

The API deliberately returns plain source metadata instead of database objects.
This keeps internal storage replaceable and prevents accidental exposure of model
vectors or private ingestion diagnostics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class UploadAccepted(BaseModel):
    document_id: str
    version_id: str
    job_id: str | None = None
    filename: str
    status: str = "QUEUED"
    duplicate: bool = False


class DocumentResponse(BaseModel):
    id: str
    filename: str
    title: str | None = None
    status: str
    size_bytes: int = 0
    mime_type: str = "application/octet-stream"
    chunk_count: int = 0
    version_id: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)
    collection_id: str = Field(default="default", min_length=1, max_length=100)
    document_ids: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank")
        return cleaned


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    collection_id: str = Field(default="default", min_length=1, max_length=100)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=8, ge=1, le=30)


class SourceResponse(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    ordinal: int = 0
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    match_count: int | None = Field(default=None, ge=1)


class AnswerDiagnostics(BaseModel):
    abstained: bool = False
    reason: str | None = None
    dense_candidates: int = 0
    lexical_candidates: int = 0
    fused_candidates: int = 0
    context_tokens: int = 0
    provider: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    diagnostics: AnswerDiagnostics = Field(default_factory=AnswerDiagnostics)


class SearchResponse(BaseModel):
    sources: list[SourceResponse]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    queue: bool
    blob_store: bool
    provider: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
