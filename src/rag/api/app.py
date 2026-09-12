"""FastAPI surface for the custom-owned RAG application."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis import Redis

from rag import __version__
from rag.adapters import TokenOverlapReranker
from rag.api.schemas import (
    AnswerDiagnostics,
    AnswerRequest,
    AnswerResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SourceResponse,
    UploadAccepted,
)
from rag.blob_store import LocalBlobStore, S3BlobStore, create_blob_store, safe_filename
from rag.config import get_settings
from rag.db import SessionLocal, check_database
from rag.ingestion import SUPPORTED_EXTENSIONS, Normalizer, ParserRegistry
from rag.models import Document, DocumentStatus, IngestionJob, VersionStatus
from rag.providers import (
    ProviderConfigurationError,
    create_embedding_provider,
    create_generator,
    create_retrieval_service,
)
from rag.repositories import ChunkRepository, DocumentRepository, NotFoundError
from rag.retrieval import (
    ChatMessage,
    CountableDocument,
    ExactCountRequest,
    RetrievalConfig,
    RetrievalFilters,
    RetrievalService,
    count_exact_occurrences,
    format_exact_count_answer,
    parse_exact_count_request,
)
from rag.worker.celery_app import ingest_document

settings = get_settings()


def _find_web_root() -> Path:
    """Locate UI assets in both source checkouts and container installations."""

    candidates = (
        Path.cwd() / "web",
        Path(__file__).resolve().parents[3] / "web",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise RuntimeError("The web asset directory could not be found.")


web_root = _find_web_root()

app = FastAPI(
    title="Index — custom RAG",
    description="A self-managed ingestion, vector retrieval, ranking, and grounded-answer API.",
    version=__version__,
)
app.mount("/static", StaticFiles(directory=web_root), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(web_root / "index.html")


@app.get("/architecture", include_in_schema=False)
def architecture() -> FileResponse:
    """Serve the interactive, code-level project architecture walkthrough."""

    return FileResponse(web_root / "architecture.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    database_ok = _database_is_healthy()
    queue_ok = _queue_is_healthy()
    blob_ok = _blob_store_is_healthy()
    healthy = database_ok and queue_ok and blob_ok
    return HealthResponse(
        status="ok" if healthy else "degraded",
        database=database_ok,
        queue=queue_ok,
        blob_store=blob_ok,
        provider=(
            f"{settings.embedding_provider}/{getattr(settings, 'generation_provider', 'openai')}"
        ),
        version=__version__,
    )


@app.post(
    "/api/documents",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(file: UploadFile = File(...)) -> UploadAccepted:
    filename = safe_filename(file.filename or "document")
    extension = Path(filename).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Choose one of: {supported}",
        )

    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"The file exceeds the {settings.max_upload_bytes} byte upload limit.",
        )

    content_hash = hashlib.sha256(content).hexdigest()
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    # Idempotency is checked before writing so repeated uploads do not create
    # extra objects or queue duplicate ingestion work.
    with SessionLocal() as session:
        existing = DocumentRepository(session).find_by_content_hash(
            tenant_id="default",
            collection_id="default",
            content_hash=content_hash,
        )
        if existing is not None:
            job = _latest_job(existing.jobs)
            return UploadAccepted(
                document_id=str(existing.document_id),
                version_id=str(existing.id),
                job_id=str(job.id) if job else None,
                filename=existing.original_filename,
                status=_enum_text(existing.status),
                duplicate=True,
            )

    blob_store = create_blob_store(settings)
    blob_key = f"default/default/sha256/{content_hash[:2]}/{content_hash}"
    try:
        blob_uri = blob_store.put_bytes(blob_key, content, mime_type)
    except Exception as error:
        raise HTTPException(
            status_code=503, detail="Original-file storage is unavailable."
        ) from error

    try:
        with SessionLocal() as session:
            registration = DocumentRepository(session).register_upload(
                filename=filename,
                content_hash=content_hash,
                blob_uri=blob_uri,
                mime_type=mime_type,
                byte_size=len(content),
                tenant_id="default",
                collection_id="default",
                provenance={"upload": "browser_or_api"},
            )
            registration.version.status = VersionStatus.QUEUED
            registration.document.status = DocumentStatus.PROCESSING
            session.commit()
            document_id = str(registration.document.id)
            version_id = str(registration.version.id)
            job_id = str(registration.job.id) if registration.job else None
            created = registration.created
    except Exception:
        # The key is content-addressed; removing it is safe only when this request
        # did not race another successful registration for the same content.
        with SessionLocal() as session:
            shared = DocumentRepository(session).find_by_content_hash(
                tenant_id="default",
                collection_id="default",
                content_hash=content_hash,
            )
        if shared is None:
            blob_store.delete(blob_uri)
        raise

    if created:
        try:
            ingest_document.delay(version_id)
        except Exception as error:
            _record_queue_failure(version_id)
            raise HTTPException(
                status_code=503,
                detail="The document was stored, but the ingestion queue is unavailable.",
            ) from error

    return UploadAccepted(
        document_id=document_id,
        version_id=version_id,
        job_id=job_id,
        filename=filename,
        status="QUEUED",
        duplicate=not created,
    )


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    with SessionLocal() as session:
        documents = DocumentRepository(session).list(tenant_id="default", limit=500)
        items = [_document_response(document) for document in documents]
    return DocumentListResponse(items=items, total=len(items))


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str) -> DocumentResponse:
    _validated_uuid(document_id, "document")
    try:
        with SessionLocal() as session:
            document = DocumentRepository(session).get(document_id, tenant_id="default")
            return _document_response(document)
    except NotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str) -> Response:
    _validated_uuid(document_id, "document")
    try:
        with SessionLocal() as session:
            repository = DocumentRepository(session)
            document = repository.get(document_id, tenant_id="default", include_deleted=True)
            blob_uris = tuple(version.blob_uri for version in document.versions)
            version_ids = tuple(str(version.id) for version in document.versions)
            repository.mark_deleting(document_id)
            session.commit()
    except NotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    blob_store = create_blob_store(settings)
    failed_blobs: list[str] = []
    for blob_uri in dict.fromkeys(blob_uris):
        try:
            blob_store.delete(blob_uri)
        except Exception:
            failed_blobs.append(blob_uri)
    if failed_blobs:
        raise HTTPException(
            status_code=503,
            detail="The document is hidden, but its original file could not yet be removed.",
        )

    with SessionLocal() as session:
        chunks = ChunkRepository(session)
        for version_id in version_ids:
            chunks.delete_for_version(version_id)
        DocumentRepository(session).mark_deleted(document_id)
        session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job_uuid = _validated_uuid(job_id, "job")
    with SessionLocal() as session:
        job = session.get(IngestionJob, job_uuid)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job was not found.")
        return {
            "id": str(job.id),
            "version_id": str(job.version_id),
            "status": _enum_text(job.status),
            "stage": _enum_text(job.stage),
            "attempt_count": job.attempt_count,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }


@app.post("/api/retrieval/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    filters = _retrieval_filters(request.collection_id, request.document_ids)
    try:
        with SessionLocal() as session:
            service = _retrieval_service(ChunkRepository(session), source_limit=request.limit)
            result = await service.search(request.query, filters=filters)
            sources = [
                _candidate_response(candidate, index + 1)
                for index, candidate in enumerate(result.candidates)
            ]
            diagnostics = _diagnostics_dict(result.diagnostics)
    except ProviderConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return SearchResponse(sources=sources, diagnostics=diagnostics)


@app.post("/api/answers", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest) -> AnswerResponse:
    filters = _retrieval_filters(request.collection_id, request.document_ids)
    exact_count_request = parse_exact_count_request(request.question)
    if exact_count_request is not None:
        try:
            return await asyncio.to_thread(
                _answer_exact_count,
                exact_count_request,
                filters,
            )
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="The complete selected documents could not be inspected for an exact count.",
            ) from error

    conversation = tuple(
        ChatMessage(role=message.role, content=message.content) for message in request.history[-6:]
    )
    try:
        with SessionLocal() as session:
            service = create_retrieval_service(ChunkRepository(session), settings)
            result = await service.answer(
                request.question,
                filters=filters,
                conversation=conversation,
            )
            sources = [_source_response(source) for source in result.sources]
    except ProviderConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    diagnostics = result.diagnostics
    return AnswerResponse(
        answer=result.answer,
        sources=sources,
        citations=list(result.cited_source_ids),
        diagnostics=AnswerDiagnostics(
            abstained=result.abstained,
            reason=result.abstention_reason,
            dense_candidates=diagnostics.dense_candidates,
            lexical_candidates=diagnostics.lexical_candidates,
            fused_candidates=diagnostics.fused_candidates,
            context_tokens=diagnostics.context_estimated_tokens,
            provider=diagnostics.embedding_provider,
            warnings=list(diagnostics.warnings),
        ),
    )


def _answer_exact_count(
    count_request: ExactCountRequest,
    filters: RetrievalFilters,
) -> AnswerResponse:
    """Count against reparsed originals so chunk overlap cannot duplicate matches."""

    with SessionLocal() as session:
        versions = DocumentRepository(session).list_ready_versions(
            tenant_id=filters.tenant_id,
            collection_id=filters.collection_id or "default",
            document_ids=filters.document_ids,
        )
        sources = tuple(
            (
                str(version.document_id),
                str(version.id),
                version.original_filename,
                version.blob_uri,
            )
            for version in versions
        )

    if not sources:
        return AnswerResponse(
            answer="No selected document is ready for an exact count.",
            diagnostics=AnswerDiagnostics(
                abstained=True,
                reason="no_ready_documents",
                provider="deterministic-exact-count",
            ),
        )

    blob_store = create_blob_store(settings)
    parser_registry = ParserRegistry()
    normalizer = Normalizer()
    documents: list[CountableDocument] = []
    for document_id, version_id, filename, blob_uri in sources:
        raw = blob_store.get_bytes(blob_uri)
        parsed = parser_registry.parse(raw, filename)
        blocks = tuple(normalizer.normalize(parsed))
        documents.append(
            CountableDocument(
                document_id=document_id,
                version_id=version_id,
                filename=filename,
                blocks=blocks,
            )
        )

    result = count_exact_occurrences(count_request, tuple(documents))
    evidence_responses = [
        SourceResponse(
            source_id=f"S{index}",
            chunk_id=str(
                uuid5(
                    NAMESPACE_URL,
                    "rag-exact-count:"
                    f"{evidence.version_id}:{evidence.block_order}:{count_request.term}",
                )
            ),
            document_id=evidence.document_id,
            filename=evidence.filename,
            text=evidence.text,
            score=1.0,
            ordinal=evidence.block_order,
            page_start=evidence.page,
            page_end=evidence.page,
            section_path=list(evidence.section_path),
            match_count=evidence.occurrence_count,
        )
        for index, evidence in enumerate(result.evidence, start=1)
    ]
    return AnswerResponse(
        answer=format_exact_count_answer(result),
        sources=evidence_responses,
        citations=[source.source_id for source in evidence_responses],
        diagnostics=AnswerDiagnostics(
            abstained=False,
            provider="deterministic-exact-count",
            warnings=["complete_source_scan"],
        ),
    )


def _retrieval_service(repository: ChunkRepository, *, source_limit: int) -> RetrievalService:
    # Search inspection can request its own result count while retaining all other
    # application defaults. The generator is unused by RetrievalService.search().
    return RetrievalService(
        repository=repository,
        embedder=create_embedding_provider(settings),
        generator=create_generator(settings),
        reranker=TokenOverlapReranker(),
        config=RetrievalConfig(
            dense_limit=settings.dense_candidate_limit,
            lexical_limit=settings.lexical_candidate_limit,
            fusion_limit=max(settings.rerank_candidate_limit, source_limit),
            source_limit=source_limit,
            max_context_tokens=settings.context_token_budget,
        ),
    )


def _retrieval_filters(collection_id: str, document_ids: Iterable[str]) -> RetrievalFilters:
    values = tuple(document_ids)
    for document_id in values:
        _validated_uuid(document_id, "document")
    return RetrievalFilters(
        tenant_id="default",
        collection_id=collection_id,
        document_ids=values,
    )


def _document_response(document: Document) -> DocumentResponse:
    versions = list(document.versions)
    selected = next(
        (version for version in versions if version.id == document.current_version_id),
        versions[-1] if versions else None,
    )
    return DocumentResponse(
        id=str(document.id),
        filename=document.filename,
        title=document.title,
        status=_enum_text(document.status),
        size_bytes=selected.byte_size if selected else 0,
        mime_type=selected.mime_type if selected else "application/octet-stream",
        chunk_count=selected.indexed_chunk_count if selected else 0,
        version_id=str(selected.id) if selected else None,
        error=selected.error_message if selected else None,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _source_response(source) -> SourceResponse:
    return SourceResponse(
        source_id=source.source_id,
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        filename=source.filename,
        text=source.text,
        score=source.score,
        ordinal=int(source.metadata.get("ordinal", 0)),
        page_start=source.page_start,
        page_end=source.page_end,
        section_path=list(source.section_path),
    )


def _candidate_response(ranked, source_number: int) -> SourceResponse:
    candidate = ranked.candidate
    return SourceResponse(
        source_id=f"S{source_number}",
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        filename=candidate.filename,
        text=candidate.text,
        score=ranked.final_score,
        ordinal=int(candidate.metadata.get("ordinal", 0)),
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        section_path=list(candidate.section_path),
    )


def _diagnostics_dict(diagnostics) -> dict[str, Any]:
    return {
        "normalized_query": diagnostics.normalized_query,
        "embedding_provider": diagnostics.embedding_provider,
        "dense_candidates": diagnostics.dense_candidates,
        "lexical_candidates": diagnostics.lexical_candidates,
        "fused_candidates": diagnostics.fused_candidates,
        "selected_candidates": diagnostics.selected_candidates,
        "reranker_used": diagnostics.reranker_used,
        "warnings": list(diagnostics.warnings),
        "timings_ms": dict(diagnostics.timings_ms),
    }


def _latest_job(jobs: Iterable[IngestionJob]) -> IngestionJob | None:
    values = list(jobs)
    return max(values, key=lambda job: job.created_at) if values else None


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _validated_uuid(value: str, label: str) -> UUID:
    try:
        return UUID(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid {label} ID.") from error


def _record_queue_failure(version_id: str) -> None:
    with SessionLocal() as session:
        try:
            version = DocumentRepository(session).get_version(version_id)
            version.status = VersionStatus.FAILED_INDEX
            version.error_code = "queue_unavailable"
            version.error_message = "The background ingestion queue was unavailable."
            version.document.status = DocumentStatus.FAILED
            session.commit()
        except Exception:
            session.rollback()


def _database_is_healthy() -> bool:
    try:
        check_database()
        return True
    except Exception:
        return False


def _queue_is_healthy() -> bool:
    if getattr(settings, "celery_task_always_eager", False):
        return True
    try:
        client = Redis.from_url(settings.effective_celery_broker_url, socket_timeout=1)
        return bool(client.ping())
    except Exception:
        return False


def _blob_store_is_healthy() -> bool:
    try:
        blob_store = create_blob_store(settings)
        if isinstance(blob_store, LocalBlobStore):
            return blob_store.root.is_dir()
        if isinstance(blob_store, S3BlobStore):
            blob_store._ensure_bucket()
        return True
    except Exception:
        return False
