# Index — a custom-owned RAG application

Index turns your files into a searchable knowledge base. The application reads and chunks each document, creates embeddings, stores those vectors in **your PostgreSQL database**, retrieves and ranks matching chunks for each question, and asks a generation model to answer only from that evidence.

This project does **not** use OpenAI hosted file search or hosted vector stores. OpenAI is an optional, replaceable provider for embeddings and answer generation.

## What works

- Upload PDF, DOCX, Markdown, HTML, and text documents
- Follow each document through ingestion states
- Structure-aware, token-bounded chunking with source provenance
- Batched embeddings stored in PostgreSQL with pgvector
- Dense semantic search plus PostgreSQL lexical search
- Reciprocal-rank fusion, deduplication, and diverse context selection
- Grounded answers with validated `[S1]` source labels
- Inspect the exact passages and retrieval scores used for an answer
- Exact word and phrase counts over complete source files without double-counting chunk overlap
- Idempotent duplicate uploads and deletion of source files plus searchable chunks
- Keyless local providers for testing the entire pipeline without an API key

The full design and production roadmap are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Fastest start: keyless local mode

Docker Desktop must be running.

```bash
cp .env.example .env
```

The example environment already selects the keyless `hash` embedding and
`extractive` answer adapters.

Then start the complete stack:

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), upload `examples/sample_handbook.md`, wait for **READY**, and ask:

> How many casual leave days do employees receive?

Local mode still exercises our parser, chunker, database, vector matching, ranking, context builder, citations, worker, and deletion logic. Its deterministic feature-hash embedding and extractive answer adapters are for development—not for judging production semantic or answer quality.

## Use OpenAI models

Set these values in `.env`:

```dotenv
OPENAI_API_KEY=your-key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
GENERATION_PROVIDER=openai
GENERATION_MODEL=gpt-5.6-luna
GENERATION_REASONING_EFFORT=none
```

Files, extracted text, metadata, and embeddings remain in your storage and database. Only chunk text sent for embedding and selected evidence sent for generation leave the application.

This mode uses a true semantic embedding model and an LLM while keeping the vector database and retrieval pipeline under your control.

Choose the embedding provider before uploading your real corpus. If you switch providers later, delete and re-upload the existing documents so stored document vectors and new question vectors are produced by the same model.

### API cost

OpenAI API billing is usage-based and requires an API Platform account with billing enabled. The cost-sensitive profile above currently uses:

- `text-embedding-3-small`: **$0.02 per 1 million input tokens** for indexing files and embedding questions.
- `gpt-5.6-luna`: **$0.20 per 1 million input tokens** and **$1.20 per 1 million output tokens** for grounded answers.

At those rates, a representative question with 2,000 input tokens and a 250-token answer costs about **$0.0007**. Even near this app's configured context and response caps, a question is roughly **$0.003 or less**. The profile disables additional reasoning tokens because this task should answer from supplied evidence. Actual cost depends on document length, retrieved context, answer length, retries, and future pricing. Exact word/phrase count questions use the local deterministic scanner and make no OpenAI model call.

Prices were checked on 2026-09-10. Verify the current [embedding price](https://developers.openai.com/api/docs/models/text-embedding-3-small) and [generation price](https://developers.openai.com/api/docs/models/gpt-5.6-luna), and configure a project spend limit before regular use.

Keep `OPENAI_API_KEY` only in the ignored local `.env` file. Never commit it or paste it into the browser UI.

## Services

| Service | Local address | Responsibility |
| --- | --- | --- |
| Web/API | `http://localhost:8000` | Upload, document management, retrieval, answers, and UI |
| PostgreSQL + pgvector | `localhost:5432` | Metadata, chunks, full-text index, and embedding vectors |
| Redis | `localhost:6379` | Background-job delivery |
| MinIO API | `localhost:9000` | Original file storage |
| MinIO console | `http://localhost:9001` | Inspect local object storage |
| Celery worker | internal | Parsing, chunking, embedding, and publishing |

## Development without Docker

Python 3.10+ and reachable PostgreSQL, Redis, and S3-compatible or local file storage are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
alembic upgrade head
uvicorn rag.api.app:app --reload
```

In another terminal:

```bash
source .venv/bin/activate
celery -A rag.worker.celery_app:celery_app worker --loglevel=INFO
```

## Tests and checks

```bash
pytest
ruff check .
```

An end-to-end Docker smoke test is also provided:

```bash
python scripts/smoke_test.py
```

## Useful endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Dependency health |
| `POST` | `/api/documents` | Upload and queue a document |
| `GET` | `/api/documents` | List documents and ingestion states |
| `GET` | `/api/documents/{id}` | Inspect one document |
| `DELETE` | `/api/documents/{id}` | Remove the document and its chunks |
| `GET` | `/api/jobs/{id}` | Inspect ingestion progress |
| `POST` | `/api/retrieval/search` | Inspect retrieval without generation |
| `POST` | `/api/answers` | Retrieve evidence and generate an answer |

## Important behavior

- Retrieval only reads the current `READY` version of a document.
- Failed or partially indexed documents never appear in search.
- The same embedding configuration is used for documents and questions.
- Changing the embedding model or dimensions requires a new index generation.
- Tenant and collection filters are applied inside database retrieval queries.
- Source IDs returned by the model are validated against the supplied context.
- When the evidence is insufficient, the application is designed to abstain instead of inventing an answer.
- Explicit count questions such as `How many times is the word "policy" written?` scan the complete parsed originals deterministically instead of asking the LLM to estimate from retrieved chunks.
- Deletion immediately hides a document, removes its original blob and chunks, and retains only a minimal soft-delete record for lifecycle bookkeeping.

## Project layout

```text
src/rag/api/          HTTP routes and schemas
src/rag/ingestion/    parsers, normalization, chunking, and orchestration
src/rag/retrieval/    search, fusion, ranking, context, and citations
src/rag/adapters/     replaceable model/provider adapters
src/rag/worker/       background tasks
migrations/           database schema
web/                  browser interface
tests/                deterministic unit tests
scripts/              live-stack smoke test
```

## Production notes

The local defaults intentionally use one tenant and collection. Before exposing the application to multiple users, add authentication, enforce per-user/tenant ACLs, enable malware scanning and parser isolation, move secrets to a secret manager, configure retention/backups, and calibrate retrieval and abstention thresholds using a labeled evaluation set.
