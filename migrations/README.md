# Database migrations

Run `alembic upgrade head` after PostgreSQL is available. The first migration
enables pgvector and creates the metadata, vector, and full-text indexes used by
the ingestion and retrieval pipelines.
