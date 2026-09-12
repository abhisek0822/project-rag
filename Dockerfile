FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# libpq supports PostgreSQL tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN addgroup --system rag \
    && adduser --system --ingroup rag --home /app rag \
    && mkdir -p /app/data/blobs \
    && chown -R rag:rag /app

USER rag
EXPOSE 8000

CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
