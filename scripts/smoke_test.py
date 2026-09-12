"""Exercise the complete running stack with a disposable document."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import httpx

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000").rstrip("/")
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "sample_handbook.md"


def main() -> None:
    unique_content = SAMPLE_PATH.read_bytes() + (f"\n\nSmoke run ID: {uuid.uuid4()}\n".encode())
    document_id: str | None = None

    with httpx.Client(base_url=BASE_URL, timeout=20) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok", health.text

        try:
            upload = _upload(client, unique_content)
            document_id = upload["document_id"]
            ready = _wait_until_ready(client, document_id)
            assert ready["chunk_count"] > 0, ready

            duplicate = _upload(client, unique_content)
            assert duplicate["duplicate"] is True, duplicate
            assert duplicate["document_id"] == document_id, duplicate

            answer_response = client.post(
                "/api/answers",
                json={
                    "question": "How many casual leave days do employees receive?",
                    "document_ids": [document_id],
                },
            )
            answer_response.raise_for_status()
            answer = answer_response.json()
            assert "12 casual leave days" in answer["answer"], answer
            assert "S1" in answer["citations"], answer
            assert answer["sources"], answer

            count_response = client.post(
                "/api/answers",
                json={
                    "question": 'How many times is the word "leave" written?',
                    "document_ids": [document_id],
                },
            )
            count_response.raise_for_status()
            count_answer = count_response.json()
            assert "7 times" in count_answer["answer"], count_answer
            assert count_answer["diagnostics"]["provider"] == "deterministic-exact-count"
            assert sum(source["match_count"] for source in count_answer["sources"]) == 7

            abstention_response = client.post(
                "/api/answers",
                json={
                    "question": "What is the office Wi-Fi password?",
                    "document_ids": [document_id],
                },
            )
            abstention_response.raise_for_status()
            abstention = abstention_response.json()
            assert abstention["diagnostics"]["abstained"] is True, abstention

            print(
                "Smoke test passed: upload, ingestion, duplicate detection, hybrid "
                "retrieval, cited answer, exact counting, and abstention."
            )
        finally:
            if document_id is not None:
                deletion = client.delete(f"/api/documents/{document_id}")
                deletion.raise_for_status()
                assert all(
                    item["id"] != document_id
                    for item in client.get("/api/documents").json()["items"]
                )
                print("Cleanup passed: disposable document is no longer searchable.")


def _upload(client: httpx.Client, content: bytes) -> dict:
    response = client.post(
        "/api/documents",
        files={"file": ("smoke-handbook.md", content, "text/markdown")},
    )
    response.raise_for_status()
    return response.json()


def _wait_until_ready(
    client: httpx.Client,
    document_id: str,
    *,
    timeout_seconds: float = 30,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/documents/{document_id}")
        response.raise_for_status()
        document = response.json()
        if document["status"] == "READY":
            return document
        if document["status"] in {"FAILED", "NEEDS_OCR"}:
            raise RuntimeError(f"Ingestion failed: {document}")
        time.sleep(0.25)
    raise TimeoutError(f"Document {document_id} did not become READY")


if __name__ == "__main__":
    main()
