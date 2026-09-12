"""Stable identities used to make ingestion safe to replay."""

from __future__ import annotations

import hashlib
import unicodedata
import uuid

_CHUNK_NAMESPACE = uuid.UUID("72d12f3e-40f3-50bb-97de-c18a39b79c46")


def canonical_text(text: str) -> str:
    """Return a stable representation for hashing without altering display text."""

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.strip().split("\n"))


def content_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def deterministic_chunk_id(version_id: str, ordinal: int, chunk_content_hash: str) -> str:
    """Derive the same UUID whenever a version is ingested the same way."""

    identity = f"{version_id}\x00{ordinal}\x00{chunk_content_hash}"
    return str(uuid.uuid5(_CHUNK_NAMESPACE, identity))
