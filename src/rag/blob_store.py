"""Original-file storage adapters.

The local adapter makes the app useful with only Python and Postgres. The S3
adapter targets MinIO in Docker and standard S3-compatible services in production.
Database rows store opaque URIs; parsers never need to know which adapter is used.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol, runtime_checkable
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from rag.config import Settings, get_settings


class BlobStoreError(RuntimeError):
    pass


class BlobNotFoundError(BlobStoreError):
    pass


@runtime_checkable
class BlobStore(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str: ...

    def put_stream(self, key: str, stream: BinaryIO, content_type: str | None = None) -> str: ...

    def get_bytes(self, blob_uri: str) -> bytes: ...

    def delete(self, blob_uri: str) -> None: ...

    def exists(self, blob_uri: str) -> bool: ...


def _safe_key(key: str) -> str:
    normalized = key.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("Blob key must be a non-empty relative path without '..'")
    return str(path)


def safe_filename(filename: str) -> str:
    """Return a display-safe suffix; IDs, not filenames, provide uniqueness."""

    basename = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip(".-")
    return (cleaned or "document")[:180]


def build_blob_key(
    *,
    tenant_id: str,
    collection_id: str,
    document_id: uuid.UUID | str,
    version_id: uuid.UUID | str,
    filename: str,
) -> str:
    """Create an immutable, non-user-controlled storage key."""

    def segment(value: object) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
        if not cleaned:
            raise ValueError("Blob key identifiers cannot be empty")
        return cleaned[:100]

    return "/".join(
        [
            segment(tenant_id),
            segment(collection_id),
            segment(document_id),
            segment(version_id),
            safe_filename(filename),
        ]
    )


class LocalBlobStore:
    scheme = "local://"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        destination = (self.root / _safe_key(key)).resolve()
        try:
            destination.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Blob key escapes the configured storage root") from error
        return destination

    def _key_from_uri(self, blob_uri: str) -> str:
        if not blob_uri.startswith(self.scheme):
            raise BlobStoreError(f"Local store cannot read URI {blob_uri!r}")
        return _safe_key(blob_uri[len(self.scheme) :])

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        del content_type
        return self.put_stream(key, io.BytesIO(data))

    def put_stream(self, key: str, stream: BinaryIO, content_type: str | None = None) -> str:
        del content_type
        safe_key = _safe_key(key)
        destination = self._path_for_key(safe_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".upload-", delete=False
            ) as temporary:
                temporary_name = temporary.name
                shutil.copyfileobj(stream, temporary, length=1024 * 1024)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        return f"{self.scheme}{safe_key}"

    def get_bytes(self, blob_uri: str) -> bytes:
        path = self._path_for_key(self._key_from_uri(blob_uri))
        try:
            return path.read_bytes()
        except FileNotFoundError as error:
            raise BlobNotFoundError(blob_uri) from error

    def delete(self, blob_uri: str) -> None:
        path = self._path_for_key(self._key_from_uri(blob_uri))
        path.unlink(missing_ok=True)
        # Remove empty key directories but never the configured root itself.
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def exists(self, blob_uri: str) -> bool:
        return self._path_for_key(self._key_from_uri(blob_uri)).is_file()


class S3BlobStore:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
        client: BaseClient | None = None,
    ) -> None:
        self.bucket = bucket
        self.client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._bucket_checked = False

    def _ensure_bucket(self) -> None:
        if self._bucket_checked:
            return
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise BlobStoreError(f"Cannot access bucket {self.bucket!r}: {code}") from error
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except ClientError as create_error:
                raise BlobStoreError(f"Cannot create bucket {self.bucket!r}") from create_error
        self._bucket_checked = True

    def _key_from_uri(self, blob_uri: str) -> str:
        parsed = urlparse(blob_uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise BlobStoreError(f"S3 store cannot read URI {blob_uri!r}")
        return _safe_key(parsed.path.lstrip("/"))

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        return self.put_stream(key, io.BytesIO(data), content_type)

    def put_stream(self, key: str, stream: BinaryIO, content_type: str | None = None) -> str:
        self._ensure_bucket()
        safe_key = _safe_key(key)
        extra_args = {"ContentType": content_type} if content_type else None
        try:
            if extra_args:
                self.client.upload_fileobj(stream, self.bucket, safe_key, ExtraArgs=extra_args)
            else:
                self.client.upload_fileobj(stream, self.bucket, safe_key)
        except ClientError as error:
            raise BlobStoreError(f"Could not store s3://{self.bucket}/{safe_key}") from error
        return f"s3://{self.bucket}/{safe_key}"

    def get_bytes(self, blob_uri: str) -> bytes:
        key = self._key_from_uri(blob_uri)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except self.client.exceptions.NoSuchKey as error:
            raise BlobNotFoundError(blob_uri) from error
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise BlobNotFoundError(blob_uri) from error
            raise BlobStoreError(f"Could not read {blob_uri}") from error

    def delete(self, blob_uri: str) -> None:
        key = self._key_from_uri(blob_uri)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            raise BlobStoreError(f"Could not delete {blob_uri}") from error

    def exists(self, blob_uri: str) -> bool:
        key = self._key_from_uri(blob_uri)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise BlobStoreError(f"Could not inspect {blob_uri}") from error


# Name the local Docker use case explicitly without coupling to the MinIO SDK.
MinioBlobStore = S3BlobStore


def create_blob_store(settings: Settings | None = None) -> BlobStore:
    configuration = settings or get_settings()
    if configuration.blob_store == "local":
        return LocalBlobStore(configuration.local_blob_path)
    return S3BlobStore(
        bucket=configuration.s3_bucket,
        endpoint_url=configuration.s3_endpoint_url,
        access_key=configuration.s3_access_key,
        secret_key=configuration.s3_secret_key,
        region=configuration.s3_region,
    )
