"""
Async wrapper for MinIO Object Storage SDK (Section 4 & Section 6.5).
Thread-offloads synchronous MinIO operations via asyncio.to_thread.
"""

import asyncio
import io
import os
import urllib.parse
from typing import Optional
from minio import Minio
from minio.error import S3Error


class MinIOStorageClient:
    def __init__(self):
        endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        parsed = urllib.parse.urlparse(endpoint)
        self.host_port = parsed.netloc or parsed.path or "localhost:9000"
        self.secure = parsed.scheme == "https"
        self.access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "documents")

        self.client = Minio(
            endpoint=self.host_port,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            # Handle standard S3 errors (e.g., bucket already owned)
            pass

    async def ensure_bucket_exists(self):
        await asyncio.to_thread(self._ensure_bucket)

    def _upload_file(self, object_name: str, data: bytes, content_type: str) -> str:
        self._ensure_bucket()
        data_stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=data_stream,
            length=len(data),
            content_type=content_type,
        )
        return object_name

    async def upload_file(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        return await asyncio.to_thread(self._upload_file, object_name, data, content_type)

    def _download_file(self, object_name: str) -> bytes:
        response = None
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.read()
        finally:
            if response:
                response.close()
                response.release_conn()

    async def download_file(self, object_name: str) -> bytes:
        return await asyncio.to_thread(self._download_file, object_name)

    def _delete_file(self, object_name: str) -> None:
        try:
            self.client.remove_object(self.bucket_name, object_name)
        except S3Error:
            pass

    async def delete_file(self, object_name: str) -> None:
        await asyncio.to_thread(self._delete_file, object_name)


_minio_storage_instance: Optional[MinIOStorageClient] = None


def get_minio_storage() -> MinIOStorageClient:
    global _minio_storage_instance
    if _minio_storage_instance is None:
        _minio_storage_instance = MinIOStorageClient()
    return _minio_storage_instance
