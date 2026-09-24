"""
Async wrapper for S3-compatible Object Storage (Cloudflare R2 / MinIO) (Section 4 & Section 6.5).
Thread-offloads synchronous S3 operations via asyncio.to_thread.
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
        # 1. Resolve environment variables (R2_* primary, MINIO_* secondary)
        endpoint = os.getenv("R2_ENDPOINT") or os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("R2_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
        bucket_name = os.getenv("R2_BUCKET_NAME") or os.getenv("MINIO_BUCKET_NAME", "documents")

        is_cloud = bool(
            os.getenv("RENDER")
            or os.getenv("VERCEL")
            or os.getenv("ENVIRONMENT") == "production"
            or os.getenv("ENV") == "production"
        )

        # Fail loudly at startup if missing required storage config in cloud deployments or when R2 is configured
        if is_cloud or os.getenv("R2_ENDPOINT"):
            missing = []
            if not endpoint or "localhost" in endpoint or "127.0.0.1" in endpoint:
                missing.append("R2_ENDPOINT (or MINIO_ENDPOINT)")
            if not access_key or access_key == "minioadmin":
                missing.append("R2_ACCESS_KEY_ID (or MINIO_ROOT_USER)")
            if not secret_key or secret_key == "minioadmin":
                missing.append("R2_SECRET_ACCESS_KEY (or MINIO_ROOT_PASSWORD)")

            if missing:
                raise RuntimeError(
                    f"R2 storage not configured: Missing or invalid storage environment variables for cloud deployment ({', '.join(missing)}). "
                    f"Please configure R2_ENDPOINT, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY."
                )

        # Local development fallbacks (only used in non-cloud environment)
        endpoint = endpoint or "http://localhost:9000"
        access_key = access_key or "minioadmin"
        secret_key = secret_key or "minioadmin"

        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            scheme = "https" if ("r2.cloudflarestorage.com" in endpoint or "supabase.co" in endpoint) else "http"
            endpoint = f"{scheme}://{endpoint}"

        parsed = urllib.parse.urlparse(endpoint)
        self.host_port = parsed.netloc or parsed.path
        self.secure = parsed.scheme == "https"
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name

        region = os.getenv("R2_REGION", "auto") if "r2.cloudflarestorage.com" in self.host_port else None

        minio_kwargs = {
            "endpoint": self.host_port,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
            "secure": self.secure,
        }
        if region:
            minio_kwargs["region"] = region

        self.client = Minio(**minio_kwargs)

        # Support path-prefixed S3 endpoints (e.g. Supabase S3: /storage/v1/s3)
        path_prefix = parsed.path.rstrip('/')
        if path_prefix:
            from minio.helpers import BaseURL, url_replace
            orig_build = getattr(BaseURL, "_orig_build", BaseURL.build)
            BaseURL._orig_build = orig_build

            def _custom_build(self_base, method, region, bucket_name=None, object_name=None, query_params=None):
                res = orig_build(self_base, method, region, bucket_name, object_name, query_params)
                prefix = getattr(self_base, "_path_prefix", None)
                if prefix:
                    return url_replace(res, path=prefix.rstrip('/') + res.path)
                return res

            BaseURL.build = _custom_build
            self.client._base_url._path_prefix = path_prefix


    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as err:
            if err.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                return
            raise


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

