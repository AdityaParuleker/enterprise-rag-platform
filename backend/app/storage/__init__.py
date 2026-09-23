"""
Storage package for MinIO Object Storage integration.
"""

from backend.app.storage.minio_client import get_minio_storage, MinIOStorageClient

__all__ = ["get_minio_storage", "MinIOStorageClient"]
