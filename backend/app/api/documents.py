"""
Documents API Endpoints (Section 5 & Section 6.5).
Provides document upload (multipart, URL, & new versioning), deduplication, state querying, deletion, and version history.
"""

import hashlib
import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, status
from pydantic import BaseModel

from backend.app.auth.rbac import get_current_user, require_permission
from backend.app.db.connection import get_db_connection
from backend.app.storage.minio_client import get_minio_storage
from backend.app.ingestion.security import IngestionSecurityValidator, SecurityValidationError
from backend.celery_worker import celery_app, process_ingestion_job

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
validator = IngestionSecurityValidator()


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _dispatch_ingestion_job(job_id_str: str):
    import asyncio
    import logging
    logger = logging.getLogger(__name__)

    # 1. Dispatch to Celery broker (for dedicated Celery worker containers)
    try:
        process_ingestion_job.delay(job_id_str)
    except Exception as queue_err:
        logger.warning(f"Celery queue dispatch warning ({queue_err}).")

    # 2. Dual-dispatch to in-process asyncio task (ensures live processing on web service)
    async def _async_run():
        try:
            from backend.app.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            await pipeline.process_job(job_id_str)
        except Exception as err:
            logger.error(f"In-process background ingestion error for job {job_id_str}: {err}")

    asyncio.create_task(_async_run())



@router.post("")
async def upload_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    document_id: Optional[str] = Form(None),
    current_user: dict = Depends(require_permission("upload_document"))
):
    """
    POST /api/v1/documents
    Accepts multipart file upload OR { url } form/JSON body.
    Supports optional document_id for adding a new version to an existing document.
    Enforces content hash deduplication, magic-byte validation, MinIO storage, and queues Celery ingestion job.
    """
    req_id = _get_request_id(request)
    tenant_id = current_user["tenant_id"]
    user_id = current_user.get("user_id") or current_user.get("sub")

    # JSON body fallback parsing
    if not file and not url:
        try:
            body_json = await request.json()
            if isinstance(body_json, dict):
                url = body_json.get("url")
                document_id = body_json.get("document_id") or document_id
        except Exception:
            pass

    if (not file and not url) or (file and url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'file' or 'url', but not both."
        )

    file_bytes: bytes = b""
    clean_filename: str = "document"
    mime_type: str = "application/octet-stream"
    source_type: str = "upload"

    if file:
        source_type = "upload"
        raw_filename = file.filename or "uploaded_file"
        file_bytes = await file.read()
        try:
            mime_type, clean_filename = validator.validate_file_bytes(file_bytes, filename=raw_filename)
        except SecurityValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    elif url:
        source_type = "url"
        try:
            file_bytes, mime_type, clean_filename = await validator.download_and_validate_url(url)
        except SecurityValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    tenant_uuid = uuid.UUID(tenant_id)
    user_uuid = uuid.UUID(user_id)

    conn = await get_db_connection()
    try:
        storage = get_minio_storage()

        # WORKFLOW A: Creating a NEW VERSION of an existing document
        if document_id:
            try:
                target_doc_uuid = uuid.UUID(document_id)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id UUID format")

            async with conn.transaction():
                existing_doc = await conn.fetchrow(
                    """
                    SELECT id, version, filename 
                    FROM documents 
                    WHERE id = $1 AND tenant_id = $2 AND status != 'DELETED'
                    FOR UPDATE
                    """,
                    target_doc_uuid, tenant_uuid
                )

                if not existing_doc:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target document not found for versioning")

                new_version = existing_doc["version"] + 1
                storage_key = f"tenants/{tenant_id}/documents/{document_id}/v{new_version}/{content_hash}.bin"
                job_uuid = uuid.uuid4()
                job_id = str(job_uuid)

                # Upload version payload to MinIO
                await storage.upload_file(storage_key, file_bytes, mime_type)

                # Update existing document header record
                await conn.execute(
                    """
                    UPDATE documents 
                    SET version = $1, content_hash = $2, storage_key = $3, status = 'QUEUED', updated_at = CURRENT_TIMESTAMP
                    WHERE id = $4
                    """,
                    new_version, content_hash, storage_key, target_doc_uuid
                )

                # Insert new version record into document_versions
                await conn.execute(
                    """
                    INSERT INTO document_versions (document_id, version, storage_key, content_hash)
                    VALUES ($1, $2, $3, $4)
                    """,
                    target_doc_uuid, new_version, storage_key, content_hash
                )

                # Create ingestion job
                await conn.execute(
                    """
                    INSERT INTO ingestion_jobs (id, document_id, status, retry_count)
                    VALUES ($1, $2, 'QUEUED', 0)
                    """,
                    job_uuid, target_doc_uuid
                )

                # Record initial ingestion event
                detail_json = json.dumps({"source": source_type, "filename": clean_filename, "version": new_version})
                await conn.execute(
                    """
                    INSERT INTO ingestion_events (job_id, stage, status, detail)
                    VALUES ($1, 'QUEUED', 'IN_PROGRESS', $2::jsonb)
                    """,
                    job_uuid, detail_json
                )

            # Dual-dispatch background job (Celery queue + in-process background worker)
            _dispatch_ingestion_job(str(job_id))



            return {
                "data": {
                    "document_id": document_id,
                    "version": new_version,
                    "job_id": job_id,
                    "status": "QUEUED"
                },
                "error": None,
                "request_id": req_id
            }

        # WORKFLOW B: New Document Upload & Deduplication Check
        existing = await conn.fetchrow(
            """
            SELECT id, status, version 
            FROM documents 
            WHERE tenant_id = $1 AND content_hash = $2 AND is_latest = TRUE AND status != 'DELETED'
            """,
            tenant_uuid, content_hash
        )

        if existing:
            return {
                "data": {
                    "document_id": str(existing["id"]),
                    "status": existing["status"],
                    "deduplicated": True,
                    "message": "Identical document content hash already exists for tenant."
                },
                "error": None,
                "request_id": req_id
            }

        doc_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()
        doc_id_str = str(doc_uuid)
        job_id_str = str(job_uuid)
        # Approved storage key format: tenants/{tenant_id}/documents/{document_id}/v{version}/{content_hash}.bin
        storage_key = f"tenants/{tenant_id}/documents/{doc_id_str}/v1/{content_hash}.bin"

        # Upload raw bytes to MinIO
        await storage.upload_file(storage_key, file_bytes, mime_type)

        async with conn.transaction():
            # Insert document record
            await conn.execute(
                """
                INSERT INTO documents (id, tenant_id, owner_id, filename, source_type, content_hash, storage_key, version, is_latest, status, visibility)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 1, TRUE, 'QUEUED', 'private')
                """,
                doc_uuid, tenant_uuid, user_uuid, clean_filename, source_type, content_hash, storage_key
            )

            # Insert initial document version
            await conn.execute(
                """
                INSERT INTO document_versions (document_id, version, storage_key, content_hash)
                VALUES ($1, 1, $2, $3)
                """,
                doc_uuid, storage_key, content_hash
            )

            # Create ingestion job
            await conn.execute(
                """
                INSERT INTO ingestion_jobs (id, document_id, status, retry_count)
                VALUES ($1, $2, 'QUEUED', 0)
                """,
                job_uuid, doc_uuid
            )

            # Record initial ingestion event
            detail_json = json.dumps({"source": source_type, "filename": clean_filename, "size_bytes": len(file_bytes)})
            await conn.execute(
                """
                INSERT INTO ingestion_events (job_id, stage, status, detail)
                VALUES ($1, 'QUEUED', 'IN_PROGRESS', $2::jsonb)
                """,
                job_uuid, detail_json
            )

        # Dual-dispatch background job (Celery queue + in-process background worker)
        _dispatch_ingestion_job(job_id_str)



        return {
            "data": {
                "document_id": doc_id_str,
                "job_id": job_id_str,
                "status": "QUEUED"
            },
            "error": None,
            "request_id": req_id
        }

    finally:
        await conn.close()


@router.get("")
async def list_documents(
    request: Request,
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(require_permission("view_document"))
):
    """List documents for current user's tenant."""
    req_id = _get_request_id(request)
    tenant_uuid = uuid.UUID(current_user["tenant_id"])

    offset = (page - 1) * limit
    conn = await get_db_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT id, filename, source_type, content_hash, version, is_latest, status, visibility, created_at, updated_at
            FROM documents
            WHERE tenant_id = $1 AND status != 'DELETED'
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            tenant_uuid, limit, offset
        )

        docs = [dict(row) for row in rows]
        for d in docs:
            d["id"] = str(d["id"])

        return {
            "data": docs,
            "error": None,
            "request_id": req_id
        }
    finally:
        await conn.close()


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    request: Request,
    current_user: dict = Depends(require_permission("view_document"))
):
    """Retrieve document details by ID."""
    req_id = _get_request_id(request)
    tenant_uuid = uuid.UUID(current_user["tenant_id"])

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id UUID format")

    conn = await get_db_connection()
    try:
        row = await conn.fetchrow(
            """
            SELECT id, tenant_id, owner_id, filename, source_type, content_hash, storage_key, version, is_latest, status, visibility, created_at, updated_at
            FROM documents
            WHERE id = $1 AND tenant_id = $2 AND status != 'DELETED'
            """,
            doc_uuid, tenant_uuid
        )

        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        doc = dict(row)
        doc["id"] = str(doc["id"])
        doc["tenant_id"] = str(doc["tenant_id"])
        doc["owner_id"] = str(doc["owner_id"])

        return {
            "data": doc,
            "error": None,
            "request_id": req_id
        }
    finally:
        await conn.close()


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    request: Request,
    current_user: dict = Depends(require_permission("view_document"))
):
    """Retrieve status and event execution history for document's ingestion job."""
    req_id = _get_request_id(request)
    tenant_uuid = uuid.UUID(current_user["tenant_id"])

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id UUID format")

    conn = await get_db_connection()
    try:
        doc = await conn.fetchrow(
            "SELECT id, status FROM documents WHERE id = $1 AND tenant_id = $2",
            doc_uuid, tenant_uuid
        )
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        job = await conn.fetchrow(
            """
            SELECT id, status, retry_count, error_code, error_message, started_at, completed_at
            FROM ingestion_jobs
            WHERE document_id = $1
            ORDER BY started_at DESC NULLS LAST
            LIMIT 1
            """,
            doc_uuid
        )

        events_data = []
        if job:
            events_rows = await conn.fetch(
                """
                SELECT id, stage, status, detail, created_at
                FROM ingestion_events
                WHERE job_id = $1
                ORDER BY created_at ASC
                """,
                job["id"]
            )
            for ev in events_rows:
                ev_dict = dict(ev)
                ev_dict["id"] = str(ev_dict["id"])
                ev_dict["detail"] = json.loads(ev_dict["detail"]) if isinstance(ev_dict["detail"], str) else ev_dict["detail"]
                events_data.append(ev_dict)

        job_dict = dict(job) if job else None
        if job_dict:
            job_dict["id"] = str(job_dict["id"])

        return {
            "data": {
                "document_id": document_id,
                "document_status": doc["status"],
                "job": job_dict,
                "events": events_data
            },
            "error": None,
            "request_id": req_id
        }
    finally:
        await conn.close()


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
    current_user: dict = Depends(require_permission("delete_document"))
):
    """Delete document, revoke Celery background tasks, and clean up MinIO storage with atomic transaction."""
    req_id = _get_request_id(request)
    tenant_uuid = uuid.UUID(current_user["tenant_id"])

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id UUID format")

    conn = await get_db_connection()
    try:
        storage_key = None
        job_ids_to_revoke = []

        async with conn.transaction():
            doc = await conn.fetchrow(
                """
                SELECT id, storage_key, status 
                FROM documents 
                WHERE id = $1 AND tenant_id = $2 FOR UPDATE
                """,
                doc_uuid, tenant_uuid
            )

            if not doc or doc["status"] == "DELETED":
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found or already deleted")

            storage_key = doc["storage_key"]

            # Active ingestion jobs to revoke
            active_jobs = await conn.fetch(
                "SELECT id FROM ingestion_jobs WHERE document_id = $1 AND status NOT IN ('COMPLETED', 'FAILED')",
                doc_uuid
            )
            job_ids_to_revoke = [str(j["id"]) for j in active_jobs]

            # Update document status to DELETED
            await conn.execute(
                "UPDATE documents SET status = 'DELETED', is_latest = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                doc_uuid
            )

            # Cancel active ingestion jobs in DB
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'CANCELLED', completed_at = CURRENT_TIMESTAMP WHERE document_id = $1 AND status NOT IN ('COMPLETED', 'FAILED')",
                doc_uuid
            )

            # Delete chunks associated with document
            await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_uuid)

        # Offload external cleanup (Celery task revocation & remote object storage deletion) to async background task
        async def _async_external_cleanup():
            for job_id_str in job_ids_to_revoke:
                try:
                    celery_app.control.revoke(job_id_str, terminate=True)
                except Exception:
                    pass
            if storage_key:
                try:
                    storage = get_minio_storage()
                    await storage.delete_file(storage_key)
                except Exception as s3_err:
                    import logging
                    logging.getLogger(__name__).warning(f"Storage cleanup warning for '{storage_key}': {s3_err}")

        asyncio.create_task(_async_external_cleanup())

        return {
            "data": {
                "message": "Document deleted successfully",
                "document_id": document_id
            },
            "error": None,
            "request_id": req_id
        }

    finally:
        await conn.close()


@router.get("/{document_id}/versions")
async def get_document_versions(
    document_id: str,
    request: Request,
    current_user: dict = Depends(require_permission("view_document"))
):
    """Retrieve version history for a document."""
    req_id = _get_request_id(request)
    tenant_uuid = uuid.UUID(current_user["tenant_id"])

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id UUID format")

    conn = await get_db_connection()
    try:
        doc = await conn.fetchrow(
            "SELECT id FROM documents WHERE id = $1 AND tenant_id = $2",
            doc_uuid, tenant_uuid
        )
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        rows = await conn.fetch(
            """
            SELECT id, version, storage_key, content_hash, created_at
            FROM document_versions
            WHERE document_id = $1
            ORDER BY version DESC
            """,
            doc_uuid
        )

        versions = [dict(row) for row in rows]
        for v in versions:
            v["id"] = str(v["id"])

        return {
            "data": versions,
            "error": None,
            "request_id": req_id
        }
    finally:
        await conn.close()


@router.post("/{document_id}/retry")
async def retry_document(
    document_id: str,
    request: Request,
    current_user: dict = Depends(require_permission("upload_document"))
):
    """
    POST /api/v1/documents/{document_id}/retry
    Idempotent re-trigger action for FAILED or stalled QUEUED documents.
    Acquires database row lock FOR UPDATE and rejects active PROCESSING states with 409 Conflict.
    """
    req_id = _get_request_id(request)
    tenant_uuid = uuid.UUID(current_user["tenant_id"])

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id UUID format")

    conn = await get_db_connection()
    try:
        new_job_id_str = None
        async with conn.transaction():
            doc = await conn.fetchrow(
                """
                SELECT id, filename, status, storage_key 
                FROM documents 
                WHERE id = $1 AND tenant_id = $2 FOR UPDATE
                """,
                doc_uuid, tenant_uuid
            )

            if not doc or doc["status"] == "DELETED":
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

            # Reject if actively processing or queued in intermediate states
            if doc["status"] in ("QUEUED", "PARSING", "CHUNKING", "EMBEDDING"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Document ingestion is actively running or queued (status: {doc['status']}). Cannot retry active job."
                )

            # Revoke any prior pending celery tasks
            active_jobs = await conn.fetch(
                "SELECT id FROM ingestion_jobs WHERE document_id = $1 AND status NOT IN ('COMPLETED', 'FAILED')",
                doc_uuid
            )
            for j in active_jobs:
                try:
                    celery_app.control.revoke(str(j["id"]), terminate=True)
                except Exception:
                    pass

            # Update existing active jobs to CANCELLED
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'CANCELLED', completed_at = CURRENT_TIMESTAMP WHERE document_id = $1 AND status NOT IN ('COMPLETED', 'FAILED')",
                doc_uuid
            )

            # Insert new ingestion job
            new_job_uuid = uuid.uuid4()
            new_job_id_str = str(new_job_uuid)

            await conn.execute(
                """
                INSERT INTO ingestion_jobs (id, document_id, status, retry_count)
                VALUES ($1, $2, 'QUEUED', 0)
                """,
                new_job_uuid, doc_uuid
            )

            # Update document status to QUEUED
            await conn.execute(
                "UPDATE documents SET status = 'QUEUED', updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                doc_uuid
            )

            # Record initial retry ingestion event
            detail_json = json.dumps({"action": "manual_retry", "filename": doc["filename"]})
            await conn.execute(
                """
                INSERT INTO ingestion_events (job_id, stage, status, detail)
                VALUES ($1, 'QUEUED', 'IN_PROGRESS', $2::jsonb)
                """,
                new_job_uuid, detail_json
            )

        # Dispatch task to Celery
        if new_job_id_str:
            process_ingestion_job.delay(new_job_id_str)

        return {
            "data": {
                "document_id": document_id,
                "job_id": new_job_id_str,
                "status": "QUEUED"
            },
            "error": None,
            "request_id": req_id
        }
    finally:
        await conn.close()

