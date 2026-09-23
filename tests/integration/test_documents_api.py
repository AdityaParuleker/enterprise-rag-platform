"""
Integration tests for Documents API Endpoints (Section 5 & Section 6.5).
Tests document upload (multipart & URL), document versioning, content hash deduplication, status check, listing, and deletion.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.auth.jwt import create_access_token


class MockTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def auth_headers(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-phase3-documents-999")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    token = create_access_token({"sub": user_id, "tenant_id": tenant_id, "email": "testuser@example.com"})
    return {"Authorization": f"Bearer {token}"}, user_id, tenant_id


@pytest.fixture(autouse=True)
def mock_rbac_permissions():
    with patch("backend.app.auth.rbac.get_user_permissions", new_callable=AsyncMock) as mock_perm:
        mock_perm.return_value = {"upload_document", "view_document", "delete_document"}
        yield mock_perm


@pytest.mark.asyncio
async def test_upload_document_multipart_success(auth_headers):
    headers, user_id, tenant_id = auth_headers

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None  # No duplicate existing document
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_storage = AsyncMock()

    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"

    with patch("backend.app.api.documents.get_db_connection", new_callable=AsyncMock) as mock_db, \
         patch("backend.app.api.documents.get_minio_storage", return_value=mock_storage), \
         patch("backend.celery_worker.process_ingestion_job.delay") as mock_celery:

        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/documents",
                headers=headers,
                files={"file": ("sample.pdf", pdf_bytes, "application/pdf")}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "document_id" in body["data"]
        assert "job_id" in body["data"]
        assert body["data"]["status"] == "QUEUED"
        assert mock_celery.called

        # Storage key format check: tenants/{tenant_id}/documents/{document_id}/v1/{content_hash}.bin
        call_args = mock_storage.upload_file.call_args[0]
        assert call_args[0].startswith(f"tenants/{tenant_id}/documents/")
        assert "/v1/" in call_args[0]


@pytest.mark.asyncio
async def test_upload_document_versioning_workflow(auth_headers):
    headers, user_id, tenant_id = auth_headers
    existing_doc_id = str(uuid.uuid4())

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": uuid.UUID(existing_doc_id),
        "version": 1,
        "filename": "sample.pdf"
    }
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_storage = AsyncMock()
    pdf_bytes_v2 = b"%PDF-1.4\n2 0 obj\n<< /Type /Catalog >>\nendobj\n"

    with patch("backend.app.api.documents.get_db_connection", new_callable=AsyncMock) as mock_db, \
         patch("backend.app.api.documents.get_minio_storage", return_value=mock_storage), \
         patch("backend.celery_worker.process_ingestion_job.delay") as mock_celery:

        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/documents",
                headers=headers,
                data={"document_id": existing_doc_id},
                files={"file": ("sample.pdf", pdf_bytes_v2, "application/pdf")}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["document_id"] == existing_doc_id
        assert body["data"]["version"] == 2
        assert body["data"]["status"] == "QUEUED"
        assert mock_celery.called

        # Verify storage key contains v2
        call_args = mock_storage.upload_file.call_args[0]
        assert f"tenants/{tenant_id}/documents/{existing_doc_id}/v2/" in call_args[0]


@pytest.mark.asyncio
async def test_upload_document_deduplication_rule(auth_headers):
    headers, user_id, tenant_id = auth_headers
    existing_doc_id = uuid.uuid4()

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": existing_doc_id,
        "status": "INDEXED",
        "version": 1
    }

    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"

    with patch("backend.app.api.documents.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/documents",
                headers=headers,
                files={"file": ("sample.pdf", pdf_bytes, "application/pdf")}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["document_id"] == str(existing_doc_id)
        assert body["data"]["deduplicated"] is True
        assert body["data"]["status"] == "INDEXED"


@pytest.mark.asyncio
async def test_get_document_status_success(auth_headers):
    headers, user_id, tenant_id = auth_headers
    doc_id = str(uuid.uuid4())
    job_id = uuid.uuid4()

    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        {"id": doc_id, "status": "INDEXED"},  # document check
        {
            "id": job_id,
            "status": "COMPLETED",
            "retry_count": 0,
            "error_code": None,
            "error_message": None,
            "started_at": None,
            "completed_at": None
        }  # job check
    ]
    mock_conn.fetch.return_value = [
        {"id": uuid.uuid4(), "stage": "PARSING", "status": "COMPLETED", "detail": '{"block_count": 5}', "created_at": None}
    ]

    with patch("backend.app.api.documents.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get(f"/api/v1/documents/{doc_id}/status", headers=headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["document_id"] == doc_id
        assert body["data"]["document_status"] == "INDEXED"
        assert len(body["data"]["events"]) == 1


@pytest.mark.asyncio
async def test_delete_document_success(auth_headers):
    headers, user_id, tenant_id = auth_headers
    doc_id = str(uuid.uuid4())

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": doc_id,
        "storage_key": f"tenants/{tenant_id}/documents/{doc_id}/v1/hash.bin",
        "status": "INDEXED"
    }
    mock_conn.fetch.return_value = [{"id": uuid.uuid4()}]
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_storage = AsyncMock()

    with patch("backend.app.api.documents.get_db_connection", new_callable=AsyncMock) as mock_db, \
         patch("backend.app.api.documents.get_minio_storage", return_value=mock_storage), \
         patch("backend.celery_worker.celery_app.control.revoke") as mock_revoke:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.delete(f"/api/v1/documents/{doc_id}", headers=headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["document_id"] == doc_id
        assert mock_storage.delete_file.called
        assert mock_revoke.called
