"""
Ingestion API Endpoint Module (Section 15 Repo Structure)
Re-exports/mounts document ingestion routes per Section 5 contract (POST /api/v1/documents).
"""

from backend.app.api.documents import router as documents_router

router = documents_router
