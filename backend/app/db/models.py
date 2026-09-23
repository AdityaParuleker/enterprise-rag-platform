"""
Backend Data Models & Entity Stubs (Section 15 Repo Structure)

Note: Standard Library dataclasses used to represent schema entities without
introducing unstated ORM dependencies (e.g. SQLAlchemy/Alembic) in Phase 0.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class Tenant:
    id: UUID
    name: str
    created_at: datetime
    plan: str = "free"
    is_active: bool = True


@dataclass
class User:
    id: UUID
    tenant_id: UUID
    email: str
    password_hash: str
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


@dataclass
class Document:
    id: UUID
    tenant_id: UUID
    owner_id: UUID
    filename: str
    source_type: str
    content_hash: str
    storage_key: str
    version: int = 1
    is_latest: bool = True
    superseded_by: Optional[UUID] = None
    status: str = "UPLOADED"
    visibility: str = "private"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    tenant_id: UUID
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    section_path: Optional[str] = None
    prev_chunk_id: Optional[UUID] = None
    next_chunk_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tsv: Optional[str] = None
    created_at: Optional[datetime] = None
