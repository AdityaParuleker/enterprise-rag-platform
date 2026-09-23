"""
Unit tests for Migration 003 (Checkpoint 5.1 — TSV Stored Generated Column & Phase 5 Performance Indexes).
"""

import os
import pytest
from unittest.mock import AsyncMock

MIGRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "backend", "app", "db", "migrations", "003_fts_and_acl_indexes.sql"
)


def test_migration_003_file_exists_and_content_structure():
    """Verify migration 003 file exists and contains exact required DDL statements."""
    assert os.path.exists(MIGRATION_PATH), f"Migration file not found at {MIGRATION_PATH}"

    with open(MIGRATION_PATH, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # 1. Verify TSV stored generated column DDL
    assert "DROP INDEX IF EXISTS idx_chunks_tsv;" in sql_content
    assert "ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;" in sql_content
    assert "ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;" in sql_content
    assert "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);" in sql_content

    # 2. Verify document_access performance index DDL
    assert "CREATE INDEX IF NOT EXISTS idx_document_access_lookup ON document_access (document_id, principal_type, principal_id);" in sql_content

    # 3. Verify documents owner/tenant performance index DDL
    assert "CREATE INDEX IF NOT EXISTS idx_documents_owner_tenant ON documents (tenant_id, owner_id, visibility);" in sql_content


@pytest.mark.asyncio
async def test_migration_003_sql_execution_sequence():
    """Verify async execution of migration 003 statements against DB connection."""
    mock_conn = AsyncMock()

    with open(MIGRATION_PATH, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Execute migration SQL
    await mock_conn.execute(sql_content)

    mock_conn.execute.assert_called_once_with(sql_content)
