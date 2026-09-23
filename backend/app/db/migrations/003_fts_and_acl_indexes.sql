-- Enterprise Knowledge Intelligence Platform — Migration 003: Full-Text Search (TSV) & ACL Performance Indexes
-- Implement Phase 5 (Section 7 & Checkpoint 5.1)

-- 1. Re-create chunks.tsv as a PostgreSQL Stored Generated Column
-- Drop existing GIN index on tsv first to allow dropping column
DROP INDEX IF EXISTS idx_chunks_tsv;

-- Drop original unpopulated tsv column
ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;

-- Add tsv as stored generated column derived strictly from chunks.text
ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

-- Re-create GIN index on generated tsv column
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

-- 2. Performance Indexes for Phase 5 Document ACL & Metadata Filtering
-- Composite index for document_access lookup
CREATE INDEX IF NOT EXISTS idx_document_access_lookup ON document_access (document_id, principal_type, principal_id);

-- Composite index for documents owner, tenant, and visibility lookup
CREATE INDEX IF NOT EXISTS idx_documents_owner_tenant ON documents (tenant_id, owner_id, visibility);
