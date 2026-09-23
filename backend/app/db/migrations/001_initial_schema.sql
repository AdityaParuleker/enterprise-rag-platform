-- Enterprise Knowledge Intelligence Platform — Migration 001: Initial Schema
-- Implement Section 4 (Database Schema) & Section 6.7 (PostgreSQL RLS Mechanism)
-- Note: Minimal reversible placeholders used for unstated SQL details (UUIDs, TIMESTAMPTZ, nullability).
-- See TRACEABILITY.md for the complete Open Questions log.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 1. Tenants
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- 2. Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ
);

-- 3. Roles
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE -- SUPER_ADMIN, TENANT_ADMIN, USER, VIEWER
);

-- 4. Permissions
CREATE TABLE permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE
);

-- 5. Role Permissions
CREATE TABLE role_permissions (
    role_id UUID NOT NULL REFERENCES roles(id),
    permission_id UUID NOT NULL REFERENCES permissions(id),
    PRIMARY KEY (role_id, permission_id)
);

-- 6. User Roles
CREATE TABLE user_roles (
    user_id UUID NOT NULL REFERENCES users(id),
    role_id UUID NOT NULL REFERENCES roles(id),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    PRIMARY KEY (user_id, role_id, tenant_id)
);

-- 7. Refresh Tokens
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);

-- 8. Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    owner_id UUID NOT NULL REFERENCES users(id),
    filename VARCHAR(550) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    storage_key VARCHAR(1024) NOT NULL,
    version INT NOT NULL DEFAULT 1,
    is_latest BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by UUID REFERENCES documents(id),
    status VARCHAR(50) NOT NULL DEFAULT 'UPLOADED',
    visibility VARCHAR(50) NOT NULL DEFAULT 'private',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 9. Document Versions
CREATE TABLE document_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    version INT NOT NULL,
    storage_key VARCHAR(1024) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 10. Document Access
CREATE TABLE document_access (
    document_id UUID NOT NULL REFERENCES documents(id),
    principal_type VARCHAR(50) NOT NULL, -- user | role | tenant
    principal_id UUID NOT NULL,
    access_level VARCHAR(50) NOT NULL, -- view | edit | owner
    PRIMARY KEY (document_id, principal_type, principal_id)
);

-- 11. Chunks & Vectors
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    page_number INT,
    section_path TEXT,
    prev_chunk_id UUID REFERENCES chunks(id),
    next_chunk_id UUID REFERENCES chunks(id),
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1024),
    tsv tsvector,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes on Chunks & Documents (Section 4)
-- HNSW Index on embedding (1024-dim, cosine distance)
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
-- GIN Index on full-text search tsvector
CREATE INDEX idx_chunks_tsv ON chunks USING gin (tsv);
-- BTree indexes for tenant filtering
CREATE INDEX idx_chunks_tenant_id ON chunks (tenant_id);
-- Recorded Open Question: Proposed index on documents(tenant_id, is_latest)
CREATE INDEX idx_documents_tenant_is_latest ON documents (tenant_id, is_latest);

-- 12. Ingestion Jobs
CREATE TABLE ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    status VARCHAR(50) NOT NULL DEFAULT 'QUEUED',
    retry_count INT NOT NULL DEFAULT 0,
    error_code VARCHAR(100),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- 13. Ingestion Events
CREATE TABLE ingestion_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ingestion_jobs(id),
    stage VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    detail JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 14. Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID NOT NULL REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    summary TEXT
);

-- 15. Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL, -- user | assistant | system
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 16. Citation Mapping
CREATE TABLE message_citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id),
    citation_index INT NOT NULL,
    chunk_id UUID NOT NULL REFERENCES chunks(id),
    document_id UUID NOT NULL REFERENCES documents(id),
    page_number INT,
    section_path TEXT
);

-- 17. Eval Datasets
CREATE TABLE eval_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 18. Eval Questions
CREATE TABLE eval_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES eval_datasets(id),
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    ground_truth_chunk_ids JSONB DEFAULT '[]'::jsonb,
    filters JSONB DEFAULT '{}'::jsonb
);

-- 19. Eval Runs
CREATE TABLE eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES eval_datasets(id),
    config_name VARCHAR(100) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- 20. Eval Results
CREATE TABLE eval_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES eval_runs(id),
    question_id UUID NOT NULL REFERENCES eval_questions(id),
    retrieved_chunk_ids JSONB DEFAULT '[]'::jsonb,
    answer TEXT NOT NULL,
    latency_ms DOUBLE PRECISION,
    cost_tokens INT,
    recall DOUBLE PRECISION,
    faithfulness DOUBLE PRECISION,
    answer_correctness DOUBLE PRECISION,
    citation_accuracy DOUBLE PRECISION,
    context_relevance DOUBLE PRECISION,
    failure_category VARCHAR(100)
);

-- 21. Audit Logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    ip_address VARCHAR(45),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    detail JSONB DEFAULT '{}'::jsonb
);

-- Row-Level Security (RLS) Policies (Section 4 & Section 6.7)
-- RLS mechanism reads session-scoped PostgreSQL variables 'app.tenant_id' set via SET LOCAL
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_chunks ON chunks
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_conversations ON conversations
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_messages ON messages
    USING (conversation_id IN (
        SELECT id FROM conversations WHERE tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID
    ));
