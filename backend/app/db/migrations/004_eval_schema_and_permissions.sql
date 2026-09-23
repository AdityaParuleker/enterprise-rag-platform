-- Enterprise Knowledge Intelligence Platform — Migration 004: Evaluation Schema & Permissions Hardening
-- Updates eval_datasets, eval_questions, eval_runs, and eval_results with tenant isolation, hashing, reproducibility, and metric columns.

-- 1. Add tenant_id and metadata columns to eval_datasets
ALTER TABLE eval_datasets ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE eval_datasets ADD COLUMN IF NOT EXISTS version VARCHAR(50) DEFAULT '1.0';
ALTER TABLE eval_datasets ADD COLUMN IF NOT EXISTS dataset_hash VARCHAR(64);

-- 2. Add tenant_id and metadata columns to eval_questions
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS question_type VARCHAR(50) DEFAULT 'factual';
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS expected_guardrail_action VARCHAR(50) DEFAULT 'ALLOW';
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS expected_access VARCHAR(50) DEFAULT 'ALLOWED';
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS difficulty VARCHAR(50) DEFAULT 'medium';

-- 3. Add tenant_id and reproducibility metadata columns to eval_runs
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS status VARCHAR(50) NOT NULL DEFAULT 'QUEUED';
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS dataset_name VARCHAR(255);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS dataset_version VARCHAR(50);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS dataset_hash VARCHAR(64);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS git_commit VARCHAR(40);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS embedding_model_version VARCHAR(50);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS reranker_model VARCHAR(100);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS reranker_model_version VARCHAR(50);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS llm_provider VARCHAR(100);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS llm_model VARCHAR(100);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS eval_provider VARCHAR(100);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS eval_model VARCHAR(100);
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS pricing_config_version VARCHAR(50);

-- 4. Add tenant_id and precision metric columns to eval_results
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS precision_at_k DOUBLE PRECISION;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS mrr DOUBLE PRECISION;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS answer_correctness_token_f1 DOUBLE PRECISION;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS answer_correctness_semantic DOUBLE PRECISION;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS citation_precision DOUBLE PRECISION;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS citation_recall DOUBLE PRECISION;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS input_tokens INT DEFAULT 0;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS output_tokens INT DEFAULT 0;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS total_tokens INT DEFAULT 0;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS estimated_cost DOUBLE PRECISION;

-- 5. Seed granular evaluation permissions
INSERT INTO permissions (name) VALUES
    ('eval:run'),
    ('eval:read'),
    ('eval:compare')
ON CONFLICT (name) DO NOTHING;

-- Grant eval permissions to SUPER_ADMIN & TENANT_ADMIN
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r, permissions p
WHERE r.name IN ('SUPER_ADMIN', 'TENANT_ADMIN')
  AND p.name IN ('eval:run', 'eval:read', 'eval:compare')
ON CONFLICT DO NOTHING;
