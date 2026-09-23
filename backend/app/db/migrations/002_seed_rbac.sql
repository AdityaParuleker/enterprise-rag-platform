-- Enterprise Knowledge Intelligence Platform — Migration 002: Seed RBAC
-- Implement Section 4 & Section 6.2 (Roles & Permissions Seeding)

-- 1. Insert System Roles
INSERT INTO roles (name) VALUES
    ('SUPER_ADMIN'),
    ('TENANT_ADMIN'),
    ('USER'),
    ('VIEWER')
ON CONFLICT (name) DO NOTHING;

-- 2. Insert System Permissions
INSERT INTO permissions (name) VALUES
    ('upload_document'),
    ('delete_document'),
    ('view_document'),
    ('chat'),
    ('view_conversations'),
    ('manage_users'),
    ('run_evaluation'),
    ('view_evaluation'),
    ('manage_tenant')
ON CONFLICT (name) DO NOTHING;

-- 3. Assign Permissions to Roles

-- SUPER_ADMIN gets all permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r, permissions p
WHERE r.name = 'SUPER_ADMIN'
ON CONFLICT DO NOTHING;

-- TENANT_ADMIN gets all permissions except system-wide manage
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r, permissions p
WHERE r.name = 'TENANT_ADMIN'
  AND p.name IN ('upload_document', 'delete_document', 'view_document', 'chat', 'view_conversations', 'manage_users', 'manage_tenant')
ON CONFLICT DO NOTHING;

-- USER gets standard user permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r, permissions p
WHERE r.name = 'USER'
  AND p.name IN ('upload_document', 'delete_document', 'view_document', 'chat', 'view_conversations')
ON CONFLICT DO NOTHING;

-- VIEWER gets read-only permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r, permissions p
WHERE r.name = 'VIEWER'
  AND p.name IN ('view_document', 'view_conversations')
ON CONFLICT DO NOTHING;
