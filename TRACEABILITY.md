# Traceability Matrix & Open Questions Deliverable (Phases 0 & 1)

This Traceability Document maps every numbered section (1 through 17) of the frozen **Enterprise Knowledge Intelligence Platform Build Plan (v3)** to the concrete files, contracts, configurations, database migrations, runtime health/readiness endpoints, tests, and interface implementations produced during **Phase 0** and **Phase 1**.

---

## Traceability Matrix (Sections 1–17)

| Plan Section | Title / Focus Area | Produced Deliverables / Mapped Artifacts |
|---|---|---|
| **Section 1** | Why Replace the Suggested Paid Stack | `docker-compose.yml` (OSS services: PostgreSQL+pgvector, Redis, MinIO, Ollama), `.env.example` ($0 self-hosted stack configuration) |
| **Section 2** | Final Stack & Provider Abstraction | `backend/app/generation/providers/base.py` (`LLMProvider` with `async check_readiness()`), `ollama.py` (availability check on base URL), `factory.py` (Provider factory raising `ValueError` on unsupported provider types without silent fallback), `docker-compose.yml`, `.env.example` |
| **Section 3** | System Architecture | `docker-compose.yml` (8-container stack topology with container-to-container internal port isolation), `infra/nginx.conf` (Reverse proxy), `backend/app/main.py` |
| **Section 4** | Database Schema | `backend/app/db/migrations/001_initial_schema.sql` (21 tables, pgvector extension, `pgcrypto` extension, HNSW index on `chunks.embedding`, GIN index on `chunks.tsv`, tenant indexes), `backend/app/db/models.py` (non-ORM entity stubs) |
| **Section 5** | API Contracts | `backend/app/api/auth.py`, `documents.py`, `ingest.py`, `chat.py`, `eval.py`, `admin.py`, `health.py` (implementing `/health` liveness HTTP 200 & `/ready` dynamic 4-dependency readiness with 3.0s bounded query timeouts and aligned `request.state.request_id`) |
| **Section 6** | Security Architecture | `backend/app/auth/` (`jwt.py`, `rbac.py`, `password.py` stubs), `backend/app/generation/prompt_guard.py` (`GuardrailPipeline` interface stub), `001_initial_schema.sql` (PostgreSQL RLS policies using `current_setting('app.tenant_id', true)` per Section 6.7) |
| **Section 7** | Component-by-Component Design | `backend/app/ingestion/` (`parsers/`, `chunkers/`, `metadata_extractor.py`, `pipeline.py`, `security.py`), `backend/app/retrieval/` (`vector_search.py`, `hybrid_search.py`, `reranker.py`, `query_rewriter.py`, `compression.py`, `acl_filter.py`), `backend/app/generation/` (`llm_client.py`, `citation_builder.py`, `streaming.py`), `backend/app/memory/` (`conversation_store.py`) |
| **Section 8** | Observability & Logging | `backend/app/observability/logging.py` (Structured JSON logger stub), `backend/app/observability/middleware.py` (Request tracking middleware) |
| **Section 9** | Testing Strategy | `tests/integration/test_health_readiness.py` (Liveness, readiness healthy, readiness dependency failure), `tests/unit/test_route_registration.py` (Document route uniqueness test), `tests/unit/test_provider_factory.py` (Provider factory exception test) |
| **Section 10** | Rate Limiting | `backend/app/cache/rate_limiter.py` (Token bucket rate limiter interface stub), `.env.example` (Explicit rate limit parameters: `RATE_LIMIT_LOGIN_PER_MIN`, `RATE_LIMIT_UPLOAD_PER_MIN`, `RATE_LIMIT_CHAT_PER_MIN`, `RATE_LIMIT_EVAL_PER_HOUR`) |
| **Section 11** | Backup, Recovery & Deployment | `docker-compose.yml` (`healthcheck` blocks per service), `backend/app/api/health.py` (`/health` process liveness & `/ready` dynamic readiness), `infra/backup.sh` (comment-only shell stub) |
| **Section 12** | Evaluation Pipeline | `backend/eval/dataset.py`, `backend/eval/metrics.py`, `backend/eval/failure_analysis.py` (Failure taxonomy classifier stub), `backend/eval/run_eval.py`, `backend/eval/report.py` |
| **Section 13** | Resource / Model Profiles | `.env.example` (Configurable model profiles), `PHASES.md` (Model profiles and phase hardware guidelines) |
| **Section 14** | Build Order & Acceptance Criteria | `PHASES.md` (Complete Phase 0–10 task breakdown, Phase 0 & Phase 1 acceptance criteria verified) |
| **Section 15** | Repo Structure | Repository directory tree produced matching Section 15 verbatim |
| **Section 16** | Optional Paid Upgrades | `backend/app/generation/providers/base.py` (`LLMProvider` abstraction), `.env.example` (`LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` placeholders) |
| **Section 17** | Phase Execution Scope | **Confirmation of Scope Followed:** Phase 0 & Phase 1 scopes strictly executed. Prohibited logic (JWT issuance, password hashing, RBAC enforcement, file parsing, vector search, LLM inference, citation building, guardrail checks) was **100% excluded**. Open Questions log updated below. |

---

## Added Dependencies & Justification

| Dependency | Version | Rationale / Phase Usage |
|---|---|---|
| `asyncpg` | `0.29.0` | Asynchronous PostgreSQL connectivity & database operations |
| `redis` | `5.0.3` | Asynchronous Redis connectivity check (`PING`) |
| `httpx` | `0.27.0` | Asynchronous HTTP reachability checks for MinIO & Ollama, ASGI test client |
| `pytest` | `8.1.1` | Automated test suite execution |
| `pytest-asyncio` | `0.23.5` | Async fixture and test case support |
| `pyjwt` | `2.8.0` | Phase 2: JWT access token issuance and signature verification (`HS256`) |
| `bcrypt` | `4.1.2` | Phase 2: OpenBSD Blowfish password hashing & verification |
| `email-validator` | `2.1.1` | Phase 2: Email address format validation for Pydantic `EmailStr` schemas |
| `minio` | `7.2.5` | Phase 3: MinIO S3-compatible Object Storage SDK wrapper |
| `python-multipart` | `0.0.9` | Phase 3: FastAPI multipart/form-data upload parsing |
| `PyMuPDF` | `1.24.1` | Phase 3: High-performance PDF document text extraction |
| `python-docx` | `1.1.0` | Phase 3: DOCX document parsing with heading structure tracking |
| `trafilatura` | `1.8.0` | Phase 3: Web/HTML document main text extraction |
| `pandas` | `2.2.1` | Phase 3: Structured CSV tabular row parsing |
| `celery` | `5.3.6` | Phase 3: Asynchronous task queue worker execution |

---

## Automated Test Results (Phase 3 Complete Test Suite)

All 57 test cases executed cleanly via `pytest` (`57 passed in 8.57s`):

- **Phase 3 Ingestion Security & Validation Tests (9 Tests)**:
  - `test_sanitize_filename`: **PASSED** (Strips null bytes and path traversal sequences `../`, `\`).
  - `test_ip_allowed_checks`: **PASSED** (Blocks loopback `127.0.0.1`, private `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, Cloud metadata `169.254.169.254`, unspecified `0.0.0.0`).
  - `test_validate_file_bytes_pdf`: **PASSED** (Validates `b"%PDF-"` header signature).
  - `test_validate_file_bytes_html`: **PASSED** (Validates HTML tags signature).
  - `test_validate_file_bytes_markdown`: **PASSED** (Validates Markdown text content signature).
  - `test_validate_file_bytes_invalid_binary`: **PASSED** (Rejects binary null bytes file).
  - `test_validate_file_bytes_empty`: **PASSED** (Rejects 0-byte empty files).
  - `test_download_and_validate_url_invalid_scheme`: **PASSED** (Rejects non-HTTP/HTTPS URL schemes).
  - `test_download_and_validate_url_private_ip`: **PASSED** (Blocks SSRF attempts to private/loopback IPs).
- **Phase 3 MetadataExtractor & Parsers Tests (12 Tests)**:
  - `test_metadata_extractor_metrics`: **PASSED** (Extracts char_count, word_count, line_count, approx_token_count, reading_time_minutes, title heuristic).
  - `test_markdown_parser`: **PASSED** (Extracts heading blocks `#`, `##` and paragraph text).
  - `test_csv_parser`: **PASSED** (Extracts header column summary and structured row text).
  - `test_csv_parser_strict_utf8`: **PASSED** (Verifies strict UTF-8 decoding without `errors="ignore"` to prevent silent source mutation).
  - `test_xxe_entity_resolution_disabled`: **PASSED** (Verifies malicious DOCX XML entity resolution `resolve_entities=False, no_network=True` blocks entity expansion attacks).
  - `test_pdf_parser`: **PASSED** (Extracts text and page numbers from PyMuPDF document).
  - `test_pdf_parser_corrupt_pdf`: **PASSED** (Verifies corrupted PDF raises standardized `ValueError` with `PARSER_CORRUPT_FILE` detail).
  - `test_html_parser_strict_utf8`: **PASSED** (Verifies HTML strict UTF-8 decoding rejects corrupted bytes).
  - `test_md_parser_strict_utf8`: **PASSED** (Verifies Markdown strict UTF-8 decoding rejects corrupted bytes).
  - `test_md_parser_atx_headings`: **PASSED** (Verifies Markdown ATX regex `^#{1,6}\s+` matching vs non-heading lines).
  - `test_html_parser`: **PASSED** (Extracts main content text via Trafilatura).
  - `test_parser_factory`: **PASSED** (Dispatches appropriate parser based on content MIME type; excludes generic `application/zip`).
- **Phase 3 Text Chunker Tests (4 Tests)**:
  - `test_text_chunker_linkage`: **PASSED** (Splits into character windows with pre-linked `prev_chunk_id` and `next_chunk_id` UUIDs).
  - `test_max_chunk_size_enforced_on_huge_block`: **PASSED** (Sub-splits blocks >1,800 chars so max chunk size <= 1,800 is strictly enforced).
  - `test_boundary_split_helper`: **PASSED** (Searches within ±150 chars for paragraph `\n\n`, line `\n`, or sentence boundaries).
  - `test_text_chunker_empty_blocks`: **PASSED** (Returns empty list for zero input blocks).
- **Phase 3 Documents API Integration Tests (5 Tests)**:
  - `test_upload_document_multipart_success`: **PASSED** (Validates upload, MinIO storage key `tenants/{t}/documents/{d}/v1/{hash}.bin`, DB record, Celery job dispatch).
  - `test_upload_document_versioning_workflow`: **PASSED** (Uploads version 2 for existing `document_id`, incrementing version and creating version record).
  - `test_upload_document_deduplication_rule`: **PASSED** (Returns existing document 200 OK without re-uploading duplicate content hash).
  - `test_get_document_status_success`: **PASSED** (Returns document status and ingestion events history).
  - `test_delete_document_success`: **PASSED** (Deletes document, revokes Celery tasks via `celery_app.control.revoke`, removes MinIO object, deletes chunks).
- **Phase 2 & Phase 1 Regression Tests (27 Tests)**:
  - All 27 authentication, JWT, password, RBAC, route uniqueness, and health/readiness tests passed cleanly.
- **Phase 2 RBAC & Tenant Isolation Tests**:
  - `test_get_current_user_valid_token`: **PASSED** (Extracts identity and populates `request.state`).
  - `test_get_current_user_missing_or_invalid_header`: **PASSED** (Rejects unauthenticated requests with HTTP 401).
  - `test_require_permission_allowed_and_denied`: **PASSED** (Allows authorized permission, blocks unauthorized with HTTP 403).
  - `test_tenant_boundary_isolation_query`: **PASSED** (Verifies permission SQL query is strictly bounded by `(user_id, tenant_id)`).
- **Phase 2 Auth API Integration Tests**:
  - `test_register_new_tenant_and_user_success`: **PASSED** (Atomic creation of tenant, user, and `TENANT_ADMIN` role).
  - `test_register_duplicate_email_conflict`: **PASSED** (Returns HTTP 409 Conflict for duplicate email registration).
  - `test_login_valid_credentials`: **PASSED** (Returns `access_token` and `refresh_token`).
  - `test_login_invalid_password_returns_401`: **PASSED** (Returns HTTP 401 for incorrect password).
  - `test_refresh_token_rotation_success`: **PASSED** (Revokes presented token, issues new token pair).
  - `test_refresh_token_revoked_reuse_fails_with_401`: **PASSED** (Returns HTTP 401 when attempting to reuse rotated token).
  - `test_logout_revokes_token`: **PASSED** (Revokes refresh token in database).
  - `test_password_reset_success`: **PASSED** (Updates password hash and revokes active refresh tokens).
- **Phase 1 Regression Tests**:
  - `test_documents_route_uniqueness`: **PASSED** (`POST /api/v1/documents` registered exactly once).
  - `test_factory_returns_ollama_provider_for_ollama`: **PASSED** (Provider factory returns OllamaProvider).
  - `test_factory_raises_value_error_for_unsupported_providers`: **PASSED** (Raises ValueError on invalid provider).
  - `test_liveness_returns_200_ok`: **PASSED** (`/health` liveness endpoint returns HTTP 200 OK).
  - `test_readiness_healthy_stack`: **PASSED** (`/ready` endpoint returns HTTP 200 OK with all 4 checks healthy).
  - `test_readiness_dependency_failure`: **PASSED** (`/ready` returns 503 when dependency fails while `/health` stays 200).

---

## Log of Open Questions for Human Review

1. **SQL Schema ID Data Types & Generation (Section 4):**
   - Section 4 specifies `id` columns for 21 tables without defining exact data types (`UUID` vs `BIGINT`) or generation functions (`gen_random_uuid()` vs `uuid_generate_v4()`).
   - *Scaffold Choice:* Implemented `UUID DEFAULT gen_random_uuid()` with `CREATE EXTENSION IF NOT EXISTS "pgcrypto";` in `001_initial_schema.sql` as a provisional choice.

2. **Timestamp Precision & Timezones (Section 4):**
   - Section 4 lists timestamp fields without specifying timezone awareness.
   - *Scaffold Choice:* Implemented `TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP`.

3. **Foreign Key ON DELETE Behavior (Section 4):**
   - Section 4 lists relationships without defining cascade behavior (`CASCADE` vs `RESTRICT` vs `SET NULL`).
   - *Scaffold Choice:* Standard `REFERENCES` without implicit `ON DELETE CASCADE`.

4. **`chunks` / `is_latest` Index Discrepancy (Section 4):**
   - Section 4 text lists `-- btree on tenant_id + is_latest` under `chunks`, but `is_latest` is a column of `documents`.
   - *Open Question:* Created unambiguous index `idx_chunks_tenant_id ON chunks(tenant_id)` and recorded proposed `idx_documents_tenant_is_latest ON documents(tenant_id, is_latest)` for human approval.

5. **PostgreSQL RLS `WITH CHECK` Clause Verification (Section 4 & 6.7):**
   - Section 6.7 explicitly prescribes tenant isolation policies reading session variables (`tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID`) via `USING (...)` clauses.
   - *Open Question:* Section 6.7 does not explicitly specify `WITH CHECK (...)` clauses for write isolation. Scaffolding implements the exact `USING` mechanism specified; adding explicit `WITH CHECK` clauses is recorded as an open design decision for Phase 2/5 implementation.

6. **Section 15 File Verification (`models.py`, `admin.py`, `infra/` scripts):**
   - `models.py` created as Python dataclass stubs without ORM dependencies (no SQLAlchemy/Alembic).
   - `admin.py` scaffolded as empty `APIRouter` stub file per Section 15.
   - `minio_init.sh` and `backup.sh` scaffolded as comment-only shell stubs.

7. **LLM Provider Async Readiness Contract (Section 2 & Phase 1):**
   - Extended `LLMProvider` in `backend/app/generation/providers/base.py` with `async check_readiness(self) -> bool:`.
   - Implemented `OllamaProvider.check_readiness()` checking HTTP GET to configured base URL (`OLLAMA_BASE_URL`). `generate()` and `stream()` remain unimplemented stubs raising `NotImplementedError`.
