# Implementation Task Breakdown & Phase Acceptance Criteria

Based on Section 14 of the frozen Build Plan.

---

## Phase 0 — Scaffolding, Contracts & Traceability (Completed & Approved)
- [x] Repository skeleton created matching Section 15 exactly
- [x] Database migration (`001_initial_schema.sql`) created matching Section 4 & Section 6.7
- [x] FastAPI route stubs (`backend/app/api/*.py`) matching Section 5 contracts verbatim
- [x] Provider & Guardrail interfaces (`LLMProvider`, `EmbeddingProvider`, `GuardrailPipeline`) created
- [x] Container configuration (`docker-compose.yml`, `infra/`) & environment variables (`.env.example`) defined
- [x] Phase 0 Traceability Matrix (`TRACEABILITY.md`) and Open Questions log produced

---

## Phase 1 — Infrastructure & Health Verification (Completed)
- [x] Docker Compose 8-service configuration verified (`postgres`, `redis`, `minio`, `ollama`, `fastapi`, `celery_worker`, `frontend`, `reverse_proxy`)
- [x] `/health` endpoint returns process liveness (`200 OK`) with 0 external network I/O
- [x] `/ready` dynamic dependency-readiness endpoint implemented with non-blocking concurrent async checks (`asyncio.gather`)
- [x] PostgreSQL async connectivity check (`asyncpg.connect` + `SELECT 1;`) verified
- [x] Redis async connectivity check (`redis.asyncio` + `.ping()`) verified
- [x] MinIO async HTTP reachability check (`/minio/health/live`) verified
- [x] LLM Provider async availability check (`OllamaProvider.check_readiness()` calling configured base URL) verified
- [x] Dependency failure handling verified (`/ready` returns `503 Service Unavailable` while `/health` remains `200 OK`)
- [x] Phase 1 automated test suite (`tests/integration/test_health_readiness.py`, `tests/unit/test_route_registration.py`) passed cleanly
- [x] Document route uniqueness verified (single registration of `POST /api/v1/documents`)

---

## Phase 2 — Authentication & RBAC (Completed — Awaiting Human Audit)
- [x] User registration (`/register`), login (`/login`), logout (`/logout`), refresh token rotation (`/refresh`), and password reset (`/password-reset`) operational
- [x] Bcrypt password hashing and JWT access token issuance/verification (`HS256`, claims `sub`, `tenant_id`, `email`, `exp`, `iat`, `jti`) enforced with non-empty `JWT_SECRET` requirement
- [x] Granular RBAC permissions (`SUPER_ADMIN`, `TENANT_ADMIN`, `USER`, `VIEWER`) checked via FastAPI dependency (`require_permission`) with strict `(user_id, tenant_id)` tenant boundary isolation
- [x] Seed migration `002_seed_rbac.sql` created and applied
- [x] Full Phase 2 test suite (`27 passed out of 27`) and Phase 1 regression verification complete

---

## Phase 3 — Document Ingestion Pipeline & File Security (Completed — Awaiting Human Audit)
- [x] User can upload PDF, DOCX, Web, Markdown, and CSV files (multipart & URL download)
- [x] Upload security enforced: magic-byte signature validation, file size limits (50MB), zip/decompression bomb protection, SSRF blocking, DNS-rebinding defense
- [x] MinIO object storage write integration complete (`MinIOStorageClient`)
- [x] Celery ingestion task worker executes full state machine (`QUEUED` -> `PARSING` -> `CHUNKING` -> `INDEXED` / `FAILED` / `CANCELLED`) with transactional cancellation lock (`FOR UPDATE`)
- [x] Multi-format parsers extract structured text blocks, section paths, page numbers, and metadata for PDF, DOCX, HTML, Markdown, and CSV
- [x] Character-window text chunker (1,200–1,800 chars, 15% overlap) with pre-linked `prev_chunk_id` and `next_chunk_id` UUIDs
- [x] Content hash deduplication rule enforced for tenant uploads
- [x] Full Phase 3 test suite (`46 passed out of 46`) and live Docker Compose runtime verification complete

---

## Phase 4 — Basic RAG & Tenant-Isolated Retrieval (Completed & Approved)
- [x] Document text chunked (~300–500 tokens, ~15% overlap)
- [x] Embeddings generated via `bge-large-en-v1.5` (1024-dim)
- [x] `chunks` table populated with vector embeddings and `tsvector`
- [x] HNSW similarity search operational with tenant-level isolation
- [x] LLM generates answer using retrieved context
- [x] Buffer-validate-stream generation flow streams answer via SSE
- [x] Answer contains deterministic citations resolved from `message_citations`

---

## Phase 5 — Hybrid Search, Metadata Filtering & Document-Level ACL (Completed & Approved)
- [x] Schema migration `003_fts_and_acl_indexes.sql` creating generated stored `tsvector` and GIN/B-tree performance indexes
- [x] Document-level ACL (`document_access` table) enforced via parameterized SQL pushdown (`DocumentACLFilter`)
- [x] Schema-grounded metadata filtering (`MetadataFilterEngine`) with allow-list validation and UTC date normalization
- [x] PostgreSQL full-text search engine (`FTSSearchEngine`) using `websearch_to_tsquery('english', $1)` with single running parameter offset tracking
- [x] Dual-branch hybrid retrieval (`HybridSearchEngine`) merging candidates via Reciprocal Rank Fusion (RRF $k=60$) with deterministic `chunk_id ASC` tie-breaking and fail-closed branch error policy
- [x] End-to-end Chat API integration (`ChatOrchestrator`) forwarding ACL/metadata filters with pre-retrieval validation and `retrieval_mode` options
- [x] Full Phase 5 test suite (`163 passed out of 163`) and Phase 1–5 regression verification complete

---

## Phase 6 — Reranking
- [ ] Top hybrid candidates scored via `bge-reranker-v2-m3` cross-encoder
- [ ] Top-n reranked chunks supplied to prompt context

---

## Phase 7 — Query Rewriting, Contextual Compression & Conversation Memory
- [ ] Multi-turn conversation history stored in PostgreSQL and cached in Redis
- [ ] LLM query rewriting resolves coreferences and decomposes multi-part queries
- [ ] Contextual compression extracts query-relevant sentences from reranked passages

---

## Phase 8 — AI & Runtime Guardrail Hardening
- [ ] Input Guardrail detects prompt injections and jailbreaks before retrieval
- [ ] Context Guardrail detects embedded secret credentials and sanitizes chunk metadata
- [ ] Output Guardrail enforces groundedness verification and deterministic citation validation
- [ ] Insufficient-evidence refusal behavior triggered via composite score when context relevance is low
- [ ] Redis token bucket rate limiting enforced across auth, ingest, chat, and eval endpoints
- [ ] Structured audit logs written to `audit_logs` table for all system actions

---

## Phase 9 — Evaluation Pipeline & Dashboard
- [x] 100 Q/A test dataset created in `eval_datasets` and `eval_questions`
- [x] `run_eval.py` harness executes automated evaluation runs
- [x] Ragas & DeepEval metrics calculated (Recall@k, Faithfulness, Context Relevance, Answer Correctness, Citation Accuracy)
- [x] Failure taxonomy classification logged to `eval_results.failure_category`
- [x] Comparison reports and Eval Dashboard operational
- [x] Conversational Query Router before RAG Retrieval implemented & verified (260/260 tests passing)

---

## Phase 10 — Deployment & Observability Hardening
- [ ] Production reference architecture deployed behind HTTPS Nginx reverse proxy
- [ ] Automated backup scripts (`pg_dump`, MinIO replication) and recovery procedures verified
- [ ] System health monitoring and alert thresholds operational
