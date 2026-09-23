"""
Ingestion Pipeline Engine (Section 6.5).
Manages document ingestion state machine (QUEUED -> PARSING -> CHUNKING -> COMPLETED/FAILED/CANCELLED)
with pre-transition cancellation checks, transactional cancellation locks, structured error codes,
and MetadataExtractor integration.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional
from backend.app.db.connection import get_db_connection
from backend.app.storage.minio_client import get_minio_storage
from backend.app.ingestion.security import IngestionSecurityValidator, SecurityValidationError
from backend.app.ingestion.parsers import get_parser_for_mime
from backend.app.ingestion.chunkers import TextChunker
from backend.app.ingestion.metadata_extractor import MetadataExtractor
from backend.app.generation.providers.factory import get_embedding_provider

logger = logging.getLogger(__name__)


class NonRetryableIngestionError(Exception):
    """Exception raised for unrecoverable ingestion errors."""
    def __init__(self, message: str, code: str = "NON_RETRYABLE_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class IngestionPipeline:
    """Core state machine and orchestrator for document parsing, metadata extraction, chunking, and vector embedding."""

    def __init__(self):
        self.validator = IngestionSecurityValidator()
        self.chunker = TextChunker()
        self.metadata_extractor = MetadataExtractor()
        self.storage = get_minio_storage()
        self.embedding_provider = get_embedding_provider()

    async def _record_event(self, conn, job_id, stage: str, status: str, detail: Optional[Dict[str, Any]] = None):
        detail_json = json.dumps(detail or {})
        job_uuid = uuid.UUID(job_id) if isinstance(job_id, str) else job_id
        await conn.execute(
            """
            INSERT INTO ingestion_events (job_id, stage, status, detail)
            VALUES ($1, $2, $3, $4::jsonb)
            """,
            job_uuid, stage, status, detail_json
        )

    async def _check_cancellation(self, conn, job_uuid: uuid.UUID, doc_uuid: uuid.UUID) -> bool:
        """Check if document or job was cancelled/deleted prior to a state transition."""
        doc_status = await conn.fetchval("SELECT status FROM documents WHERE id = $1", doc_uuid)
        job_status = await conn.fetchval("SELECT status FROM ingestion_jobs WHERE id = $1", job_uuid)
        if doc_status in ("DELETED", "CANCELLED") or job_status == "CANCELLED":
            await conn.execute("UPDATE ingestion_jobs SET status = 'CANCELLED', completed_at = CURRENT_TIMESTAMP WHERE id = $1", job_uuid)
            await self._record_event(conn, job_uuid, "CANCELLATION_CHECK", "CANCELLED", {
                "reason": f"Ingestion cancelled (doc_status={doc_status}, job_status={job_status})"
            })
            return True
        return False

    async def process_job(self, job_id: str) -> bool:
        """
        Process an ingestion job end-to-end.
        Returns True if completed successfully, False if cancelled or non-retryable error occurred.
        Raises exception for retryable errors so Celery can retry.
        """
        document_id = None
        doc_uuid = None
        conn = await get_db_connection()
        try:
            job_uuid = uuid.UUID(job_id) if isinstance(job_id, str) else job_id

            # 1. Fetch job record
            job = await conn.fetchrow("SELECT document_id, status, retry_count FROM ingestion_jobs WHERE id = $1", job_uuid)
            if not job:
                logger.error(f"Ingestion job {job_id} not found")
                return False

            doc_uuid = job["document_id"]
            document_id = str(doc_uuid)
            retry_count = job["retry_count"]

            # Pre-transition check 1: Before PARSING
            if await self._check_cancellation(conn, job_uuid, doc_uuid):
                logger.info(f"Cancellation detected before PARSING for job {job_id}. Aborting.")
                return False

            # 2. Fetch document record
            doc = await conn.fetchrow("SELECT tenant_id, storage_key, filename, source_type, status FROM documents WHERE id = $1", doc_uuid)
            if not doc:
                return False

            tenant_uuid = doc["tenant_id"]
            tenant_id = str(tenant_uuid)
            storage_key = doc["storage_key"]
            filename = doc["filename"]
            source_type = doc["source_type"]

            # 3. Transition stage -> PARSING
            await conn.execute("UPDATE ingestion_jobs SET status = 'PARSING', started_at = CURRENT_TIMESTAMP WHERE id = $1", job_uuid)
            await conn.execute("UPDATE documents SET status = 'PARSING', updated_at = CURRENT_TIMESTAMP WHERE id = $1", doc_uuid)
            await self._record_event(conn, job_uuid, "PARSING", "IN_PROGRESS", {"filename": filename})

            # 4. Download raw file bytes from MinIO
            try:
                file_bytes = await self.storage.download_file(storage_key)
            except Exception as e:
                logger.error(f"Failed to download object '{storage_key}' from MinIO: {e}")
                raise e  # Transient storage issue is retryable

            # 5. Magic-byte security validation
            try:
                mime_type, sanitized_name = self.validator.validate_file_bytes(file_bytes, filename=filename)
            except SecurityValidationError as e:
                logger.warning(f"Security validation failed for job {job_id} [{e.code}]: {e.message}")
                raise NonRetryableIngestionError(e.message, code=e.code)

            # 6. Extract text blocks using parser
            try:
                parser = get_parser_for_mime(mime_type)
                blocks = parser.parse(file_bytes)
            except Exception as e:
                logger.warning(f"Parsing failed for job {job_id}: {e}")
                raise NonRetryableIngestionError(f"Document parsing failed: {str(e)}", code="PARSER_CORRUPT_FILE")

            # Extract document metadata via MetadataExtractor
            full_text = "\n\n".join([b.text for b in blocks])
            doc_metadata = self.metadata_extractor.extract_metadata(
                content=full_text,
                source_type=source_type,
                filename=filename,
                blocks=blocks
            )

            await self._record_event(conn, job_uuid, "PARSING", "COMPLETED", {
                "block_count": len(blocks),
                "mime_type": mime_type,
                "metadata": doc_metadata
            })

            # Pre-transition check 2: Before CHUNKING
            if await self._check_cancellation(conn, job_uuid, doc_uuid):
                logger.info(f"Cancellation detected before CHUNKING for job {job_id}. Aborting.")
                return False

            # 7. Transition stage -> CHUNKING
            await conn.execute("UPDATE ingestion_jobs SET status = 'CHUNKING' WHERE id = $1", job_uuid)
            await conn.execute("UPDATE documents SET status = 'CHUNKING', updated_at = CURRENT_TIMESTAMP WHERE id = $1", doc_uuid)
            await self._record_event(conn, job_uuid, "CHUNKING", "IN_PROGRESS")

            # 8. Create character-window chunks with linked UUIDs & approx_token_count
            chunks = self.chunker.chunk_blocks(blocks, document_id=document_id, tenant_id=tenant_id)
            await self._record_event(conn, job_uuid, "CHUNKING", "COMPLETED", {"chunk_count": len(chunks)})

            # Pre-transition check 3: Before EMBEDDING
            if await self._check_cancellation(conn, job_uuid, doc_uuid):
                logger.info(f"Cancellation detected before EMBEDDING for job {job_id}. Aborting.")
                return False

            # 9. Transition stage -> EMBEDDING
            await conn.execute("UPDATE ingestion_jobs SET status = 'EMBEDDING' WHERE id = $1", job_uuid)
            await conn.execute("UPDATE documents SET status = 'EMBEDDING', updated_at = CURRENT_TIMESTAMP WHERE id = $1", doc_uuid)
            await self._record_event(conn, job_uuid, "EMBEDDING", "IN_PROGRESS", {"chunk_count": len(chunks)})

            # 10. GENERATE EMBEDDINGS OUTSIDE DATABASE TRANSACTION
            chunk_texts = [c["text"] for c in chunks]
            try:
                embeddings = self.embedding_provider.embed_batch(chunk_texts)
            except ValueError as e:
                logger.error(f"Non-retryable embedding generation error for job {job_id}: {e}")
                raise NonRetryableIngestionError(str(e), code="EMBEDDING_FAILURE")
            except Exception as e:
                logger.warning(f"Transient embedding provider error for job {job_id}: {e}")
                raise e  # Transient network/provider error is retryable

            if len(embeddings) != len(chunks):
                raise NonRetryableIngestionError(
                    f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}",
                    code="EMBEDDING_COUNT_MISMATCH"
                )

            for idx, vec in enumerate(embeddings):
                if len(vec) != 1024:
                    raise NonRetryableIngestionError(
                        f"EMBEDDING_DIMENSION_MISMATCH: expected 1024, got {len(vec)}",
                        code="EMBEDDING_DIMENSION_MISMATCH"
                    )
                chunks[idx]["embedding"] = vec

            await self._record_event(conn, job_uuid, "EMBEDDING", "COMPLETED", {"embedding_count": len(embeddings)})

            # 11. TRANSACTIONAL ATOMIC PERSISTENCE & CANCELLATION GUARANTEE
            # Open short transaction to lock document & job rows FOR UPDATE before final chunk & vector commit
            async with conn.transaction():
                locked_doc_status = await conn.fetchval(
                    "SELECT status FROM documents WHERE id = $1 FOR UPDATE", doc_uuid
                )
                locked_job_status = await conn.fetchval(
                    "SELECT status FROM ingestion_jobs WHERE id = $1 FOR UPDATE", job_uuid
                )

                if locked_doc_status in ("DELETED", "CANCELLED") or locked_job_status == "CANCELLED":
                    logger.info(f"Cancellation/deletion detected during final commit for document {document_id}. Aborting commit.")
                    await conn.execute(
                        "UPDATE ingestion_jobs SET status = 'CANCELLED', completed_at = CURRENT_TIMESTAMP WHERE id = $1", job_uuid
                    )
                    await self._record_event(conn, job_uuid, "TRANSACTION_CHECK", "CANCELLED", {
                        "reason": "Document or job was deleted/cancelled prior to chunk insertion commit"
                    })
                    return False

                # Two-pass insertion to prevent foreign key constraint violations on next_chunk_id
                for chunk in chunks:
                    metadata_json = json.dumps(chunk["metadata"])
                    c_uuid = uuid.UUID(chunk["id"])
                    vec_param = str(chunk["embedding"]) if chunk["embedding"] is not None else None

                    await conn.execute(
                        """
                        INSERT INTO chunks (id, document_id, tenant_id, chunk_index, text, page_number, section_path, metadata, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::vector)
                        """,
                        c_uuid,
                        doc_uuid,
                        tenant_uuid,
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk["page_number"],
                        chunk["section_path"],
                        metadata_json,
                        vec_param
                    )

                for chunk in chunks:
                    c_uuid = uuid.UUID(chunk["id"])
                    prev_c_uuid = uuid.UUID(chunk["prev_chunk_id"]) if chunk["prev_chunk_id"] else None
                    next_c_uuid = uuid.UUID(chunk["next_chunk_id"]) if chunk["next_chunk_id"] else None
                    if prev_c_uuid or next_c_uuid:
                        await conn.execute(
                            """
                            UPDATE chunks
                            SET prev_chunk_id = $1, next_chunk_id = $2
                            WHERE id = $3
                            """,
                            prev_c_uuid,
                            next_c_uuid,
                            c_uuid
                        )

                # Transition document to INDEXED and job to COMPLETED
                await conn.execute(
                    "UPDATE documents SET status = 'INDEXED', updated_at = CURRENT_TIMESTAMP WHERE id = $1", doc_uuid
                )
                await conn.execute(
                    "UPDATE ingestion_jobs SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP WHERE id = $1", job_uuid
                )
                await self._record_event(conn, job_uuid, "INGESTION", "COMPLETED", {"total_chunks": len(chunks)})

            logger.info(f"Ingestion job {job_id} for document {document_id} completed successfully ({len(chunks)} chunks & 1024-dim vectors inserted).")
            return True

        except NonRetryableIngestionError as e:
            err_code = getattr(e, "code", "NON_RETRYABLE_ERROR")
            err_msg = str(e)
            logger.error(f"Non-retryable ingestion error [{err_code}] for job {job_id}: {err_msg}")
            await conn.execute(
                """
                UPDATE ingestion_jobs 
                SET status = 'FAILED', error_code = $2, error_message = $3, completed_at = CURRENT_TIMESTAMP 
                WHERE id = $1
                """,
                job_uuid, err_code, err_msg
            )
            if doc_uuid:
                await conn.execute(
                    "UPDATE documents SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE id = $1", doc_uuid
                )
            # Dead-letter event structure matching specification
            await self._record_event(conn, job_uuid, "FAILED", "FAILURE", {
                "error_code": err_code,
                "error_message": err_msg,
                "retries": retry_count
            })
            return False

        except Exception as e:
            job = await conn.fetchrow("SELECT retry_count FROM ingestion_jobs WHERE id = $1", job_uuid)
            current_retries = job["retry_count"] if job else 0
            if current_retries < 3:
                new_retries = current_retries + 1
                await conn.execute("UPDATE ingestion_jobs SET retry_count = $2 WHERE id = $1", job_uuid, new_retries)
                await self._record_event(conn, job_uuid, "INGESTION", "RETRYING", {"retry_count": new_retries, "error": str(e)})
                logger.warning(f"Retryable error for job {job_id} (retry {new_retries}/3): {e}")
                raise e
            else:
                err_code = "MAX_RETRIES_EXCEEDED"
                err_msg = f"Exceeded maximum retries (3): {str(e)}"
                await conn.execute(
                    """
                    UPDATE ingestion_jobs 
                    SET status = 'FAILED', error_code = $2, error_message = $3, completed_at = CURRENT_TIMESTAMP 
                    WHERE id = $1
                    """,
                    job_uuid, err_code, err_msg
                )
                if doc_uuid:
                    await conn.execute(
                        "UPDATE documents SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE id = $1", doc_uuid
                    )
                await self._record_event(conn, job_uuid, "FAILED", "FAILURE", {
                    "error_code": err_code,
                    "error_message": err_msg,
                    "retries": current_retries
                })
                return False

        finally:
            await conn.close()

