"""
Celery Task Worker Engine (Section 2 & Section 6.5).
Executes asynchronous ingestion background jobs with retries, startup validation, and stalled-job watchdog.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from pathlib import Path

# Ensure project root is in sys.path when running from backend directory
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from celery import Celery
from celery.signals import worker_ready
from backend.app.db.connection import get_db_connection
from backend.app.generation.providers.factory import validate_provider_configs
from backend.app.ingestion.pipeline import IngestionPipeline, NonRetryableIngestionError

logger = logging.getLogger(__name__)

redis_url = os.getenv("REDIS_URL")
if redis_url:
    broker_url = redis_url
else:
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    auth_str = f":{redis_password}@" if redis_password else ""
    broker_url = f"redis://{auth_str}{redis_host}:{redis_port}/0"

celery_app = Celery(
    "ekp_tasks",
    broker=broker_url,
    backend=broker_url,
)


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "check-stalled-ingestion-jobs-every-minute": {
            "task": "backend.celery_worker.check_stalled_ingestion_jobs",
            "schedule": 60.0,
        },
    },
)


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Fail-fast validation at Celery worker startup."""
    logger.info("Executing Celery worker fail-fast provider configuration validation...")
    try:
        validate_provider_configs()
        logger.info("Provider configuration validation passed successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Celery Worker Startup Provider Validation Failed: {e}")
        raise e


async def _mark_job_failed_in_db(job_id: str, error_msg: str, error_code: str = "WORKER_CRASH"):
    """Emergency helper to persist FAILED status to DB if pipeline crashes prior to internal error handling."""
    conn = None
    try:
        conn = await get_db_connection()
        job_uuid = uuid.UUID(job_id) if isinstance(job_id, str) else job_id
        
        job = await conn.fetchrow("SELECT document_id FROM ingestion_jobs WHERE id = $1", job_uuid)
        doc_uuid = job["document_id"] if job else None

        await conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'FAILED', error_code = $2, error_message = $3, completed_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            job_uuid, error_code, error_msg
        )

        if doc_uuid:
            await conn.execute(
                "UPDATE documents SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                doc_uuid
            )

        detail_json = json.dumps({"error_code": error_code, "error_message": error_msg})
        await conn.execute(
            """
            INSERT INTO ingestion_events (job_id, stage, status, detail)
            VALUES ($1, 'FAILED', 'FAILURE', $2::jsonb)
            """,
            job_uuid, detail_json
        )
        logger.info(f"Persisted emergency FAILED status for job {job_id} to database.")
    except Exception as db_err:
        logger.error(f"Failed to persist emergency FAILED status for job {job_id}: {db_err}")
    finally:
        if conn:
            await conn.close()


@celery_app.task(bind=True, max_retries=3)
def process_ingestion_job(self, job_id: str):
    """Celery background task for running ingestion pipeline on a document job."""
    logger.info(f"Starting Celery task for ingestion job {job_id}")

    # Top-level try block wrapping EVERYTHING including pipeline instantiation & setup
    try:
        pipeline = IngestionPipeline()
        success = asyncio.run(pipeline.process_job(job_id))
        return {"job_id": job_id, "success": success}
    except NonRetryableIngestionError as e:
        logger.error(f"Non-retryable error processing job {job_id}: {e}")
        asyncio.run(_mark_job_failed_in_db(job_id, str(e), error_code="NON_RETRYABLE_ERROR"))
        return {"job_id": job_id, "success": False, "error": str(e)}
    except Exception as exc:
        attempt = self.request.retries
        if attempt >= self.max_retries:
            err_tb = f"{str(exc)}\n{traceback.format_exc()}"
            logger.error(f"Max retries reached for job {job_id}. Marking FAILED: {exc}")
            asyncio.run(_mark_job_failed_in_db(job_id, f"Max retries exceeded: {err_tb}", error_code="MAX_RETRIES_EXCEEDED"))
            return {"job_id": job_id, "success": False, "error": str(exc)}
        
        retry_delays = [5, 15, 45]
        delay = retry_delays[attempt] if attempt < len(retry_delays) else 45
        logger.warning(f"Retrying Celery task for job {job_id} in {delay} seconds (attempt {attempt + 1}/3): {exc}")
        
        # If initialization failed (e.g. ValueError during pipeline init), persist failure on last retry
        try:
            raise self.retry(exc=exc, countdown=delay)
        except Exception:
            if attempt >= 2:
                asyncio.run(_mark_job_failed_in_db(job_id, f"Initialization error: {str(exc)}", error_code="INITIALIZATION_FAILURE"))
            raise


@celery_app.task
def check_stalled_ingestion_jobs():
    """Watchdog task to detect and mark stalled ingestion jobs without progress for > 15 minutes as FAILED."""
    async def _run_watchdog():
        conn = await get_db_connection()
        try:
            # Select jobs in active/queued states where updated_at is older than 15 minutes
            stalled_jobs = await conn.fetch(
                """
                SELECT j.id, j.document_id, j.status
                FROM ingestion_jobs j
                JOIN documents d ON j.document_id = d.id
                WHERE j.status IN ('QUEUED', 'PARSING', 'CHUNKING', 'EMBEDDING', 'PROCESSING')
                  AND d.status != 'DELETED'
                  AND d.updated_at < (CURRENT_TIMESTAMP - INTERVAL '15 minutes')
                """
            )
            for row in stalled_jobs:
                job_id_str = str(row["id"])
                logger.warning(f"Watchdog: Found stalled ingestion job {job_id_str} (status: {row['status']}). Marking FAILED.")
                await _mark_job_failed_in_db(
                    job_id_str,
                    error_msg="stalled - no progress update within 15 minute timeout (watchdog)",
                    error_code="STALLED_TIMEOUT"
                )
        except Exception as e:
            logger.error(f"Error in check_stalled_ingestion_jobs watchdog: {e}")
        finally:
            await conn.close()

    asyncio.run(_run_watchdog())
