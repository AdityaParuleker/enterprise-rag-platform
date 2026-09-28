"""
ARQ Task Worker Engine.
Executes asynchronous ingestion background jobs natively on asyncio with retries, startup validation, and stalled-job watchdog.
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

from arq import cron
from arq.connections import RedisSettings
from arq.worker import Retry

from backend.app.db.connection import get_db_connection
from backend.app.generation.providers.factory import validate_provider_configs
from backend.app.ingestion.pipeline import IngestionPipeline, NonRetryableIngestionError

logger = logging.getLogger(__name__)


def get_redis_settings() -> RedisSettings:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return RedisSettings.from_dsn(redis_url)
    
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "")
    return RedisSettings(
        host=redis_host,
        port=redis_port,
        password=redis_password if redis_password else None
    )


async def startup(ctx: dict):
    """Fail-fast validation at ARQ worker startup."""
    logger.info("Executing ARQ worker fail-fast provider configuration validation...")
    try:
        validate_provider_configs()
        logger.info("Provider configuration validation passed successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: ARQ Worker Startup Provider Validation Failed: {e}")
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


async def process_ingestion_job(ctx: dict, job_id: str):
    """ARQ background task for running ingestion pipeline on a document job."""
    logger.info(f"Starting ARQ task for ingestion job {job_id}")
    job_try = ctx.get("job_try", 1)

    try:
        pipeline = IngestionPipeline()
        success = await pipeline.process_job(job_id)
        return {"job_id": job_id, "success": success}
    except NonRetryableIngestionError as e:
        logger.error(f"Non-retryable error processing job {job_id}: {e}")
        await _mark_job_failed_in_db(job_id, str(e), error_code="NON_RETRYABLE_ERROR")
        return {"job_id": job_id, "success": False, "error": str(e)}
    except Exception as exc:
        if job_try >= 3:
            err_tb = f"{str(exc)}\n{traceback.format_exc()}"
            logger.error(f"Max retries reached for job {job_id}. Marking FAILED: {exc}")
            await _mark_job_failed_in_db(job_id, f"Max retries exceeded: {err_tb}", error_code="MAX_RETRIES_EXCEEDED")
            return {"job_id": job_id, "success": False, "error": str(exc)}

        retry_delays = [5, 15, 45]
        delay = retry_delays[job_try - 1] if (job_try - 1) < len(retry_delays) else 45
        logger.warning(f"Retrying ARQ task for job {job_id} in {delay} seconds (attempt {job_try}/3): {exc}")
        raise Retry(defer=delay)


async def check_stalled_ingestion_jobs(ctx: dict):
    """Watchdog task to detect and mark stalled ingestion jobs without progress for > 15 minutes as FAILED."""
    conn = None
    try:
        conn = await get_db_connection()
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
        if conn:
            await conn.close()


class WorkerSettings:
    functions = [process_ingestion_job]
    cron_jobs = [cron(check_stalled_ingestion_jobs, minute=None)]
    on_startup = startup
    redis_settings = get_redis_settings()
    max_jobs = 10
