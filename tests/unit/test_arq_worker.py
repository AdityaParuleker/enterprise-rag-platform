import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.arq_worker import process_ingestion_job, check_stalled_ingestion_jobs, startup, get_redis_settings, NonRetryableIngestionError
from arq.worker import Retry


@pytest.mark.asyncio
async def test_arq_get_redis_settings(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    settings = get_redis_settings()
    assert settings.host == "localhost"
    assert settings.port == 6379


@pytest.mark.asyncio
async def test_arq_startup_success():
    with patch("backend.arq_worker.validate_provider_configs") as mock_val:
        await startup({})
        assert mock_val.called


@pytest.mark.asyncio
async def test_arq_process_ingestion_job_success():
    ctx = {"job_try": 1}
    job_id = "test-job-uuid-123"

    with patch("backend.arq_worker.IngestionPipeline") as mock_pipeline_cls:
        mock_instance = AsyncMock()
        mock_instance.process_job.return_value = True
        mock_pipeline_cls.return_value = mock_instance

        res = await process_ingestion_job(ctx, job_id)
        assert res["success"] is True
        assert res["job_id"] == job_id


@pytest.mark.asyncio
async def test_arq_process_ingestion_job_retry_on_transient_error():
    ctx = {"job_try": 1}
    job_id = "test-job-uuid-456"

    with patch("backend.arq_worker.IngestionPipeline") as mock_pipeline_cls:
        mock_instance = AsyncMock()
        mock_instance.process_job.side_effect = RuntimeError("Database temporary error")
        mock_pipeline_cls.return_value = mock_instance

        with pytest.raises(Retry) as exc_info:
            await process_ingestion_job(ctx, job_id)

        assert exc_info.value.defer_score == 5000


@pytest.mark.asyncio
async def test_arq_process_ingestion_job_non_retryable_failure():
    ctx = {"job_try": 1}
    job_id = "test-job-uuid-789"

    with patch("backend.arq_worker.IngestionPipeline") as mock_pipeline_cls, \
         patch("backend.arq_worker._mark_job_failed_in_db", new_callable=AsyncMock) as mock_mark_failed:
        mock_instance = AsyncMock()
        mock_instance.process_job.side_effect = NonRetryableIngestionError("Invalid document format")
        mock_pipeline_cls.return_value = mock_instance

        res = await process_ingestion_job(ctx, job_id)
        assert res["success"] is False
        assert mock_mark_failed.called
