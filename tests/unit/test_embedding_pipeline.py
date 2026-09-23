"""
Unit tests for IngestionPipeline Embedding Stage & Vector Persistence (Checkpoint 4.3).
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.app.ingestion.pipeline import IngestionPipeline, NonRetryableIngestionError


class MockTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.mark.asyncio
async def test_pipeline_embedding_stage_and_vector_persistence():
    pipeline = IngestionPipeline()
    job_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    tenant_uuid = uuid.uuid4()
    job_id = str(job_uuid)

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    # Return initial job
    mock_conn.fetchrow.side_effect = [
        {"document_id": doc_uuid, "status": "QUEUED", "retry_count": 0},  # Job fetch
        {"tenant_id": tenant_uuid, "storage_key": "key.bin", "filename": "doc.pdf", "source_type": "upload", "status": "QUEUED"},  # Doc fetch
    ]
    # Cancellation checks return active status (not DELETED/CANCELLED)
    mock_conn.fetchval.side_effect = [
        "QUEUED", "QUEUED",  # Cancellation check 1 (Parsing)
        "PARSING", "PARSING",  # Cancellation check 2 (Chunking)
        "CHUNKING", "CHUNKING",  # Cancellation check 3 (Embedding)
        "EMBEDDING", "EMBEDDING",  # Transaction lock check (Final commit)
    ]

    mock_storage = AsyncMock()
    mock_storage.download_file.return_value = b"%PDF-1.4\nSample content"
    pipeline.storage = mock_storage

    fake_vector = [0.1] * 1024
    mock_embed_provider = MagicMock()
    mock_embed_provider.embed_batch.return_value = [fake_vector]
    pipeline.embedding_provider = mock_embed_provider

    with patch("backend.app.ingestion.pipeline.get_db_connection", AsyncMock(return_value=mock_conn)), \
         patch.object(pipeline.validator, "validate_file_bytes", return_value=("application/pdf", "doc.pdf")), \
         patch("backend.app.ingestion.pipeline.get_parser_for_mime") as mock_get_parser:

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [MagicMock(text="Sample paragraph text", page_number=1, section_title="Intro")]
        mock_get_parser.return_value = mock_parser

        result = await pipeline.process_job(job_id)
        assert result is True

        # Verify embedding provider embed_batch was called with chunk text
        mock_embed_provider.embed_batch.assert_called_once()
        assert "Sample paragraph text" in mock_embed_provider.embed_batch.call_args[0][0][0]

        # Verify SQL execute transitions through EMBEDDING stage to INDEXED
        execute_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert any("status = 'EMBEDDING'" in sql for sql in execute_calls)
        assert any("status = 'INDEXED'" in sql for sql in execute_calls)
        assert any("status = 'COMPLETED'" in sql for sql in execute_calls)


@pytest.mark.asyncio
async def test_pipeline_embedding_failure_does_not_produce_indexed():
    pipeline = IngestionPipeline()
    job_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    tenant_uuid = uuid.uuid4()
    job_id = str(job_uuid)

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_conn.fetchrow.side_effect = [
        {"document_id": doc_uuid, "status": "QUEUED", "retry_count": 0},
        {"tenant_id": tenant_uuid, "storage_key": "key.bin", "filename": "doc.pdf", "source_type": "upload", "status": "QUEUED"},
    ]
    mock_conn.fetchval.side_effect = [
        "QUEUED", "QUEUED",
        "PARSING", "PARSING",
        "CHUNKING", "CHUNKING",
    ]

    mock_storage = AsyncMock()
    mock_storage.download_file.return_value = b"%PDF-1.4\nSample content"
    pipeline.storage = mock_storage

    # Embedding provider throws Non-retryable ValueError
    mock_embed_provider = MagicMock()
    mock_embed_provider.embed_batch.side_effect = ValueError("Configured embedding model is not installed")
    pipeline.embedding_provider = mock_embed_provider

    with patch("backend.app.ingestion.pipeline.get_db_connection", AsyncMock(return_value=mock_conn)), \
         patch.object(pipeline.validator, "validate_file_bytes", return_value=("application/pdf", "doc.pdf")), \
         patch("backend.app.ingestion.pipeline.get_parser_for_mime") as mock_get_parser:

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [MagicMock(text="Sample paragraph text", page_number=1, section_title="Intro")]
        mock_get_parser.return_value = mock_parser

        result = await pipeline.process_job(job_id)
        assert result is False

        # Verify job and document status were set to FAILED (NEVER INDEXED)
        execute_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert not any("status = 'INDEXED'" in sql for sql in execute_calls)
        assert any("SET status = 'FAILED'" in sql for sql in execute_calls)


@pytest.mark.asyncio
async def test_pipeline_dimension_mismatch_fails_job():
    pipeline = IngestionPipeline()
    job_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    tenant_uuid = uuid.uuid4()
    job_id = str(job_uuid)

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_conn.fetchrow.side_effect = [
        {"document_id": doc_uuid, "status": "QUEUED", "retry_count": 0},
        {"tenant_id": tenant_uuid, "storage_key": "key.bin", "filename": "doc.pdf", "source_type": "upload", "status": "QUEUED"},
    ]
    mock_conn.fetchval.side_effect = [
        "QUEUED", "QUEUED",
        "PARSING", "PARSING",
        "CHUNKING", "CHUNKING",
    ]

    mock_storage = AsyncMock()
    mock_storage.download_file.return_value = b"%PDF-1.4\nSample content"
    pipeline.storage = mock_storage

    # Embedding provider returns 512 dimensions instead of 1024
    invalid_vector = [0.1] * 512
    mock_embed_provider = MagicMock()
    mock_embed_provider.embed_batch.return_value = [invalid_vector]
    pipeline.embedding_provider = mock_embed_provider

    with patch("backend.app.ingestion.pipeline.get_db_connection", AsyncMock(return_value=mock_conn)), \
         patch.object(pipeline.validator, "validate_file_bytes", return_value=("application/pdf", "doc.pdf")), \
         patch("backend.app.ingestion.pipeline.get_parser_for_mime") as mock_get_parser:

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [MagicMock(text="Sample paragraph text", page_number=1, section_title="Intro")]
        mock_get_parser.return_value = mock_parser

        result = await pipeline.process_job(job_id)
        assert result is False

        # Verify job and document status were set to FAILED
        execute_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert not any("status = 'INDEXED'" in sql for sql in execute_calls)
        assert any("EMBEDDING_DIMENSION_MISMATCH" in str(call[0]) for call in mock_conn.execute.call_args_list)


@pytest.mark.asyncio
async def test_pipeline_cancellation_before_embedding():
    pipeline = IngestionPipeline()
    job_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    tenant_uuid = uuid.uuid4()
    job_id = str(job_uuid)

    mock_conn = AsyncMock()

    mock_conn.fetchrow.side_effect = [
        {"document_id": doc_uuid, "status": "QUEUED", "retry_count": 0},
        {"tenant_id": tenant_uuid, "storage_key": "key.bin", "filename": "doc.pdf", "source_type": "upload", "status": "QUEUED"},
    ]
    # Check 1 (Parsing): OK, Check 2 (Chunking): OK, Check 3 (Embedding): CANCELLED
    mock_conn.fetchval.side_effect = [
        "QUEUED", "QUEUED",
        "PARSING", "PARSING",
        "CANCELLED", "CANCELLED",
    ]

    mock_storage = AsyncMock()
    mock_storage.download_file.return_value = b"%PDF-1.4\nSample content"
    pipeline.storage = mock_storage

    mock_embed_provider = MagicMock()
    pipeline.embedding_provider = mock_embed_provider

    with patch("backend.app.ingestion.pipeline.get_db_connection", AsyncMock(return_value=mock_conn)), \
         patch.object(pipeline.validator, "validate_file_bytes", return_value=("application/pdf", "doc.pdf")), \
         patch("backend.app.ingestion.pipeline.get_parser_for_mime") as mock_get_parser:

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [MagicMock(text="Sample paragraph text", page_number=1, section_title="Intro")]
        mock_get_parser.return_value = mock_parser

        result = await pipeline.process_job(job_id)
        assert result is False

        # Verify embedding provider was NEVER called after cancellation
        mock_embed_provider.embed_batch.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_multi_chunk_ordering_and_count():
    """Verify 1:1 chunk-to-embedding count matching and deterministic ordering."""
    pipeline = IngestionPipeline()
    job_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    tenant_uuid = uuid.uuid4()
    job_id = str(job_uuid)

    c1_id = str(uuid.uuid4())
    c2_id = str(uuid.uuid4())

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_conn.fetchrow.side_effect = [
        {"document_id": doc_uuid, "status": "QUEUED", "retry_count": 0},
        {"tenant_id": tenant_uuid, "storage_key": "key.bin", "filename": "doc.pdf", "source_type": "upload", "status": "QUEUED"},
    ]
    mock_conn.fetchval.side_effect = [
        "QUEUED", "QUEUED",
        "PARSING", "PARSING",
        "CHUNKING", "CHUNKING",
        "EMBEDDING", "EMBEDDING",
    ]

    mock_storage = AsyncMock()
    mock_storage.download_file.return_value = b"%PDF-1.4\nSample content"
    pipeline.storage = mock_storage

    pipeline.chunker = MagicMock()
    pipeline.chunker.chunk_blocks.return_value = [
        {"id": c1_id, "chunk_index": 0, "text": "Paragraph 1 text", "page_number": 1, "section_path": "Intro", "prev_chunk_id": None, "next_chunk_id": c2_id, "metadata": {}},
        {"id": c2_id, "chunk_index": 1, "text": "Paragraph 2 text", "page_number": 1, "section_path": "Body", "prev_chunk_id": c1_id, "next_chunk_id": None, "metadata": {}},
    ]

    vec1 = [0.1] * 1024
    vec2 = [0.2] * 1024
    mock_embed_provider = MagicMock()
    mock_embed_provider.embed_batch.return_value = [vec1, vec2]
    pipeline.embedding_provider = mock_embed_provider

    with patch("backend.app.ingestion.pipeline.get_db_connection", AsyncMock(return_value=mock_conn)), \
         patch.object(pipeline.validator, "validate_file_bytes", return_value=("application/pdf", "doc.pdf")), \
         patch("backend.app.ingestion.pipeline.get_parser_for_mime") as mock_get_parser:

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [MagicMock(text="Raw text", page_number=1, section_title="Intro")]
        mock_get_parser.return_value = mock_parser

        result = await pipeline.process_job(job_id)
        assert result is True

        # Verify batch embed received texts in order
        batch_input = mock_embed_provider.embed_batch.call_args[0][0]
        assert len(batch_input) == 2
        assert batch_input[0] == "Paragraph 1 text"
        assert batch_input[1] == "Paragraph 2 text"

        # Verify SQL execute inserted 2 chunk rows with vector parameters
        insert_calls = [call for call in mock_conn.execute.call_args_list if "INSERT INTO chunks" in str(call[0][0])]
        assert len(insert_calls) == 2
        assert str(vec1) in insert_calls[0][0]
        assert str(vec2) in insert_calls[1][0]


@pytest.mark.asyncio
async def test_pipeline_cancellation_during_final_transaction_commit():
    """Verify cancellation detected inside transaction lock FOR UPDATE aborts chunk insertion."""
    pipeline = IngestionPipeline()
    job_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()
    tenant_uuid = uuid.uuid4()
    job_id = str(job_uuid)

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    mock_conn.fetchrow.side_effect = [
        {"document_id": doc_uuid, "status": "QUEUED", "retry_count": 0},
        {"tenant_id": tenant_uuid, "storage_key": "key.bin", "filename": "doc.pdf", "source_type": "upload", "status": "QUEUED"},
    ]
    # Check 1 (Parsing): OK, Check 2 (Chunking): OK, Check 3 (Embedding): OK,
    # Final Commit Transaction Lock: DELETED!
    mock_conn.fetchval.side_effect = [
        "QUEUED", "QUEUED",
        "PARSING", "PARSING",
        "CHUNKING", "CHUNKING",
        "DELETED", "DELETED",
    ]

    mock_storage = AsyncMock()
    mock_storage.download_file.return_value = b"%PDF-1.4\nSample content"
    pipeline.storage = mock_storage

    fake_vector = [0.1] * 1024
    mock_embed_provider = MagicMock()
    mock_embed_provider.embed_batch.return_value = [fake_vector]
    pipeline.embedding_provider = mock_embed_provider

    with patch("backend.app.ingestion.pipeline.get_db_connection", AsyncMock(return_value=mock_conn)), \
         patch.object(pipeline.validator, "validate_file_bytes", return_value=("application/pdf", "doc.pdf")), \
         patch("backend.app.ingestion.pipeline.get_parser_for_mime") as mock_get_parser:

        mock_parser = MagicMock()
        mock_parser.parse.return_value = [MagicMock(text="Sample paragraph text", page_number=1, section_title="Intro")]
        mock_get_parser.return_value = mock_parser

        result = await pipeline.process_job(job_id)
        assert result is False

        # Verify no chunk insert SQL was executed and job status became CANCELLED
        insert_calls = [call for call in mock_conn.execute.call_args_list if "INSERT INTO chunks" in str(call[0][0])]
        assert len(insert_calls) == 0
        execute_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert any("status = 'CANCELLED'" in sql for sql in execute_calls)
        assert not any("status = 'INDEXED'" in sql for sql in execute_calls)

