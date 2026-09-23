"""
Unit tests for MetadataFilterEngine (Checkpoint 5.3 — Schema-Grounded Metadata Filtering).
"""

import uuid
from datetime import datetime
import pytest
from fastapi import HTTPException

from backend.app.retrieval.metadata_filter import MetadataFilterEngine


def test_metadata_filter_unrecognized_key_rejection():
    engine = MetadataFilterEngine()

    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"unrecognized_field": "val"})

    assert exc_info.value.status_code == 400
    assert "Unrecognized metadata filter key" in exc_info.value.detail


def test_metadata_filter_document_id_validation():
    engine = MetadataFilterEngine()
    doc_id1 = str(uuid.uuid4())
    doc_id2 = str(uuid.uuid4())

    # Valid single
    res = engine.parse_and_validate({"document_id": doc_id1})
    assert res["document_id"] == [uuid.UUID(doc_id1)]

    # Valid list
    res = engine.parse_and_validate({"document_id": [doc_id1, doc_id2]})
    assert res["document_id"] == [uuid.UUID(doc_id1), uuid.UUID(doc_id2)]

    # Invalid UUID
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"document_id": "not-a-uuid"})
    assert exc_info.value.status_code == 400
    assert "Invalid document_id UUID format" in exc_info.value.detail


def test_metadata_filter_source_type_validation():
    engine = MetadataFilterEngine()

    # Valid single & list
    res = engine.parse_and_validate({"source_type": "upload"})
    assert res["source_type"] == ["upload"]

    res = engine.parse_and_validate({"source_type": ["upload", "url"]})
    assert res["source_type"] == ["upload", "url"]

    # Invalid empty item
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"source_type": [""]})
    assert exc_info.value.status_code == 400


def test_metadata_filter_page_number_validation():
    engine = MetadataFilterEngine()

    # Integer
    res = engine.parse_and_validate({"page_number": 3})
    assert res["page_number"] == {"eq": 3}

    # Range
    res = engine.parse_and_validate({"page_number": {"min": 2, "max": 10}})
    assert res["page_number"] == {"min": 2, "max": 10}

    # Min > Max
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"page_number": {"min": 10, "max": 2}})
    assert exc_info.value.status_code == 400
    assert "min (10) cannot be greater than max (2)" in exc_info.value.detail

    # Boolean rejection
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"page_number": True})
    assert exc_info.value.status_code == 400


def test_metadata_filter_created_at_validation():
    engine = MetadataFilterEngine()

    # Valid range
    payload = {
        "created_at": {
            "start_date": "2026-01-01T00:00:00Z",
            "end_date": "2026-12-31T23:59:59Z"
        }
    }
    res = engine.parse_and_validate(payload)
    assert isinstance(res["created_at"]["start"], datetime)
    assert isinstance(res["created_at"]["end"], datetime)

    # Invalid ISO format
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"created_at": {"start_date": "not-a-date"}})
    assert exc_info.value.status_code == 400
    assert "Invalid created_at start_date ISO format" in exc_info.value.detail

    # Start > End
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({
            "created_at": {
                "start_date": "2026-12-31T00:00:00Z",
                "end_date": "2026-01-01T00:00:00Z"
            }
        })
    assert exc_info.value.status_code == 400


def test_metadata_filter_build_where_clauses_parameterization():
    engine = MetadataFilterEngine()
    doc_id = str(uuid.uuid4())
    payload = {
        "document_id": doc_id,
        "source_type": "upload",
        "page_number": {"min": 1, "max": 5},
        "created_at": {"start_date": "2026-01-01T00:00:00Z"}
    }
    validated = engine.parse_and_validate(payload)

    conds, params = engine.build_where_clauses(validated, param_offset=4)

    assert len(conds) == 5
    assert conds[0] == "c.document_id = ANY($4::uuid[])"
    assert conds[1] == "d.source_type = ANY($5::text[])"
    assert conds[2] == "c.page_number >= $6"
    assert conds[3] == "c.page_number <= $7"
    assert conds[4] == "d.created_at >= $8"

    assert len(params) == 5
    assert params[0] == [uuid.UUID(doc_id)]
    assert params[1] == ["upload"]
    assert params[2] == 1
    assert params[3] == 5
    assert isinstance(params[4], datetime)


def test_metadata_filter_mixed_timezone_created_at_validation():
    """Bug 1 Test: Verify tz-aware start_date + naive end_date parses & compares safely without TypeError."""
    engine = MetadataFilterEngine()

    payload = {
        "created_at": {
            "start_date": "2026-01-01T00:00:00Z",
            "end_date": "2026-01-05T00:00:00"
        }
    }
    res = engine.parse_and_validate(payload)
    assert res["created_at"]["start"].tzinfo is not None
    assert res["created_at"]["end"].tzinfo is not None
    assert res["created_at"]["start"] < res["created_at"]["end"]

    # Invalid range with mixed offset (start > end)
    invalid_payload = {
        "created_at": {
            "start_date": "2026-01-10T00:00:00Z",
            "end_date": "2026-01-05T00:00:00"
        }
    }
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate(invalid_payload)
    assert exc_info.value.status_code == 400
    assert "created_at start_date cannot be after end_date" in exc_info.value.detail


def test_metadata_filter_empty_string_date_rejection():
    """Bug 2 Test: Verify empty string start_date raises HTTP 400 instead of being silently ignored."""
    engine = MetadataFilterEngine()

    payload = {
        "created_at": {
            "start_date": "",
            "end_date": "2026-01-01T00:00:00Z"
        }
    }
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate(payload)
    assert exc_info.value.status_code == 400
    assert "start_date must be a non-empty ISO date string" in exc_info.value.detail


def test_metadata_filter_source_type_overlong_rejection():
    """Bug 3 Test: Verify source_type exceeding 50 chars raises HTTP 400 before DB execution."""
    engine = MetadataFilterEngine()

    overlong_source = "a" * 51
    with pytest.raises(HTTPException) as exc_info:
        engine.parse_and_validate({"source_type": overlong_source})
    assert exc_info.value.status_code == 400
    assert "exceeds maximum length of 50 characters" in exc_info.value.detail

