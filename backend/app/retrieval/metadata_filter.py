"""
Schema-Grounded V1 Metadata Filter Engine (Phase 5 — Checkpoint 5.3)
Parses, validates, and parameterizes V1 metadata filters against exact SQL schema columns.
Authoritative allow-list validator returning HTTP 400 / ValueError for malformed payloads.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from fastapi import HTTPException, status

ALLOWED_KEYS = {"document_id", "source_type", "page_number", "created_at"}


class MetadataFilterEngine:
    """
    Authoritative validator and parameterized SQL generator for V1 metadata filters.
    Grounds all filters strictly in existing database columns:
    - document_id -> chunks.document_id (UUID)
    - source_type -> documents.source_type (VARCHAR(50))
    - page_number -> chunks.page_number (INT)
    - created_at  -> documents.created_at (TIMESTAMPTZ)
    """

    def parse_and_validate(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate input filter payload dictionary against schema allow-list.

        Raises:
            HTTPException(400): If unknown key, malformed value, or unsupported structure is encountered.
        """
        if not filters:
            return {}

        if not isinstance(filters, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Metadata filters payload must be a JSON object/dictionary."
            )

        validated = {}
        for key, val in filters.items():
            if key not in ALLOWED_KEYS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unrecognized metadata filter key '{key}'. Allowed keys: {sorted(list(ALLOWED_KEYS))}"
                )

            if val is None:
                continue

            if key == "document_id":
                validated["document_id"] = self._validate_document_id(val)
            elif key == "source_type":
                validated["source_type"] = self._validate_source_type(val)
            elif key == "page_number":
                validated["page_number"] = self._validate_page_number(val)
            elif key == "created_at":
                validated["created_at"] = self._validate_created_at(val)

        return validated

    def _validate_document_id(self, val: Any) -> List[uuid.UUID]:
        """Validate single UUID string or list of UUID strings."""
        items = val if isinstance(val, list) else [val]
        if not items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_id filter list cannot be empty."
            )
        uuids = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="document_id filter items must be non-empty UUID strings."
                )
            try:
                uuids.append(uuid.UUID(item.strip()))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid document_id UUID format: '{item}'"
                )
        return uuids

    def _validate_source_type(self, val: Any) -> List[str]:
        """Validate single string or list of strings against VARCHAR(50) limit."""
        items = val if isinstance(val, list) else [val]
        if not items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_type filter list cannot be empty."
            )
        sources = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="source_type filter items must be non-empty strings."
                )
            clean_item = item.strip()
            if len(clean_item) > 50:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"source_type item '{clean_item[:10]}...' exceeds maximum length of 50 characters."
                )
            sources.append(clean_item)
        return sources

    def _validate_page_number(self, val: Any) -> Dict[str, int]:
        """
        Validate integer or dict containing exact page number or min/max page range.
        """
        if isinstance(val, bool):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="page_number filter value cannot be a boolean."
            )

        if isinstance(val, int):
            if val < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"page_number must be >= 1, got {val}"
                )
            return {"eq": val}

        if isinstance(val, dict):
            res = {}
            min_val = val.get("min") if val.get("min") is not None else val.get("min_page")
            max_val = val.get("max") if val.get("max") is not None else val.get("max_page")
            eq_val = val.get("eq")

            if eq_val is not None:
                if isinstance(eq_val, bool) or not isinstance(eq_val, int) or eq_val < 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid page_number eq value: {eq_val}"
                    )
                res["eq"] = eq_val

            if min_val is not None:
                if isinstance(min_val, bool) or not isinstance(min_val, int) or min_val < 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid page_number min value: {min_val}"
                    )
                res["min"] = min_val

            if max_val is not None:
                if isinstance(max_val, bool) or not isinstance(max_val, int) or max_val < 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid page_number max value: {max_val}"
                    )
                res["max"] = max_val

            if "min" in res and "max" in res and res["min"] > res["max"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"page_number min ({res['min']}) cannot be greater than max ({res['max']})."
                )

            if not res:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="page_number filter object must specify 'eq', 'min', or 'max'."
                )
            return res

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page_number filter must be an integer or object with min/max/eq."
        )

    def _validate_created_at(self, val: Any) -> Dict[str, datetime]:
        """
        Validate dict containing ISO 8601 date strings for range filtering.
        Normalizes all parsed datetimes to timezone-aware UTC objects for safe comparison and TIMESTAMPTZ compatibility.
        """
        if not isinstance(val, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="created_at filter must be an object with start_date and/or end_date ISO strings."
            )

        res = {}
        start_raw = val.get("start_date") if val.get("start_date") is not None else val.get("start")
        end_raw = val.get("end_date") if val.get("end_date") is not None else val.get("end")

        if start_raw is not None:
            if not isinstance(start_raw, str) or not start_raw.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="created_at start_date must be a non-empty ISO date string."
                )
            try:
                dt = datetime.fromisoformat(start_raw.strip().replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    from datetime import timezone
                    dt = dt.replace(tzinfo=timezone.utc)
                res["start"] = dt
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid created_at start_date ISO format: '{start_raw}'"
                )

        if end_raw is not None:
            if not isinstance(end_raw, str) or not end_raw.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="created_at end_date must be a non-empty ISO date string."
                )
            try:
                dt = datetime.fromisoformat(end_raw.strip().replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    from datetime import timezone
                    dt = dt.replace(tzinfo=timezone.utc)
                res["end"] = dt
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid created_at end_date ISO format: '{end_raw}'"
                )

        if "start" in res and "end" in res:
            try:
                if res["start"] > res["end"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="created_at start_date cannot be after end_date."
                    )
            except TypeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid created_at range comparison: {e}"
                )

        if not res:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="created_at filter object must specify 'start_date' or 'end_date'."
            )
        return res

    def build_where_clauses(
        self,
        validated_filters: Dict[str, Any],
        param_offset: int = 1
    ) -> Tuple[List[str], List[Any]]:
        """
        Construct parameterized SQL WHERE clause conditions and parameters.
        """
        if param_offset < 1:
            raise ValueError("param_offset must be >= 1")

        if not validated_filters:
            return [], []

        conditions = []
        params = []
        curr_idx = param_offset

        # 1. document_id -> chunks.document_id
        if "document_id" in validated_filters:
            doc_uuids = validated_filters["document_id"]
            conditions.append(f"c.document_id = ANY(${curr_idx}::uuid[])")
            params.append(doc_uuids)
            curr_idx += 1

        # 2. source_type -> documents.source_type
        if "source_type" in validated_filters:
            sources = validated_filters["source_type"]
            conditions.append(f"d.source_type = ANY(${curr_idx}::text[])")
            params.append(sources)
            curr_idx += 1

        # 3. page_number -> chunks.page_number
        if "page_number" in validated_filters:
            p_val = validated_filters["page_number"]
            if "eq" in p_val:
                conditions.append(f"c.page_number = ${curr_idx}")
                params.append(p_val["eq"])
                curr_idx += 1
            else:
                if "min" in p_val:
                    conditions.append(f"c.page_number >= ${curr_idx}")
                    params.append(p_val["min"])
                    curr_idx += 1
                if "max" in p_val:
                    conditions.append(f"c.page_number <= ${curr_idx}")
                    params.append(p_val["max"])
                    curr_idx += 1

        # 4. created_at -> documents.created_at
        if "created_at" in validated_filters:
            c_val = validated_filters["created_at"]
            if "start" in c_val:
                conditions.append(f"d.created_at >= ${curr_idx}")
                params.append(c_val["start"])
                curr_idx += 1
            if "end" in c_val:
                conditions.append(f"d.created_at <= ${curr_idx}")
                params.append(c_val["end"])
                curr_idx += 1

        return conditions, params
