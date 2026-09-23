"""
Metadata Extractor Module (Section 6.5 & Section 7.4).
Extracts document and chunk-level metadata including word count, token estimates,
reading time, title heuristics, and structure stats.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.app.ingestion.parsers.base import ParsedBlock


class MetadataExtractor:
    """Extractor for rich document and chunk metadata."""

    def extract_metadata(
        self,
        content: str,
        source_type: str = "upload",
        filename: str = "",
        blocks: Optional[List[ParsedBlock]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata metrics from raw text content and parsed blocks.
        """
        text = content or ""
        char_count = len(text)
        words = re.findall(r'\w+', text)
        word_count = len(words)
        line_count = len(text.split("\n")) if text else 0
        approx_token_count = char_count // 4
        reading_time_minutes = round(word_count / 200.0, 2) if word_count > 0 else 0.0

        # Title heuristic from blocks or first non-empty line
        title = filename or "Untitled Document"
        if blocks:
            for b in blocks:
                if b.block_type == "heading" and b.text.strip():
                    title = b.text.strip()
                    break
        elif text:
            first_line = text.split("\n")[0].strip()
            if first_line and len(first_line) < 100:
                title = first_line.lstrip("#").strip()

        return {
            "title": title,
            "filename": filename,
            "source_type": source_type,
            "char_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "approx_token_count": approx_token_count,
            "reading_time_minutes": reading_time_minutes,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }
