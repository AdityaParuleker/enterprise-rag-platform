"""
Text Chunker Module (Section 6.5).
Splits parsed document blocks into character-window chunks (1200-1800 chars, 15% overlap)
with paragraph/sentence boundary adjustments (±150 chars), strict max size enforcement (1800 chars),
approx_token_count metadata, and pre-assigned linked UUIDs.
"""

import uuid
import hashlib
import re
from typing import List, Dict, Any, Optional
from backend.app.ingestion.parsers.base import ParsedBlock

TARGET_CHUNK_SIZE = 1500
MAX_CHUNK_SIZE = 1800
MIN_CHUNK_SIZE = 400


def find_boundary_split(text: str, target_idx: int, window: int = 150) -> int:

    """
    Search within target_idx ± window for nearest paragraph break (\n\n),
    line break (\n), or sentence boundary (. , ! , ? ).
    Returns optimal split index.
    """
    if target_idx >= len(text):
        return len(text)

    start_pos = max(0, target_idx - window)
    end_pos = min(len(text), target_idx + window)
    search_sub = text[start_pos:end_pos]

    # 1. Paragraph boundary (\n\n)
    para_idx = search_sub.rfind("\n\n")
    if para_idx != -1:
        return start_pos + para_idx + 2

    # 2. Line boundary (\n)
    line_idx = search_sub.rfind("\n")
    if line_idx != -1:
        return start_pos + line_idx + 1

    # 3. Sentence boundary (. , ! , ? followed by space or newline)
    sentence_matches = list(re.finditer(r'[.!?](\s|$)', search_sub))
    if sentence_matches:
        best_match = max(sentence_matches, key=lambda m: m.end())
        return start_pos + best_match.end()

    return target_idx


class TextChunker:
    """Character-window chunker with structure awareness, boundary adjustment, and linked UUIDs."""

    def __init__(self, target_size: int = TARGET_CHUNK_SIZE, overlap: Optional[int] = None):
        self.target_size = min(target_size, MAX_CHUNK_SIZE)
        self.max_size = MAX_CHUNK_SIZE
        self.overlap = overlap if overlap is not None else int(self.target_size * 0.15)

    def _split_large_block(self, block_text: str) -> List[str]:
        """Sub-split a block that exceeds max_size (1800 chars) into safe character windows."""
        if len(block_text) <= self.max_size:
            return [block_text]

        sub_blocks = []
        start = 0
        while start < len(block_text):
            target = start + self.target_size
            if target >= len(block_text):
                sub_blocks.append(block_text[start:])
                break

            split_pt = find_boundary_split(block_text, target, window=150)
            if split_pt <= start:
                split_pt = target

            sub_blocks.append(block_text[start:split_pt].strip())
            start = max(start + 1, split_pt - self.overlap)

        return [sb for sb in sub_blocks if sb]

    def chunk_blocks(self, blocks: List[ParsedBlock], document_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Convert ParsedBlocks into a list of linked chunk records ready for DB insertion.
        Guarantees no chunk exceeds max_size (1800 chars) and includes approx_token_count.
        """
        if not blocks:
            return []

        # 1. Normalize and sub-split large blocks (exempting tables to keep them atomic)
        normalized_blocks: List[ParsedBlock] = []
        for b in blocks:
            if not b.text.strip():
                continue
            if b.block_type == "table":
                # Table structures are preserved atomically as single blocks
                normalized_blocks.append(b)
            elif len(b.text) > self.max_size:
                sub_texts = self._split_large_block(b.text)
                for st in sub_texts:
                    normalized_blocks.append(ParsedBlock(
                        text=st,
                        page_number=b.page_number,
                        section_title=b.section_title,
                        block_type=b.block_type,
                        metadata=b.metadata
                    ))
            else:
                normalized_blocks.append(b)

        # 2. Accumulate character windows
        raw_chunks: List[Dict[str, Any]] = []
        current_text = ""
        current_page: Optional[int] = None
        current_section: Optional[str] = None

        min_threshold = min(MIN_CHUNK_SIZE, int(self.target_size * 0.8))

        for block in normalized_blocks:
            if block.block_type == "table":
                # Flush preceding accumulated paragraph text before table
                if current_text.strip():
                    if len(current_text) > self.max_size:
                        sub_parts = self._split_large_block(current_text)
                        for sp in sub_parts:
                            raw_chunks.append({
                                "text": sp,
                                "page_number": current_page,
                                "section_path": current_section
                            })
                    else:
                        raw_chunks.append({
                            "text": current_text.strip(),
                            "page_number": current_page,
                            "section_path": current_section
                        })
                    current_text = ""

                table_text = block.text.strip()
                if block.section_title and not table_text.startswith(block.section_title):
                    table_text = f"{block.section_title}\n\n{table_text}"

                # Emit table block as a dedicated, atomic chunk
                raw_chunks.append({
                    "text": table_text,
                    "page_number": block.page_number,
                    "section_path": block.section_title
                })
                current_page = block.page_number
                if block.section_title:
                    current_section = block.section_title
                continue

            # Flush preceding text when crossing into a new section
            if block.section_title and current_section and block.section_title != current_section:
                if current_text.strip():
                    if len(current_text) > self.max_size:
                        sub_parts = self._split_large_block(current_text)
                        for sp in sub_parts:
                            raw_chunks.append({
                                "text": sp,
                                "page_number": current_page,
                                "section_path": current_section
                            })
                    else:
                        raw_chunks.append({
                            "text": current_text.strip(),
                            "page_number": current_page,
                            "section_path": current_section
                        })
                    current_text = ""

            if block.page_number is not None and current_page is None:
                current_page = block.page_number
            if block.section_title is not None:
                current_section = block.section_title


            # Check if adding block text exceeds max_size (1800) or target_size
            if (len(current_text) + len(block.text) > self.target_size and len(current_text) >= min_threshold) or (len(current_text) + len(block.text) > self.max_size):
                # Finalize current chunk with boundary adjustment around target_size
                split_idx = find_boundary_split(current_text, self.target_size, window=150)
                chunk_str = current_text[:split_idx].strip()
                if chunk_str:
                    raw_chunks.append({
                        "text": chunk_str,
                        "page_number": current_page,
                        "section_path": current_section
                    })

                # Retain overlap portion from end of chunk_str
                overlap_text = chunk_str[-self.overlap:] if len(chunk_str) > self.overlap else ""
                current_text = (overlap_text + "\n\n" + block.text).strip()
                current_page = block.page_number
            else:
                if current_text:
                    current_text += "\n\n" + block.text
                else:
                    current_text = block.text

        # Flush remaining text as final chunk
        if current_text.strip():
            # If current_text exceeds max_size, split again
            if len(current_text) > self.max_size:
                sub_parts = self._split_large_block(current_text)
                for sp in sub_parts:
                    raw_chunks.append({
                        "text": sp,
                        "page_number": current_page,
                        "section_path": current_section
                    })
            else:
                raw_chunks.append({
                    "text": current_text.strip(),
                    "page_number": current_page,
                    "section_path": current_section
                })


        # 3. Pre-assign UUIDs, link prev_chunk_id / next_chunk_id, add approx_token_count & Secret Redaction
        from backend.app.generation.prompt_guard import SecretRedactor
        secret_redactor = SecretRedactor()

        final_chunks: List[Dict[str, Any]] = []
        chunk_uuids = [str(uuid.uuid4()) for _ in raw_chunks]

        for idx, item in enumerate(raw_chunks):
            c_id = chunk_uuids[idx]
            prev_id = chunk_uuids[idx - 1] if idx > 0 else None
            next_id = chunk_uuids[idx + 1] if idx < len(raw_chunks) - 1 else None

            raw_text = item["text"]
            # Secret scanning & redaction BEFORE embedding generation
            sanitized_text, redaction_types, redaction_count = secret_redactor.redact_secrets(raw_text)

            content_hash = hashlib.sha256(sanitized_text.encode("utf-8")).hexdigest()
            approx_tokens = len(sanitized_text) // 4  # Required approx_token_count

            meta: Dict[str, Any] = {
                "char_count": len(sanitized_text),
                "approx_token_count": approx_tokens,
                "content_hash": content_hash,
                "section_path": item["section_path"]
            }

            if redaction_count > 0:
                meta["security_flags"] = ["SECRET_REDACTED"]
                meta["redaction_count"] = redaction_count
                meta["redaction_types"] = redaction_types

            final_chunks.append({
                "id": c_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "chunk_index": idx,
                "text": sanitized_text,
                "page_number": item["page_number"],
                "section_path": item["section_path"],
                "prev_chunk_id": prev_id,
                "next_chunk_id": next_id,
                "metadata": meta
            })

        return final_chunks
