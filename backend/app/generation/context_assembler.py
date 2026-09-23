"""
Context Assembler Module (Phase 4 — Basic RAG)
Implements deterministic RAG context assembly with 4000-char budget enforcement.
"""

import os
from typing import List, Dict, Any, Optional

# INVARIANT (RAG Benchmark Regression Decision): 8,000 character context budget.
# Provides necessary headroom for multi-chunk retrieval without triggering hallucinations or refusal degradation.
DEFAULT_MAX_CONTEXT_CHARS = 8000


class ContextAssembler:
    """
    Assembles retrieved vector-search chunks into a deterministic, character-bounded
    context block for RAG prompt construction.
    """

    def __init__(self, max_chars: Optional[int] = None):
        if max_chars is None:
            env_val = os.getenv("RAG_MAX_CONTEXT_CHARS")
            self.max_chars = int(env_val) if env_val and env_val.isdigit() else DEFAULT_MAX_CONTEXT_CHARS
        else:
            self.max_chars = max_chars

    def format_chunk(self, chunk: Dict[str, Any], index: int) -> str:
        """
        Format a single retrieved chunk with clear structural metadata delimiters.
        """
        chunk_id = chunk.get("chunk_id", "")
        doc_id = chunk.get("document_id", "")
        doc_ver = chunk.get("document_version", "")
        page = chunk.get("page_number")
        page_str = str(page) if page is not None else "N/A"
        section = chunk.get("section_path") or "N/A"
        text = chunk.get("text", "")

        header = (
            f"--- Context Entry {index + 1} ---\n"
            f"[Chunk ID: {chunk_id} | Document ID: {doc_id} | Version: {doc_ver} | Page: {page_str} | Section: {section}]\n"
        )
        return f"{header}{text}"

    def assemble(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assemble chunks sequentially until adding the next complete chunk would exceed max_chars.

        Args:
            search_results: List of chunk dictionaries returned by VectorSearchEngine.

        Returns:
            Dict containing:
                - context_text: Fully assembled context string
                - included_chunks: List of included chunk dicts with metadata preserved
                - excluded_chunks: List of excluded chunk dicts
                - total_chars: Length of context_text
                - truncated_by_budget: Boolean indicating if budget limit excluded any chunks
        """
        if not search_results:
            return {
                "context_text": "",
                "included_chunks": [],
                "excluded_chunks": [],
                "total_chars": 0,
                "truncated_by_budget": False,
            }

        formatted_chunks = []
        included_chunks = []
        excluded_chunks = []
        current_len = 0
        truncated = False

        # Structural wrapper preamble
        preamble = "=== BEGIN UNTRUSTED RETRIEVED CONTEXT ===\n"
        postamble = "\n=== END UNTRUSTED RETRIEVED CONTEXT ==="
        wrapper_len = len(preamble) + len(postamble)

        # Effective character budget available for chunk contents + separators
        effective_budget = self.max_chars - wrapper_len
        if effective_budget <= 0:
            return {
                "context_text": "",
                "included_chunks": [],
                "excluded_chunks": list(search_results),
                "total_chars": 0,
                "truncated_by_budget": True,
            }

        for idx, chunk in enumerate(search_results):
            if truncated:
                excluded_chunks.append(chunk)
                continue

            chunk_str = self.format_chunk(chunk, idx)
            sep = "\n\n" if formatted_chunks else ""
            candidate_addition_len = len(sep) + len(chunk_str)

            if current_len + candidate_addition_len <= effective_budget:
                formatted_chunks.append(chunk_str)
                included_chunks.append(chunk)
                current_len += candidate_addition_len
            else:
                # Adding this complete chunk exceeds character budget
                truncated = True
                excluded_chunks.append(chunk)

        if not formatted_chunks:
            return {
                "context_text": "",
                "included_chunks": [],
                "excluded_chunks": list(search_results),
                "total_chars": 0,
                "truncated_by_budget": True,
            }

        joined_chunks = "\n\n".join(formatted_chunks)
        assembled_text = f"{preamble}{joined_chunks}{postamble}"

        return {
            "context_text": assembled_text,
            "included_chunks": included_chunks,
            "excluded_chunks": excluded_chunks,
            "total_chars": len(assembled_text),
            "truncated_by_budget": truncated,
        }
