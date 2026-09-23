"""
PDF Document Parser using PyMuPDF (fitz) (Section 6.5).
"""

import fitz  # PyMuPDF
from typing import List, Any
from backend.app.ingestion.parsers.base import BaseParser, ParsedBlock



def table_to_markdown(extracted_table: List[List[Any]]) -> str:
    if not extracted_table or not extracted_table[0]:
        return ""
    header = extracted_table[0]
    lines = ["| " + " | ".join(str(cell or "").strip() for cell in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in extracted_table[1:]:
        lines.append("| " + " | ".join(str(cell or "").strip() for cell in row) + " |")
    return "\n".join(lines)


class PDFParser(BaseParser):
    def parse(self, file_bytes: bytes) -> List[ParsedBlock]:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Corrupted or invalid PDF file: {str(e)}")

        blocks: List[ParsedBlock] = []
        try:
            current_section = None

            for page_num, page in enumerate(doc, start=1):
                # Detect tables using PyMuPDF (fitz)
                tabs = page.find_tables()
                tables = list(tabs.tables) if hasattr(tabs, "tables") else []
                tab_rects = [tab.bbox for tab in tables]
                emitted_tables = set()

                text_page = page.get_text("blocks")
                for b in text_page:
                    r = fitz.Rect(b[:4])
                    matching_table_idx = None
                    for idx, tr in enumerate(tab_rects):
                        if r.intersects(tr):
                            matching_table_idx = idx
                            break

                    if matching_table_idx is not None:
                        if matching_table_idx not in emitted_tables:
                            emitted_tables.add(matching_table_idx)
                            table_md = table_to_markdown(tables[matching_table_idx].extract())
                            if table_md.strip():
                                blocks.append(ParsedBlock(
                                    text=table_md.strip(),
                                    page_number=page_num,
                                    section_title=current_section,
                                    block_type="table"
                                ))
                    else:
                        text = b[4].strip()
                        if not text:
                            continue

                        # Lightweight section tracking heuristic
                        lines = text.split("\n")
                        first_line = lines[0].strip()
                        if len(lines) == 1 and len(first_line) < 100 and not first_line.endswith("."):
                            current_section = first_line

                        blocks.append(ParsedBlock(
                            text=text,
                            page_number=page_num,
                            section_title=current_section,
                            block_type="paragraph"
                        ))
        finally:
            doc.close()

        return blocks

