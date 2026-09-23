"""
HTML Document Parser using Trafilatura (Section 6.5).
"""

import trafilatura
from typing import List
from backend.app.ingestion.parsers.base import BaseParser, ParsedBlock


class HTMLParser(BaseParser):
    def parse(self, file_bytes: bytes) -> List[ParsedBlock]:
        try:
            html_str = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Corrupted or invalid UTF-8 encoding in HTML file: {str(e)}")

        extracted_text = trafilatura.extract(html_str, include_formatting=False, include_links=False)

        if not extracted_text:
            # Fallback simple strip if trafilatura returns None
            import re
            extracted_text = re.sub(r'<[^>]+>', ' ', html_str)

        blocks: List[ParsedBlock] = []
        paragraphs = [p.strip() for p in extracted_text.split("\n\n") if p.strip()]

        current_section = None
        for p in paragraphs:
            # Lightweight section tracking heuristic (short line without trailing period)
            lines = p.split("\n")
            if len(lines) == 1 and len(lines[0]) < 80 and not lines[0].endswith("."):
                current_section = lines[0].strip()

            blocks.append(ParsedBlock(
                text=p,
                section_title=current_section,
                block_type="paragraph"
            ))

        return blocks
