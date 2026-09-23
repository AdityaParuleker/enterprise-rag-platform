"""
Markdown Document Parser (Section 6.5).
"""

import re
from typing import List
from backend.app.ingestion.parsers.base import BaseParser, ParsedBlock


class MarkdownParser(BaseParser):
    def parse(self, file_bytes: bytes) -> List[ParsedBlock]:
        try:
            text_str = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Corrupted or invalid UTF-8 encoding in Markdown file: {str(e)}")

        lines = text_str.split("\n")

        blocks: List[ParsedBlock] = []
        current_paragraph: List[str] = []
        current_section = None

        def flush_paragraph():
            nonlocal current_paragraph
            if current_paragraph:
                para_text = "\n".join(current_paragraph).strip()
                if para_text:
                    blocks.append(ParsedBlock(
                        text=para_text,
                        section_title=current_section,
                        block_type="paragraph"
                    ))
                current_paragraph = []

        for line in lines:
            stripped = line.strip()
            heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
            if heading_match:
                flush_paragraph()
                heading_text = heading_match.group(1).strip()
                current_section = heading_text
                blocks.append(ParsedBlock(
                    text=heading_text,
                    section_title=current_section,
                    block_type="heading"
                ))
            elif not stripped:
                flush_paragraph()
            else:
                current_paragraph.append(line)

        flush_paragraph()
        return blocks
