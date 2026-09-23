"""
DOCX Document Parser using python-docx with XXE Entity Resolution Defense (Section 6.5).
"""

import io
import docx
from lxml import etree
from typing import List
from backend.app.ingestion.parsers.base import BaseParser, ParsedBlock

# Configure default lxml parser globally to disable entity resolution and network access
safe_xml_parser = etree.XMLParser(resolve_entities=False, no_network=True, dtd_validation=False)
etree.set_default_parser(safe_xml_parser)

# Ensure python-docx oxml parser uses safe_xml_parser
try:
    docx.oxml.parser = safe_xml_parser
except AttributeError:
    pass


class DOCXParser(BaseParser):
    def parse(self, file_bytes: bytes) -> List[ParsedBlock]:
        blocks: List[ParsedBlock] = []

        # Enforce safe XML parser prior to parsing
        etree.set_default_parser(safe_xml_parser)

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Corrupted or invalid DOCX file: {str(e)}")

        current_section = None

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            style_name = paragraph.style.name if paragraph.style else ""
            if "Heading" in style_name:
                current_section = text
                blocks.append(ParsedBlock(
                    text=text,
                    section_title=current_section,
                    block_type="heading"
                ))
            else:
                blocks.append(ParsedBlock(
                    text=text,
                    section_title=current_section,
                    block_type="paragraph"
                ))

        # Extract table text
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_rows.append(" | ".join(row_cells))
            if table_rows:
                table_text = "\n".join(table_rows)
                blocks.append(ParsedBlock(
                    text=table_text,
                    section_title=current_section,
                    block_type="table"
                ))

        return blocks
