"""
CSV Document Parser using pandas (Section 6.5).
Strictly validates UTF-8 text encoding without silently discarding invalid bytes.
"""

import io
import pandas as pd
from typing import List
from backend.app.ingestion.parsers.base import BaseParser, ParsedBlock


class CSVParser(BaseParser):
    def parse(self, file_bytes: bytes) -> List[ParsedBlock]:
        blocks: List[ParsedBlock] = []

        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            # Fallback simple line parser for raw CSV text if pandas parsing fails
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError as err:
                raise ValueError(f"Corrupted or invalid UTF-8 encoding in CSV document: {str(err)}")

            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for idx, line in enumerate(lines, start=1):
                blocks.append(ParsedBlock(
                    text=line,
                    block_type="paragraph",
                    metadata={"row_index": idx}
                ))
            return blocks

        columns = list(df.columns)
        header_summary = f"Columns: {', '.join(str(c) for c in columns)}"
        blocks.append(ParsedBlock(
            text=header_summary,
            block_type="heading"
        ))

        for idx, row in df.iterrows():
            row_items = []
            for col in columns:
                val = row[col]
                if pd.notna(val):
                    row_items.append(f"{col}: {val}")
            if row_items:
                row_text = f"Row {idx + 1}: " + " | ".join(row_items)
                blocks.append(ParsedBlock(
                    text=row_text,
                    block_type="paragraph",
                    metadata={"row_index": idx + 1}
                ))

        return blocks
