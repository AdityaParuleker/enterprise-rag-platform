"""
Base Parser Interface & Data Classes (Section 6.5).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ParsedBlock:
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    block_type: str = "paragraph"  # heading, paragraph, table, code
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes) -> List[ParsedBlock]:
        """Parse raw file bytes into a list of structured ParsedBlocks."""
        pass
