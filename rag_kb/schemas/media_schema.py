from .base_schema import BaseEntry
from typing import Optional, List


class MediaEntry(BaseEntry):
    example: Optional[str] = None
    channel_platform: Optional[str] = None
    era: Optional[str] = None
    cultural_significance: Optional[str] = None
    common_references: Optional[List[str]] = None
    embed_text: Optional[str] = None
