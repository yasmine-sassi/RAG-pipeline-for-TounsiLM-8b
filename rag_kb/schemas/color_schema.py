from .base_schema import BaseEntry
from typing import Optional


class ColorEntry(BaseEntry):
    color_family: Optional[str] = None
    cultural_significance: Optional[str] = None
