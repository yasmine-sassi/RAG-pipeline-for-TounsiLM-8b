from .base_schema import BaseEntry
from typing import Optional


class RitualEntry(BaseEntry):
    occasion: Optional[str] = None
    expected_response: Optional[str] = None
    tone: Optional[str] = None
