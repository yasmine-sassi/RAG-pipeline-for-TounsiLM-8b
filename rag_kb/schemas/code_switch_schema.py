from .base_schema import BaseEntry
from typing import Optional


class CodeSwitchEntry(BaseEntry):
    origin_language: Optional[str] = None
    origin_word: Optional[str] = None
    tunisian_equivalent: Optional[str] = None
    switch_type: Optional[str] = None
    domain: Optional[str] = None
