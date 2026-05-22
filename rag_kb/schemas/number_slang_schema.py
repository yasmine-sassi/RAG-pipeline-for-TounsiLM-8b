from .base_schema import BaseEntry
from typing import Optional


class NumberSlangEntry(BaseEntry):
    msa_equivalent: Optional[str] = None
