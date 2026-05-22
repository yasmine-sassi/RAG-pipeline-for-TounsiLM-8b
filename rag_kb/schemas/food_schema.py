from .base_schema import BaseEntry
from typing import Optional


class FoodEntry(BaseEntry):
    description: Optional[str] = None
    regional_variation: Optional[str] = None
    when_eaten: Optional[str] = None
    similar_to: Optional[str] = None
    embed_text: Optional[str] = None
