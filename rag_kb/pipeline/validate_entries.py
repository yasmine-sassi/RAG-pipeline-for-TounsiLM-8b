"""
validate_entries.py

Validates JSON entries against Pydantic schemas.
Bad data = bad AI. This is critical.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from rag_kb.schemas.expression_schema import ExpressionEntry
from rag_kb.schemas.proverb_schema import ProverbEntry
from rag_kb.schemas.food_schema import FoodEntry
from rag_kb.schemas.ritual_schema import RitualEntry
from rag_kb.schemas.code_switch_schema import CodeSwitchEntry
from rag_kb.schemas.media_schema import MediaEntry
from rag_kb.schemas.color_schema import ColorEntry
from rag_kb.schemas.number_slang_schema import NumberSlangEntry

SCHEMA_MAP = {
    "expression": ExpressionEntry,
    "number_slang": NumberSlangEntry,
    "proverb": ProverbEntry,
    "food": FoodEntry,
    "ingredient": FoodEntry,
    "ritual": RitualEntry,
    "code_switch": CodeSwitchEntry,
    "tv_series": MediaEntry,
    "movie": MediaEntry,
    "film": MediaEntry,
    "color": ColorEntry,
}


def validate_file(file_path: str) -> Tuple[List[dict], List[Tuple[str, str]]]:
    valid_entries = []
    errors = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        errors.append(("FILE", f"File not found: {file_path}"))
        return [], errors
    except json.JSONDecodeError as e:
        errors.append(("FILE", f"Invalid JSON: {e}"))
        return [], errors

    if not isinstance(data, list):
        errors.append(("FILE", "Root must be a JSON array"))
        return [], errors

    for entry in data:
        entry_id = entry.get("id", "UNKNOWN")
        entry_type = entry.get("type", "UNKNOWN")

        schema_class = SCHEMA_MAP.get(entry_type)
        if schema_class is None:
            errors.append((entry_id, f"Unknown type: {entry_type!r}. Known types: {list(SCHEMA_MAP)}"))
            continue

        try:
            validated = schema_class(**entry)
            valid_entries.append(validated.model_dump())
        except Exception as e:
            errors.append((entry_id, str(e)))

    return valid_entries, errors


def print_validation_report(valid_entries: List[dict], errors: List[Tuple[str, str]]):
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"\n✓ Valid entries: {len(valid_entries)}")

    if errors:
        print(f"\n✗ Errors: {len(errors)}")
        print("-" * 60)
        for entry_id, error_msg in errors:
            print(f"  [{entry_id}] {error_msg}")
    else:
        print("\n✓ No errors found!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    valid, errors = validate_file("data/expressions.json")
    print_validation_report(valid, errors)
