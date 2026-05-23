import re
from typing import Optional

# Arabizi digit-to-letter substitutions for query normalization
_ARABIZI_MAP = {
    '3': 'a',   # ع
    '7': 'h',   # ح
    '5': 'kh',  # خ
    '9': 'q',   # ق
    '8': 'gh',  # غ
    '2': '',    # ء  (drop)
}

# Keywords that signal a specific entry type — used for automatic query routing
_ROUTE_KEYWORDS: dict[str, list[str]] = {
    'food': [
        'اكل', 'طعام', 'طبق', 'مطبخ', 'طبيخ', 'وجبة',
        'akl', 'matabkh', 'tbikh', 'makla',
        'harissa', 'هريسة', 'couscous', 'كسكسي',
        'food', 'dish', 'recipe', 'recette',
    ],
    'proverb': [
        'مثل', 'قولة', 'حكمة', 'أمثال',
        'methel', 'qawla', 'hikma', 'proverbe', 'proverb',
    ],
    'expression': [
        'معنى', 'تعبير', 'كلمة', 'اصطلاح',
        'maana', 'taabir', 'klma', 'meaning', 'signifie', 'veut dire',
    ],
    'color': [
        'لون', 'ألوان', 'لوان',
        'loun', 'lwana', 'color', 'colour', 'couleur',
    ],
    'ritual': [
        'تحية', 'عادة', 'تقليد', 'مناسبة',
        'tahya', 'ada', 'tqlid', 'greeting', 'ritual', 'occasion',
    ],
    'code_switch': [
        'فرنسية', 'بالفرنسي', 'مقتبس',
        'fransawi', 'français', 'french', 'emprunt',
    ],
}


class QueryRewriter:
    """Generates query variants and auto-detects entry type for routing."""

    def rewrite(self, query: str, n_variants: int = 2) -> list[str]:
        variants = [query]

        # Variant 1: normalize Arabizi digits to letters
        normalized = _normalize_arabizi(query)
        if normalized != query:
            variants.append(normalized)

        # Variant 2: strip punctuation
        cleaned = re.sub(r'[؟?!،,\.]+', ' ', query).strip()
        if cleaned not in variants:
            variants.append(cleaned)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                unique.append(v)

        return unique[:n_variants + 1]

    def detect_type(self, query: str) -> Optional[str]:
        """Return the most likely entry_type for the query, or None."""
        q = query.lower()
        for entry_type, keywords in _ROUTE_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return entry_type
        return None


def _normalize_arabizi(text: str) -> str:
    for digit, letter in _ARABIZI_MAP.items():
        text = text.replace(digit, letter)
    return text
