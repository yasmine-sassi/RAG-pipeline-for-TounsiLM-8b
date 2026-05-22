def build_embed_text(entry: dict) -> str:
    if entry.get("embed_text"):
        return "passage: " + entry["embed_text"]

    entry_type = entry.get("type", "")

    parts = [
        entry.get("term_arabic", ""),
        entry.get("term_arabizi", ""),
        entry.get("meaning", ""),
        entry.get("meaning_fr", ""),
        entry.get("example", ""),
        entry.get("usage_context", ""),
    ]

    if entry_type == "proverb":
        parts.extend([
            entry.get("literal_meaning", ""),
            entry.get("real_meaning", ""),
            entry.get("when_used", ""),
            entry.get("msa_equivalent", ""),
        ])
    elif entry_type in ("expression", "number_slang"):
        parts.extend([
            entry.get("origin", ""),
            entry.get("msa_equivalent", ""),
        ])
    elif entry_type == "food":
        parts.extend([
            entry.get("description", ""),
            entry.get("regional_variation", ""),
            entry.get("when_eaten", ""),
            entry.get("similar_to", ""),
        ])
    elif entry_type == "ritual":
        parts.extend([
            entry.get("occasion", ""),
            entry.get("tone", ""),
            entry.get("expected_response", ""),
        ])
    elif entry_type == "code_switch":
        parts.extend([
            entry.get("origin_language", ""),
            entry.get("origin_word", ""),
            entry.get("tunisian_equivalent", ""),
            entry.get("domain", ""),
            entry.get("switch_type", ""),
        ])
    elif entry_type in ("tv_series", "movie"):
        parts.extend([
            entry.get("cultural_significance", ""),
            entry.get("era", ""),
            entry.get("channel_platform", ""),
        ])
        refs = entry.get("common_references", [])
        if isinstance(refs, list):
            parts.append(" ".join(refs))
    elif entry_type == "color":
        parts.extend([
            entry.get("color_family", ""),
            entry.get("cultural_significance", ""),
        ])

    return "passage: " + " ".join(p for p in parts if p and isinstance(p, str))
