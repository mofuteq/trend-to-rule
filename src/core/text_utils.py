import unicodedata


def normalize_text_nfkc(text: str) -> str:
    """Normalize text with Unicode NFKC."""
    return unicodedata.normalize("NFKC", text)
