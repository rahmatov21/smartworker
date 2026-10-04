"""
Text Processing Utilities & Unicode Normalization.
Provides regex tokenization, punctuation stripping, case folding,
and keyword extraction.
"""

import re
import unicodedata
from typing import List, Set, Optional


DEFAULT_STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "is", "was", "are", "were", "it", "this", "that", "by", "as", "from"
}


def normalize_unicode(text: str) -> str:
    """Normalizes Unicode text to NFKC standard form."""
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, remove_digits: bool = False) -> str:
    """Cleans text by stripping punctuation and normalizing whitespace."""
    if not text or not isinstance(text, str):
        return ""
    normalized = normalize_unicode(text)
    pattern = r"[^\w\s]" if not remove_digits else r"[^\w\s]|\d"
    cleaned = re.sub(pattern, " ", normalized)
    return " ".join(cleaned.split())


def tokenize(text: str, stop_words: Optional[Set[str]] = None) -> List[str]:
    """Tokenizes text into lowercase tokens filtering stop words."""
    if not text:
        return []
    stops = stop_words if stop_words is not None else DEFAULT_STOP_WORDS
    cleaned = clean_text(text).lower()
    tokens = cleaned.split()
    return [t for t in tokens if t not in stops]


def extract_keywords(tokens: List[str], top_k: int = 5) -> List[str]:
    """Extracts the most frequent keywords from a token list."""
    if not tokens:
        return []
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    sorted_keywords = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in sorted_keywords[:top_k]]
