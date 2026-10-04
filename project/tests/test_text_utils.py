"""
Tests for Text Utilities: Normalization, Cleaning, Tokenization.
"""

from project.src.text_utils import (
    clean_text,
    normalize_unicode,
    tokenize,
    extract_keywords,
)


def test_unicode_normalization():
    accented = "Café naïve résumé"
    norm = normalize_unicode(accented)
    assert norm == accented
    assert normalize_unicode("") == ""


def test_clean_text_strips_punctuation():
    text = "Hello, world! Welcome to AI (Artificial Intelligence)..."
    cleaned = clean_text(text)
    assert "," not in cleaned
    assert "!" not in cleaned
    assert "..." not in cleaned
    assert "Hello world Welcome to AI Artificial Intelligence" == cleaned


def test_tokenize_filters_stop_words():
    text = "The quick brown fox is running in a green forest"
    tokens = tokenize(text)
    assert "the" not in tokens
    assert "is" not in tokens
    assert "in" not in tokens
    assert "a" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_extract_keywords():
    tokens = ["ai", "ai", "model", "model", "model", "data"]
    top = extract_keywords(tokens, top_k=2)
    assert top == ["model", "ai"]
