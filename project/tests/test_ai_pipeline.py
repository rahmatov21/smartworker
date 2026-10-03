import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
import functools
"""
Unit tests for AI Text Processing & Retrieval Pipeline.
"""

import pytest
from project.src.ai_pipeline import TextPipeline


def test_tokenize_basic():
    pipeline = TextPipeline()
    tokens = pipeline.tokenize("The quick brown fox jumps over the lazy dog.")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "the" not in tokens  # Stop word filtered
    assert len(tokens) == 7


def test_term_frequency_calculation():
    pipeline = TextPipeline()
    tokens = ["apple", "banana", "apple", "cherry"]
    tf = pipeline.term_frequency(tokens)
    assert tf["apple"] == 0.5
    assert tf["banana"] == 0.25
    assert tf["cherry"] == 0.25


def test_cosine_similarity_identical():
    pipeline = TextPipeline()
    vec1 = {"hello": 0.5, "world": 0.5}
    sim = pipeline.cosine_similarity(vec1, vec1)
    assert pytest.approx(sim, 0.001) == 1.0


def test_cosine_similarity_orthogonal():
    pipeline = TextPipeline()
    vec1 = {"apple": 1.0}
    vec2 = {"orange": 1.0}
    sim = pipeline.cosine_similarity(vec1, vec2)
    assert sim == 0.0


def test_search_ranking():
    pipeline = TextPipeline()
    docs = [
        "Machine learning algorithms improve through experience.",
        "Culinary recipes for Italian pasta and pizza.",
        "Deep neural network learning and artificial intelligence.",
    ]
    query = "machine learning and intelligence"
    results = pipeline.search(query, docs)
    # The first document or third document should have highest similarity
    top_doc_idx = results[0][0]
    assert top_doc_idx in (0, 2)
    assert results[0][1] > results[-1][1]
