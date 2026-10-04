"""
Unit tests for AI Text Processing & Retrieval Pipeline.
Tests basic tokenization, similarity ranking, BM25, RRF hybrid search,
boundary conditions, and metrics.
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


def test_shannon_entropy_calculation():
    pipeline = TextPipeline()
    assert pipeline.shannon_entropy(["word", "word", "word"]) == 0.0
    assert pipeline.shannon_entropy(["a", "b", "c", "d"]) == 2.0


def test_lexical_diversity_calculation():
    pipeline = TextPipeline()
    assert pipeline.lexical_diversity(["unique", "words", "here"]) == 1.0
    assert pipeline.lexical_diversity(["repeat", "repeat"]) == 0.5


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
    top_doc_idx = results[0][0]
    assert top_doc_idx in (0, 2)
    assert results[0][1] > results[-1][1]


def test_bm25_search_ranking():
    pipeline = TextPipeline()
    docs = [
        "Python programming language for software development.",
        "Statistical methods and probabilistic regression models.",
        "Advanced Python modules and asynchronous event loops.",
    ]
    query = "Python programming"
    results = pipeline.bm25_search(query, docs)
    assert len(results) == 3
    assert results[0][0] in (0, 2)


def test_hybrid_search():
    pipeline = TextPipeline()
    docs = [
        "Information retrieval systems and vector space models.",
        "Cooking gourmet meals with authentic ingredients.",
        "Search engine ranking algorithms and keyword indices.",
    ]
    query = "search information retrieval"
    results = pipeline.hybrid_search(query, docs)
    assert len(results) == 3
    assert results[0][0] in (0, 2)


def test_pipeline_boundary_empty_string():
    pipeline = TextPipeline()
    res = pipeline.process("")
    assert res["tokens"] == []
    assert res["word_count"] == 0


def test_pipeline_unicode_handling():
    pipeline = TextPipeline()
    res = pipeline.process("Hello 🌍 世界! Café naïve.")
    assert "hello" in res["tokens"]
    assert res["word_count"] > 0


def test_pipeline_batch_processing():
    pipeline = TextPipeline()
    batch = pipeline.batch_process(["First sentence.", "Second sentence."])
    assert len(batch) == 2
    assert batch[0]["word_count"] == 2


def test_pipeline_levenshtein_distance():
    pipeline = TextPipeline()
    assert pipeline.levenshtein_distance("kitten", "sitting") == 3
    assert pipeline.levenshtein_distance("same", "same") == 0


def test_pipeline_ngrams_extraction():
    pipeline = TextPipeline()
    tokens = ["deep", "learning", "neural", "network"]
    bigrams = pipeline.ngrams(tokens, 2)
    assert len(bigrams) == 3
    assert bigrams[0] == ("deep", "learning")


def test_pipeline_cache_stats():
    pipeline = TextPipeline()
    pipeline.process("Cached test text query.")
    pipeline.process("Cached test text query.")  # Repeat hits cache
    stats = pipeline.cache_stats()
    assert stats["cache_active"] is True
    assert stats["hits"] >= 1
