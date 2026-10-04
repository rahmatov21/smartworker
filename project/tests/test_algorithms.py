"""
Tests for Algorithmic Core: BM25, RRF, Levenshtein, Entropy, N-grams.
"""

from project.src.algorithms import (
    levenshtein_distance,
    jaccard_similarity,
    extract_ngrams,
    shannon_entropy,
    lexical_diversity,
    bm25_score,
    reciprocal_rank_fusion,
)


def test_levenshtein_distance_cases():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("hello", "hello") == 0
    assert levenshtein_distance("", "test") == 4
    assert levenshtein_distance("test", "") == 4


def test_jaccard_similarity():
    assert jaccard_similarity(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0
    assert jaccard_similarity(["a", "b"], ["b", "c"]) == 1.0 / 3.0


def test_extract_ngrams():
    tokens = ["deep", "learning", "neural", "network"]
    bigrams = extract_ngrams(tokens, 2)
    assert len(bigrams) == 3
    assert bigrams[0] == ("deep", "learning")
    assert bigrams[-1] == ("neural", "network")

    trigrams = extract_ngrams(tokens, 3)
    assert len(trigrams) == 2
    assert trigrams[0] == ("deep", "learning", "neural")


def test_shannon_entropy_cases():
    assert shannon_entropy(["word", "word", "word"]) == 0.0
    assert shannon_entropy(["a", "b", "c", "d"]) == 2.0
    assert shannon_entropy([]) == 0.0


def test_lexical_diversity_cases():
    assert lexical_diversity(["test", "test", "test"]) == round(1.0 / 3.0, 4)
    assert lexical_diversity(["one", "two", "three"]) == 1.0
    assert lexical_diversity([]) == 0.0


def test_bm25_scoring():
    corpus = [
        ["deep", "learning", "model"],
        ["quantum", "computing", "algorithm"],
        ["deep", "neural", "networks", "and", "deep", "learning"],
    ]
    query = ["deep", "learning"]

    score0 = bm25_score(query, corpus[0], corpus)
    score1 = bm25_score(query, corpus[1], corpus)
    score2 = bm25_score(query, corpus[2], corpus)

    assert score0 > score1
    assert score2 > score1


def test_reciprocal_rank_fusion():
    list1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
    list2 = [(2, 0.95), (1, 0.85), (4, 0.5)]

    fused = reciprocal_rank_fusion([list1, list2])
    doc_ids = [doc_id for doc_id, _ in fused]
    assert 1 in doc_ids
    assert 2 in doc_ids
    assert len(fused) == 4
