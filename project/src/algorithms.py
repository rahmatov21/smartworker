"""
Algorithmic Core for Text Processing & Semantic Ranking.
Implements edit distance, lexical diversity, Shannon entropy, BM25 scoring,
and Reciprocal Rank Fusion (RRF).
"""

import math
from typing import Dict, List, Tuple, Set, Any


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Computes Jaccard set overlap similarity."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


def extract_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generates contiguous n-grams from a sequence of tokens."""
    if not tokens or n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def shannon_entropy(tokens: List[str]) -> float:
    """Calculates Shannon entropy, in bits, of token probability distribution."""
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def lexical_diversity(tokens: List[str]) -> float:
    """Calculates Type-Token Ratio (TTR) lexical diversity."""
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    corpus_docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Computes BM25 relevance score for a document against a query.
    """
    if not query_tokens or not doc_tokens or not corpus_docs:
        return 0.0

    total_docs = len(corpus_docs)
    avg_dl = sum(len(d) for d in corpus_docs) / total_docs if total_docs > 0 else 1.0
    doc_len = len(doc_tokens)

    doc_counts: Dict[str, int] = {}
    for t in doc_tokens:
        doc_counts[t] = doc_counts.get(t, 0) + 1

    score = 0.0
    for q in query_tokens:
        # Calculate Inverse Document Frequency (IDF)
        n_q = sum(1 for d in corpus_docs if q in d)
        idf = math.log((total_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)

        f_q = doc_counts.get(q, 0)
        numerator = f_q * (k1 + 1.0)
        denominator = f_q + k1 * (1.0 - b + b * (doc_len / avg_dl))
        if denominator > 0:
            score += idf * (numerator / denominator)

    return round(max(0.0, score), 4)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Combines multiple ranked retrieval lists using Reciprocal Rank Fusion (RRF).
    """
    scores: Dict[int, float] = {}
    for r_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(r_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, round(score, 6)) for doc_id, score in fused]
