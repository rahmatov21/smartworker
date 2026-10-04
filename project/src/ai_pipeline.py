"""
AI Text Processing & Semantic Retrieval Pipeline.
Production-grade modular search, feature engineering, and lexical analytics.
"""

import math
import re
from typing import Dict, List, Tuple, Any, Optional

try:
    from project.src.algorithms import (
        levenshtein_distance as _levenshtein,
        extract_ngrams as _extract_ngrams,
        shannon_entropy as _entropy,
        lexical_diversity as _diversity,
        bm25_score as _bm25,
        reciprocal_rank_fusion as _rrf,
    )
    from project.src.cache import LRUCache
    from project.src.text_utils import clean_text, normalize_unicode, tokenize as _tokenize
    from project.src.models import SearchResult, PipelineMetrics
except (ImportError, ValueError):
    try:
        from .algorithms import (
            levenshtein_distance as _levenshtein,
            extract_ngrams as _extract_ngrams,
            shannon_entropy as _entropy,
            lexical_diversity as _diversity,
            bm25_score as _bm25,
            reciprocal_rank_fusion as _rrf,
        )
        from .cache import LRUCache
        from .text_utils import clean_text, normalize_unicode, tokenize as _tokenize
        from .models import SearchResult, PipelineMetrics
    except (ImportError, ValueError):
        _levenshtein = None
        _extract_ngrams = None
        _entropy = None
        _diversity = None
        _bm25 = None
        _rrf = None
        LRUCache = None


class TextPipeline:
    def __init__(self, cache_size: int = 1024, cache_ttl: float = 300.0):
        self.stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "was"
        }
        self.cache = LRUCache(maxsize=cache_size, ttl_seconds=cache_ttl) if LRUCache else None

    def tokenize(self, text: str) -> List[str]:
        """Splits text into lowercase alphanumeric tokens, filtering stop words."""
        if not text or not isinstance(text, str):
            return []

        # Check cache
        if self.cache:
            cached = self.cache.get(f"tok:{text}")
            if cached is not None:
                return cached

        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if t not in self.stop_words]

        if self.cache:
            self.cache.set(f"tok:{text}", tokens)
        return tokens

    def term_frequency(self, tokens: List[str]) -> Dict[str, float]:
        """Calculates normalized term frequency."""
        if not tokens:
            return {}
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        return {k: v / total for k, v in counts.items()}

    def shannon_entropy(self, tokens: List[str]) -> float:
        """Calculates Shannon entropy, in bits, for a token sequence."""
        if _entropy is not None:
            return _entropy(tokens)
        if not tokens:
            return 0.0
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def lexical_diversity(self, tokens: List[str]) -> float:
        """Calculates lexical diversity as unique terms divided by total terms."""
        if _diversity is not None:
            return _diversity(tokens)
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculates Levenshtein edit distance between two strings."""
        if _levenshtein is not None:
            return _levenshtein(s1, s2)
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dels = curr[j] + 1
                subs = prev[j] + (c1 != c2)
                curr.append(min(ins, dels, subs))
            prev = curr
        return prev[-1]

    def ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Generates contiguous n-grams from a sequence of tokens."""
        if _extract_ngrams is not None:
            return _extract_ngrams(tokens, n)
        if not tokens or n < 1 or len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Computes cosine similarity between two term-frequency sparse vectors."""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(vec1[k] * vec2[k] for k in vec1 if k in vec2)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Calculates token overlap similarity between two texts."""
        tokens_a = set(self.tokenize(text_a))
        tokens_b = set(self.tokenize(text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return float(intersection) / float(union) if union > 0 else 0.0

    def similarity_matrix(self, texts: List[str]) -> List[List[float]]:
        """Computes pairwise similarity matrix across document texts."""
        return [[self.similarity(t1, t2) for t2 in texts] for t1 in texts]

    def cache_stats(self) -> Dict[str, Any]:
        """Returns operational metrics and hit/miss stats of the active cache."""
        if self.cache:
            stats = self.cache.stats()
            stats["cache_active"] = True
            return stats
        return {"cache_active": False, "status": "cache_disabled"}

    def process(self, text: str) -> Dict[str, Any]:
        """Processes raw text into token representation and term frequencies."""
        tokens = self.tokenize(text)
        tf = self.term_frequency(tokens)
        return {
            "tokens": tokens,
            "term_frequency": tf,
            "word_count": len(tokens),
            "unique_terms": len(tf),
            "entropy": self.shannon_entropy(tokens),
            "lexical_diversity": self.lexical_diversity(tokens),
        }

    def search(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Ranks documents by cosine similarity against query."""
        query_vec = self.term_frequency(self.tokenize(query))
        ranked = []
        for idx, doc in enumerate(documents):
            doc_vec = self.term_frequency(self.tokenize(doc))
            sim = self.cosine_similarity(query_vec, doc_vec)
            ranked.append((idx, round(sim, 4)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def bm25_search(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Ranks documents using BM25 Okapi probabilistic retrieval model."""
        if _bm25 is None:
            return self.search(query, documents)

        q_tokens = self.tokenize(query)
        tokenized_corpus = [self.tokenize(d) for d in documents]
        ranked = []
        for idx, doc_tokens in enumerate(tokenized_corpus):
            score = _bm25(q_tokens, doc_tokens, tokenized_corpus)
            ranked.append((idx, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def hybrid_search(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Fuses Vector Cosine and BM25 search rankings using Reciprocal Rank Fusion."""
        if _rrf is None:
            return self.search(query, documents)

        vec_ranks = self.search(query, documents)
        bm25_ranks = self.bm25_search(query, documents)
        return _rrf([vec_ranks, bm25_ranks])

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes multiple text documents in a single batch invocation."""
        return [self.process(t) for t in texts]
