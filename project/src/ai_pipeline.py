import functools
"""
AI Text Processing & Semantic Retrieval Pipeline.
Starter project codebase for autonomous self-improvement.
"""

import functools
import math
import re
from typing import Dict, List, Tuple, Any


class TextPipeline:
    def __init__(self):
        self.stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "was"
        }

    def tokenize(self, text: str) -> List[str]:
        """Splits text into lowercase alphanumeric tokens, filtering stop words."""
        if not text or not isinstance(text, str):
            return []
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()
        return [t for t in tokens if t not in self.stop_words]

    def term_frequency(self, tokens: List[str]) -> Dict[str, float]:
        """Calculates normalized term frequency."""
        if not tokens:
            return {}
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        return {k: v / total for k, v in counts.items()}

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

    def process(self, text: str) -> Dict[str, Any]:
        """Processes raw text into token representation and term frequencies."""
        tokens = self.tokenize(text)
        tf = self.term_frequency(tokens)
        return {
            "tokens": tokens,
            "term_frequency": tf,
            "word_count": len(tokens),
            "unique_terms": len(tf),
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

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes multiple text documents in a batch."""
        return [self.process(t) for t in texts]

