"""
Data Models and Structured Representations for AI Pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    doc_id: int
    score: float
    text: str
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineMetrics:
    tokens_count: int
    unique_terms: int
    shannon_entropy: float
    lexical_diversity: float
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
