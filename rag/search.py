"""Keyword (TF-IDF) retrieval over the chunked corpus."""
import json
from pathlib import Path

from minsearch import Index

CHUNKS_FILE = Path("data/processed/chunks.json")


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def build_index(chunks: list[dict]) -> Index:
    # 'section' is None when the Item-detection regex found nothing
    # (best-effort tagging, see chunk_filings.py). sklearn's vectorizer
    # requires strings, so we normalize None -> "" at the index
    # boundary. The empty string is simply ignored by TF-IDF (no terms
    # to count), which is exactly the behavior we want.
    for chunk in chunks:
        chunk["section"] = chunk["section"] or ""

    index = Index(
        text_fields=["text", "section"],
        keyword_fields=["ticker", "form"],
    )
    index.fit(chunks)
    return index


def keyword_search(index: Index, query: str, filters: dict | None = None,
                   num_results: int = 5) -> list[dict]:
    """Unified retriever interface: (query, filters, k) -> chunks.

    Both retrievers expose the same signature so evaluation (and later
    hybrid search) can treat them interchangeably.
    """
    return index.search(query=query, filter_dict=filters or {},
                        num_results=num_results)