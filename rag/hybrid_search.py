"""
...

STATUS: IMPLEMENTED AND EVALUATED, NOT ADOPTED.

On hand-written questions, hybrid+rule_filter scores 0.200 hit@5 (0.500
expanded) against 0.400 (0.800) for dense retrieval with the same
filters. Down-weighting keyword to 0.5 improves hybrid to 0.300 (0.700)
— the trend points toward weight zero, i.e. dense retrieval alone.

Mechanism: RRF rewards agreement between retrievers, which is evidence
only when both are comparably strong. Here keyword search scores 0.000
hit@5 on realistic phrasing, so consensus amplified generic chunks that
both retrievers rank moderately, pushing down the relevant chunk that
only dense retrieval identified. Lowering RRF_K from 60 to 10 and
CANDIDATE_DEPTH from 100 to 25 helped (overall hit@5 0.280 -> 0.305),
confirming the diagnosis, but not enough to overturn the verdict.
"""

from rag.search import keyword_search
from rag.vector_search import VectorIndex

RRF_K = 10  # was 60 — sharpens the advantage of top positions


# Each retriever is queried deeper than the final result count: a chunk
# ranked 40th by one retriever and 2nd by the other should still be able
# to surface. Fusing only two top-10 lists would discard exactly the
# complementary evidence hybrid search exists to exploit.

CANDIDATE_DEPTH = 25  # was 100 — fewer weak candidates entering the pool


class HybridIndex:
    """Wraps both retrievers behind the shared (query, filters, k) interface."""

    def __init__(self, keyword_index, vector_index: VectorIndex,
                 keyword_weight: float = 1.0, vector_weight: float = 1.0):
        self.keyword_index = keyword_index
        self.vector_index = vector_index
        # Weights allow biasing the fusion toward the retriever that
        # measures better on realistic questions. Equal weights are the
        # neutral starting point; the eval decides whether to shift them.
        self.weights = {"keyword": keyword_weight, "vector": vector_weight}

    def search(self, query: str, filters: dict | None = None,
               num_results: int = 10) -> list[dict]:
        runs = {
            "keyword": keyword_search(self.keyword_index, query, filters,
                                      CANDIDATE_DEPTH),
            "vector": self.vector_index.search(query, filters,
                                               CANDIDATE_DEPTH),
        }

        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}

        for source, results in runs.items():
            weight = self.weights[source]
            for rank, chunk in enumerate(results, start=1):
                cid = chunk["id"]
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
                chunks_by_id[cid] = chunk

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [chunks_by_id[cid] for cid in ranked[:num_results]]


def build_hybrid_index(chunks: list[dict], keyword_index,
                       **weights) -> HybridIndex:
    """Convenience constructor reusing an already-built keyword index."""
    return HybridIndex(keyword_index, VectorIndex(), **weights)
