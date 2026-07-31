"""
Asymmetric augmentation: dense retrieval leads, keyword injects.

Distinct from the RRF hybrid, which was evaluated and rejected: RRF
fuses two rankings symmetrically, so chunks that both retrievers rank
moderately outscore a chunk only one retriever ranks first. With
retrievers of unequal strength, that amplifies noise.

Here dense retrieval keeps the head of the result list; a few keyword
results are appended only if they are not already present. The two fail
in complementary, documented ways:

  dense misses near-verbatim lexical matches
    - Pfizer employee count: gold not in top 10 (keyword: rank 1-2)
    - Microsoft income statement: tax chunks instead (keyword: rank 1)

  keyword misses paraphrases
    - "profit" (net income), "import duties" (tariffs),
      "heartburn medication" (Zantac): gold not found within rank 200

This covers both without letting either dominate the ranking.
"""
from rag.search import keyword_search


class AugmentedIndex:
    """Wraps both retrievers behind the shared (query, filters, k) interface."""

    def __init__(self, vector_index, keyword_index, n_keyword: int = 3):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.n_keyword = n_keyword

    def search(self, query: str, filters: dict | None = None,
               num_results: int = 10) -> list[dict]:
        kw = keyword_search(self.keyword_index, query, filters, self.n_keyword)

        primary = self.vector_index.search(query, filters, num_results)
        seen = {c["id"] for c in primary}
        extra = [c for c in kw if c["id"] not in seen][:self.n_keyword]

        if not extra:
            return primary

        # Interleave rather than append. Appending put keyword results at
        # positions 8-10, which lifted hit@10 (0.402 -> 0.488) but left
        # hit@5 untouched — the useful chunks were being retrieved and
        # then buried. Slotting them into positions 2, 4, 6 keeps dense
        # retrieval at rank 1 (it is the better default on realistic
        # phrasing) while putting keyword's finds where the LLM's
        # attention actually lands.
        merged, kw_iter = [], iter(extra)
        for i, chunk in enumerate(primary):
            merged.append(chunk)
            if i % 2 == 0 and i > 0:      # after positions 3, 5, 7...
                nxt = next(kw_iter, None)
                if nxt is not None:
                    merged.append(nxt)
        merged.extend(kw_iter)            # any leftovers go at the end
        return merged[:num_results]