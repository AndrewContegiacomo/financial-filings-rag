"""

...

STATUS: IMPLEMENTED AND EVALUATED, NOT ADOPTED.

Hand-written questions, hit@5: 0.300 with reranking vs 0.400 without
(0.700 vs 0.800 expanded). Reducing depth 30 -> 15 did not help.

The premise was wrong. hit@1 = 0.110 suggested plenty of room to
promote golds, but that figure is dominated by the 72 synthetic items.
On the 10 realistic questions the first stage already ranks 7 of 10
golds within position 3 — there is nothing to promote, only something
to break. Per-question ranks confirm it: 4 questions got worse, 2 better.

The two questions pushed out of the top 10 entirely ("profit" and
"import duties") are precisely the ones with the widest vocabulary gap
from the filings' wording. ms-marco-MiniLM is trained on web passage
ranking, where question and passage share vocabulary heavily; on
financial terminology it falls into the same trap as keyword search,
rewarding surface overlap.

Lesson for choosing what to optimize: estimate headroom on the subset
you actually care about, not on the aggregate.

Cross-encoder reranking of first-stage retrieval results.

WHY HERE: with vector+rule_filter, hit@1 is 0.110 while hit@10 is 0.402.
In roughly three cases out of ten the gold chunk is already among the
candidates but badly positioned — the one failure mode a reranker
addresses. Unlike hybrid fusion or query rewriting, this margin is
measured rather than assumed.

BI-ENCODER VS CROSS-ENCODER: the retrieval model (all-MiniLM-L6-v2)
encodes question and chunk independently, which is what makes searching
4,631 chunks instant — but it must compress a chunk's meaning into 384
numbers without knowing the question. A cross-encoder takes the pair as
a single input, so attention can relate every question token to every
chunk token ("long-term borrowing" can match "Long-term debt $40,152").
Far more accurate, and far more expensive: nothing can be precomputed,
so it only works on a shortlist.

Standard two-stage design: bi-encoder for breadth, cross-encoder for
depth.

COST: runs locally on Apple Silicon, no API calls, no quota. Scoring 30
pairs takes well under a second on an M2.
"""
from sentence_transformers import CrossEncoder

# Trained on MS MARCO passage ranking. Small (~80MB) and CPU-friendly;
# stronger cross-encoders exist but would strain an 8GB machine.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many first-stage candidates to rerank. Deep enough that golds
# sitting outside the top 10 can still be rescued, shallow enough to
# stay fast. Reranking cannot recover what the first stage never
# retrieved, so this depth is the ceiling on what reranking can fix.
DEFAULT_DEPTH = 15  # was 30 before

_model = None


def get_model() -> CrossEncoder:
    """Lazy singleton — loading the model costs seconds, do it once."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], num_results: int = 10) -> list[dict]:
    """Reorder chunks by cross-encoder relevance to the query."""
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    scores = get_model().predict(pairs)

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order[:num_results]]


class RerankedIndex:
    """Wraps any retriever with the shared (query, filters, k) interface,
    adding a cross-encoder reranking stage on top."""

    def __init__(self, base_index, depth: int = DEFAULT_DEPTH):
        self.base_index = base_index
        self.depth = depth

    def search(self, query: str, filters: dict | None = None,
               num_results: int = 10) -> list[dict]:
        candidates = self.base_index.search(query, filters, self.depth)
        return rerank(query, candidates, num_results)