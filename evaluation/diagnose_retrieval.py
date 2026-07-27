"""
Diagnostic: WHERE does the gold chunk rank across the full corpus?

Hit-rate answers "is it in the top k". This answers "how far off are we",
which is what determines the cure:
  - rank 6-20   -> near miss: raise k, add reranking
  - rank 100+   -> buried: metadata filtering, hybrid, better embeddings
  - not ranked  -> the chunk itself may be unusable: revisit chunking

No API calls: retrieval is local and deterministic, so this can be run
freely and repeatedly.
"""
import json
from pathlib import Path
from statistics import median

from rag.search import load_chunks, build_index, keyword_search
from rag.vector_search import VectorIndex

EVAL_FILE = Path("data/eval/eval_set.json")
DEEP_K = 200  # how far down to look before giving up


def best_gold_rank(retrieved, gold_ids):
    gold = set(gold_ids)
    for i, c in enumerate(retrieved, 1):
        if c["id"] in gold:
            return i
    return None


def summarize(name, ranks):
    found = [r for r in ranks if r is not None]
    buckets = {
        "top5": sum(1 for r in found if r <= 5),
        "6-20": sum(1 for r in found if 5 < r <= 20),
        "21-100": sum(1 for r in found if 20 < r <= 100),
        f"101-{DEEP_K}": sum(1 for r in found if r > 100),
        f">{DEEP_K}": len(ranks) - len(found),
    }
    med = median(found) if found else float("nan")
    print(f"{name:<26} median_rank={med:>6.0f}  " +
          "  ".join(f"{k}={v}" for k, v in buckets.items()))


def main():
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    chunks = load_chunks()
    kw = build_index(chunks)
    vec = VectorIndex()

    configs = {
        "keyword": lambda q, it: keyword_search(kw, q, None, DEEP_K),
        "vector": lambda q, it: vec.search(q, None, DEEP_K),
        "keyword+filter": lambda q, it: keyword_search(
            kw, q, {"ticker": it["ticker"], "form": it["form"]}, DEEP_K),
        "vector+filter": lambda q, it: vec.search(
            q, {"ticker": it["ticker"], "form": it["form"]}, DEEP_K),
    }

    for subset in ["manual", "synthetic"]:
        items = [i for i in eval_set if i["origin"] == subset]
        print(f"\n=== {subset.upper()} (n={len(items)}) — rank of best gold ===")
        for name, retrieve in configs.items():
            ranks = [best_gold_rank(retrieve(i["question"], i),
                                    i["gold_chunk_ids"]) for i in items]
            summarize(name, ranks)

    # Per-question detail on the realistic subset: which questions are
    # hopeless vs nearly there.
    print("\n=== MANUAL, per question (rank, keyword / vector, unfiltered) ===")
    for i in [x for x in eval_set if x["origin"] == "manual"]:
        rk = best_gold_rank(keyword_search(kw, i["question"], None, DEEP_K),
                            i["gold_chunk_ids"])
        rv = best_gold_rank(vec.search(i["question"], None, DEEP_K),
                            i["gold_chunk_ids"])
        print(f"  kw={str(rk):>5}  vec={str(rv):>5}   {i['question'][:70]}")


if __name__ == "__main__":
    main()
    