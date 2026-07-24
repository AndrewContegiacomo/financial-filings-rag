"""
Retrieval evaluation: hit-rate@k and MRR@k for each configuration.

No LLM calls here — retrieval evaluation is pure information retrieval,
so it runs in seconds and can be repeated freely. This is why retrieval
is evaluated separately from generation: fast, deterministic, cheap.

The 'oracle filter' configurations use the gold chunk's own ticker/form
as a hard filter. A real system does NOT have this information — these
runs are an UPPER BOUND, included to quantify how much metadata
inference from the question would be worth (Phase 4, agentic step).
They must never be reported as system performance.
"""
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from rag.search import load_chunks, build_index, keyword_search
from rag.vector_search import VectorIndex

EVAL_FILE = Path("data/eval/eval_set.json")
OUT_FILE = Path("data/eval/retrieval_results.json")

K_VALUES = [1, 5, 10]
MAX_K = max(K_VALUES)


def rank_of_gold(retrieved: list[dict], gold_id: str) -> int | None:
    """1-based position of the gold chunk, or None if not retrieved."""
    for i, chunk in enumerate(retrieved, start=1):
        if chunk["id"] == gold_id:
            return i
    return None


def score(ranks: list[int | None]) -> dict:
    """Compute hit-rate and MRR at every k from a list of gold ranks."""
    n = len(ranks)
    out = {}
    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        mrr = sum(1 / r for r in ranks if r is not None and r <= k)
        out[f"hit_rate@{k}"] = round(hits / n, 3)
        out[f"mrr@{k}"] = round(mrr / n, 3)
    out["n"] = n
    return out


def main() -> None:
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    chunks = load_chunks()

    print("Building indexes...")
    kw_index = build_index(chunks)
    vec_index = VectorIndex()

    # Each configuration is a function: question + item -> ranked chunks.
    # 'item' is passed so oracle configs can read the gold's metadata.
    configs = {
        "keyword": lambda q, it: keyword_search(kw_index, q, None, MAX_K),
        "vector": lambda q, it: vec_index.search(q, None, MAX_K),
        "keyword+oracle_filter": lambda q, it: keyword_search(
            kw_index, q, {"ticker": it["ticker"], "form": it["form"]}, MAX_K
        ),
        "vector+oracle_filter": lambda q, it: vec_index.search(
            q, {"ticker": it["ticker"], "form": it["form"]}, MAX_K
        ),
    }

    # ranks[config] = list of gold ranks, aligned with eval_set order
    ranks = defaultdict(list)
    for item in tqdm(eval_set, desc="Evaluating"):
        for name, retrieve in configs.items():
            retrieved = retrieve(item["question"], item)
            ranks[name].append(rank_of_gold(retrieved, item["gold_chunk_id"]))

    # --- Overall results ---
    results = {"overall": {name: score(r) for name, r in ranks.items()}}

    # --- Breakdowns: the interesting part ---
    # Slicing by 'kind' tests the hypothesis from the Q1 case study:
    # keyword should hold up on narrative prose and collapse on
    # figure-heavy tables, where embeddings should do relatively better.
    for dimension in ["kind", "origin", "ticker", "form"]:
        groups = defaultdict(list)
        for i, item in enumerate(eval_set):
            groups[item[dimension]].append(i)

        results[dimension] = {
            value: {
                name: score([ranks[name][i] for i in idxs])
                for name in configs
            }
            for value, idxs in groups.items()
        }

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Console report ---
    print("\n=== OVERALL ===")
    header = f"{'config':<24}" + "".join(
        f"{'hit@'+str(k):>9}{'mrr@'+str(k):>9}" for k in K_VALUES
    )
    print(header)
    for name, s in results["overall"].items():
        row = f"{name:<24}" + "".join(
            f"{s[f'hit_rate@{k}']:>9.3f}{s[f'mrr@{k}']:>9.3f}" for k in K_VALUES
        )
        print(row)

    print("\n=== BY KIND (hit_rate@5) ===")
    for value, per_config in results["kind"].items():
        n = per_config["keyword"]["n"]
        line = f"{value:<12} (n={n:>2})  " + "  ".join(
            f"{name}={s['hit_rate@5']:.3f}" for name, s in per_config.items()
        )
        print(line)

    print(f"\nFull results -> {OUT_FILE}")


if __name__ == "__main__":
    main()