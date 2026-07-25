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


def rank_of_gold(retrieved: list[dict], gold_ids: list[str]) -> int | None:
    """1-based position of the FIRST gold chunk found, or None.

    With multiple valid golds, the best-ranked one is what matters: the
    question is whether a correct answer reached the LLM's context, not
    which particular copy of it did.
    """
    gold_set = set(gold_ids)
    for i, chunk in enumerate(retrieved, start=1):
        if chunk["id"] in gold_set:
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

    
    # Two gold conventions, computed side by side:
    #   strict   = primary gold only (symmetric across origins)
    #   expanded = all annotated golds (curated for manual items only)
    ranks = {"strict": defaultdict(list), "expanded": defaultdict(list)}

    for item in tqdm(eval_set, desc="Evaluating"):
        for name, retrieve in configs.items():
            retrieved = retrieve(item["question"], item)
            ranks["strict"][name].append(
                rank_of_gold(retrieved, item["gold_chunk_ids"][:1])
            )
            ranks["expanded"][name].append(
                rank_of_gold(retrieved, item["gold_chunk_ids"])
            )

    # --- Aggregate both gold conventions ---
    results = {}
    for variant, per_config in ranks.items():
        block = {"overall": {name: score(r) for name, r in per_config.items()}}

        for dimension in ["kind", "origin", "ticker", "form"]:
            groups = defaultdict(list)
            for i, item in enumerate(eval_set):
                groups[item[dimension]].append(i)

            block[dimension] = {
                value: {
                    name: score([per_config[name][i] for i in idxs])
                    for name in configs
                }
                for value, idxs in groups.items()
            }
        results[variant] = block

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- Console report ---
    # 'strict' is the headline convention: primary gold only, applied
    # uniformly, so synthetic and manual items are directly comparable.
    print("\n=== OVERALL (strict — primary gold only) ===")
    print(f"{'config':<24}" + "".join(
        f"{'hit@'+str(k):>9}{'mrr@'+str(k):>9}" for k in K_VALUES))
    for name, s in results["strict"]["overall"].items():
        print(f"{name:<24}" + "".join(
            f"{s[f'hit_rate@{k}']:>9.3f}{s[f'mrr@{k}']:>9.3f}" for k in K_VALUES))

    print("\n=== BY KIND (strict, hit_rate@5) ===")
    for value, per_config in results["strict"]["kind"].items():
        n = per_config["keyword"]["n"]
        print(f"{str(value):<12} (n={n:>2})  " + "  ".join(
            f"{name}={s['hit_rate@5']:.3f}" for name, s in per_config.items()))

    # The reason the manual control subset exists: if synthetic questions
    # inherit vocabulary from the chunk they were written from, keyword
    # search should look relatively better on them than on hand-written
    # ones. This is where that shows up — or doesn't.
    print("\n=== BY ORIGIN (strict, hit_rate@5) — lexical leakage check ===")
    for value, per_config in results["strict"]["origin"].items():
        n = per_config["keyword"]["n"]
        print(f"{value:<12} (n={n:>2})  " + "  ".join(
            f"{name}={s['hit_rate@5']:.3f}" for name, s in per_config.items()))

    # Multiple golds are curated for manual items only, so this gap is
    # readable only within that subset. It quantifies how much the
    # single-gold convention understates real retrieval performance.
    print("\n=== MANUAL SUBSET: strict vs expanded golds (hit_rate@5) ===")
    for name in configs:
        s = results["strict"]["origin"]["manual"][name]["hit_rate@5"]
        e = results["expanded"]["origin"]["manual"][name]["hit_rate@5"]
        print(f"{name:<24} strict={s:.3f}  expanded={e:.3f}  (+{e - s:.3f})")

    print(f"\nFull results -> {OUT_FILE}")
    
if __name__ == "__main__":
    main()