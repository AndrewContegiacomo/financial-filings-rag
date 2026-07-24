"""
Post-hoc patch of the eval set:
  1. Adds the 'kind' field (narrative / financial), recomputed from the
     gold chunk — no need to regenerate questions to get it.
  2. Merges hand-written questions from field testing, tagged
     origin='manual'. These are the control subset: written by a human
     who did NOT read the chunk, so they carry no lexical leakage and
     act as a sanity check against the synthetic ones.
"""
import json
from pathlib import Path

from evaluation.generate_eval_set import classify_source

CHUNKS_FILE = Path("data/processed/chunks.json")
EVAL_FILE = Path("data/eval/eval_set.json")

# Questions written by hand during Phase 2 field testing, with golds
# identified manually (see notes.md).
MANUAL = [
    {
        "question": "What were Apple's total net sales in the last fiscal year?",
        "gold_chunk_id": "AAPL_10K_2025-10-31_0104",
    },
]


def main() -> None:
    chunks = {c["id"]: c for c in json.loads(CHUNKS_FILE.read_text())}
    eval_set = json.loads(EVAL_FILE.read_text())

    for item in eval_set:
        gold = chunks[item["gold_chunk_id"]]
        item["kind"] = classify_source(gold)

    for m in MANUAL:
        gold = chunks[m["gold_chunk_id"]]
        eval_set.append(
            {
                "question": m["question"],
                "gold_chunk_id": m["gold_chunk_id"],
                "ticker": gold["ticker"],
                "form": gold["form"],
                "section": gold["section"],
                "kind": classify_source(gold),
                "origin": "manual",
            }
        )

    EVAL_FILE.write_text(json.dumps(eval_set, indent=2), encoding="utf-8")

    from collections import Counter
    print(f"{len(eval_set)} items")
    print("by kind:  ", Counter(i["kind"] for i in eval_set))
    print("by origin:", Counter(i["origin"] for i in eval_set))
    print("by ticker:", Counter(i["ticker"] for i in eval_set))


if __name__ == "__main__":
    main()