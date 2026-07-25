"""
Post-processing of the eval set:
  1. Migrates the schema from a single 'gold_chunk_id' to a list
     'gold_chunk_ids' (first element = primary gold).
  2. Adds the 'kind' field, recomputed from the primary gold chunk.
  3. Merges hand-written control questions (origin='manual').

WHY A LIST OF GOLDS: in financial filings the same figure legitimately
appears in several places — Microsoft's net income shows up in the MD&A
summary, the income statement, the cash flow statement, the equity
statement and the EPS note. Plus, the 40-word chunk overlap means an
answer often straddles two consecutive chunks. Counting only one of
them as correct systematically understates retrieval performance.

ASYMMETRY WARNING: multiple golds are curated only for the manual
subset (hand-annotating 70+ synthetic items is not worth the effort).
Evaluation therefore reports BOTH a 'strict' variant (primary gold
only — symmetric, used for synthetic vs manual comparisons) and an
'expanded' variant (all golds).
"""
import json
from collections import Counter
from pathlib import Path

from evaluation.generate_eval_set import classify_source

CHUNKS_FILE = Path("data/processed/chunks.json")
EVAL_FILE = Path("data/eval/eval_set.json")

# Hand-written control questions. Written BEFORE looking at the corpus,
# using everyday business vocabulary rather than filing terminology;
# golds were then located with evaluation/find_gold.py (plain substring
# search — deliberately independent of the retrievers under test, to
# avoid selecting only questions the retrievers already answer well).
MANUAL = [
    {
        "question": "What were Apple's total net sales in the last fiscal year?",
        "gold_chunk_ids": ["AAPL_10K_2025-10-31_0104", "AAPL_10K_2025-10-31_0105"],
    },
    {
        "question": "How much did Apple spend on R&D last year?",
        "gold_chunk_ids": ["AAPL_10K_2025-10-31_0107", "AAPL_10K_2025-10-31_0108",
                           "AAPL_10K_2025-10-31_0121"],
    },
    {
        "question": "How much profit did Microsoft make in fiscal 2025?",
        "gold_chunk_ids": ["MSFT_10K_2025-07-30_0148", "MSFT_10K_2025-07-30_0184",
                           "MSFT_10K_2025-07-30_0186", "MSFT_10K_2025-07-30_0189",
                           "MSFT_10K_2025-07-30_0219"],
    },
    {
        "question": "How much cash and cash equivalents did Apple hold at the end of fiscal 2025?",
        "gold_chunk_ids": ["AAPL_10K_2025-10-31_0123", "AAPL_10K_2025-10-31_0126",
                           "AAPL_10K_2025-10-31_0128"],
    },
    {
        "question": "How much did Microsoft's Azure business grow in fiscal 2025?",
        "gold_chunk_ids": ["MSFT_10K_2025-07-30_0152", "MSFT_10K_2025-07-30_0138",
                           "MSFT_10K_2025-07-30_0139"],
    },
    {
        "question": "What is Apple's best-selling product line?",
        "gold_chunk_ids": ["AAPL_10K_2025-10-31_0105"],
    },
    {
        "question": "How much did JPMorgan pay in common stock dividends during 2025?",
        "gold_chunk_ids": ["JPM_10K_2026-02-13_0397"],
    },
    {
        "question": "What lawsuits is Pfizer facing over its former heartburn medication?",
        "gold_chunk_ids": ["PFE_10K_2026-02-26_0559", "PFE_10K_2026-02-26_0560",
                           "PFE_10K_2026-02-26_0561", "PFE_10K_2026-02-26_0562"],
    },
    {
        "question": "How could new U.S. import duties affect Apple's business?",
        "gold_chunk_ids": ["AAPL_10K_2025-10-31_0033", "AAPL_10K_2025-10-31_0034",
                           "AAPL_10K_2025-10-31_0031"],
    },
    {
        "question": "How much long-term borrowing did Microsoft have outstanding at the end of fiscal 2025?",
        "gold_chunk_ids": ["MSFT_10K_2025-07-30_0244", "MSFT_10K_2025-07-30_0245",
                           "MSFT_10K_2025-07-30_0242", "MSFT_10K_2025-07-30_0185",
                           "MSFT_10K_2025-07-30_0186"],
    },
]


def main() -> None:
    chunks = {c["id"]: c for c in json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))}
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    # --- Migrate + enrich synthetic items ---
    for item in eval_set:
        if "gold_chunk_id" in item:          # old schema
            item["gold_chunk_ids"] = [item.pop("gold_chunk_id")]
        primary = chunks[item["gold_chunk_ids"][0]]
        item["kind"] = classify_source(primary)

    # --- Append manual items (idempotent) ---
    existing = {i["question"] for i in eval_set}
    for m in MANUAL:
        if m["question"] in existing:
            continue
        missing = [g for g in m["gold_chunk_ids"] if g not in chunks]
        if missing:
            # Corpus and eval set out of sync — fail loudly rather than
            # silently dropping a control question.
            raise SystemExit(f"Unknown gold chunk(s): {missing}")

        primary = chunks[m["gold_chunk_ids"][0]]
        eval_set.append({
            "question": m["question"],
            "gold_chunk_ids": m["gold_chunk_ids"],
            "ticker": primary["ticker"],
            "form": primary["form"],
            "section": primary["section"],
            "kind": classify_source(primary),
            "origin": "manual",
        })

    EVAL_FILE.write_text(json.dumps(eval_set, indent=2), encoding="utf-8")

    print(f"{len(eval_set)} items")
    print("by origin:", Counter(i["origin"] for i in eval_set))
    print("by kind:  ", Counter(i["kind"] for i in eval_set))
    print("by ticker:", Counter(i["ticker"] for i in eval_set))
    multi = sum(1 for i in eval_set if len(i["gold_chunk_ids"]) > 1)
    print(f"items with >1 gold: {multi}")
    
if __name__ == "__main__":
    main()