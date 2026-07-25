"""
Helper for hand-labeling gold chunks — deliberately NOT a retriever.

Uses plain substring matching over the corpus, so gold selection is
independent of the systems being evaluated. Using keyword or vector
search to pick golds would introduce selection bias: the eval set would
only ever contain questions those retrievers can already answer.

Usage:
    python -m evaluation.find_gold "research and development" --ticker AAPL --form 10K
"""
import argparse
import json
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pattern", help="literal text to search for (case-insensitive)")
    p.add_argument("--ticker")
    p.add_argument("--form")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--width", type=int, default=300, help="chars of preview")
    args = p.parse_args()

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    needle = args.pattern.lower()

    hits = [
        c for c in chunks
        if needle in c["text"].lower()
        and (not args.ticker or c["ticker"] == args.ticker)
        and (not args.form or c["form"] == args.form)
    ]

    print(f"{len(hits)} matching chunks\n")
    for c in hits[: args.limit]:
        pos = c["text"].lower().find(needle)
        start = max(0, pos - args.width // 3)
        print(f"--- {c['id']}  [{c['section']}]")
        print(f"    ...{c['text'][start:start + args.width]}...\n")


if __name__ == "__main__":
    main()