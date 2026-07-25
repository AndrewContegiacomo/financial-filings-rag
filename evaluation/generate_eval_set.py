"""
Generates a synthetic evaluation set: (question -> gold chunk) pairs.

Method: sample chunks from the corpus, then ask an LLM to write the
question that chunk answers. The source chunk is the gold answer by
construction, so no manual annotation is needed.

Known bias — LEXICAL LEAKAGE: a question written while looking at the
chunk tends to reuse its wording, which unfairly favors keyword search
over dense retrieval. The generation prompt actively fights this
(paraphrase, no distinctive terminology), and hand-written questions
from field testing are kept as a separate control subset.
"""
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHUNKS_FILE = Path("data/processed/chunks.json")
OUT_FILE = Path("data/eval/eval_set.json")

MODEL = "llama-3.3-70b-versatile"

# Chunks sampled per (ticker, form) stratum. 4 companies x 2 form types
# x 5 = 40 questions, in the 30-50 range the rubric asks for.
PER_STRATUM = 5

# Sleep between API calls: the Groq free tier is rate limited, and
# hitting the limit mid-run would leave a partial eval set.
SLEEP = 1.5

SEED = 42  # sampling must be reproducible: same corpus -> same eval set

COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft",
    "JPM": "JPMorgan Chase", "PFE": "Pfizer",
}

GENERATION_PROMPT = """You are helping build an evaluation set for a
question-answering system over SEC filings.

Below is an excerpt from {company}'s {form} filed on {date}.

Write ONE question that this excerpt answers.

Requirements:
- The question must target a SPECIFIC fact, figure, or detail found in
  this excerpt — something that no other part of the filing could
  answer identically. Avoid broad thematic questions ("what risks does
  X face?") that dozens of passages would answer equally well.
- Write it the way an external analyst would ask it, WITHOUT having
  read the excerpt: do NOT reuse the distinctive terminology, phrasing
  or sentence structure of the text. Use natural, common business
  vocabulary instead.
- Make it self-contained: name the company and the period.
- One single question, no preamble, no explanation, no quotes.

EXCERPT:
{text}

QUESTION:"""


MONEY_RE = re.compile(r"\d{1,3},\d{3}")

def classify_source(chunk: dict) -> str | None:
    """Classify a chunk as a question source, or reject it.

    Two deliberate categories, because they stress retrieval very
    differently:
      - 'narrative': flowing prose (risk factors, business description)
      - 'financial': figure-bearing text, typically linearized tables

    The first version of this filter required 60% alphabetic tokens,
    which silently excluded ALL financial tables — i.e. exactly the
    hard cases the Q1 case study exposed. Sampling both strata keeps
    the eval set representative of real usage.
    """
    text = chunk["text"]
    if len(text) < 600:
        return None

    tokens = text.split()
    alpha_ratio = sum(
        1 for t in tokens if re.fullmatch(r"[A-Za-z]{3,}", t)
    ) / len(tokens)

    has_figures = len(MONEY_RE.findall(text)) >= 3

    if has_figures and alpha_ratio > 0.25:
        return "financial"   # numbers, but still some readable labels
    if alpha_ratio > 0.6:
        return "narrative"
    return None              # headers, TOC entries, pure number dumps


# Raised from {3, 2}: at n=17 per stratum a single question was worth
# ~6 points of hit-rate. 8 strata x 9 = 72 synthetic questions.
PER_STRATUM = {"narrative": 5, "financial": 4}


def sample_chunks(chunks: list[dict]) -> list[dict]:
    strata = defaultdict(list)
    for c in chunks:
        kind = classify_source(c)
        if kind:
            strata[(c["ticker"], c["form"], kind)].append(c)

    rng = random.Random(SEED)
    sampled = []
    for (ticker, form, kind), group in sorted(strata.items()):
        n = min(PER_STRATUM[kind], len(group))
        picked = rng.sample(group, n)
        for c in picked:
            c["_kind"] = kind          # carried into the eval set
        sampled.extend(picked)
        print(f"  {ticker} {form} {kind}: {len(group)} eligible -> {n}")
    return sampled

def generate_question(chunk: dict) -> str:
    prompt = GENERATION_PROMPT.format(
        company=COMPANY_NAMES.get(chunk["ticker"], chunk["ticker"]),
        form=chunk["form"],
        date=chunk["date"],
        text=chunk["text"],
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # a little variety, but still mostly stable
    )
    return resp.choices[0].message.content.strip().strip('"')


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    print("Sampling source chunks:")
    sampled = sample_chunks(chunks)
    print(f"\nGenerating {len(sampled)} questions...\n")

    eval_set = []
    for chunk in tqdm(sampled, desc="Generating"):
        try:
            question = generate_question(chunk)
        except Exception as exc:
            # Never let one failed call kill the whole run: skip and
            # keep going, the loss of one item is acceptable.
            print(f"\n  ! skipped {chunk['id']}: {exc}")
            continue

        eval_set.append(
            {
                "question": question,
                "gold_chunk_id": chunk["id"],
                "ticker": chunk["ticker"],
                "form": chunk["form"],
                "section": chunk["section"],
                "origin": "synthetic",
            }
        )
        time.sleep(SLEEP)

    OUT_FILE.write_text(json.dumps(eval_set, indent=2), encoding="utf-8")
    print(f"\n{len(eval_set)} question/gold pairs -> {OUT_FILE}")


if __name__ == "__main__":
    main()