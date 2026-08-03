"""
Generation evaluation: compares prompt strategies on identical context.

DESIGN: retrieval is held fixed — augmented search with rule-inferred
filters at k=10, i.e. the production configuration — so the prompt is
the only variable. The previous round ran on keyword@k=5, the weakest
configuration measured, which is why the gold chunk reached the context
in only 4 of 18 cases and made the comparison nearly meaningless.

SCORING is conditioned on what the context actually contained: with a
gold chunk retrieved, the answer should be correct, grounded and cited;
without one, an explicit refusal is the correct behaviour. Scoring only
"was the answer right" would re-measure retrieval, since a prompt cannot
extract a figure it was never given.

JUDGE: same model as generation. Judging was moved to a smaller model in
the previous round to save quota, and it failed — correct answers
($35,934 for Apple's cash, 15% for the opex ratio) were scored
incorrect, and near-identical refusals received opposite labels. Sample
size is reduced instead. Self-preference bias remains and is documented
rather than solved: both prompts face the same judge, so the comparison
holds even if absolute values are optimistic.

RATE LIMITS: the Groq free tier caps tokens per DAY, so sleeping between
calls does not help against it. Results are checkpointed after every
scored pair and a completed (question, prompt) pair is never re-run — an
interrupted job resumes rather than starting over.
"""
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError, InternalServerError
from tqdm import tqdm

from rag.augmented_search import AugmentedIndex
from rag.query_analysis import infer_filters
from rag.rag import format_context
from rag.search import load_chunks, build_index
from rag.vector_search import VectorIndex

load_dotenv()

# Evaluation keeps its own client on purpose: rag/llm_client.py serves
# the production path and shouldn't be bent to fit measurement needs.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EVAL_FILE = Path("data/eval/eval_set.json")
OUT_FILE = Path("data/eval/llm_results.json")

MODEL = "llama-3.3-70b-versatile"
TOP_K = 10
SLEEP = 2.0

# All 10 hand-written items (the realistic ones) plus a few synthetic.
# Three prompts x 2 calls each puts ~14 items near the daily token
# budget; the constraint is quota, not methodology.
N_SYNTHETIC = 4


PROMPT_A = """You are a financial analyst assistant. Answer the QUESTION
using only the information in the CONTEXT below, extracted from SEC filings.

Rules:
- Base your answer strictly on the CONTEXT. Do not use outside knowledge.
- After each claim, cite the source in brackets: [TICKER FORM DATE, SECTION].
- If the CONTEXT does not contain the answer, say so explicitly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


PROMPT_B = """You are a financial analyst assistant answering questions
about SEC filings.

Work through these steps before answering:
1. RELEVANCE: identify which of the context blocks, if any, actually
   address the question. Most blocks are usually irrelevant.
2. EXTRACT: from the relevant blocks only, pull out the specific facts,
   figures or statements that answer the question.
3. ANSWER: state the answer, citing each claim as
   [TICKER FORM DATE, SECTION].

If step 1 finds no relevant block, do not attempt an answer: reply that
the filings provided do not contain this information. Never infer a
figure that is not explicitly present.

Output only the final answer, not your intermediate steps.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# Adds an explicit ban on DERIVING figures. The previous round showed
# both A and B fabricating whenever they computed: A multiplied share
# counts by average prices across unrelated rows, B subtracted values
# with mismatched scopes and reported $92.8B of buybacks in two months.
# Neither prompt forbade calculation — only invention.
PROMPT_C = """You are a financial analyst assistant answering questions
about SEC filings.

Work through these steps before answering:
1. RELEVANCE: identify which of the context blocks, if any, actually
   address the question. Most blocks are usually irrelevant.
2. EXTRACT: from the relevant blocks only, pull out the specific facts,
   figures or statements that answer the question.
3. ANSWER: state the answer, citing each claim as
   [TICKER FORM DATE, SECTION].

Rules on figures:
- Report only numbers that appear LITERALLY in the context. Do not add,
  subtract, multiply or compute percentages, even when the arithmetic
  looks trivial.
- If answering would require calculation, say which figures the context
  provides and state that the comparison is not stated in the filings.
- If step 1 finds no relevant block, reply that the filings provided do
  not contain this information.

Output only the final answer, not your intermediate steps.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PROMPTS = {
    "A_strict_contract": PROMPT_A,
    "B_explicit_triage": PROMPT_B,
    "C_no_derived_figures": PROMPT_C,
}


JUDGE_PROMPT = """You are evaluating an AI assistant that answers
questions about SEC filings using retrieved document excerpts.

QUESTION: {question}

REFERENCE PASSAGE (the excerpt that contains the answer):
{gold_text}

WAS THE REFERENCE PASSAGE AVAILABLE TO THE ASSISTANT? {gold_available}

ASSISTANT'S ANSWER:
{answer}

Reply with ONLY a JSON object, no other text:

{{"refused": 0 or 1,
  "grounded": 0 or 1,
  "correct": 0 or 1,
  "computed": 0 or 1}}

- "refused": 1 if the assistant declined to answer, stating the
  information was not available. 0 if it gave a substantive answer.
- "grounded": 1 if every factual claim is attributable to filing content
  and carries a source citation. Score 1 for a clean refusal.
- "correct": 1 if the answer conveys the fact stated in the REFERENCE
  PASSAGE. A correct answer phrased differently still counts. If the
  assistant refused, score 0.
- "computed": 1 if the assistant produced a figure by calculating it
  (a difference, a percentage, a total) rather than quoting a number
  present in the text. 0 otherwise.
"""

_SCORE_FIELDS = ("refused", "grounded", "correct", "computed")


def ask(prompt: str, retries: int = 3) -> str | None:
    """Call the API, backing off on transient failures.

    A 429 on the per-minute bucket clears within a minute; a 429 on the
    daily budget exhausts the retries and returns None, which is correct
    — the run should stop and resume tomorrow, with completed work
    already on disk.
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, InternalServerError):
            if attempt == retries - 1:
                return None
            print("\n  rate limited — waiting 60s")
            time.sleep(60)
    return None


def judge(question: str, gold_text: str, answer: str,
          gold_available: bool) -> dict:
    """Score one answer. Gold text is NOT truncated: the previous round
    cut it at 800 characters, which could hide the very figure the judge
    was meant to verify."""
    raw = ask(JUDGE_PROMPT.format(
        question=question,
        gold_text=gold_text,
        gold_available="YES" if gold_available else "NO",
        answer=answer,
    ))
    if raw is None:
        return {k: None for k in _SCORE_FIELDS}

    # Models sometimes wrap JSON in prose or code fences despite
    # instructions — extract the object rather than failing the run.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {k: None for k in _SCORE_FIELDS}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {k: None for k in _SCORE_FIELDS}
    return {k: parsed.get(k) for k in _SCORE_FIELDS}


def load_checkpoint() -> tuple[list[dict], set]:
    """Reload previous results so an interrupted run can resume.

    Keyed by (question, prompt): every pair already scored is skipped.
    """
    if not OUT_FILE.exists():
        return [], set()
    records = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    done = {(r["question"], r["prompt"]) for r in records}
    return records, done


def report(records: list[dict]) -> None:
    def rate(rows: list[dict], field: str) -> float:
        vals = [r[field] for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n=== GENERATION EVALUATION ===")
    for name in PROMPTS:
        rows = [r for r in records if r["prompt"] == name]
        if not rows:
            continue
        avail = [r for r in rows if r["gold_available"]]
        missing = [r for r in rows if not r["gold_available"]]

        print(f"\n{name}  (n={len(rows)})")
        print(f"  gold IN context     (n={len(avail):>2}): "
              f"correct={rate(avail, 'correct'):.2f}  "
              f"grounded={rate(avail, 'grounded'):.2f}  "
              f"refused={rate(avail, 'refused'):.2f}")
        print(f"  gold NOT in context (n={len(missing):>2}): "
              f"refused={rate(missing, 'refused'):.2f}  "
              f"grounded={rate(missing, 'grounded'):.2f}")
        print(f"  computed a figure  : {rate(rows, 'computed'):.2f}")

    # The manual subset is the realistic one: questions phrased without
    # borrowing the filings' vocabulary.
    manual = [r for r in records if r["origin"] == "manual"]
    if manual:
        print("\n--- hand-written questions only ---")
        for name in PROMPTS:
            rows = [r for r in manual if r["prompt"] == name]
            if not rows:
                continue
            avail = [r for r in rows if r["gold_available"]]
            missing = [r for r in rows if not r["gold_available"]]
            print(f"{name:<24} "
                  f"gold in ctx (n={len(avail)}): "
                  f"correct={rate(avail, 'correct'):.2f}   "
                  f"gold absent (n={len(missing)}): "
                  f"refused={rate(missing, 'refused'):.2f}   "
                  f"computed={rate(rows, 'computed'):.2f}")


def main() -> None:
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}

    print("Building index...")
    index = AugmentedIndex(VectorIndex(), build_index(chunks))

    manual = [i for i in eval_set if i["origin"] == "manual"]
    synthetic = [i for i in eval_set if i["origin"] == "synthetic"][:N_SYNTHETIC]
    subset = manual + synthetic

    records, done = load_checkpoint()
    if done:
        print(f"Resuming: {len(done)} (question, prompt) pairs already scored")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    for item in tqdm(subset, desc="Generating & judging"):
        # Retrieval is deterministic, so it is recomputed on resume
        # rather than checkpointed — only expensive work is persisted.
        filters = infer_filters(item["question"])
        retrieved = index.search(item["question"], filters or None, TOP_K)
        context = format_context(retrieved)

        gold_ids = set(item["gold_chunk_ids"])
        gold_available = any(c["id"] in gold_ids for c in retrieved)
        gold_text = by_id[item["gold_chunk_ids"][0]]["text"]

        for name, template in PROMPTS.items():
            if (item["question"], name) in done:
                continue

            answer = ask(template.format(context=context,
                                         question=item["question"]))
            if answer is None:
                print("\n  token budget exhausted — stopping; "
                      "re-run to resume")
                report(records)
                return
            time.sleep(SLEEP)

            scores = judge(item["question"], gold_text, answer,
                           gold_available)
            time.sleep(SLEEP)

            records.append({
                "question": item["question"],
                "origin": item["origin"],
                "prompt": name,
                "gold_available": gold_available,
                "n_filters": len(filters),
                "answer": answer,
                **scores,
            })
            done.add((item["question"], name))

            # Write after every scored pair: an API job that loses hours
            # of work to one 429 is a design flaw, not bad luck.
            OUT_FILE.write_text(json.dumps(records, indent=2),
                                encoding="utf-8")

    report(records)
    print(f"\nFull results -> {OUT_FILE}")


if __name__ == "__main__":
    main()