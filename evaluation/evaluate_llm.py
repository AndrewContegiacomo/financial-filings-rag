"""
Generation evaluation: compares two prompt strategies on the same
retrieved context.

DESIGN: retrieval is held FIXED (keyword, top-5, no oracle) so the only
variable is the prompt. Otherwise a difference could not be attributed.

WHAT "CORRECT" MEANS HERE: retrieval evaluation showed the gold chunk
often does NOT reach the context. Scoring only "was the answer right"
would just re-measure retrieval — a prompt cannot extract a figure it
was never given. So each item is scored against what the context
actually contained:
  - gold in context  -> expect a grounded, correct, cited answer
  - gold not present -> expect an explicit refusal
A prompt that always answers looks better on a naive metric and is in
fact the dangerous one: in finance a fabricated number is the worst
possible failure.

JUDGE MODEL: judging runs on a different, smaller model than generation.
Two reasons — it halves the load on the 70B model's rate-limit bucket,
and it removes part of the self-preference bias that appears when a
model grades its own output.

RATE LIMITS: the Groq free tier caps *tokens per day* (100k), not just
per minute, so sleeping between calls does not help against it. The
subset size below is chosen to fit that budget, results are checkpointed
after every call, and a completed (question, prompt) pair is never
re-run — an interrupted job resumes instead of starting over.
"""
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from tqdm import tqdm

from rag.search import load_chunks, build_index, keyword_search
from rag.rag import format_context

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EVAL_FILE = Path("data/eval/eval_set.json")
OUT_FILE = Path("data/eval/llm_results.json")

GEN_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "llama-3.1-8b-instant"

TOP_K = 5
SLEEP = 1.5

# Subset sizing is a quota decision, not a methodological one: all 10
# hand-written items are kept (they are the realistic ones), plus a
# sample of synthetic items. ~18 items x 2 prompts x 2 calls fits within
# the free tier's daily token budget.
N_SYNTHETIC = 8

# Gold passages are truncated before being sent to the judge: the first
# few hundred words are enough to verify a fact, and full chunks would
# roughly double token consumption.
GOLD_TEXT_CHARS = 800


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

PROMPTS = {"A_strict_contract": PROMPT_A, "B_explicit_triage": PROMPT_B}


JUDGE_PROMPT = """You are evaluating an AI assistant that answers
questions about SEC filings using retrieved document excerpts.

QUESTION: {question}

REFERENCE PASSAGE (the excerpt that truly contains the answer):
{gold_text}

WAS THE REFERENCE PASSAGE AVAILABLE TO THE ASSISTANT? {gold_available}

ASSISTANT'S ANSWER:
{answer}

Score the answer on three criteria. Reply with ONLY a JSON object, no
other text:

{{"refused": 0 or 1,
  "grounded": 0 or 1,
  "correct": 0 or 1}}

- "refused": 1 if the assistant declined to answer, stating the
  information was not available. 0 if it gave a substantive answer.
- "grounded": 1 if every factual claim is attributable to filing content
  and carries a source citation; 0 if any claim is unsupported or
  uncited. Score 1 for a clean refusal.
- "correct": 1 if the answer conveys the fact stated in the REFERENCE
  PASSAGE. 0 otherwise. If the assistant refused, score 0.
"""


def ask(prompt: str, model: str = GEN_MODEL, retries: int = 3) -> str:
    """Call the API, backing off on rate limits instead of crashing.

    A 429 on the per-minute bucket clears within a minute; a 429 on the
    daily budget will exhaust the retries and raise, which is correct —
    at that point the run should stop and resume tomorrow (already
    completed work is on disk).
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            print("\n  rate limited — waiting 60s")
            time.sleep(60)


def judge(question: str, gold_text: str, answer: str,
          gold_available: bool) -> dict:
    raw = ask(
        JUDGE_PROMPT.format(
            question=question,
            gold_text=gold_text[:GOLD_TEXT_CHARS],
            gold_available="YES" if gold_available else "NO",
            answer=answer,
        ),
        model=JUDGE_MODEL,
    )
    # Models sometimes wrap JSON in prose or code fences despite
    # instructions — extract the object rather than failing the run.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"refused": None, "grounded": None, "correct": None}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {"refused": None, "grounded": None, "correct": None}
    return {k: parsed.get(k) for k in ("refused", "grounded", "correct")}


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
    print("\n=== GENERATION EVALUATION ===")

    def rate(rows: list[dict], field: str) -> float:
        vals = [r[field] for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

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

    # The manual subset is the realistic one: questions phrased without
    # borrowing the filing's vocabulary.
    manual = [r for r in records if r["origin"] == "manual"]
    if manual:
        print("\n--- manual questions only ---")
        for name in PROMPTS:
            rows = [r for r in manual if r["prompt"] == name]
            if not rows:
                continue
            avail = [r for r in rows if r["gold_available"]]
            missing = [r for r in rows if not r["gold_available"]]
            print(f"{name:<20} gold in ctx (n={len(avail)}): "
                  f"correct={rate(avail, 'correct'):.2f}   "
                  f"gold absent (n={len(missing)}): "
                  f"refused={rate(missing, 'refused'):.2f}")


def main() -> None:
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}

    print("Building index...")
    index = build_index(chunks)

    manual = [i for i in eval_set if i["origin"] == "manual"]
    synthetic = [i for i in eval_set if i["origin"] == "synthetic"][:N_SYNTHETIC]
    subset = manual + synthetic

    records, done = load_checkpoint()
    if done:
        print(f"Resuming: {len(done)} (question, prompt) pairs already scored")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    for item in tqdm(subset, desc="Generating & judging"):
        # Retrieval is deterministic, so it can be recomputed cheaply on
        # resume rather than checkpointed.
        retrieved = keyword_search(index, item["question"], None, TOP_K)
        context = format_context(retrieved)

        gold_ids = set(item["gold_chunk_ids"])
        gold_available = any(c["id"] in gold_ids for c in retrieved)
        gold_text = by_id[item["gold_chunk_ids"][0]]["text"]

        for name, template in PROMPTS.items():
            if (item["question"], name) in done:
                continue

            answer = ask(template.format(context=context,
                                         question=item["question"]))
            time.sleep(SLEEP)

            scores = judge(item["question"], gold_text, answer, gold_available)
            time.sleep(SLEEP)

            records.append({
                "question": item["question"],
                "origin": item["origin"],
                "prompt": name,
                "gold_available": gold_available,
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