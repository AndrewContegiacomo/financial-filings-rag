"""
LLM-assisted query understanding: form-type inference and query
rewriting, used only where deterministic rules abstain.

RULES FIRST, LLM SECOND: rule-based inference already resolves ticker
82/82 and form 47/82 at zero cost and zero latency. The LLM is called
only for the residual — it fills gaps rather than replacing rules, which
keeps the common path deterministic and cheap.

CACHING: every result is persisted keyed by question text. Evaluation
re-runs the same 82 questions repeatedly; without a cache each run would
consume most of the free tier's daily token budget. With it, only new
questions cost anything.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from rag.query_analysis import infer_ticker, infer_form

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
CACHE_FILE = Path("data/processed/query_analysis_cache.json")

FORM_PROMPT = """A user is asking a question about SEC filings. Decide
whether the answer lives in an annual report (10-K) or a quarterly
report (10-Q).

Guidance:
- Full-year figures, risk factors, employee counts, auditor information,
  properties: 10-K.
- Figures for a three/six/nine month period, or tied to a quarter-end
  date: 10-Q.
- If the question could plausibly be answered by either, reply UNKNOWN.
  Answering UNKNOWN is correct and expected when the question does not
  specify a period.

Reply with exactly one word: 10K, 10Q, or UNKNOWN.

QUESTION: {question}

ANSWER:"""

REWRITE_PROMPT = """Rewrite this question using the terminology that SEC
filings actually use, so it can be matched against filing text.

Rules:
- Replace everyday business words with the accounting terms found in
  filings (for example: profit -> net income; spending on research ->
  research and development expenses; borrowing -> debt).
- Keep the company name and any period unchanged.
- Add no facts and make no assumptions about the answer.
- Output the rewritten question only, nothing else.

QUESTION: {question}

REWRITTEN:"""


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


_cache = _load_cache()


def _ask(prompt: str, cache_key: str) -> str:
    if cache_key in _cache:
        return _cache[cache_key]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    out = resp.choices[0].message.content.strip()
    _cache[cache_key] = out
    _save_cache(_cache)
    return out


def infer_form_llm(question: str) -> str | None:
    """Ask the model for the form type. Returns None on UNKNOWN.

    Same precision-over-recall stance as the rules: a wrong hard filter
    makes the answer unreachable, so an explicit 'no idea' is preferred
    to a guess — hence UNKNOWN is offered as a first-class answer.
    """
    raw = _ask(FORM_PROMPT.format(question=question), f"form::{question}")
    token = raw.upper().replace("-", "").strip(".")
    return token if token in {"10K", "10Q"} else None


def rewrite_query(question: str) -> str:
    """Rewrite into filing vocabulary; fall back to the original on
    anything unexpected (empty or suspiciously long output)."""
    out = _ask(REWRITE_PROMPT.format(question=question), f"rewrite::{question}")
    if not out or len(out) > 3 * len(question) + 100:
        return question
    return out


def infer_filters_llm(question: str) -> dict:
    """Rules first, LLM only for what rules leave undecided."""
    filters = {}
    ticker = infer_ticker(question)     # rules resolve this 82/82
    if ticker:
        filters["ticker"] = ticker

    form = infer_form(question) or infer_form_llm(question)
    if form:
        filters["form"] = form
    return filters