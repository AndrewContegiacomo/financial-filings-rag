"""
Tools the agent can call. Each one does retrieval and arithmetic in
Python; the model only chooses parameters and phrases the result.

WHY: generation evaluation showed both prompt variants fabricating
figures whenever they computed something — multiplying share counts by
average prices across unrelated rows, subtracting values with different
scopes. The model is unreliable at arithmetic and reliable at deciding
what to look up, so the split follows that boundary. Numbers are
extracted verbatim and every derived figure is computed by code.
"""
import os
import re

from dotenv import load_dotenv
from groq import Groq

from rag.query_analysis import infer_ticker
from rag.vector_search import VectorIndex
from rag.rag import format_context
from rag.query_analysis import infer_form
from rag.search import load_chunks, build_index, keyword_search




load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
LOOKUP_K = 8

_index = None
_kw_index = None

def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index


EXTRACT_PROMPT = """Extract one specific figure from these SEC filing
excerpts.

COMPANY: {ticker}
METRIC: {metric}
PERIOD: {period}

EXCERPTS:
{context}

Reply with ONLY a JSON object:
{{"value": <number, no separators or currency symbols>,
  "unit": "<millions|billions|percent|units>",
  "period_in_text": "<the period label the excerpt attaches to this figure,
                      copied exactly — e.g. 'Year Ended June 30, 2025',
                      'Three Months Ended December 27, 2025'>",
  "verbatim": "<the exact fragment containing BOTH the label and the
                number, max 20 words>",
  "source": "<the [TICKER FORM DATE, SECTION] tag of the block>"}}

The figure must be for {period} exactly. If the excerpts only state it
for a different period — a quarter instead of a year, or a different
year — reply {{"value": null}}. A six-month or quarterly figure is NOT
an annual figure. Do not compute, estimate, or infer.
"""

def get_index():
    """Keyword retrieval for metric lookup — deliberately NOT the dense
    retriever used elsewhere.

    Evaluation by chunk kind: on figure-bearing chunks keyword scores
    0.487 hit@5 against 0.205 for dense. Dense retrieval was chosen as
    the system default because it handles paraphrased questions, but this
    tool never receives a paraphrase: the model supplies filing
    terminology ("net income", "research and development expenses") by
    design, which is precisely where exact lexical matching wins.

    Observed failure with dense retrieval here: "MSFT net income fiscal
    2024" returned tax-discussion chunks ("income before income taxes",
    "income taxes paid") and never surfaced the income statement.
    """
    global _kw_index
    if _kw_index is None:
        _kw_index = build_index(load_chunks())
    return _kw_index

def _period_matches(requested: str, stated: str | None) -> bool:
    """Cross-check the period the model says it read against the one asked for.

    The failure mode this guards against: income statements show three
    years side by side ("101,832  88,136  72,361"), so reading the wrong
    column produces a confident, cited, wrong number — worse than no
    answer at all.

    Deliberately permissive: it only rejects when both sides mention
    years AND they don't overlap. A missing or year-less period label is
    not treated as a mismatch, because the goal is catching clear errors,
    not enforcing a format the model was never given.
    """
    years = set(re.findall(r"20\d\d", requested))
    stated_years = set(re.findall(r"20\d\d", stated or ""))
    if not years or not stated_years:
        return True
    return bool(years & stated_years)

def lookup_metric(ticker: str, metric: str, period: str) -> dict:
    """Retrieve and extract a single stated figure.

    The 'verbatim' field is the audit trail: it forces the model to point
    at the text it read the number from, which makes a fabricated value
    visible instead of plausible.
    """
    query = f"{ticker} {metric} {period}"
    # The period parameter IS the form signal: "fiscal 2025" means the
    # annual report, "Q1 2026" the quarterly one. Without this filter the
    # extractor happily reads a six-month figure out of a 10-Q and labels
    # it as a fiscal year value — observed, not hypothetical.
    filters = {"ticker": ticker}
    form = infer_form(period)
    if form:
        filters["form"] = form

    # The period is already enforced by the form filter; repeating it in
    # the query text actively hurts. Income-statement chunks state their
    # label once inside a table whose columns are bare years
    # ("2025 2024 2023"), while tax-discussion chunks repeat "fiscal year
    # 2024" several times and win on term frequency — the same pattern
    # documented in the Q1 case study. With the period removed, the
    # income statement ranks first.
    results = keyword_search(get_index(), metric, filters, LOOKUP_K)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(
            ticker=ticker, metric=metric, period=period,
            context=format_context(results),
        )}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"value": None, "error": "unparseable extraction"}

    try:
        extracted = json.loads(match.group())
    except json.JSONDecodeError:
        return {"value": None, "error": "invalid JSON"}

    # Guard runs after parsing, before the value can reach compare_periods
    # and be turned into a percentage. Rejecting here means the caller
    # sees "not found" instead of a wrong figure.
    if extracted.get("value") is not None:
        if not _period_matches(period, extracted.get("period_in_text")):
            return {
                "value": None,
                "error": (
                    f"period mismatch: asked for {period!r}, "
                    f"extracted figure is labelled "
                    f"{extracted.get('period_in_text')!r}"
                ),
                "rejected": extracted,
            }

    return extracted


def compare_periods(ticker: str, metric: str,
                    period_a: str, period_b: str) -> dict:
    """Look up a metric in two periods and compute the change IN PYTHON.

    This is the whole point of the tool: the model never performs the
    subtraction or the percentage. Both are exact by construction.
    """
    a = lookup_metric(ticker, metric, period_a)
    b = lookup_metric(ticker, metric, period_b)

    if a.get("value") is None or b.get("value") is None:
        return {
            "error": "figure not found for one or both periods",
            period_a: a, period_b: b,
        }

    change = b["value"] - a["value"]
    pct = (change / a["value"] * 100) if a["value"] else None

    return {
        "metric": metric, "ticker": ticker,
        period_a: a, period_b: b,
        "absolute_change": round(change, 2),
        "percent_change": round(pct, 2) if pct is not None else None,
    }


# Tool schemas exposed to the model. Descriptions matter more than they
# look: they are the only instructions the model gets about when to use
# each tool and what belongs in each field.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_metric",
            "description": (
                "Look up one financial figure as stated in a company's SEC "
                "filings. Use for questions about a single value in a single "
                "period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string",
                               "enum": ["AAPL", "MSFT", "JPM", "PFE"]},
                    "metric": {"type": "string", "description":
                               "The figure, in filing terminology "
                               "(e.g. 'net sales', 'net income', "
                               "'research and development expenses')"},
                    "period": {"type": "string", "description":
                               "e.g. 'fiscal 2025', 'Q1 2026'"},
                },
                "required": ["ticker", "metric", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": (
                "Compare one financial metric across two periods. Returns "
                "both values plus the absolute and percentage change, "
                "computed exactly. Always use this instead of looking up "
                "two figures and doing the arithmetic yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string",
                               "enum": ["AAPL", "MSFT", "JPM", "PFE"]},
                    "metric": {"type": "string"},
                    "period_a": {"type": "string", "description": "earlier period"},
                    "period_b": {"type": "string", "description": "later period"},
                },
                "required": ["ticker", "metric", "period_a", "period_b"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_metric": lookup_metric,
    "compare_periods": compare_periods,
}