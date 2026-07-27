"""
Rule-based inference of metadata filters from a question.

Retrieval evaluation showed metadata filtering to be the single largest
lever (vector on hand-written questions: median gold rank 6 -> 1 with a
ticker+form filter). The oracle configuration used the gold's own
metadata, which a real system does not have. This module recovers that
information from the question text itself.

NO LLM CALLS: company names appear verbatim in questions and the
annual/quarterly distinction is expressed in a small, closed vocabulary.
Rules cover most of it at zero cost and zero latency. An LLM-based
extractor is a later refinement, worth adding only if measurement shows
the rules leaving value on the table.

DESIGN PRINCIPLE — PRECISION OVER RECALL: these filters are HARD
constraints. A wrong ticker filter makes the answer unreachable
(guaranteed miss), while no filter merely leaves it lower-ranked. So
every ambiguous case returns no filter rather than a guess.
"""
import re

# Aliases people actually use, mapped to corpus tickers. Order matters
# only in that longer, more specific strings should not be shadowed by
# shorter ones — matching is done on word boundaries to avoid that.
COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft", "azure", "windows", "xbox", "linkedin"],
    "JPM": ["jpmorgan", "jp morgan", "j.p. morgan", "chase", "jpm"],
    "PFE": ["pfizer", "pfe", "comirnaty", "paxlovid", "zantac"],
}

# Terms that place a question in an annual report vs a quarterly one.
# Note the asymmetry: a 10-K covers the full year AND contains prior-year
# comparatives, so annual signals are reliable; quarterly signals are
# reliable only when explicit.
ANNUAL_SIGNALS = [
    r"\bfiscal year\b", r"\bfull year\b", r"\bannual\b", r"\bannually\b",
    r"\bfy\s?20\d\d\b", r"\bfor the year\b", r"\byear ended\b",
    r"\bper year\b", r"\blast year\b",
    # "fiscal 2025" without the word "year" — found in live use, not in
    # the eval set: a reminder that rule coverage gaps surface from
    # actual questions, not from the questions you designed.
    r"\bfiscal\s+20\d\d\b",
]

QUARTERLY_SIGNALS = [
    r"\bquarter\b", r"\bquarterly\b", r"\bq[1-4]\b", r"\bthree months\b",
    r"\bsix months\b", r"\bnine months\b", r"\bfirst quarter\b",
    r"\bsecond quarter\b", r"\bthird quarter\b", r"\bfourth quarter\b",
]


def infer_ticker(question: str) -> str | None:
    """Return a ticker only if EXACTLY ONE company is mentioned.

    Two companies means a comparison — filtering to either one would hide
    half the answer. Zero companies means we have no evidence. Both cases
    return None: no filter.
    """
    q = question.lower()
    found = {
        ticker
        for ticker, aliases in COMPANY_ALIASES.items()
        if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases)
    }
    return found.pop() if len(found) == 1 else None


def infer_form(question: str) -> str | None:
    """Return '10K' or '10Q' only when the signal is unambiguous."""
    q = question.lower()
    annual = any(re.search(p, q) for p in ANNUAL_SIGNALS)
    quarterly = any(re.search(p, q) for p in QUARTERLY_SIGNALS)

    # Mixed signals ("Azure growth in the fourth quarter of fiscal 2025")
    # genuinely span both document types — don't choose.
    if annual == quarterly:
        return None
    return "10K" if annual else "10Q"


def infer_filters(question: str) -> dict:
    """Build a filter dict from a question. Empty dict = no filtering.

    Each field is inferred independently: a confident ticker is still
    applied even when the form type is unclear (partial filtering already
    shrinks the candidate pool by ~4x on this corpus).
    """
    filters = {}
    ticker = infer_ticker(question)
    if ticker:
        filters["ticker"] = ticker
    form = infer_form(question)
    if form:
        filters["form"] = form
    return filters


if __name__ == "__main__":
    # Quick manual check against representative questions.
    samples = [
        "How much did Apple spend on R&D last year?",
        "How much profit did Microsoft make in fiscal 2025?",
        "What was Apple's effective tax rate for the three months ended December 27, 2025?",
        "How does Apple's margin compare to Microsoft's?",   # two companies
        "What are the main risks in the filings?",           # no company
        "How much did Microsoft's Azure business grow in fiscal 2025?",
    ]
    for s in samples:
        print(f"{str(infer_filters(s)):<40} {s}")