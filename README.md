# Financial Filings RAG

A RAG application with agentic capabilities for querying SEC filings
(10-K / 10-Q) in natural language, with source-cited answers.

## Problem

Financial filings are long (100+ pages), dense, and written in legal
prose. Extracting a specific fact — revenue growth, identified risk
factors, segment performance — means manually digging through them.
A general-purpose LLM alone can't help reliably: it doesn't know the
latest filings, and in finance a hallucinated number is worse than no
answer.

This project answers natural-language questions over a corpus of real
SEC filings, grounding every claim in the source documents and citing
them explicitly.

## Data

**Source:** [SEC EDGAR](https://www.sec.gov/search-filings) — the SEC's
public, free filing database (no registration required).

**Corpus:** the latest 10-K (annual report) and the two latest 10-Q
(quarterly reports) for four companies chosen across different sectors:

| Ticker | Company        | Sector          |
|--------|----------------|-----------------|
| AAPL   | Apple          | Tech (hardware) |
| MSFT   | Microsoft      | Software/cloud  |
| JPM    | JPMorgan Chase | Banking         |
| PFE    | Pfizer         | Pharma          |

Sector diversity is deliberate: partially overlapping companies
(AAPL/MSFT) and clearly distinct ones let the evaluation observe
retrieval behavior in both easy and hard regimes. Multiple filing
periods per company enable year-over-year comparisons (agentic
component, Phase 4).

**Reproducibility:** raw documents are not committed. The corpus is
rebuilt from scratch by the ingestion scripts (see below). Filings are
identified deterministically (latest N per type), and chunk IDs are
stable across re-runs.

**Processing:** HTML filings are cleaned (scripts/styles stripped,
table text kept — key figures live in tables) and split into chunks of
200 words with a 40-word overlap. Chunk size is dictated by the
embedding model's 256-token input limit. Each chunk carries metadata
(ticker, form type, filing date, and the 10-K "Item" section when
detectable), which powers filtering and source citations downstream.

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (or plain venv+pip)
- A free [Groq](https://console.groq.com) API key

### Setup

```bash
git clone https://github.com/AndrewContegiacomo/financial-filings-rag.git
cd financial-filings-rag
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # then fill in your values
```

`.env` requires two variables:
- `GROQ_API_KEY` — your Groq API key
- `SEC_USER_AGENT` — your contact info (`Name Surname email@example.com`),
  required by the SEC's fair-use policy

### Build the corpus

```bash
python ingestion/download_filings.py   # ~12 filings from EDGAR
python ingestion/chunk_filings.py     # -> data/processed/chunks.json
```