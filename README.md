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

## Architecture

SEC EDGAR ──> download_filings.py ──> data/raw/*.html
│
chunk_filings.py
│
v
data/processed/chunks.json
│ │
keyword index embeddings
(TF-IDF, minsearch) (MiniLM, precomputed .npy)
│ │
└────────┬───────────┘
v
retrieval (top-k, metadata filters)
│
v
prompt with context + source metadata
│
v
LLM (Groq, Llama 3.3 70B, temp=0)
│
v
answer with citations
[TICKER FORM DATE, SECTION]

**Two interchangeable retrievers** over the same chunked corpus:

- **Keyword (TF-IDF)** via [minsearch](https://github.com/alexeygrigorev/minsearch):
  strong on exact financial terminology, blind to synonyms.
- **Dense (embeddings)** via `all-MiniLM-L6-v2` (sentence-transformers,
  runs locally): captures paraphrases ("revenue" ≈ "net sales"), but
  struggles with linearized financial tables.

Both support **hard metadata filtering** (ticker, form type), applied
before ranking rather than left to similarity scores. Both are kept —
and evaluated head-to-head in Phase 3 — because early testing showed
neither dominates: each fails on cases the other handles (see
`notes.md` for a documented case study).

**Generation:** the prompt enforces three rules — answer only from the
retrieved context, cite every claim as `[TICKER FORM DATE, SECTION]`,
and explicitly refuse when the context is insufficient. Verified on
out-of-corpus questions (e.g. asking about a company not in the corpus
correctly yields a refusal, not a hallucination). `temperature=0` for
reproducible, factual answers.

## How to run

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
python ingestion/chunk_filings.py      # -> data/processed/chunks.json
python -m rag.vector_search            # -> data/processed/embeddings.npy
```

### Ask questions (interactive CLI)

```bash
python -m rag.rag
```

## Evaluation

### Evaluation set

41 question → gold-chunk pairs (`data/eval/eval_set.json`), built by
sampling chunks from the corpus and asking an LLM to write the question
each chunk answers — the source chunk is the gold by construction, so no
manual annotation is required.

Sampling is **stratified** along two axes, and reproducible (fixed seed):

- **Company × form type**, so that a single large filing can't dominate.
- **Chunk kind**: `narrative` (flowing prose — risk factors, business
  description) vs `financial` (figure-bearing text, mostly linearized
  tables). An earlier version of the quality filter required 60%
  alphabetic tokens, which silently excluded every financial table —
  precisely the hard cases. Sampling both strata keeps the set
  representative.

### Retrieval results (v1 — baseline)

Metrics: **hit-rate@k** (is the gold chunk in the top k?) and **MRR@k**
(mean reciprocal rank of the gold — rewards position, not just presence,
because context tokens cost money and LLM attention degrades on
mid-context material).

| Configuration | hit@1 | hit@5 | MRR@5 | hit@10 | MRR@10 |
|---|---|---|---|---|---|
| keyword (TF-IDF) | 0.171 | **0.415** | 0.263 | 0.610 | 0.289 |
| vector (MiniLM) | 0.122 | 0.317 | 0.193 | 0.366 | 0.201 |
| keyword + oracle filter* | 0.317 | 0.659 | 0.445 | 0.756 | 0.456 |
| vector + oracle filter* | 0.171 | 0.390 | 0.253 | 0.463 | 0.261 |

\* **Oracle filter = upper bound, not system performance.** These runs
restrict retrieval to the gold chunk's own ticker and form type —
information a real system does not have. They are reported to quantify
what automatic metadata inference from the question would be worth.

### What the numbers show

**Keyword beats dense retrieval on this corpus.** Contrary to the usual
expectation, TF-IDF outperforms embeddings at every k. Financial
questions are dense in exact, rare terms ("noninterest expense", "eSLR
buffer", drug and subsidiary names) — exactly where high IDF pays off.

**The two retrievers fail differently.** Going from k=5 to k=10,
keyword gains +19.5 points while vector gains +4.9. When keyword misses,
the gold is usually just outside the cut; when dense retrieval misses,
the gold is buried far down (rank 231/4631 in the documented case
study). Practical consequence: reranking a top-10 candidate set has real
headroom on keyword and almost none on vector.

**Dense retrieval collapses on financial tables** (hit@5 by chunk kind):

| Chunk kind | keyword | vector |
|---|---|---|
| narrative (n=24) | 0.500 | 0.458 |
| financial (n=17) | 0.294 | **0.118** |

On prose the two are comparable. On figure-bearing text the sentence
embedding model breaks down: a linearized table row is not a sentence.
Even with the oracle filter, dense retrieval only reaches 0.176 on
financial chunks — restricting to the right document doesn't help if the
text can't be embedded meaningfully.

**Metadata filtering is the single largest lever** (+24 points on
keyword hit@5, +59% relative) — larger than any retrieval technique
tested. Inferring ticker and form type from the question is a
straightforward LLM task, which makes it the highest-value target for
the agentic component.

### Known limitations

- **Small sample.** At n=17 for the financial stratum, one question is
  worth 5.9 points. Directions are consistent with qualitative
  evidence; exact magnitudes are not precise.
- **Lexical leakage.** Questions written by an LLM that could see the
  chunk tend to reuse its vocabulary, which structurally favors keyword
  matching. The generation prompt actively counteracts this, but part
  of the keyword advantage may still be an artifact. Hand-written
  control questions are being added to test this.
- **Single gold chunk.** Broad questions ("what risks does X face?")
  are legitimately answered by several passages, but only one counts as
  correct — this understates performance on the narrative stratum.
- Questions occasionally cite the SEC **filing date** as if it were the
  reporting period, an artifact of the generation prompt.
