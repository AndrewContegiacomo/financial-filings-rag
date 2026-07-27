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

82 question → gold-chunk pairs (`data/eval/eval_set.json`), in two subsets
that serve different purposes.

**72 synthetic questions.** Chunks are sampled from the corpus and an LLM
writes the question each chunk answers — the source chunk is the gold by
construction. Sampling is stratified (company × form type × chunk kind)
and reproducible via a fixed seed.

**10 hand-written control questions.** These exist to test a specific
threat to validity: a question written by a model that can see the chunk
tends to reuse its vocabulary, which structurally favors keyword
matching. Each control question was written *before* looking at the
corpus, using everyday business language, and its gold was then located
with plain substring search (`evaluation/find_gold.py`) — deliberately
**not** with either retriever, since selecting golds with the systems
under test would only keep questions those systems already answer.

| # | Question | Filing wording | Gap |
|---|---|---|---|
| 1 | What were Apple's total net sales in the last fiscal year? | net sales | — |
| 2 | How much did Apple spend on R&D last year? | research and development | spend → expenses |
| 3 | How much profit did Microsoft make in fiscal 2025? | net income | profit → net income |
| 4 | How much cash and cash equivalents did Apple hold at the end of fiscal 2025? | cash and cash equivalents | — |
| 5 | How much did Microsoft's Azure business grow in fiscal 2025? | Azure and other cloud services revenue growth | business → revenue |
| 6 | What is Apple's best-selling product line? | net sales by category (iPhone) | best-selling → highest net sales |
| 7 | How much did JPMorgan pay in common stock dividends during 2025? | dividends declared on common stock | pay → declared |
| 8 | What lawsuits is Pfizer facing over its former heartburn medication? | Zantac | **heartburn medication → Zantac** |
| 9 | How could new U.S. import duties affect Apple's business? | tariffs | **import duties → tariffs** |
| 10 | How much long-term borrowing did Microsoft have outstanding at the end of fiscal 2025? | long-term debt | borrowing → debt |

Questions 8 and 9 are the sharpest cases: "heartburn medication" and
"import duties" appear **nowhere** in the filings. A user who doesn't know
the corpus writes exactly this way.

Three of the ten initial drafts had to be reformulated because locating
the gold revealed they were ambiguous — "how much cash does Apple have
available?" has two defensible answers ($35.9B cash and equivalents vs
$132.4B including marketable securities). Synthetic questions never show
this failure mode, which is not a virtue: they are written from the answer.

**Multiple golds.** In financial filings the same figure legitimately
appears in several places — Microsoft's net income shows up in the MD&A
summary, the income statement, the cash flow statement, the equity
statement and the EPS note — and the 40-word chunk overlap often splits
an answer across two consecutive chunks. Control questions are therefore
annotated with all valid golds (up to 5). Metrics are reported in two
variants: **strict** (primary gold only, applied uniformly — the only
convention under which subsets are comparable) and **expanded** (all
annotated golds).

### Retrieval results

Metrics: **hit-rate@k** (does a gold chunk reach the top k?) and **MRR@k**
(mean reciprocal rank — rewards position, since context tokens cost money
and LLM attention degrades on mid-context material).

**By question origin** (hit@5, strict) — the headline result:

| Configuration | Synthetic (n=72) | Hand-written (n=10) |
|---|---|---|
| keyword (TF-IDF) | 0.417 | **0.000** |
| vector (MiniLM) | 0.181 | 0.100 |
| keyword + oracle filter* | 0.556 | 0.300 |
| vector + oracle filter* | 0.278 | **0.400** |

\* **Oracle filter = upper bound, not system performance.** Retrieval is
restricted to the gold chunk's own ticker and form type — information a
real system does not have. Reported to quantify what automatic metadata
inference from the question would be worth.

**Keyword search scores zero on hand-written questions.** Under the
synthetic rate of 0.417, the probability of 0 hits in 10 is about 0.5% —
this is a real effect, not sampling noise. An earlier version of this
evaluation, run on synthetic questions only, concluded that TF-IDF
outperformed embeddings. That conclusion was an artifact: it measured
vocabulary overlap that the synthetic generation process had introduced.
On realistic phrasing the ranking reverses.

**The single-gold convention penalized dense retrieval specifically**
(manual subset, hit@5):

| Configuration | strict | expanded | Δ |
|---|---|---|---|
| keyword | 0.000 | 0.100 | +0.100 |
| vector | 0.100 | 0.400 | +0.300 |
| keyword + oracle | 0.300 | 0.400 | +0.100 |
| vector + oracle | 0.400 | **0.800** | +0.400 |

Embeddings retrieve semantically correct passages that happen not to be
the annotated primary — the income statement rather than the MD&A
summary. Those are legitimate answers being scored as failures. Keyword
misses, by contrast, are genuine misses.

**Overall baseline** (strict; dominated by the 72 synthetic items, so the
by-origin split above is the more informative view):

| Configuration | hit@1 | hit@5 | MRR@5 | hit@10 | MRR@10 |
|---|---|---|---|---|---|
| keyword | 0.183 | 0.366 | 0.245 | 0.476 | 0.260 |
| vector | 0.098 | 0.171 | 0.122 | 0.280 | 0.136 |
| keyword + oracle* | 0.280 | 0.524 | 0.365 | 0.634 | 0.380 |
| vector + oracle* | 0.134 | 0.293 | 0.188 | 0.402 | 0.203 |

### Where retrieval actually fails

Hit-rate answers "is the gold in the top k". To choose a fix, the useful
question is "how far off is it" — a near miss and a burial call for
different remedies. Rank of the best gold across the whole corpus
(4,631 chunks), hand-written subset:

| Configuration | median rank | top-5 | 6–20 | 21–100 | >100 |
|---|---|---|---|---|---|
| keyword | 21 | 1 | 2 | 2 | 5 |
| vector | 6 | 4 | 5 | 0 | 1 |
| keyword + filter | 18 | 4 | 1 | 4 | 1 |
| vector + filter | **1** | **8** | 1 | 0 | 1 |

With metadata filtering, dense retrieval ranks the gold **first** in half
the questions and within the top 5 in 8 of 10. Nine of ten golds sit
within rank 12 — these are near misses, not burials, which means the
remedies are configuration-level rather than architectural.

The three questions where keyword search finds nothing in 200 results
are exactly the ones designed with the widest vocabulary gap: "profit"
(filing says *net income*), "best-selling" (*net sales by category*),
"import duties" (*tariffs*). Dense retrieval finds all three within
rank 12.

### The conclusion that mattered

The system was not underperforming — it was **configured on the wrong
retriever**. Keyword search was chosen as the default early on, on
intuition, and the first evaluation round appeared to confirm it. That
round used synthetic questions only, which inherit the source chunk's
vocabulary and hand keyword matching an advantage that does not exist in
real use. Only the hand-written control subset exposed the mistake.

The cost of an unrepresentative evaluation set is not imprecise numbers.
It is making architectural decisions that look validated.

### Generation evaluation

Two prompt strategies were compared on identical retrieved context
(retrieval held fixed, so the prompt is the only variable):

- **A — strict contract:** answer only from context, cite every claim,
  refuse when insufficient.
- **B — explicit triage:** first identify which context blocks are
  relevant, then extract, then answer; refuse if step one finds nothing.

Scoring is conditioned on what the context actually contained: when a
gold chunk was retrieved, the answer should be correct, grounded and
cited; when it was not, the correct behaviour is an explicit refusal.
Scoring only "was the answer right" would re-measure retrieval, since a
prompt cannot extract a figure it was never given.

**Result: B refuses cleanly, A does not.** With no gold in context, B
refused in 86% of cases (100% on hand-written questions) against 7% for
A. A's failure mode is not hallucination so much as verbose
pseudo-refusal: paragraphs stating the answer is unavailable while
citing irrelevant chunks — worse for the reader and ambiguous to score.

**Both prompts fabricate when they do arithmetic.** Asked for share
repurchases over a sub-period, A multiplied share counts by average
prices across unrelated rows; B subtracted two figures from different
scopes and reported $92.8B of buybacks in two months. Neither prompt
forbids *deriving* figures — only inventing them. Restricting the model
to values stated verbatim is a requirement for the next iteration.

**Judge reliability: the automated scores are not trustworthy.** To stay
within the free tier's daily token budget, judging ran on a smaller
model than generation. Manual inspection shows it failed: correct
answers ($35,934 for Apple's cash, 15% for operating expenses as a share
of net sales) were scored incorrect, and near-identical refusals
received opposite `refused` labels. The prompt comparison above rests on
behaviour that is directly verifiable in the saved outputs
(`data/eval/llm_results.json`), not on the judge's scores. A re-run with
a stronger judge on a smaller subset is the pending item.

A second flaw in this round: `gold_available` was derived from the
annotated gold only. Synthetic items carry a single gold, so a retrieved
chunk that legitimately contained the answer still counted as "gold
absent" — which inflates the refusal metric in B's favour.

### Known limitations

- **Small control subset** (n=10): one question is worth 10 points.
  Direction is robust, magnitude is not.
- **Confound:** the control questions skew toward figure-bearing chunks
  (~7/10), so part of the keyword collapse may be chunk type rather than
  vocabulary. Keyword scores 0.333 on financial chunks overall vs 0.000
  on the control set, so vocabulary appears to dominate.
- **Multiple golds are curated for control questions only**, so the
  strict/expanded comparison is valid only within that subset.
- Synthetic questions occasionally cite the SEC **filing date** as if it
  were the reporting period, an artifact of the generation prompt.
- Section tags are best-effort and sometimes wrong (a cross-reference can
  update the running state); one eval item has an unclassified chunk kind.
  - **Generation evaluation is provisional.** Judged on 18 items with an
  underpowered judge model; conclusions are drawn from manual reading of
  the outputs rather than the automated scores.
- **Free-tier token budget (100k/day)** constrains generation
  evaluation subset size. This is an operational limit, not a
  methodological choice.