# Financial Filings RAG

> 🔗 **Live demo:** https://financial-filings-rag-rlcq58txd8kpc6p2blkajz.streamlit.app/
>
> Free-tier hosting puts the app to sleep when idle — the first visit
> takes about a minute to wake it, and the first question another few
> seconds to load the embedding model. Subsequent questions are fast.
>
> It also runs on a free-tier API key with a daily token budget. If the
> app reports the model unavailable, the quota is exhausted — the code
> and evaluation results below are unaffected.

![Interface](docs/screenshot-answer.png)

*The question uses everyday wording ("heartburn medication"); the filings
only ever say "Zantac". Each claim carries the filing and section it
came from.*

![Sources](docs/screenshot-sources.png)

*Every answer exposes the retrieved passages, so any claim can be checked
against the source text.*

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

**Corpus:** 14 filings (annual 10-K and quarterly 10-Q reports) for four
companies chosen across different sectors:

| Ticker | Company        | Sector          |
|--------|----------------|-----------------|
| AAPL   | Apple          | Tech (hardware) |
| MSFT   | Microsoft      | Software/cloud  |
| JPM    | JPMorgan Chase | Banking         |
| PFE    | Pfizer         | Pharma          |

Sector diversity is deliberate: partially overlapping companies
(AAPL/MSFT) and clearly distinct ones let the evaluation observe
retrieval behaviour in both easy and hard regimes. Multiple filing
periods per company enable year-over-year comparisons.

**Corpus membership is explicit.** `data/corpus_manifest.json` records
every filing by SEC accession number, together with its filing date and
its report date (the period covered — the two differ by months, and only
the latter identifies the fiscal year). Discovery adds newly published
filings and never removes: a filing that once belonged to the corpus
stays in it, so evaluation gold IDs remain valid as the corpus grows.

An earlier version defined the corpus implicitly as "the N most recent
filings per form, plus whatever was on disk". That held locally, where
old files accumulated, and broke on the first clean-checkout pipeline
run: the runner fetched only the latest N, silently dropping older
filings the evaluation set references.

**Processing:** HTML filings are cleaned (scripts and styles stripped,
table text kept — key figures live in tables) and split into chunks of
200 words with a 40-word overlap. Chunk size is dictated by the
embedding model's 256-token input limit, not by preference. Each chunk
carries metadata — ticker, form type, filing date, fiscal year, and the
10-K "Item" section when detectable — which powers filtering and source
citations downstream.

**Reproducibility:** raw filings (~70MB of HTML) are not committed and
are rebuilt by the ingestion pipeline. The processed artifacts
(`chunks.json`, `embeddings.npy`) *are* committed, because the deployed
app needs a corpus at container start and regenerating it would mean
hitting the SEC API and re-embedding on every restart. Chunk IDs are
`{filename}_{index}`, so adding a filing leaves existing IDs untouched.

## Architecture

### Ingestion (scheduled weekly, or run on demand)

```text
SEC EDGAR API
      │
      ▼
download_filings.py ──────▶ data/corpus_manifest.json
      │                     data/raw/*.html
      ▼
chunk_filings.py ─────────▶ data/processed/chunks.json
      │
      ├──▶ TF-IDF index (in memory, built at startup)
      │
      └──▶ vector_search.py ──▶ data/processed/embeddings.npy
```

### Query path A — natural-language questions (default)

```text
question
   │
   ▼
query_analysis.py          rules only, no LLM call
   │  infers ticker + form type + fiscal year
   ▼
augmented search           dense (MiniLM) leads, keyword interleaved
   │                       k=10, hard metadata filters
   ▼
prompt + context           context-only, cite each claim, refuse if absent
   │
   ▼
Groq / Llama 3.3 70B       temperature=0
   │
   ▼
answer + [TICKER FORM DATE, SECTION]
```

### Query path B — quantitative and comparative questions (agentic)

```text
question
   │
   ▼
tool selection             model picks tool + parameters
   │
   ├──▶ lookup_metric(ticker, metric, period)
   │         │
   │         ▼
   │    keyword search     TF-IDF, filtered by ticker + form
   │         │
   │         ▼
   │    verbatim extraction   value + period label, quoted from source
   │         │
   │         ▼
   │    _period_matches()     rejects wrong-column reads
   │
   └──▶ compare_periods(...)  two lookups, arithmetic in Python
             │
             ▼
        answer + citations
```
**Two retrieval strategies, each used where it measures better.**

- **Augmented retrieval for natural-language questions.** Dense
  retrieval (`all-MiniLM-L6-v2`) leads, with a few keyword results
  interleaved. Users phrase questions in everyday vocabulary ("profit",
  "import duties", "heartburn medication") while filings use accounting
  terms ("net income", "tariffs", "Zantac"); keyword search alone fails
  to surface the gold at all on 3 of 10 such questions, while dense
  retrieval alone misses near-verbatim lexical matches.
- **Keyword retrieval (TF-IDF) inside the agentic tools.** Tools receive
  filing terminology by construction — the model supplies `metric="net
  income"`, not a paraphrase — and on figure-bearing chunks keyword
  scores 0.487 hit@5 against 0.205 for dense. Using dense retrieval here
  returned tax-discussion passages and never surfaced the income
  statement.

**Metadata filtering** (`rag/query_analysis.py`) infers ticker, form type
and fiscal year from the question with rules and no LLM call: company
aliases include product names (Azure, Zantac, Comirnaty), and a closed
vocabulary separates annual from quarterly wording. Ticker resolves on
82/82 evaluation questions, form on 47/82, with one wrong inference in
total. Filters are applied as hard constraints before ranking. Ambiguous
cases produce no filter deliberately — a wrong hard filter makes the
answer unreachable, while no filter merely leaves it lower-ranked.

**Generation** enforces context-only answers, per-claim citations, and
explicit refusal when the context is insufficient, at `temperature=0`.

**Agentic path** (`rag/tools.py`, `rag/agent.py`) handles quantitative
and comparative questions. The model chooses which figure to look up;
the tool retrieves it, requires the model to quote the source fragment
and the period label verbatim, and performs all arithmetic in Python.
This split exists because generation evaluation showed prompts
fabricating figures whenever they computed — multiplying share counts by
average prices across unrelated rows, subtracting values with mismatched
scopes. A code-level guard rejects an extraction whose declared period
doesn't match the requested one, the failure mode of three-year
comparative tables, and a second guard rejects comparisons whose two
values differ by more than tenfold.

**API access** is centralized in `rag/llm_client.py`, which retries
transient failures (rate limits, capacity errors) with exponential
backoff and returns `None` on persistent failure so callers degrade into
a message rather than an exception. The RAG path still displays retrieved
passages when generation is unavailable.

**Monitoring** (`monitoring/store.py`, `dashboard.py`) logs every query
with latency split between retrieval and generation, the filters
inferred, which path handled it, and whether it was refused. Users rate
answers, and the dashboard reports volume, latency separated by cold
start, filter coverage, routing and outcomes.

## How to run

### Prerequisites
- Python 3.12 (ML libraries lag newer releases — this is a hard
  requirement, not a preference)
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

### Build or refresh the corpus

```bash
python -m pipeline.run_pipeline
```

Downloads any filing listed in the manifest that isn't on disk, discovers
newly published ones, then chunks and embeds. Idempotent: with nothing
new and artifacts present it exits in seconds without rebuilding. The
processed artifacts ship with the repo, so this is only needed to refresh.

### Run the app

```bash
streamlit run app.py          # http://localhost:8501
streamlit run dashboard.py    # http://localhost:8502
```

Or the interactive CLI:

```bash
python -m rag.rag             # RAG path
python -m rag.agent           # agentic path
```

### Run with Docker

```bash
docker compose up --build
```

Starts three services: the app (8501), the monitoring dashboard (8502)
and Postgres. `monitoring/store.py` selects its backend from
`DATABASE_URL` — Postgres under compose, SQLite otherwise.

### Reproduce the evaluation

```bash
python -m evaluation.evaluate_retrieval   # local, no API calls
python -m evaluation.diagnose_retrieval   # rank distribution
python -m evaluation.evaluate_llm         # needs API quota
```

### Automated ingestion

`.github/workflows/update-corpus.yml` runs the pipeline weekly and
commits regenerated artifacts back to the repo, which in turn triggers a
Streamlit Cloud redeploy. Schedule, pipeline and deployment are one chain
rather than three mechanisms. Requires a `SEC_USER_AGENT` repository
secret.

## Evaluation

### Evaluation set

82 question → gold-chunk pairs (`data/eval/eval_set.json`), in two
subsets that serve different purposes.

**72 synthetic questions.** Chunks are sampled from the corpus and an LLM
writes the question each chunk answers — the source chunk is the gold by
construction. Sampling is stratified (company × form type × chunk kind)
and reproducible via a fixed seed.

**10 hand-written control questions.** These exist to test a specific
threat to validity: a question written by a model that can see the chunk
tends to reuse its vocabulary, which structurally favours keyword
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
"import duties" appear **nowhere** in the filings. A user who doesn't
know the corpus writes exactly this way.

Three of the ten initial drafts had to be reformulated because locating
the gold revealed they were ambiguous — "how much cash does Apple have
available?" has two defensible answers ($35.9B cash and equivalents vs
$132.4B including marketable securities). Synthetic questions never show
this failure mode, which is not a virtue: they are written from the
answer.

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

Metrics: **hit-rate@k** (does a gold chunk reach the top k?) and
**MRR@k** (mean reciprocal rank — rewards position, since context tokens
cost money and LLM attention degrades on mid-context material).

Unless stated otherwise, the tables below were measured on a 4,631-chunk
corpus. The corpus has since grown to 5,034 chunks; see *Corpus growth*
below for what changed.

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
a real effect, not sampling noise. An earlier version of this evaluation,
run on synthetic questions only, concluded that TF-IDF outperformed
embeddings. That conclusion was an artifact: it measured vocabulary
overlap the synthetic generation process had introduced. On realistic
phrasing the ranking reverses.

**The single-gold convention penalized dense retrieval specifically**
(hand-written subset, hit@5):

| Configuration | strict | expanded | Δ |
|---|---|---|---|
| keyword | 0.000 | 0.100 | +0.100 |
| vector | 0.100 | 0.400 | +0.300 |
| keyword + oracle | 0.300 | 0.400 | +0.100 |
| vector + oracle | 0.400 | **0.800** | +0.400 |

Embeddings retrieve semantically correct passages that happen not to be
the annotated primary — the income statement rather than the MD&A
summary. Those are legitimate answers scored as failures. Keyword misses,
by contrast, are genuine misses.

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
different remedies. Rank of the best gold across the whole corpus,
hand-written subset:

| Configuration | median rank | top-5 | 6–20 | 21–100 | >100 |
|---|---|---|---|---|---|
| keyword | 21 | 1 | 2 | 2 | 5 |
| vector | 6 | 4 | 5 | 0 | 1 |
| keyword + filter | 18 | 4 | 1 | 4 | 1 |
| vector + filter | **1** | **8** | 1 | 0 | 1 |

With metadata filtering, dense retrieval ranks the gold **first** in half
the questions and within the top 5 in 8 of 10. Nine of ten golds sit
within rank 12 — near misses, not burials, which means the remedies are
configuration-level rather than architectural.

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

### Corpus growth

The scheduled pipeline added Microsoft's FY2026 10-K and an Apple 10-Q on
its first real run, growing the corpus from 4,631 to 5,034 chunks. No
gold IDs were lost, but retrieval degraded sharply: hand-written hit@5
fell from 0.400 to 0.100.

Cause: the corpus now holds two Microsoft 10-Ks, and the filter
`{ticker: MSFT, form: 10K}` no longer isolates one document. The
competing chunks are the *same section of the same document one year
later* — Microsoft's MD&A keeps its structure and wording year over year
and changes only the figures, so the two are near-identical both
lexically and semantically. "Fiscal 2025" is a weak semantic signal
against wholesale textual similarity.

Adding `fiscal_year` as a third filter dimension — derived from the
filing's report date, not its submission date — recovered most of it
(0.300 strict / 0.700 expanded). The cost is that a year mentioned
incidentally now excludes three quarters of the corpus: keyword retrieval
on narrative questions dropped from 0.452 to 0.333.

This failure mode is invisible to static benchmarks. Real corporate
documents are time series of near-identical text, and temporal
disambiguation is a metadata problem, not a semantic retrieval one.

## Retrieval improvements: what was adopted and what wasn't

Seven techniques were implemented and measured against the 82-item
evaluation set. Headline figures are hit@5 on the 10 hand-written
questions (strict / expanded golds), since those are phrased without
borrowing the filings' vocabulary.

| Technique | Result | Adopted |
|---|---|---|
| Rule-based metadata filtering | 0.100 → 0.400 / 0.800 | Y |
| Dense retrieval as default, k=10 | see above | Y |
| **Augmented retrieval** (dense + interleaved keyword) | hit@5 0.256 → 0.305, hit@10 0.402 → 0.488 | Y |
| Agentic metric tools | qualitative — see below | Y |
| Hybrid search (RRF fusion) | 0.200 / 0.500 | N |
| LLM-based form inference | 0.400 / 0.800 (identical to rules) | N |
| Query rewriting | 0.500 / 0.800, 20× latency | N |
| Cross-encoder reranking | 0.300 / 0.700 | N |

**Rule-based filtering was the largest single gain**, closing roughly
half the gap to an oracle that knows the answer's metadata. Notably, one
two-word regex addition (`fiscal 20\d\d`, matching "fiscal 2025" without
the word "year") moved hand-written hit@5 from 0.200 to 0.400 — a larger
gain than any retrieval technique tested. Rule coverage mattered more
than algorithmic sophistication.

**Augmented retrieval was the one technique from outside the rules that
helped**, and it did not come from the literature. Dense retrieval keeps
the head of the result list; up to three keyword results are interleaved
at positions 4, 7 and 10 if not already present. This lifts hit@5 from
0.256 to 0.305 and hit@10 from 0.402 to 0.488, with a clear split by
question type: **+15 points on figure-bearing questions, −5 on narrative
ones**.

The asymmetry is the point. RRF fuses two rankings symmetrically, so
chunks both retrievers rank moderately outrank a chunk only one ranks
first — with retrievers of unequal strength that amplifies noise. Here
dense retrieval keeps rank 1 and its ordering, and keyword only
contributes candidates that would otherwise never appear. An earlier
version appended the keyword results instead of interleaving them: hit@10
improved identically while hit@5 did not move at all, because the useful
chunks were retrieved and then buried below the cut.

The idea came from using the application, not from the evaluation set.
"How many people does Pfizer employ?" returned a refusal even though the
answer sits in the corpus; dense retrieval missed it entirely while
keyword search ranked it first.

**Hybrid search underperformed dense retrieval alone.** RRF rewards
agreement between retrievers, which is evidence only when both are
comparably strong; keyword scores 0.000 on realistic phrasing, so
consensus amplified generic chunks. Tuning RRF_K from 60 to 10 improved
it (overall hit@5 0.280 → 0.305) without changing the verdict.
Down-weighting keyword trends toward weight zero — that is, toward dense
retrieval alone.

**LLM form inference matched the rules exactly.** The 35 questions where
rules abstain are genuinely period-ambiguous ("in 2025" fits either
document type); abstaining was the correct answer, and a model cannot
manufacture information the question doesn't contain.

**Reranking was based on a mis-measured premise.** hit@1 of 0.110
suggested room to promote golds, but that figure is dominated by the 72
synthetic items. On realistic questions the first stage already ranks 7
of 10 golds within position 3 — nothing to promote, only something to
break, and 4 questions got worse. The two pushed out of the top 10
entirely were the ones with the widest vocabulary gap; `ms-marco-MiniLM`
is trained on web passage ranking, where question and passage share
vocabulary, so on financial terminology it falls into the same trap as
keyword search.

The general lesson: **estimate headroom on the subset you care about,
not on the aggregate.**

### Agentic tools

`compare_periods` answers questions like "how much did Apple's net sales
grow from fiscal 2024 to fiscal 2025?" by looking up each figure
separately and computing the change in Python. Building it surfaced three
bugs that aggregate metrics had hidden:

1. Without a form-type filter, the extractor read a six-month figure from
   a 10-Q and reported it as an annual value.
2. With the filter but using dense retrieval, "MSFT net income fiscal
   2024" returned tax-discussion chunks and never surfaced the income
   statement.
3. Including the period in the keyword query buried the income statement:
   it states its label once inside a table with bare year columns, while
   tax passages repeat "fiscal year 2024" and win on term frequency.

All three were found by inspecting what the tool actually retrieved, at
zero API cost. The fix — keyword retrieval, form filter, period excluded
from the query text — produces the correct figures (Microsoft FY2024
$88,136M → FY2025 $101,832M, +15.54%).

## Generation evaluation

Three prompt strategies compared on identical retrieved context
(retrieval held fixed at the production configuration — augmented
search, inferred filters, k=10 — so the prompt is the only variable):

- **A — strict contract:** answer only from context, cite every claim,
  refuse when insufficient.
- **B — explicit triage:** identify relevant blocks, extract, then
  answer; refuse if nothing relevant.
- **C — no derived figures:** as B, plus an explicit ban on computing any
  number not stated literally in the text.

Scoring is conditioned on what the context contained: with a gold chunk
retrieved, the answer should be correct, grounded and cited; without one,
an explicit refusal is correct. Scoring only "was the answer right" would
re-measure retrieval, since a prompt cannot extract a figure it was never
given.

**Results** (7 hand-written questions, 21 answers — the run stopped on
the free tier's daily token limit):

| | A | B | C |
|---|---|---|---|
| correct, gold in context (n=5) | 5/5 | 5/5 | 5/5 |
| refused, gold absent (n=2) | 2/2 | 2/2 | 2/2 |
| grounded | 6/7 | 7/7 | 7/7 |
| computed a figure | 0 | 0 | 0 |

**All three prompts answer correctly when the evidence is present and
refuse when it is not.** The meaningful change between this round and the
previous one was retrieval, not prompting: the gold chunk reached the
context in 5 of 7 cases here against 4 of 18 when the same evaluation ran
on keyword search at k=5.

**The `computed` metric is untested.** It exists because both A and B
were observed fabricating figures whenever they calculated — multiplying
share counts by average prices, subtracting values with mismatched scopes
to report $92.8B of buybacks in two months. None of these seven questions
invites arithmetic, so the score is zero for all three prompts and says
nothing about whether C's restriction works. Testing it requires
questions that tempt calculation.

**The one measurable difference** is a single grounding failure: asked
about Apple's cash, A answered "$35,934 million in cash, cash
equivalents, and restricted cash and cash equivalents" — the figure is
right but the phrasing conflates two balance-sheet lines. B and C give
the number plainly.

B is used in production. On this evidence the justification is verbosity
and that one grounding slip, not a difference in accuracy.

**An earlier round of this evaluation was discarded.** To stay within the
token budget, judging ran on a smaller model than generation; manual
inspection showed it scoring correct answers as incorrect and giving
near-identical refusals opposite labels. Judging returned to the
generation model and the sample was reduced instead. The judge is still
the same model family that produced the answers, so self-preference bias
remains — the comparison between prompts holds, the absolute values are
likely optimistic.

## Monitoring

Every query is logged with latency split between retrieval and
generation, the filters inferred, which path handled it, the number of
passages retrieved, and whether the answer was a refusal. Users rate
answers directly in the interface.

The dashboard's five views were chosen from what this project learned to
worry about rather than from what is easy to plot:

- **Query volume** over time
- **Latency, cold start separated from warm** — the first query of a
  process loads the embedding model and builds the keyword index;
  averaging the two hides which component is slow
- **Filter coverage** — how many constraints each search had, since
  metadata filtering proved the largest driver of retrieval quality
- **Routing** — RAG vs agentic path, to see how often the keyword-based
  router fires
- **Outcomes** — answered, refused, errored; a system that refuses too
  readily fails as surely as one that fabricates

## Known limitations

- **Small control subset** (n=10): one question is worth 10 points.
  Direction is robust, magnitude is not.
- **Confound:** the control questions skew toward figure-bearing chunks
  (~7/10), so part of the keyword collapse may be chunk type rather than
  vocabulary. Keyword scores 0.333 on financial chunks overall vs 0.000
  on the control set, so vocabulary appears to dominate.
- **Multiple golds are curated for control questions only**, so the
  strict/expanded comparison is valid only within that subset.
- **Generation evaluation covers 7 of the 10 hand-written questions**,
  cut short by the free tier's daily token limit. The `computed` metric
  is untested because none of those questions requires arithmetic.
- **Free-tier token budget (100k/day)** constrains evaluation subset
  sizes throughout. This is an operational limit, not a methodological
  choice.
- **Guards are per-tool, and the model can route around them.** A
  plausibility check rejects comparisons whose two values differ by more
  than tenfold. Asked about Pfizer revenue, the model was blocked by that
  check and then obtained the same wrong figure through the single-figure
  lookup tool, which has no equivalent guard. Validation currently sits
  where a value is consumed rather than where it is produced.
- **Generic metric names break the agentic lookup.** `metric="revenue"`
  retrieves product-returns adjustments and segment lines rather than the
  income-statement total, because the term appears hundreds of times in a
  filing. Specific names ("net income", "research and development
  expenses") work reliably.
- **Cross-company comparisons are not guarded for comparability.** Asked
  which company has the highest revenue, the system compared a full year
  for one company against nine months for another and answered
  confidently. Neither retrieval path enforces matching periods.
- **Routing is heuristic.** "What caused the increase in operating
  expenses?" is routed to the analytical tools because it contains
  "increase", though it is a causal narrative question. The interface
  exposes a manual override.
- Synthetic questions occasionally cite the SEC **filing date** as if it
  were the reporting period, an artifact of the generation prompt.
- **Section tags are best-effort** and sometimes wrong (a cross-reference
  can update the running state); one evaluation item has an unclassified
  chunk kind.
- **The dashboard container exits with SIGSEGV on Apple Silicon** when
  the app processes a query concurrently. The same code runs correctly
  outside Docker and the compose stack otherwise works end to end —
  events written by the app are read back by the dashboard. The crash
  occurs in native code with no Python traceback and was not diagnosed
  further.

## Where to find each rubric criterion

| Criterion | Where |
|---|---|
| Problem description | *Problem* |
| Retrieval flow | *Architecture*, `rag/rag.py` |
| Retrieval evaluation | *Retrieval results* — 4 configurations compared |
| LLM evaluation | *Generation evaluation* — 3 prompts compared |
| Interface | Live demo link, `app.py` |
| Ingestion pipeline | *Automated ingestion*, `pipeline/`, `.github/workflows/` |
| Monitoring | *Monitoring*, `dashboard.py` — 5 charts + user feedback |
| Containerization | *Run with Docker* — app, dashboard and Postgres |
| Reproducibility | *How to run*, pinned dependencies, fixed seeds, committed artifacts |
| Best practices — hybrid search | *Retrieval improvements*, `rag/hybrid_search.py` |
| Best practices — reranking | *Retrieval improvements*, `rag/rerank.py` |
| Best practices — query rewriting | *Retrieval improvements*, `rag/llm_query_analysis.py` |
| Bonus — cloud deployment | Live demo link at the top |