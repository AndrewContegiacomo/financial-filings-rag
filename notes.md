# Working notes — experiments & known issues

## Case study: "Apple total revenue" (the Q1 question)

**Question:** What were Apple's total net sales in the last fiscal year?
**Gold chunk:** `AAPL_10K_2025-10-31_0104` (Item 7)
**Gold answer:** Total net sales FY2025 = $416,161M (FY2024: $391,035M, FY2023: $383,285M)

### Results so far

| # | Configuration | Outcome |
|---|--------------|---------|
| 1 | Keyword (TF-IDF), no filter, query "total revenue" | Gold not in top-5. Retrieved mostly narrative chunks + one MSFT chunk ("revenue" is MSFT's term; Apple says "net sales") |
| 2 | Keyword, no filter, query "total net sales" | All top-5 AAPL, but gold still missing: 10-Q chunks win (label repeated in every quarterly table inflates TF) |
| 3 | Vector (MiniLM), no filter | Gold ranked **231 / 4631**. Top-5 all 10-Qs — "annual vs quarterly" is semantically fuzzy for embeddings |
| 4 | Vector + hard filter (ticker=AAPL, form=10K) | Top-5 all Item 7/8 — right neighborhood, gold still not top-5 |
| 5 | Vector + filter + query "Apple total net sales fiscal year 2025" | **Adjacent chunk (0105) ranked #1** — one overlap-window away from gold |

### Diagnosis

- **TF-IDF fails on:** synonyms (revenue ≠ net sales) + verbose narrative
  chunks outscoring the dense table row.
- **Embeddings fail on:** linearized tables (country names + figures embed
  as semantic noise vs. a natural-language question) + annual/quarterly
  distinction being a nuance in embedding space.
- **Metadata filters don't fail:** `form=10K` is a fact, not a similarity.
  Biggest single improvement observed.
- Gold at rank 231 ⇒ reranking top-k alone cannot fix this case; the
  first-stage candidate set must improve (hybrid / filtering) first.
- v1 of the eval set was skewed towards narrative sections due to the quality filter.
- Some "kind" labels are questionable ("preferred stock dividends", "eSLR buffer cap" → tagged as narrative, but clearly financial). The heuristic looks at the chunk rather than the question, so mixed chunks end up in the wrong category. It’s not an issue for now—we’ll use it as a rough lens, not as ground truth—but make a note of it.

### Implications for Phase 3/4

- This question is **eval-set item #1** (hard case, fully diagnosed).
- Motivates, with evidence: hybrid search (lexical "net sales" + semantic
  question), metadata inference from the question ("fiscal year" → 10-K —
  candidate agentic task), possibly table-aware chunking.
- Eval should measure hit-rate at multiple k (gold may enter filtered
  top-10 even where top-5 misses).

## Known issues

- **Section tagging misfires:** `AAPL_10K_2025-10-31_0108` tagged
  `Item 1A` but content is Operating Expenses (Item 7). Likely a
  "see Item 1A" cross-reference updated the running state. Best-effort
  by design; revisit if section tags become load-bearing for filtering.

## Eval set candidates (from field testing)

1. Apple total net sales FY2025 → gold `AAPL_10K_2025-10-31_0104` ← hard
2. JPM interest-rate risks → keyword search handled well (easy case)
3. Pfizer pipeline performance → partial (descriptive chunks only)
4. Tesla revenue → out-of-corpus refusal test (behavioral, no gold chunk)

## Retrieval evaluation v1

Eval set: 41 items (40 synthetic + 1 manual), stratified by
company × form × kind. Full results: `data/eval/retrieval_results.json`.

| config | hit@1 | hit@5 | mrr@5 | hit@10 | mrr@10 |
|---|---|---|---|---|---|
| keyword | 0.171 | 0.415 | 0.263 | 0.610 | 0.289 |
| vector | 0.122 | 0.317 | 0.193 | 0.366 | 0.201 |
| keyword+oracle | 0.317 | 0.659 | 0.445 | 0.756 | 0.456 |
| vector+oracle | 0.171 | 0.390 | 0.253 | 0.463 | 0.261 |

By kind, hit@5: financial (n=17) kw 0.294 / vec 0.118 —
narrative (n=24) kw 0.500 / vec 0.458.

Sanity check: hit@1 == MRR@1 in every row, as expected by definition
(a gold at rank 1 contributes 1/1). Implementation behaves correctly.

### Hypotheses confirmed / refuted

- CONFIRMED: dense retrieval breaks on linearized tables (kind=financial
  0.118 vs 0.458 narrative; oracle filter barely helps at 0.176).
- CONFIRMED: metadata filtering is the biggest single lever (+24 pts).
- REFUTED (mildly surprising): vector > keyword. The opposite holds
  here, consistently across k and across both strata.
- NEW: keyword's k=5→k=10 jump (+19.5 pts) vs vector's (+4.9) means
  near-misses are a keyword-specific pattern ⇒ reranking should be
  applied on top of keyword/hybrid candidates, not dense-only.

### Open issues affecting these numbers

- `kind` labels are heuristic and computed from the CHUNK, not the
  question: several clearly financial questions ("preferred stock
  dividends", "eSLR buffer cap") sit in the narrative bucket. Treat the
  split as a lens, not ground truth.
- Lexical leakage untested: only 1 manual question, so the
  origin-based breakdown is not usable yet. → next step.

### Next actions

1. Add ~10 hand-written questions with deliberately non-document
   vocabulary; re-run and compare synthetic vs manual to measure leakage.
2. Grow the synthetic set (PER_STRATUM up) for tighter estimates.
3. Fix ITEM_RE (`Item 85` bug) and the filing-date-as-period prompt
   artifact at the next corpus rebuild — requires regenerating the eval
   set, so only after this evaluation round is closed.

   ## Retrieval evaluation v2 — measuring our own bias

Eval set v2: 82 items (72 synthetic + 10 hand-written control), multi-gold
schema. NOT comparable with v1: the synthetic sample was regenerated from
scratch (same seed, larger n ⇒ entirely different draw), so v1 numbers are
kept only as methodological record.

### Headline: v1's main conclusion was an artifact

hit@5, strict (primary gold only), by origin:

| config | synthetic (n=72) | manual (n=10) |
|---|---|---|
| keyword | 0.417 | **0.000** |
| vector | 0.181 | 0.100 |
| keyword+oracle | 0.556 | 0.300 |
| vector+oracle | 0.278 | **0.400** |

Keyword search finds the primary gold in 0/10 hand-written questions.
Under the synthetic rate of 0.417, P(0 of 10) ≈ 0.005 — this is not
small-sample noise. With the oracle filter the ranking REVERSES:
vector (0.400) > keyword (0.300).

⇒ v1's "keyword beats dense retrieval" was measuring lexical leakage,
not retrieval quality.

### Single-gold convention penalized dense retrieval specifically

Manual subset, hit@5:

| config | strict | expanded | delta |
|---|---|---|---|
| keyword | 0.000 | 0.100 | +0.100 |
| vector | 0.100 | 0.400 | +0.300 |
| keyword+oracle | 0.300 | 0.400 | +0.100 |
| vector+oracle | 0.400 | **0.800** | +0.400 |

Asymmetry explained: embeddings retrieve semantically correct chunks that
weren't the annotated primary — income statement instead of MD&A summary,
cash flow statement instead of balance sheet. All legitimate answers,
scored as misses. Keyword misses are genuine misses (+0.100 only).

Second measurement embedded here: the single-gold convention understates
real hit-rate by up to 40 points on figure-based questions, where the same
number legitimately appears in 4-5 places.

### Overall v2 baseline (strict)

| config | hit@1 | hit@5 | mrr@5 | hit@10 | mrr@10 |
|---|---|---|---|---|---|
| keyword | 0.183 | 0.366 | 0.245 | 0.476 | 0.260 |
| vector | 0.098 | 0.171 | 0.122 | 0.280 | 0.136 |
| keyword+oracle | 0.280 | 0.524 | 0.365 | 0.634 | 0.380 |
| vector+oracle | 0.134 | 0.293 | 0.188 | 0.402 | 0.203 |

Overall numbers are dominated by the 72 synthetic items, i.e. by the
biased subset. Treat the by-origin split as the informative view.

By kind (strict, hit@5): financial (n=39) kw 0.333 / vec 0.128 —
narrative (n=42) kw 0.405 / vec 0.214.

### Caveats

- n=10 for the manual subset: one question = 10 points.
- Manual set skews financial (~7/10), a partial confound with the
  lexical effect — though keyword scores 0.333 on financial overall vs
  0.000 on manual, so vocabulary is doing most of the work.
- Multi-gold curated for manual items only ⇒ strict/expanded comparison
  valid only within that subset.
- One item has kind=None (classify_source returned None on its gold).

### Implications for Phase 4

1. The two retrievers fail on disjoint question types (exact-term vs
   paraphrased), which motivates hybrid search. NOTE: superseded in
   priority by the close-out diagnostics below — vector+filter alone
   already ranks the gold first on half the manual questions, so
   configuration fixes come first and hybrid is a refinement on top.
2. Metadata filtering remains the largest single lever and helps dense
   retrieval most (vector+oracle 0.800 expanded on manual).
3. Any future eval set needs hand-written or paraphrase-forced questions.
   Purely synthetic generation cannot measure retrieval honestly.

## Phase 3 close-out — diagnosis and verdict

### Rank diagnostics (no API cost — pure local retrieval)

Manual subset (n=10), rank of best gold in full corpus:

| config | median | top5 | 6-20 | 21-100 | >100 |
|---|---|---|---|---|---|
| keyword | 21 | 1 | 2 | 2 | 5 |
| vector | 6 | 4 | 5 | 0 | 1 |
| keyword+filter | 18 | 4 | 1 | 4 | 1 |
| vector+filter | **1** | **8** | 1 | 0 | 1 |

Synthetic subset shows the opposite ordering (keyword median 7, vector
12) — consistent with the leakage finding. The two subsets disagree
because one of them is not measuring reality.

Per-question (manual, unfiltered, keyword / vector):

| question | kw | vec |
|---|---|---|
| Apple total net sales | 7 | 6 |
| Apple R&D spend | 168 | 2 |
| Microsoft profit | None | 12 |
| Apple cash & equivalents | 4 | 5 |
| Azure growth | 13 | 7 |
| Apple best-selling line | None | 10 |
| JPM common dividends | 130 | None |
| Pfizer heartburn lawsuits | 21 | 3 |
| US import duties | None | 8 |
| MSFT long-term borrowing | 26 | 3 |

### Verdict: misconfiguration, not a broken system

Failure type is NEAR MISS (9/10 manual golds within rank 12 on vector),
not burial. Cures are configuration-level:

1. vector as default retriever instead of keyword — zero cost
2. rule-based metadata filter (company names appear verbatim in
   questions; "fiscal year"/"quarter" separates 10-K from 10-Q) — no LLM
   needed, moves median rank 6 -> 1
3. k from 5 to 10 — 10 chunks x 200 words = 2k words, trivial for a
   128k-context model

Corpus rebuild NOT needed: the suspicion that 200-word chunks + MiniLM
were the bottleneck is refuted — with the right configuration MiniLM
ranks the gold first. Half a day saved by diagnosing before fixing.

Residual hard case: JPM common dividends (kw=130, vec=None). Gold 0397
is a capital-table block; likely genuinely hard to match by any method.
1 case in 10, does not drive priorities.

### Generation eval v1 — judge unreliable, findings partial

Ran 18 items x 2 prompts. Judge (llama-3.1-8b-instant, chosen to save
quota) produced invalid scores: correct answers marked incorrect
($35,934 cash, 15% opex ratio), inconsistent `refused` labels on
equivalent refusals. Possibly compounded by gold_text[:800] truncation
hiding the figure from the judge.

What survives, verified by reading outputs directly:
- B (explicit triage) refuses cleanly: 100% on manual items with no gold
  vs 11% for A. A produces verbose pseudo-refusals citing irrelevant
  chunks.
- BOTH prompts fabricate when they compute. Share-repurchase question:
  A multiplied share counts by average prices ($12.99B), B subtracted
  mismatched scopes and claimed $92.8B of buybacks in two months.
  Neither prompt forbids DERIVING figures, only inventing them. → next
  prompt iteration must restrict to verbatim values.

Design flaw in this round: `gold_available` computed from annotated gold
only. Synthetic items have one gold, so retrieved chunks that did answer
the question counted as "gold absent", inflating B's refusal metric.

Also note: the whole generation eval ran on keyword@k=5 — the worst
configuration, as the diagnostics later showed. gold_in_context was 4/18
largely because of that. Re-run belongs on the fixed retrieval.

### Phase 3 status

- Retrieval evaluation: DONE (2 rubric points).
- LLM evaluation: two prompts compared, conclusions drawn from manual
  reading; automated scoring pending a stronger judge (2 rubric points,
  to be consolidated).

## Operational constraint: free-tier token budget

Groq free tier caps tokens per DAY (100k), not just per minute. The
first generation-eval run consumed ~99k tokens on 21 of 30 items and
died on a 429 — with no checkpointing, all of it was lost.

Fixes: results written after every scored pair; judging moved to
llama-3.1-8b-instant (separate rate-limit bucket, and it removes part
of the self-preference bias); subset reduced to 18 items; gold
passages truncated to 800 chars.

Lesson: for API jobs, estimate token cost BEFORE running, and never
write results only at the end of the loop.

Rule coverage matters more than technique: adding `fiscal 20\d\d` to the
annual-signal patterns (found by testing a live question, not from the
eval set) moved vector+rule_filter on manual questions from 0.200 to
0.400 strict, 0.600 to 0.800 expanded. Larger than any retrieval
technique tested so far.

## Phase 4 — six techniques, two adopted

hit@5 on manual subset (strict / expanded), baseline vector+rule_filter
0.400 / 0.800:

| technique | manual | overall hit@5 | verdict |
|---|---|---|---|
| rule filtering | 0.400/0.800 | 0.256 | ADOPTED |
| hybrid RRF (k=60) | 0.200/0.500 | 0.280 | rejected |
| hybrid RRF (k=10, depth 25) | 0.200/0.500 | 0.305 | rejected |
| hybrid vector-weighted | 0.300/0.700 | 0.341 | rejected |
| LLM form inference | 0.400/0.800 | 0.268 | rejected |
| query rewriting | 0.500/0.800 | 0.293 | rejected |
| rules on rewritten query | 0.500/0.800 | 0.293 | rejected |
| rerank depth 30 | 0.300/0.700 | 0.329 | rejected |
| rerank depth 15 | 0.300/0.700 | 0.317 | rejected |

Note the recurring pattern: techniques that help the synthetic subset
hurt or don't move the manual one. Reranking gained +10 points on
synthetic (0.236 → 0.333) while losing on manual. Cross-encoders do
lexical-plus-semantic matching, so leakage rewards them the same way it
rewards TF-IDF.

### Per-question ranks with/without rerank (manual)

base:   1, 1, 2, 1, 2, 10, None, 3, 5, 1
rerank: 1, 2, None, 2, 2, 5, None, 2, None, 1

7/10 golds already within rank 3 before reranking. The headroom I sized
from aggregate hit@1 did not exist on the subset that matters.

### Retrieval saturation

vector+rule_filter on manual questions: median rank 1-2, 7/10 within
rank 3, 0.800 expanded hit@5. Four independent techniques failed to
improve it. Conclusion: retrieval on realistic questions is near the
ceiling of this architecture. Low aggregate numbers come from the
synthetic subset, which we know is biased. Remaining true failures:
JPM dividends (unrecoverable, capital-table chunk) and the strict vs
expanded gap (annotation convention, not system behaviour).

### Agentic tool debugging — three bugs, zero API cost to find

1. No form filter → six-month 10-Q figure reported as annual
   (MSFT "fiscal 2024" = 48,375 instead of 88,136).
2. Form filter + dense retrieval → tax chunks ("income before income
   taxes", "income taxes paid"); income statement never retrieved.
   Dense retrieval sees the topic "income/taxes" and cannot distinguish
   the specific line item.
3. Keyword + period in query text → same Q1 pattern: income statement
   says "Net income" once in a table with bare year columns
   ("2025 2024 2023"), tax passages repeat "fiscal year 2024" and win
   on TF. Removing the period from the query put the income statement
   at rank 1.

Each bug was found by printing what the tool retrieved — local, free,
repeatable. No API call was spent on diagnosis.

Key insight, and it was already in the data: the by-kind breakdown
(financial n=39: keyword 0.487 vs vector 0.205) told us weeks ago that
keyword wins on figure-bearing chunks. We only used it once the tool
forced the question. Different retrievers for different tasks —
dense for paraphrased natural language, keyword for filing terminology.

### Guard against wrong-column extraction

Income statements show three years side by side. The extraction prompt
now requires `period_in_text` (the period label as written in the
excerpt), and `_period_matches` rejects the value in code when the
declared and requested years don't overlap. Rejected extractions are
kept under a `rejected` key for monitoring: how often the guard fires is
itself a metric worth tracking in Phase 7.

### Centralizing API access (rag/llm_client.py)

Trigger: a Groq 503 ("model over capacity") surfaced as a raw stack
trace in the Streamlit UI mid-question. Not our bug, but our problem —
transient provider failures are normal and the interface has to survive
them.

Four modules were calling the API independently, each needing the same
retry handling. Consolidated into `call_llm` / `call_llm_text`:
- retries 429, 503 and connection errors with exponential backoff
  (5s, 10s, 20s)
- returns None on persistent failure instead of raising, so callers
  degrade into a message; the RAG path still shows the retrieved
  passages when generation is unavailable
- builds the client lazily via `get_client()`

The lazy construction fixed a second, unrelated fragility: modules used
to build their Groq client at import time from os.environ, which meant
the Streamlit secrets bridge had to run before any `from rag import ...`
line. Order-dependent imports are a trap; now the key only has to be
present before the first question.

Evaluation code (`generate_eval_set.py`, `evaluate_llm.py`) keeps its
own client deliberately: it needs non-zero temperature for generation
and drives two models with its own checkpointing. Bending the
production helper to fit measurement needs would serve neither.

Refactor note: `tools.py` had accumulated several partial edits and was
in a non-running state — a duplicated `get_index`, a half-replaced LLM
call with literal `...` arguments, a missing `re.search` line before
`match.group()`, and a `time.sleep` without `import time`. Lesson: after
three or four incremental patches to the same file, re-read it whole
rather than trusting that each patch composed.

### Governance failure: the model routes around a guard

Observed while testing the plausibility guard on
"How did Pfizer's revenue change from 2024 to 2025?":

1. `compare_periods(metric="revenue")` → rejected, 103x gap
2. `compare_periods(metric="revenue (consolidated statements of
   income)")` → not found
3. `lookup_metric(metric="revenue", period="2024")` → **771**, the same
   wrong value the guard had just blocked

The guard protects `compare_periods`. The model reached the identical
bad figure through `lookup_metric`, where no equivalent check exists.

Guarding one tool is not enough when the model can obtain the same
result through another. Mitigated weakly for now with a system-prompt
instruction not to work around rejections; the strong fix would be
either shared state (a rejected metric stays rejected for the turn) or
applying the same suspicion to generic metric names inside
`lookup_metric`. Left as future work and documented as a limitation.

This generalizes: validation belongs at the boundary where the value is
produced, not at the boundary where it happens to be consumed.

## Phase 9 — deployment

Committed the processed artifacts (chunks.json ~6MB, embeddings.npy
~7MB) while keeping raw filings out. Regenerating the corpus on every
container start would mean hitting the SEC API and re-embedding — the
derived artifacts are the deployment payload, the scripts remain the
source of truth.

Two build failures worth recording:
- torch pulled the CUDA runtime by default (~2.5GB, useless without a
  GPU). Fixed with `--extra-index-url` pointing at the CPU wheel index.
- Streamlit Cloud defaulted to Python 3.14; torch failed on import with
  a ModuleNotFoundError inside its own package tree. ML libraries lag
  new Python releases by months — the interpreter version is part of
  reproducibility, and we had pinned it locally (3.12) without
  declaring it to the deployment platform.

  ## Corpus growth broke the metadata filters (Phase 6, first pipeline run)

The scheduled pipeline picked up two new filings on its first real run:
MSFT_10K_2026-07-29 and AAPL_10Q_2026-07-31. Corpus went 4,631 -> 5,034
chunks. All eval gold IDs survived (chunk IDs are per-file, so adding
documents doesn't renumber existing ones), but the metrics dropped:

| manual hit@5 | 4,631 chunks | 5,034 chunks |
|---|---|---|
| vector+rule_filter | 0.400 / 0.800 | **0.100 / 0.600** |
| augmented+rule_filter | 0.400 / 0.700 | 0.200 / 0.600 |
| vector+rule_filter+rewrite | 0.500 / 0.800 | 0.400 / 0.800 |

Cause: the corpus now holds TWO Microsoft 10-Ks. The filter
{ticker: MSFT, form: 10K} used to isolate one document; it now leaves
two, and three of the ten manual questions target Microsoft fiscal 2025.

Retrieval trace for "How much profit did Microsoft make in fiscal 2025?":
ranks 1-3 are all MSFT_10K_2026-07-29 (MD&A), the gold sits at rank 6.
The competing chunks are the SAME SECTION of the SAME DOCUMENT one year
later — 2026-07-29_0150 vs 2025-07-30_0149. Microsoft's MD&A keeps its
structure and wording year over year and changes only the figures, so
the two are near-identical lexically and semantically. No embedding
model can separate them from the question alone: "fiscal 2025" is a weak
semantic signal against wholesale textual similarity.

**The unstated assumption**: metadata filtering worked because the
corpus contained exactly one 10-K per company. That was never written
down anywhere — the ingestion pipeline broke it on its first cycle.

Fix (not implemented, scoped): add `fiscal_year` as a third filter
dimension, derived from filing content or period-of-report metadata, and
teach query_analysis to infer it. Half a day, touching chunking, filters
and the eval set.

Note: the one configuration that degraded least is query rewriting
(0.400/0.800), which we had rejected. Plausible mechanism — rewriting
into filing terminology raises the weight of specific terms and reduces
dependence on the filter. Not a verdict at n=10, but worth revisiting.

Broader point: this failure mode is invisible to static benchmarks.
Real