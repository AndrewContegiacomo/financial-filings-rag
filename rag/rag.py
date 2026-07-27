"""End-to-end RAG flow: question -> retrieve -> prompt -> LLM -> answer."""
import os

from dotenv import load_dotenv
from groq import Groq

from rag.search import load_chunks, build_index, keyword_search
from rag.vector_search import VectorIndex
from rag.query_analysis import infer_filters

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

TOP_K = 10  # number of chunks to retrieve for context

# Three key clauses:
# 1. answer ONLY from the provided context (hallucination guard)
# 2. cite sources using the metadata we carried through the pipeline
# 3. say "I don't know" when the context is insufficient — for financial
#    data, a wrong number is far worse than no number.
PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the QUESTION
using only the information in the CONTEXT below, extracted from SEC filings.

Rules:
- Base your answer strictly on the CONTEXT. Do not use outside knowledge.
- After each claim, cite the source in brackets: [TICKER FORM DATE, SECTION].
- If the CONTEXT does not contain the answer, say so explicitly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_context(results: list[dict]) -> str:
    """Serialize retrieved chunks, prefixing each with its source metadata
    so the model can produce the citations the prompt demands."""
    blocks = []
    for r in results:
        source = f"[{r['ticker']} {r['form']} {r['date']}, {r['section'] or 'n/a'}]"
        blocks.append(f"{source}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(vec_index: VectorIndex, question: str) -> dict:
    """Answer a question over the filings corpus.

    Dense retrieval is the default: evaluation on hand-written questions
    (the ones phrased without borrowing the filings' vocabulary) showed
    keyword search failing to surface the gold at all on 3 of 10, while
    dense retrieval ranked all of them within the top 12. The opposite
    ordering seen on synthetic questions was an artifact of those
    questions inheriting their source chunk's wording.

    Filters are inferred from the question (no LLM), which closes about
    half the gap to an oracle that knows the answer's metadata.

    k=10 rather than 5: 10 chunks x 200 words is ~2k words of context,
    negligible for a 128k-context model, and hit@10 runs 10-13 points
    above hit@5 across configurations.
    """
    filters = infer_filters(question)
    results = vec_index.search(question, filters or None, TOP_K)
    prompt = PROMPT_TEMPLATE.format(
        context=format_context(results), question=question
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [r["id"] for r in results],
        "filters_applied": filters,
    }


if __name__ == "__main__":
    print("Loading index...")
    index = VectorIndex()
    print("Ready. Empty line to quit.\n")
    while True:
        q = input("Question> ").strip()
        if not q:
            break
        result = answer(index, q)
        print("\n" + result["answer"])
        if result["filters_applied"]:
            print("\nFilters inferred:", result["filters_applied"])
        print("Retrieved chunks:", ", ".join(result["sources"]), "\n")