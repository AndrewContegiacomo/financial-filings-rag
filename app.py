"""
Streamlit chat interface for the filings RAG system.

STREAMLIT EXECUTION MODEL: the whole script re-runs on every
interaction. Two consequences shape this file:
  - expensive objects (embedding model, index) go behind
    @st.cache_resource so they load once, not per message;
  - conversation history lives in st.session_state, the only place
    values survive a re-run. Everything on screen is re-rendered from
    that list each time.
"""
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from monitoring.store import log_query, log_feedback

st.set_page_config(
    page_title="Filings Analyst",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_dotenv()

# st.secrets is Streamlit Cloud's mechanism; .env covers local runs.
# Order no longer matters for correctness — llm_client builds its client
# lazily on first use — but the key must be present before the first
# question is answered. Check the file exists first: touching st.secrets
# without one emits a warning even when the exception is handled.
if "GROQ_API_KEY" not in os.environ:
    secrets_paths = [
        Path(".streamlit/secrets.toml"),
        Path.home() / ".streamlit/secrets.toml",
    ]
    if any(p.exists() for p in secrets_paths):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


st.markdown("""
<style>
    /* Strip Streamlit's default chrome for a cleaner, app-like surface */
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 3rem;
        padding-bottom: 7rem;
        max-width: 46rem;
    }

    /* Flatten chat bubbles: plain text blocks read better for long,
       citation-heavy answers than high-contrast bubbles */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.75rem 0;
    }

    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .app-subtitle {
        color: #71767d;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    /* Small monospace tag for retrieval metadata under an answer */
    .meta {
        color: #8b9199;
        font-size: 0.78rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        margin-top: 0.4rem;
    }

    [data-testid="stExpander"] details {
        border: 1px solid rgba(130,140,150,0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Cached resources -------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_index():
    """Augmented retrieval: dense results with a few keyword results
    interleaved. Chosen over dense alone for +15 points hit@5 on
    figure-bearing questions, at a cost of ~5 points on narrative ones.
    """
    from rag.vector_search import VectorIndex
    from rag.search import load_chunks, build_index
    from rag.augmented_search import AugmentedIndex
    return AugmentedIndex(VectorIndex(), build_index(load_chunks()))


@st.cache_resource(show_spinner=False)
def load_modules():
    from rag import rag, agent
    return rag, agent


# Module-level, not session_state: @st.cache_resource is shared across
# browser sessions, so "has this process loaded the model yet" is a
# process-level fact. Using session_state marked a reloaded page as cold
# when the model was already in memory, mixing the two populations in
# the latency chart.
_WARM = {"value": False}


# --- Routing ----------------------------------------------------------

COMPARATIVE_TERMS = [
    "grow", "growth", "grew", "increase", "decrease", "decline",
    "change", "compare", "comparison", "versus", " vs ", "difference",
    "year over year", "yoy", "higher than", "lower than", "more than",
]


def looks_analytical(question: str) -> bool:
    """Route between the RAG path and the agentic tools.

    Kept deliberately simple, and overridable from the UI. A router is
    needed because sending a narrative question to the agent with
    tool_choice='auto' risks the model answering without calling any
    tool — i.e. from parametric knowledge with no filings context, the
    exact failure this system exists to prevent.

    Known false positive: "What caused the increase in operating
    expenses?" is a causal, narrative question that this heuristic sends
    to the tools.
    """
    q = question.lower()
    return any(term in q for term in COMPARATIVE_TERMS)


# --- Rendering helpers ------------------------------------------------

def render_answer(text: str) -> None:
    """Escape dollar signs before markdown rendering.

    st.markdown treats paired '$' as LaTeX math delimiters, so an answer
    containing two figures silently swallows everything between them.
    Financial answers contain dollar amounts constantly.
    """
    st.markdown(text.replace("$", r"\$"))


def render_sources(chunks: list[dict]) -> None:
    """One expander for all passages, rather than one per passage —
    ten stacked expanders bury the answer they are meant to support."""
    with st.expander(f"Sources ({len(chunks)})"):
        for i, c in enumerate(chunks, start=1):
            section = c["section"] or "section n/a"
            st.markdown(
                f"**{i}. {c['ticker']} {c['form']} {c['date']}** · {section}"
            )
            st.markdown(
                f"<div style='color:#6b7178; font-size:0.86rem; "
                f"line-height:1.5'>{c['text']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='meta'>{c['id']}</div>", unsafe_allow_html=True
            )
            if i < len(chunks):
                st.divider()


def render_feedback(query_id: int) -> None:
    """Rating buttons for one answer.

    Keyed by query_id so every answer in the conversation keeps its own
    buttons across re-runs — Streamlit redraws the whole page on each
    interaction, and widgets sharing a key would collide.
    """
    col1, col2, _ = st.columns([1.2, 1.2, 6])
    if col1.button("Good answer", key=f"up_{query_id}",
                   use_container_width=True):
        log_feedback(query_id, 1)
        st.toast("Thanks for the feedback")
    if col2.button("Bad answer", key=f"down_{query_id}",
                   use_container_width=True):
        log_feedback(query_id, -1)
        st.toast("Thanks for the feedback")


# --- Header -----------------------------------------------------------

st.markdown('<div class="app-title">Filings Analyst</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Questions about SEC filings for Apple, '
    'Microsoft, JPMorgan Chase and Pfizer. Every answer is grounded in '
    'the filings and cited.</div>',
    unsafe_allow_html=True,
)


# --- Conversation state -----------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history: session_state is the source of truth, the screen is
# just its projection.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"],
                         avatar=":material/person:" if msg["role"] == "user"
                         else ":material/analytics:"):
        render_answer(msg["content"])
        if msg.get("meta"):
            st.markdown(f"<div class='meta'>{msg['meta']}</div>",
                        unsafe_allow_html=True)
        if msg.get("chunks"):
            render_sources(msg["chunks"])
        if msg.get("query_id"):
            render_feedback(msg["query_id"])


# --- Empty state ------------------------------------------------------

if not st.session_state.messages:
    st.markdown(
        "<div style='color:#8b9199; font-size:0.86rem; margin-bottom:0.6rem'>"
        "Try asking</div>", unsafe_allow_html=True
    )
    examples = [
        "What lawsuits is Pfizer facing over its former heartburn medication?",
        "How could new U.S. import duties affect Apple's business?",
        "How much did Apple's net sales grow from fiscal 2024 to fiscal 2025?",
    ]
    for ex in examples:
        # Buttons write into session_state and trigger a re-run, which is
        # how Streamlit turns a click into "the user asked this".
        if st.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
            st.session_state.pending = ex
            st.rerun()


# --- Input ------------------------------------------------------------

typed = st.chat_input("Ask about the filings")

# Routing is heuristic and therefore fallible; an explicit override keeps
# the user in control without putting a mode selector in front of every
# question.
force_tools = st.toggle(
    "Force analytical tools",
    help="Look up figures individually and compute any arithmetic in code. "
         "Used automatically for questions about growth or differences.",
)

question = typed or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        render_answer(question)

    with st.chat_message("assistant", avatar=":material/analytics:"):
        # Cold start is tracked explicitly: the first query of a process
        # pays for loading the embedding model and building the keyword
        # index, and averaging it with warm queries makes the latency
        # figure meaningless.
        cold = not _WARM["value"]

        try:
            rag_mod, agent_mod = load_modules()
            index = load_index()
            _WARM["value"] = True
        except FileNotFoundError:
            st.error(
                "Corpus not found. Build it with:\n\n"
                "```\npython -m pipeline.run_pipeline\n```"
            )
            st.stop()

        error = None
        if force_tools or looks_analytical(question):
            path = "agent"
            t0 = time.perf_counter()
            with st.spinner("Looking up figures"):
                text = agent_mod.run(question, verbose=False)
            gen_ms = int((time.perf_counter() - t0) * 1000)
            retrieval_ms, chunks, filters = 0, None, {}
            meta = ("analytical tools · figures looked up individually, "
                    "arithmetic in code")
        else:
            path = "rag"
            t0 = time.perf_counter()
            with st.spinner("Searching filings"):
                result = rag_mod.answer(index, question)
            gen_ms = int((time.perf_counter() - t0) * 1000)
            text = result["answer"]
            chunks = result["chunks"]
            filters = result["filters_applied"]
            retrieval_ms = result.get("retrieval_ms", 0)
            scope = (" · " + " ".join(f"{k}={v}" for k, v in filters.items())
                     if filters else "")
            meta = f"{len(chunks)} passages retrieved{scope}"

        # A refusal is a legitimate outcome, not a failure — but the rate
        # matters: a system that refuses too often is as useless as one
        # that fabricates.
        refused = "do not contain" in text or "does not contain" in text

        query_id = log_query(
            question=question, path=path, filters=filters,
            retrieval_ms=retrieval_ms, generation_ms=gen_ms,
            cold_start=cold, n_chunks=len(chunks) if chunks else 0,
            refused=refused, error=error,
        )

        render_answer(text)
        st.markdown(f"<div class='meta'>{meta}</div>", unsafe_allow_html=True)
        if chunks:
            render_sources(chunks)
        render_feedback(query_id)

    st.session_state.messages.append({
        "role": "assistant", "content": text, "meta": meta,
        "chunks": chunks, "query_id": query_id,
    })