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
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

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
        try:
            rag_mod, agent_mod = load_modules()
            index = load_index()
        except FileNotFoundError:
            st.error(
                "Corpus not found. Build it with:\n\n"
                "```\npython ingestion/download_filings.py\n"
                "python ingestion/chunk_filings.py\n"
                "python -m rag.vector_search\n```"
            )
            st.stop()

        if force_tools or looks_analytical(question):
            with st.spinner("Looking up figures"):
                text = agent_mod.run(question, verbose=False)
            meta = ("analytical tools · figures looked up individually, "
                    "arithmetic in code")
            chunks = None
        else:
            with st.spinner("Searching filings"):
                result = rag_mod.answer(index, question)
            text = result["answer"]
            chunks = result["chunks"]
            # Filters are hard constraints: a wrong inference makes the
            # answer unreachable, so the user should be able to see when
            # the search was narrowed.
            applied = result["filters_applied"]
            scope = (" · " + " ".join(f"{k}={v}" for k, v in applied.items())
                     if applied else "")
            meta = f"{len(chunks)} passages retrieved{scope}"

        render_answer(text)
        st.markdown(f"<div class='meta'>{meta}</div>", unsafe_allow_html=True)
        if chunks:
            render_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant", "content": text, "meta": meta, "chunks": chunks,
    })