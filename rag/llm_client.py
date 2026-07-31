"""
Single entry point for Groq calls, with retry and graceful failure.

WHY CENTRALIZED: three modules (rag, tools, agent) call the API, and
each needed the same handling for transient errors — 429 (rate limit)
and 503 (model over capacity) both surfaced as stack traces in the UI.
Duplicating the retry logic three times would also mean three places to
touch when Phase 7 adds latency and token logging.

FAILURE MODE: returns None rather than raising, so callers can degrade
into a useful message instead of crashing the interface.
"""
import os
import time

from dotenv import load_dotenv
from groq import Groq, InternalServerError, RateLimitError, APIConnectionError

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

_client = None


def get_client() -> Groq:
    """Lazy client construction.

    Deliberately not a module-level constant: building the client at
    import time reads os.environ before the caller has had a chance to
    set it, which made the Streamlit secrets bridge order-dependent and
    fragile.
    """
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def call_llm(messages: list[dict], retries: int = 3, **kwargs):
    """Call the chat completions endpoint, retrying transient failures.

    Returns the message object on success, None on persistent failure.
    Accepts **kwargs so callers can pass tools/tool_choice without this
    helper needing to know about them.
    """
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=kwargs.pop("model", MODEL),
                messages=messages,
                temperature=kwargs.pop("temperature", 0.0),
                **kwargs,
            )
            return resp.choices[0].message
        except (RateLimitError, InternalServerError, APIConnectionError) as exc:
            if attempt == retries - 1:
                print(f"  [llm] giving up after {retries} attempts: "
                      f"{type(exc).__name__}")
                return None
            wait = 2 ** attempt * 5      # 5s, 10s, 20s
            print(f"  [llm] {type(exc).__name__}, retrying in {wait}s")
            time.sleep(wait)
    return None


def call_llm_text(prompt: str, **kwargs) -> str | None:
    """Convenience wrapper for single-prompt calls returning plain text."""
    msg = call_llm([{"role": "user", "content": prompt}], **kwargs)
    return msg.content.strip() if msg else None
