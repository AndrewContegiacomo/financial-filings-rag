"""
Cleans the raw HTML filings downloaded from SEC EDGAR and splits them
into fixed-size, overlapping text chunks enriched with metadata.

Output: a single JSON file (data/processed/chunks.json) where each chunk has:
  - a deterministic ID (stable across re-runs, required by the eval set later)
  - ticker / form type / filing date (parsed from the filename)
  - the 10-K/10-Q "Item" section it belongs to, when detectable
  - the chunk text itself
"""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

# Anchor paths to the project root rather than the working directory:
# these scripts run both directly and as subprocesses of the pipeline,
# and a relative path silently resolves differently depending on where
# the caller started.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
OUT_FILE = PROJECT_ROOT / "data/processed/chunks.json"

# Chunk size is dictated by the embedding model:
# all-MiniLM-L6-v2 truncates input at 256 tokens (~200 English words).
# Anything beyond that limit would be silently ignored by vector search,
# so chunks larger than ~200 words would have an "invisible" second half.
CHUNK_WORDS = 200

# Consecutive chunks share their last/first 40 words. This prevents
# sentences that fall exactly on a chunk boundary from being split in a
# way that makes them unretrievable. Costs some storage, buys semantic
# continuity.
OVERLAP_WORDS = 40

# Item numbers in 10-K/10-Q filings run from 1 to 16. Accepting any
# two-digit number produced bogus sections like "Item 85".
ITEM_RE = re.compile(r"\bITEM\s+(1[0-6]|[1-9])(A?)\b[.\s]", re.IGNORECASE)


def html_to_text(path: Path) -> str:
    """Extract plain text from a filing's HTML.

    Design choice: we strip scripts/styles but deliberately KEEP table
    text, even though get_text() linearizes it poorly. Key financial
    figures (revenue, net income) often live in tables — ugly text that
    is present beats clean text that is missing.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    # Remove elements that contain no human-readable content.
    for tag in soup(["script", "style", "head"]):
        tag.decompose()  # removes the tag AND its content from the tree

    # separator=" " avoids gluing words together when adjacent HTML tags
    # have no whitespace between them (e.g. "</td><td>").
    text = soup.get_text(separator=" ")

    # Collapse runs of whitespace/newlines into single spaces. Filings
    # are full of layout-driven whitespace that carries no meaning.
    return re.sub(r"\s+", " ", text).strip()


def parse_filename(path: Path) -> dict:
    """Recover metadata encoded in the filename by the download script.

    Filenames follow the convention TICKER_FORM_DATE.html
    (e.g. AAPL_10K_2025-10-31.html), so the filename itself is our
    metadata store — no extra lookup needed.
    """
    ticker, form, date = path.stem.split("_")
    return {"ticker": ticker, "form": form, "date": date}


def chunk_text(text: str) -> list[dict]:
    """Split text into overlapping fixed-size chunks, tagging each one
    with the 10-K "Item" section it falls under (when detectable).

    Fixed-size chunking was chosen over purely structural (per-section)
    chunking because it is robust: it works on any document regardless
    of how each company formats its HTML. Section detection is layered
    on top as metadata only.
    """
    words = text.split()
    chunks = []

    # Tracks the most recently seen section header. Carried forward as
    # state: a chunk belongs to the last section that started before it.
    current_item = None

    # Each new chunk starts OVERLAP_WORDS before the previous one ended.
    step = CHUNK_WORDS - OVERLAP_WORDS

    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + CHUNK_WORDS])

        # If one or more section headers appear inside this chunk, the
        # LAST one is the section the following text belongs to.
        matches = ITEM_RE.findall(piece)
        if matches:
            # ITEM_RE has two capture groups (number, optional 'A'), so
            # findall returns tuples like ('1', 'A') — join them back
            # into "1A" before formatting.
            current_item = f"Item {''.join(matches[-1]).upper()}"

        chunks.append({"text": piece, "section": current_item})

        # Stop once the window has reached the end of the document
        if start + CHUNK_WORDS >= len(words):
            break

    return chunks


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    files = sorted(RAW_DIR.glob("*.html"))
    if not files:
        # Fail fast with an actionable message instead of producing an
        # empty output file that would break downstream steps silently.
        raise SystemExit("No files in data/raw/ — run download_filings.py first")

    for path in tqdm(files, desc="Chunking"):
        meta = parse_filename(path)
        text = html_to_text(path)

        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append(
                {
                    # Deterministic ID: same corpus in -> same IDs out.
                    # The evaluation set (Phase 3) will reference these
                    # IDs, so they must be stable across re-runs.
                    "id": f"{path.stem}_{i:04d}",
                    "ticker": meta["ticker"],
                    "form": meta["form"],
                    "date": meta["date"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                }
            )

    # Single JSON file for the whole corpus: downstream steps (indexing,
    # evaluation) load everything with one call. At a few thousand
    # chunks this is a few MB — fine. Revisit only if the corpus grows
    # by orders of magnitude.
    OUT_FILE.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\n{len(all_chunks)} chunks -> {OUT_FILE}")


if __name__ == "__main__":
    main()