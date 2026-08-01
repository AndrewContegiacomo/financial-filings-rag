"""Dense (embedding-based) retrieval over the chunked corpus.

Complements keyword search: TF-IDF is blind to synonyms ("revenue" vs
"net sales" — see the Q1 case study), while embeddings map semantically
close text to nearby vectors regardless of exact wording.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data/processed/chunks.json"
EMB_FILE = PROJECT_ROOT / "data/processed/embeddings.npy"

# Small, fast, runs locally on Apple Silicon. Its 256-token input limit
# is what dictated our 200-word chunk size back in the chunking phase.
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy singleton: loading the model takes seconds, do it once


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_embeddings() -> None:
    """Embed every chunk and persist the matrix to disk.

    Run once after (re)building the corpus. Row i of the matrix is the
    embedding of chunks[i] — alignment by position, which is why chunk
    order must be deterministic (it is: files are sorted, IDs are stable).
    """
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    model = get_model()
    # normalize_embeddings=True -> unit-length vectors, so cosine
    # similarity later reduces to a plain dot product.
    emb = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(EMB_FILE, emb)
    print(f"{emb.shape[0]} embeddings ({emb.shape[1]} dims) -> {EMB_FILE}")


class VectorIndex:
    def __init__(self):
        if not CHUNKS_FILE.exists() or not EMB_FILE.exists():
            raise FileNotFoundError(
                f"Corpus artifacts missing ({CHUNKS_FILE}, {EMB_FILE}). "
                "Run: python -m pipeline.run_pipeline"
            )
        self.chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.emb = np.load(EMB_FILE)
        assert len(self.chunks) == self.emb.shape[0], (
            "chunks.json and embeddings.npy are out of sync — "
            "re-run the pipeline after changing the corpus"
        )

    def search(self, query: str, filters: dict | None = None,
               num_results: int = 5) -> list[dict]:
        q = get_model().encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q  # cosine similarity, all chunks at once

        # Hard metadata constraints applied BEFORE ranking, by pushing
        # excluded chunks below any achievable score.
        if filters:
            mask = np.array([
                any(c[field] != value for field, value in filters.items())
                for c in self.chunks
            ])
            scores = scores.copy()
            scores[mask] = -1.0

        top = np.argsort(scores)[::-1][:num_results]
        return [self.chunks[i] for i in top]


if __name__ == "__main__":
    build_embeddings()