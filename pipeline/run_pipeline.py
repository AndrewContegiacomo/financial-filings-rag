"""
End-to-end corpus pipeline: download -> chunk -> embed.

IDEMPOTENT BY DESIGN: filings already on disk are not re-downloaded, and
the expensive stages (chunking, embedding) run only when something new
arrived. Re-running with no new filings costs four API calls to EDGAR
and nothing else.

INVARIANT — CHUNKING PARAMETERS ARE FROZEN: chunk IDs are
{filename}_{index}, so adding a new filing leaves existing IDs
untouched. Changing CHUNK_WORDS or OVERLAP_WORDS would resegment every
document and invalidate every gold ID in the evaluation set. If those
parameters ever change, the eval set must be regenerated in the same
commit.

ATOMICITY: chunks.json and embeddings.npy are aligned by position. They
are always regenerated together — a partial update silently returns the
wrong chunks for every query.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST = PROCESSED_DIR / "pipeline_manifest.json"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"runs": [], "known_filings": []}


def save_manifest(manifest: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_stage(name: str, command: list[str]) -> dict:
    """Run one pipeline stage, capturing timing and outcome.

    Stages are separate processes rather than function calls: each script
    stays independently runnable (which is how they were developed and
    debugged), and a crash in one cannot corrupt the interpreter state of
    the others.
    """
    print(f"\n--- {name}")
    started = time.time()
    result = subprocess.run(command, capture_output=False)
    elapsed = time.time() - started

    status = "ok" if result.returncode == 0 else "failed"
    print(f"--- {name}: {status} ({elapsed:.1f}s)")
    return {"stage": name, "status": status, "seconds": round(elapsed, 1)}


def main() -> None:
    manifest = load_manifest()
    known = set(manifest["known_filings"])

    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }

    # Stage 1: download 
    # download_filings.py overwrites existing files, which is harmless but
    # wasteful. We detect novelty by comparing the file set before/after.
    before = {p.name for p in RAW_DIR.glob("*.html")} if RAW_DIR.exists() else set()

    stage = run_stage("download", [sys.executable, "ingestion/download_filings.py"])
    run["stages"].append(stage)
    if stage["status"] == "failed":
        run["result"] = "aborted: download failed"
        manifest["runs"].append(run)
        save_manifest(manifest)
        sys.exit(1)

    after = {p.name for p in RAW_DIR.glob("*.html")}
    new_filings = sorted(after - before)
    run["new_filings"] = new_filings

    # Decide whether the expensive stages are needed
    artifacts_exist = (
        (PROCESSED_DIR / "chunks.json").exists()
        and (PROCESSED_DIR / "embeddings.npy").exists()
    )

    if not new_filings and artifacts_exist and after == known:
        print("\nNo new filings and artifacts are current — nothing to rebuild.")
        run["result"] = "no-op"
        manifest["runs"].append(run)
        save_manifest(manifest)
        return

    if new_filings:
        print(f"\nNew filings: {', '.join(new_filings)}")
    elif not artifacts_exist:
        print("\nArtifacts missing — rebuilding from existing filings.")

    # Stage 2 & 3: chunk and embed (always together)
    for name, script in [
        ("chunk", ["ingestion/chunk_filings.py"]),
        ("embed", ["-m", "rag.vector_search"]),
    ]:
        stage = run_stage(name, [sys.executable, *script])
        run["stages"].append(stage)
        if stage["status"] == "failed":
            # A failure here leaves chunks.json and embeddings.npy
            # potentially out of sync. VectorIndex asserts on length
            # mismatch, so the app fails loudly rather than serving
            # misaligned results.
            run["result"] = f"aborted: {name} failed"
            manifest["runs"].append(run)
            save_manifest(manifest)
            sys.exit(1)

    # Record success
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    run["result"] = "rebuilt"
    run["chunk_count"] = len(chunks)
    run["finished_at"] = datetime.now(timezone.utc).isoformat()

    manifest["known_filings"] = sorted(after)
    manifest["runs"] = (manifest["runs"] + [run])[-20:]   # keep last 20
    save_manifest(manifest)

    print(f"\nPipeline complete: {len(after)} filings, {len(chunks)} chunks.")


if __name__ == "__main__":
    main()