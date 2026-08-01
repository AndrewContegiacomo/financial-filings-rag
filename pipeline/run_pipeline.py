"""
End-to-end corpus pipeline: download -> chunk -> embed.

CORPUS MEMBERSHIP is defined by data/corpus_manifest.json, not by what
happens to be on disk. The expensive stages run only when the manifest
grows or when artifacts are missing, so a re-run with nothing new costs
four EDGAR index calls and nothing else.

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

# Anchor paths to the project root: this module runs via `python -m` from
# the repo root locally and from a different cwd in CI.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
CORPUS_MANIFEST = PROJECT_ROOT / "data/corpus_manifest.json"
RUN_LOG = PROCESSED_DIR / "pipeline_manifest.json"


def load_run_log() -> dict:
    if RUN_LOG.exists():
        return json.loads(RUN_LOG.read_text(encoding="utf-8"))
    return {"runs": []}


def save_run_log(log: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")


def count_manifest_filings() -> int:
    """Number of filings the corpus manifest declares.

    Growth here means genuinely new filings. The previous check compared
    files on disk before and after downloading, which reported all 14 as
    "new" on a clean checkout — correct locally, misleading in CI.
    """
    if not CORPUS_MANIFEST.exists():
        return 0
    return len(json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))["filings"])


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


def abort(log: dict, run: dict, reason: str) -> None:
    """Record a failed run and exit non-zero so CI reports the failure."""
    run["result"] = reason
    log["runs"].append(run)
    save_run_log(log)
    sys.exit(1)


def main() -> None:
    log = load_run_log()
    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }

    # Stage 1: discover and download 
    manifest_before = count_manifest_filings()

    stage = run_stage("download",
                      [sys.executable, "ingestion/download_filings.py"])
    run["stages"].append(stage)
    if stage["status"] == "failed":
        abort(log, run, "aborted: download failed")

    manifest_after = count_manifest_filings()
    new_count = manifest_after - manifest_before
    run["new_filings"] = new_count
    run["corpus_filings"] = manifest_after

    # Decide whether the expensive stages are needed
    artifacts_exist = (
        (PROCESSED_DIR / "chunks.json").exists()
        and (PROCESSED_DIR / "embeddings.npy").exists()
    )

    if new_count == 0 and artifacts_exist:
        print("\nNo new filings and artifacts are current — nothing to rebuild.")
        run["result"] = "no-op"
        log["runs"].append(run)
        save_run_log(log)
        return

    if new_count:
        print(f"\n{new_count} new filing(s) in manifest.")
    else:
        # Clean checkout: manifest unchanged but nothing built yet.
        print("\nArtifacts missing — rebuilding from manifest.")

    # Stages 2 & 3: chunk and embed (always together)
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
            abort(log, run, f"aborted: {name} failed")

    # Record success
    chunks = json.loads(
        (PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8")
    )
    run["result"] = "rebuilt"
    run["chunk_count"] = len(chunks)
    run["finished_at"] = datetime.now(timezone.utc).isoformat()

    log["runs"] = (log["runs"] + [run])[-20:]   # keep the last 20 runs
    save_run_log(log)

    print(f"\nPipeline complete: {manifest_after} filings, {len(chunks)} chunks.")


if __name__ == "__main__":
    main()