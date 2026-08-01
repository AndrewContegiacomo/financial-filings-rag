"""
Downloads the filings listed in the corpus manifest, and discovers new
ones from SEC EDGAR.

WHY A MANIFEST: the corpus used to be defined implicitly as "the N most
recent filings per form, plus whatever happened to be on disk". That
worked locally, where old files accumulated, and broke the first time
the pipeline ran on a clean checkout: the runner downloaded only the
latest N, silently dropping older filings that the evaluation set
references.

The manifest makes membership explicit and versioned. Filings are keyed
by SEC accession number — unique, immutable, and independent of how
recent they are. Discovery ADDS to the manifest and never removes:
a filing that once belonged to the corpus stays in it, so gold chunk IDs
remain valid as new filings arrive.

The manifest also records reportDate (the period covered) alongside
filingDate (when it was submitted). These differ by months and only the
former identifies the fiscal year — Microsoft's FY2026 10-K was filed in
July 2026 for a year ended June 2026, Apple's FY2025 10-K in October
2025 for a year ended September 2025.
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
MANIFEST_FILE = PROJECT_ROOT / "data/corpus_manifest.json"

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
if not SEC_USER_AGENT:
    raise SystemExit(
        "Missing SEC_USER_AGENT in .env "
        "(format: 'Name Surname email@example.com'). "
        "The SEC requires it to identify API users."
    )

HEADERS = {"User-Agent": SEC_USER_AGENT}

COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "JPM":  "0000019617",
    "PFE":  "0000078003",
}

# How many recent filings of each type to DISCOVER. This bounds growth
# per run; it no longer defines the corpus, which is the manifest.
DISCOVER = {"10-K": 1, "10-Q": 2}


def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {"filings": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")


def get_filings_index(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def discover(ticker: str, cik: str, manifest: dict) -> list[str]:
    """Add newly published filings to the manifest. Returns added keys.

    EDGAR returns its index in columnar form — parallel lists, newest
    first — so zip() reassembles them row by row.
    """
    index = get_filings_index(cik)
    recent = index["filings"]["recent"]
    counts = {k: 0 for k in DISCOVER}
    added = []

    for form, acc, doc, filed, reported in zip(
        recent["form"], recent["accessionNumber"],
        recent["primaryDocument"], recent["filingDate"],
        recent["reportDate"],
    ):
        if form not in DISCOVER or counts[form] >= DISCOVER[form]:
            continue
        counts[form] += 1

        if acc in manifest["filings"]:
            continue

        manifest["filings"][acc] = {
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "accession": acc,
            "document": doc,
            "filing_date": filed,
            "report_date": reported,   # the period covered, not the submission
        }
        added.append(acc)

        if all(counts[f] >= n for f, n in DISCOVER.items()):
            break

    return added


def filename_for(entry: dict) -> str:
    """TICKER_FORM_FILINGDATE.html — the convention chunking parses.

    Kept unchanged despite report_date now being available: chunk IDs
    derive from this filename, and changing it would invalidate every
    gold ID in the evaluation set.
    """
    form = entry["form"].replace("-", "")
    return f"{entry['ticker']}_{form}_{entry['filing_date']}.html"


def download(entry: dict) -> bool:
    """Fetch a filing unless it is already on disk. Returns True if
    downloaded."""
    out_path = RAW_DIR / filename_for(entry)
    if out_path.exists():
        return False

    acc_clean = entry["accession"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(entry['cik'])}/{acc_clean}/{entry['document']}"
    )
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    out_path.write_text(resp.text, encoding="utf-8")
    print(f"  ↓ {out_path.name}  ({len(resp.text)/1e6:.1f} MB)")
    return True


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    before = len(manifest["filings"])

    print("Discovering new filings...")
    for ticker, cik in COMPANIES.items():
        added = discover(ticker, cik, manifest)
        if added:
            for acc in added:
                e = manifest["filings"][acc]
                print(f"  + {ticker} {e['form']} filed {e['filing_date']} "
                      f"(period {e['report_date']})")
        time.sleep(0.2)   # stay well under the SEC's 10 req/s limit

    save_manifest(manifest)

    print(f"\nManifest: {len(manifest['filings'])} filings "
          f"({len(manifest['filings']) - before} new)")

    print("\nDownloading missing files...")
    fetched = 0
    for entry in manifest["filings"].values():
        if download(entry):
            fetched += 1
            time.sleep(0.2)

    print(f"{fetched} downloaded, "
          f"{len(manifest['filings']) - fetched} already present.")


if __name__ == "__main__":
    main()