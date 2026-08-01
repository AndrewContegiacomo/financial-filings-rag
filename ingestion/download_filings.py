"""
Downloads the latest 10-K and 10-Q filings from SEC EDGAR for a
configured set of companies.

EDGAR is a free, public API with two requirements we must honor:
  1. A User-Agent header identifying who is making the requests
     (loaded from .env — it's personal config, not code).
  2. A rate limit of max 10 requests/second.

Output: one HTML file per filing in data/raw/, named with the
convention TICKER_FORM_DATE.html so that downstream steps can recover
metadata from the filename alone.
"""
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Anchor paths to the project root rather than the working directory:
# these scripts run both directly and as subprocesses of the pipeline,
# and a relative path silently resolves differently depending on where
# the caller started.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
OUT_FILE = PROJECT_ROOT / "data/processed/chunks.json"

# Configuration 

# The SEC requires a real contact in the User-Agent so they can reach
# you if your script misbehaves. It lives in .env because it varies per
# person: anyone cloning this repo should plug in their OWN contact
# without touching the source code.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
if not SEC_USER_AGENT:
    # Fail fast at startup with an actionable message, instead of
    # letting the SEC reject us mid-run with an opaque 403.
    raise SystemExit(
        "Missing SEC_USER_AGENT in .env "
        "(format: 'Name Surname email@example.com'). "
        "The SEC requires it to identify API users."
    )

HEADERS = {"User-Agent": SEC_USER_AGENT}

# Companies were chosen across different sectors (tech hardware,
# software, banking, pharma) on purpose: a corpus where documents
# discuss different topics vs. overlapping ones (AAPL/MSFT) lets the
# evaluation phase observe retrieval behaving in both regimes.
# The CIK (Central Index Key) is the SEC's 10-digit company identifier.
COMPANIES = {
    "AAPL": "0000320193",   # Apple
    "MSFT": "0000789019",   # Microsoft
    "JPM":  "0000019617",   # JPMorgan Chase
    "PFE":  "0000078003",   # Pfizer
}

# One annual report (10-K) plus two quarterlies (10-Q) per company:
# multiple periods are what enables the agentic year-over-year /
# quarter-over-quarter comparisons planned for Phase 4.
WANTED = {"10-K": 1, "10-Q": 2}

OUT_DIR = Path("data/raw")
# ----------------------------------------------------------------------


def get_filings_index(cik: str) -> dict:
    """Fetch the master index of ALL filings for one company.

    This endpoint returns metadata only (types, dates, accession
    numbers) — not the documents themselves. From this index we build
    the URLs of the actual documents.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()  # turn HTTP errors into visible exceptions
    return resp.json()


def pick_filings(index: dict) -> list[dict]:
    """Select the N most recent filings of each wanted type.

    EDGAR returns the index in a COLUMNAR format: instead of a list of
    filing objects, it gives parallel lists ('form', 'accessionNumber',
    ...) all of the same length, already sorted newest-first. zip()
    stitches them back together row by row. This layout is a common
    API optimization to shrink JSON payloads.
    """
    recent = index["filings"]["recent"]
    picked, counts = [], {k: 0 for k in WANTED}

    for form, acc, doc, date in zip(
        recent["form"], recent["accessionNumber"],
        recent["primaryDocument"], recent["filingDate"],
    ):
        if form in WANTED and counts[form] < WANTED[form]:
            picked.append(
                {"form": form, "accession": acc, "document": doc, "date": date}
            )
            counts[form] += 1
        # Early exit once we have everything we need — no reason to
        # walk through years of remaining filing history.
        if all(counts[f] >= n for f, n in WANTED.items()):
            break
    return picked


def download_filing(cik: str, filing: dict, ticker: str) -> None:
    """Download the main HTML document of a single filing.

    The archive URL is built from the accession number (with dashes
    stripped) and the primary document name found in the index.
    """
    acc_clean = filing["accession"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_clean}/{filing['document']}"
    )
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    # Filename convention TICKER_FORM_DATE.html: the filename IS the
    # metadata store. The chunking step parses it to tag every chunk
    # with ticker/form/date, which ultimately powers source citations
    # in the RAG answers.
    fname = f"{ticker}_{filing['form'].replace('-', '')}_{filing['date']}.html"
    out_path = OUT_DIR / fname
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"  ✓ {fname}  ({len(resp.text)/1e6:.1f} MB)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in COMPANIES.items():
        print(f"\n{ticker} (CIK {cik})")
        index = get_filings_index(cik)
        for filing in pick_filings(index):
            download_filing(cik, filing, ticker)
            # Stay far below the SEC's 10 req/s limit. Overkill for 12
            # downloads, but the politeness is built in for when this
            # becomes an automated pipeline in Phase 6.
            time.sleep(0.2)


if __name__ == "__main__":
    main()