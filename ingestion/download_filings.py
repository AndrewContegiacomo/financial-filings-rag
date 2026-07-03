"""Download 10-K and 10-Q filings from SEC EDGAR for the configured companies."""
import json
import time
from pathlib import Path

import requests

import os
from dotenv import load_dotenv

load_dotenv()

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
if not SEC_USER_AGENT:
    raise SystemExit(
        "SEC_USER_AGENT is missing from the .env file "
        "(format: 'First Last email@example.com'). "
        "The SEC requires it to identify API users."
    )

HEADERS = {"User-Agent": SEC_USER_AGENT}

COMPANIES = {
    "AAPL": "0000320193",   # Apple
    "MSFT": "0000789019",   # Microsoft
    "JPM":  "0000019617",   # JPMorgan Chase
    "PFE":  "0000078003",   # Pfizer
}

# Number of recent filings per type
WANTED = {"10-K": 1, "10-Q": 2}

OUT_DIR = Path("data/raw")


def get_filings_index(cik: str) -> dict:
    #Download the full filing index for a company.
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pick_filings(index: dict) -> list[dict]:
    #From the index, select the N most recent filings for each requested type.
    recent = index["filings"]["recent"]
    picked, counts = [], {k: 0 for k in WANTED}
    # The lists in 'recent' are parallel and already sorted newest first.
    for form, acc, doc, date in zip(
        recent["form"], recent["accessionNumber"],
        recent["primaryDocument"], recent["filingDate"],
    ):
        if form in WANTED and counts[form] < WANTED[form]:
            picked.append(
                {"form": form, "accession": acc, "document": doc, "date": date}
            )
            counts[form] += 1
        if all(counts[f] >= n for f, n in WANTED.items()):
            break
    return picked


def download_filing(cik: str, filing: dict, ticker: str) -> None:
    #Download the main HTML document for a filing.
    acc_clean = filing["accession"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_clean}/{filing['document']}"
    )
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

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
            time.sleep(0.2)  # stay well below the SEC's 10 req/s limit


if __name__ == "__main__":
    main()