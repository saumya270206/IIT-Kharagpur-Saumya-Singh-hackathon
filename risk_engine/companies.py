"""Company reference data used for entity resolution (text -> ticker)."""

from __future__ import annotations

import re
from typing import Optional

# ticker -> canonical name + aliases that may appear in free text
COMPANIES: dict[str, dict[str, object]] = {
    "AAPL": {"name": "Apple", "aliases": ["apple", "iphone", "tim cook", "app store"]},
    "MSFT": {"name": "Microsoft", "aliases": ["microsoft", "azure", "windows", "copilot"]},
    "NVDA": {"name": "NVIDIA", "aliases": ["nvidia", "geforce", "cuda", "jensen huang"]},
    "AMZN": {"name": "Amazon", "aliases": ["amazon", "aws", "amazon web services", "prime"]},
    "GOOG": {"name": "Alphabet", "aliases": ["alphabet", "google", "gmail", "waymo"]},
    "NFLX": {"name": "Netflix", "aliases": ["netflix"]},
    "PYPL": {"name": "PayPal", "aliases": ["paypal", "venmo"]},
    "DIS": {"name": "Walt Disney", "aliases": ["walt disney", "disney", "disney+", "espn"]},
    "BA": {"name": "Boeing", "aliases": ["boeing", "737 max"]},
    "COST": {"name": "Costco", "aliases": ["costco"]},
    "INTC": {"name": "Intel", "aliases": ["intel corp", "intel"]},
    "TSM": {"name": "Taiwan Semiconductor", "aliases": ["taiwan semiconductor", "tsmc"]},
    "META": {"name": "Meta Platforms", "aliases": ["meta platforms", "facebook", "instagram", "threads"]},
    "TSLA": {"name": "Tesla", "aliases": ["tesla", "autopilot", "cybertruck", "elon musk"]},
    "JPM": {"name": "JPMorgan Chase", "aliases": ["jpmorgan", "j p morgan", "jp morgan", "chase bank"]},
    "BAC": {"name": "Bank of America", "aliases": ["bank of america", "merrill lynch"]},
    "WMT": {"name": "Walmart", "aliases": ["walmart", "sam's club", "sams club"]},
    "XOM": {"name": "Exxon Mobil", "aliases": ["exxon mobil", "exxonmobil", "exxon", "esso"]},
    "JNJ": {"name": "Johnson & Johnson", "aliases": ["johnson & johnson", "johnson and johnson", "j&j"]},
    "PG": {"name": "Procter & Gamble", "aliases": ["procter & gamble", "procter and gamble", "p&g"]},
    "UNH": {"name": "UnitedHealth Group", "aliases": ["unitedhealth", "united health", "optum"]},
    "V": {"name": "Visa", "aliases": ["visa inc", "visa card"]},
    "ORCL": {"name": "Oracle", "aliases": ["oracle", "java"]},
    "CRM": {"name": "Salesforce", "aliases": ["salesforce", "slack"]},
    "AMD": {"name": "Advanced Micro Devices", "aliases": ["advanced micro devices", "amd", "ryzen"]},
    "KO": {"name": "Coca-Cola", "aliases": ["coca-cola", "coca cola", "coke"]},
    "PFE": {"name": "Pfizer", "aliases": ["pfizer", "comirnaty"]},
}

_CASH_TAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_TICKER_ALIAS = {"GOOGL": "GOOG"}   # same company, other share class
_PATTERNS: dict[str, re.Pattern[str]] = {
    ticker: re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in sorted(
            [str(meta["name"]).lower(), *[str(a) for a in meta["aliases"]]], key=len, reverse=True)
        ) + r")(?![a-z0-9])",
        re.IGNORECASE,
    )
    for ticker, meta in COMPANIES.items()
}


def match_company(text: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve a ticker from cashtags or company names/aliases in free text.

    Returns (ticker, company_name) or (None, None) when nothing matches.
    Cashtags win over name matching (explicit intent).
    """
    for m in _CASH_TAG_RE.finditer(text):
        cand = _TICKER_ALIAS.get(m.group(1).upper(), m.group(1).upper())
        if cand in COMPANIES:
            return cand, str(COMPANIES[cand]["name"])

    # longest aliases first so "bank of america" beats a hypothetical "bank"
    best: tuple[int, Optional[str]] = (-1, None)
    for ticker, pat in _PATTERNS.items():
        m = pat.search(text)
        if m and len(m.group(0)) > best[0]:
            best = (len(m.group(0)), ticker)
    if best[1]:
        return best[1], str(COMPANIES[best[1]]["name"])
    return None, None
