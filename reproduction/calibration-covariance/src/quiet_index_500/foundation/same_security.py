"""Same-security rules: where a ticker's downloaded history is, or is not, the security held on 30 Sep 2026.

Decided by Sai on 13 Oct 2026 and first written in notebooks/02_prices.ipynb: recover the same security where
feasible; otherwise exclude transparently, with no substitute company and no invented history. One row per
exception, with the evidence behind it. This module is now the single place the rules are written down; the
build writes them to data/processed/same_security_rules_2026-09-30.csv.

Columns
- history_start: first session that belongs to this security (earlier prices are a predecessor or pre-listing).
- history_end: last session kept (later prices are dropped).
- price_source: "yahoo" (the main pull), "yahoo:<symbol>" or "fmp:<symbol>" (a recovered history from the supplement file).
- market_cap_valid_from: first date on which the vendor market cap uses the right share basis.
- evidence: the facts behind the decision.
"""

import pandas as pd

same_security_rule_records = [
    {"ticker": "WBD", "history_start": None, "history_end": "2026-09-30", "price_source": "fmp:WBD", "market_cap_valid_from": "2022-04-11",
     "evidence": "One SEC registrant since 2008 (CIK 1437107: Discovery Communications, Discovery Inc., Warner Bros. Discovery); DISCA renamed WBD "
                 "on 11 Apr 2022 with no price break. Yahoo holds a single bar (5 Oct 2026), which FMP's close matches; FMP adjusted returns match "
                 "Yahoo's to 0.4bp a day on 25 test names. Kept through the snapshot date as Sai instructed. FMP market cap before the rename uses "
                 "the post-merger share count (180-335% above SEC filings) and is rejected."},
    {"ticker": "PSKY", "history_start": "2025-08-07", "history_end": None, "price_source": "yahoo:SKYD", "market_cap_valid_from": None,
     "evidence": "SEC CIK 2041610 became Paramount Skydance Corp on 7 Aug 2025 and Skydance Corp on 6 Oct 2026; FMP records PSKY renamed SKYD on "
                 "6 Oct 2026. Yahoo and FMP agree exactly from 7 Aug 2025. Earlier SKYD prices are Paramount Global's and are excluded."},
    {"ticker": "SW", "history_start": "2024-07-08", "history_end": None, "price_source": "yahoo", "market_cap_valid_from": None,
     "evidence": "New registrant (CIK 2005951, created Feb 2024); NYSE registration (8-A12B) filed 5 Jul 2024; last zero-volume day 5 Jul 2024. "
                 "Earlier Yahoo prices are not this security."},
    {"ticker": "AMCR", "history_start": "2019-06-11", "history_end": None, "price_source": "yahoo", "market_cap_valid_from": None,
     "evidence": "New registrant Amcor plc (CIK 1748790); NYSE registration filed 7 Jun 2019; last zero-volume day 10 Jun 2019. "
                 "Earlier Yahoo prices are not this security."},
    {"ticker": "FERG", "history_start": "2021-03-08", "history_end": None, "price_source": "yahoo", "market_cap_valid_from": None,
     "evidence": "Ferguson plc NYSE trading from 8 Mar 2021, exchanged 1:1 into Ferguson Enterprises (CIK 2011641) on 1 Aug 2024 under an SEC "
                 "successor filing (8-K12B), same ticker, no price break. Sai chose to keep the plc NYSE years; earlier prices are excluded."},
    {"ticker": "FOX", "history_start": "2019-03-19", "history_end": None, "price_source": "yahoo", "market_cap_valid_from": None,
     "evidence": "Fox Corp (CIK 1754301) regular-way trading from 19 Mar 2019; 12-18 Mar was when-issued. The symbol belonged to "
                 "21st Century Fox before March 2019. Sai chose regular-way."},
    {"ticker": "FOXA", "history_start": "2019-03-19", "history_end": None, "price_source": "yahoo", "market_cap_valid_from": None,
     "evidence": "As FOX."},
]


def same_security_rules():
    """The same-security rules as a ticker-indexed frame, built the same way notebook 02 built it (so its CSV is unchanged)."""
    return pd.DataFrame(same_security_rule_records).set_index("ticker")  # a fresh frame on every call, so no caller can edit the shared rules
