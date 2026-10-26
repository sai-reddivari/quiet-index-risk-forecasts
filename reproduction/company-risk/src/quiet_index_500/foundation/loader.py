"""Shared study loader: the single implementation of the price cleaning first written in notebooks/02_prices.ipynb.

Every later analysis reads prices, returns, factors and market cap through these functions, never through the raw
Yahoo or FMP files. Each input file is checked against the sha256 in its manifest before it is used.

Cleaning, in order (identical to notebook 02, which now calls this module)
1. Read the frozen 2026-10-12 Yahoo vintage by explicit name (adjusted close, split-adjusted close, volume). The
   2026-10-13 leftovers are never read.
2. Replace the columns of recovered tickers (WBD from FMP, PSKY from Yahoo under SKYD) with the supplement file.
3. Apply the same-security windows (same_security.py): sessions before history_start or after history_end are missing.
4. Zero-volume rule: a session with no trades is missing (inside the same-security window; outside it the window is
   the reason).
5. Corrections (apply_corrections=True): the extreme-event ledger, if present, is applied to the cleaned adjusted
   close (corrections.py). Returns are always recomputed from the corrected panel.
6. Daily simple total return r_t = P_t / P_t-1 - 1 on the adjusted close; missing whenever either price is missing,
   so missing sessions are never bridged.
7. Truncate to through= (default settings.research_price_cutoff, 2026-09-30). Corrections are applied before the
   truncation so a back adjustment is the same whatever the cutoff.

Every loader function takes through= and apply_corrections=; functions whose output does not depend on one of them
accept it so every call in the study reads the same way.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation.corrections import apply_extreme_event_ledger, empty_extreme_event_ledger, read_extreme_event_ledger
from quiet_index_500.foundation.manifests import verify_file_against_manifest
from quiet_index_500.foundation import same_security

# folders, read at call time so tests can point them at synthetic files
raw_data_folder = settings.raw_data_folder
processed_data_folder = settings.processed_data_folder
expected_universe_size = 503  # security records in the 30 Sep 2026 IVV snapshot

factor_columns = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]
gics_columns = ["gics_sector_code", "gics_sector", "gics_industry_group_code", "gics_industry_group",
                "gics_industry_code", "gics_industry", "gics_sub_industry_code", "gics_sub_industry"]


# ---------- file locations (the vintage is named explicitly, never "newest matching file") ----------

def universe_tickers_path():
    return processed_data_folder / f"universe_tickers_{settings.holdings_date}.csv"


def universe_weights_path():
    return processed_data_folder / f"universe_weights_{settings.holdings_date}.csv"


def ivv_equity_holdings_path():
    return processed_data_folder / f"ivv_equity_holdings_{settings.holdings_date}.csv"


def adjusted_close_path():
    return raw_data_folder / f"prices_adjusted_close_through_{settings.price_vintage_last_session}.csv"


def split_adjusted_close_path():
    return raw_data_folder / f"prices_close_split_adjusted_through_{settings.price_vintage_last_session}.csv"


def volume_path():
    return raw_data_folder / f"prices_volume_through_{settings.price_vintage_last_session}.csv"


def price_pull_manifest_path():
    return raw_data_folder / f"prices_through_{settings.price_vintage_last_session}.manifest.json"


def supplement_path():
    return raw_data_folder / f"prices_supplement_through_{settings.price_vintage_last_session}.csv"


def spy_prices_path():
    return raw_data_folder / f"prices_spy_through_{settings.price_vintage_last_session}.csv"


def smh_prices_path():
    return raw_data_folder / f"prices_smh_through_{settings.price_vintage_last_session}.csv"


def market_cap_raw_path():
    return raw_data_folder / f"market_cap_fmp_through_{settings.price_vintage_last_session}.csv"


def factors_path():
    return processed_data_folder / settings.ff5_mom_factors_file


# ---------- helpers ----------

def _through_timestamp(through):
    """The cutoff as a Timestamp; a cutoff after the frozen vintage is an error, not a silent truncation."""
    through_timestamp = pd.Timestamp(through)
    if through_timestamp > pd.Timestamp(settings.price_vintage_last_session):
        raise ValueError(f"through={through} is after the price vintage's last session {settings.price_vintage_last_session}")
    return through_timestamp


@lru_cache(maxsize=None)
def _read_verified_csv(path_text, manifest_path_text, index_column):
    """Read a CSV once per session after checking its fingerprint; callers receive copies."""
    verify_file_against_manifest(Path(path_text), Path(manifest_path_text) if manifest_path_text else None)
    if index_column is None:
        return pd.read_csv(path_text)
    return pd.read_csv(path_text, index_col=index_column, parse_dates=index_column == "date")


def _verified_csv(file_path, index_column=None, manifest_path=None):
    return _read_verified_csv(str(file_path), str(manifest_path) if manifest_path else None, index_column).copy()


def clear_cache():
    """Forget every file read in this session (after a file legitimately changes, or between tests)."""
    _read_verified_csv.cache_clear()
    _cleaned_full_vintage.cache_clear()


# ---------- universe, weights, identifiers ----------

def load_universe_tickers(through=settings.research_price_cutoff, apply_corrections=True):
    """The 503 snapshot tickers in the frozen order, as a tuple (fingerprint-checked against notebook 01's manifest)."""
    universe_tickers = tuple(_verified_csv(universe_tickers_path())["ticker"])  # a tuple cannot be edited, so the list stays constant
    assert len(universe_tickers) == expected_universe_size and len(set(universe_tickers)) == expected_universe_size
    return universe_tickers


def load_snapshot_weights(through=settings.research_price_cutoff, apply_corrections=True):
    """Precise snapshot weights (decimal, sum to 1) indexed by ticker, in universe order."""
    universe_weights = _verified_csv(universe_weights_path(), index_column="ticker")
    return universe_weights["weight"].reindex(list(load_universe_tickers()))  # same order as the price columns


def representative_security_by_ticker(cik_by_ticker, snapshot_weights):
    """For each ticker, its issuer's representative security: the class with the largest snapshot weight."""
    representative_by_cik = snapshot_weights.groupby(cik_by_ticker).idxmax()  # one ticker per CIK
    return cik_by_ticker.map(representative_by_cik)


def load_identifiers(through=settings.research_price_cutoff, apply_corrections=True):
    """One row per snapshot security: cik, issuer, snapshot weight, representative-security flag and the four GICS levels.

    GICS is the 30 Sep 2026 classification applied retrospectively. The representative security is fixed for the study
    (largest snapshot weight within the issuer: GOOGL, FOXA, NWSA; every other issuer has one class).
    """
    universe_tickers = list(load_universe_tickers())
    ivv_equity_holdings = _verified_csv(ivv_equity_holdings_path(), index_column="ticker").reindex(universe_tickers)
    snapshot_weights = load_snapshot_weights()
    cik_by_ticker = ivv_equity_holdings["cik"].astype("int64")
    representative_security = representative_security_by_ticker(cik_by_ticker, snapshot_weights)
    identifiers = pd.DataFrame({
        "cik": cik_by_ticker,
        "issuer_name": ivv_equity_holdings["issuer_name"],
        "ivv_name": ivv_equity_holdings["Name"],
        "snapshot_weight": snapshot_weights,
        "issuer_snapshot_weight": snapshot_weights.groupby(cik_by_ticker).transform("sum"),  # both classes together, for company-level weighting
        "share_classes_of_issuer": cik_by_ticker.map(cik_by_ticker.value_counts()),
        "representative_security": representative_security,
        "is_representative_security": representative_security == cik_by_ticker.index,
    }).rename_axis("ticker")
    return identifiers.join(ivv_equity_holdings[gics_columns])


# ---------- prices ----------

@dataclass
class PricePanels:
    """Cleaned price panels (date index, one column per ticker) and the masks that explain every missing value."""
    adjusted_close: pd.DataFrame          # splits and dividends; total-return basis; corrections applied when switched on
    split_adjusted_close: pd.DataFrame    # splits only; the price for market cap and dollar volume
    volume: pd.DataFrame                  # shares traded
    zero_volume_days: pd.DataFrame        # True where the zero-volume rule removed the session
    before_same_security_start: pd.DataFrame  # True before a security's same-security history starts
    after_history_end: pd.DataFrame       # True after a recorded history end (WBD after 2026-09-30)
    quarantined_returns: pd.DataFrame     # True where a ledger quarantine made the return missing
    quarantined_prices: pd.DataFrame      # True where a ledger quarantine made the adjusted close missing
    applied_corrections: pd.DataFrame     # one row per ledger action taken


def with_supplement(vendor_panel, supplement, field_name):
    """The Yahoo panel with each recovered ticker's column replaced by its supplementary series (notebook 02's rule)."""
    combined_panel = vendor_panel.copy()
    for ticker, rows in supplement.groupby("ticker"):
        combined_panel[ticker] = rows.set_index("date")[field_name].reindex(vendor_panel.index)
    return combined_panel


def clean_price_panels(adjusted_close_raw, split_adjusted_close_raw, volume_raw, supplement, rules):
    """Supplement, same-security windows and the zero-volume rule, applied to the downloaded panels.

    Returns a dict with adjusted_close, split_adjusted_close, volume, zero_volume_days, before_same_security_start and
    after_history_end, every one aligned to the downloaded sessions and tickers.
    """
    assert adjusted_close_raw.index.is_unique and adjusted_close_raw.index.is_monotonic_increasing  # one row per session, in date order
    assert adjusted_close_raw.index.equals(volume_raw.index) and adjusted_close_raw.index.equals(split_adjusted_close_raw.index)
    assert supplement["date"].isin(adjusted_close_raw.index).all()  # every supplementary date is a session of the main pull

    adjusted_close_vendor = with_supplement(adjusted_close_raw, supplement, "adjusted_close")
    split_adjusted_close_vendor = with_supplement(split_adjusted_close_raw, supplement, "close_split_adjusted")
    volume_vendor = with_supplement(volume_raw, supplement, "volume")

    rules_in_panel = rules[rules.index.isin(adjusted_close_vendor.columns)]  # a rule for a ticker outside the panel has nothing to act on
    before_same_security_start = pd.DataFrame(False, index=adjusted_close_vendor.index, columns=adjusted_close_vendor.columns)
    after_history_end = before_same_security_start.copy()
    for ticker, rule in rules_in_panel.iterrows():
        if pd.notna(rule["history_start"]):
            before_same_security_start[ticker] = adjusted_close_vendor.index < pd.Timestamp(rule["history_start"])
        if pd.notna(rule["history_end"]):
            after_history_end[ticker] = adjusted_close_vendor.index > pd.Timestamp(rule["history_end"])
    outside_same_security_history = before_same_security_start | after_history_end

    zero_volume_days = (volume_vendor == 0) & ~outside_same_security_history  # days outside the history are excluded for that reason instead
    return {
        "adjusted_close": adjusted_close_vendor.mask(zero_volume_days | outside_same_security_history),
        "split_adjusted_close": split_adjusted_close_vendor.mask(zero_volume_days | outside_same_security_history),
        "volume": volume_vendor.mask(zero_volume_days | outside_same_security_history),
        "zero_volume_days": zero_volume_days,
        "before_same_security_start": before_same_security_start,
        "after_history_end": after_history_end,
    }


@lru_cache(maxsize=None)
def _cleaned_full_vintage():
    """The cleaned panels over the whole 2026-10-12 vintage, before corrections and truncation (computed once)."""
    universe_tickers = load_universe_tickers()
    adjusted_close_raw = _verified_csv(adjusted_close_path(), index_column="date", manifest_path=price_pull_manifest_path())
    split_adjusted_close_raw = _verified_csv(split_adjusted_close_path(), index_column="date", manifest_path=price_pull_manifest_path())
    volume_raw = _verified_csv(volume_path(), index_column="date", manifest_path=price_pull_manifest_path())
    assert tuple(adjusted_close_raw.columns) == universe_tickers  # same tickers, same order as the frozen universe
    supplement = _verified_csv(supplement_path())
    supplement["date"] = pd.to_datetime(supplement["date"])
    return clean_price_panels(adjusted_close_raw, split_adjusted_close_raw, volume_raw, supplement, same_security.same_security_rules())


def load_price_panels(through=settings.research_price_cutoff, apply_corrections=True):
    """Cleaned adjusted close, split-adjusted close and volume with their masks, through the cutoff (PricePanels)."""
    through_timestamp = _through_timestamp(through)
    cleaned_panels = {name: panel.copy() for name, panel in _cleaned_full_vintage().items()}
    ledger = read_extreme_event_ledger() if apply_corrections else empty_extreme_event_ledger()
    corrected = apply_extreme_event_ledger(cleaned_panels["adjusted_close"], ledger)  # on the full vintage, then truncated
    within_cutoff = cleaned_panels["adjusted_close"].index <= through_timestamp
    return PricePanels(
        adjusted_close=corrected.adjusted_close[within_cutoff],
        split_adjusted_close=cleaned_panels["split_adjusted_close"][within_cutoff],
        volume=cleaned_panels["volume"][within_cutoff],
        zero_volume_days=cleaned_panels["zero_volume_days"][within_cutoff],
        before_same_security_start=cleaned_panels["before_same_security_start"][within_cutoff],
        after_history_end=cleaned_panels["after_history_end"][within_cutoff],
        quarantined_returns=corrected.quarantined_returns[within_cutoff],
        quarantined_prices=corrected.quarantined_prices[within_cutoff],
        applied_corrections=corrected.applied_corrections,
    )


def daily_simple_returns_from_prices(adjusted_close, quarantined_returns=None):
    """r_t = P_t / P_t-1 - 1 between consecutive sessions; missing on the first row, after any missing price and where quarantined."""
    daily_simple_returns = adjusted_close / adjusted_close.shift(1) - 1  # a missing price on either day leaves the return missing: no bridging
    if quarantined_returns is not None and quarantined_returns.to_numpy().any():
        daily_simple_returns = daily_simple_returns.mask(quarantined_returns)
    assert daily_simple_returns.min().min() > -1  # no return at or below -100%, which would mean a zero or negative price
    return daily_simple_returns


def load_daily_returns(through=settings.research_price_cutoff, apply_corrections=True):
    """Daily simple total returns of the 503 securities from the cleaned (and corrected) adjusted close."""
    price_panels = load_price_panels(through=through, apply_corrections=apply_corrections)
    return daily_simple_returns_from_prices(price_panels.adjusted_close, price_panels.quarantined_returns)


def _etf_daily_returns(prices_path, return_name, through):
    """Daily simple total return of one ETF from Yahoo's adjusted close, zero-volume sessions missing (notebook 02's rule)."""
    etf_prices_raw = _verified_csv(prices_path, index_column="date")
    stock_sessions = _cleaned_full_vintage()["adjusted_close"].index
    assert etf_prices_raw.index.equals(stock_sessions)  # the ETF and the stocks share every session
    etf_adjusted_close = etf_prices_raw["adjusted_close"].mask(etf_prices_raw["volume"] == 0)
    etf_daily_returns = (etf_adjusted_close / etf_adjusted_close.shift(1) - 1).rename(return_name)
    return etf_daily_returns[etf_daily_returns.index <= _through_timestamp(through)]


def load_spy_returns(through=settings.research_price_cutoff, apply_corrections=True):
    """SPY daily simple total return (missing on the first session); the benchmark for ordinary and downside beta."""
    return _etf_daily_returns(spy_prices_path(), "spy_return", through)


def load_smh_returns(through=settings.research_price_cutoff, apply_corrections=True):
    """SMH daily simple total return (missing on the first session); minus rf it is the SMH factor."""
    return _etf_daily_returns(smh_prices_path(), "smh_return", through)


# ---------- factors and excess returns ----------

def load_factors(through=settings.research_price_cutoff, apply_corrections=True):
    """French FF5 plus mom, daily, decimal, CRSP vintage 202608, date-indexed, rows on or before the cutoff."""
    factors = _verified_csv(factors_path(), index_column="date")[factor_columns]
    return factors[factors.index <= _through_timestamp(through)]


def factor_data_last_date(through=settings.research_price_cutoff, apply_corrections=True):
    """The factor cutoff read from the data: the last date of the factor file on or before through (2026-08-31)."""
    return load_factors(through=through).index.max()


def excess_returns(returns, risk_free_rate):
    """Returns minus rf on the same date, subtracted once; dates without an rf (after the factor cutoff) are missing."""
    risk_free_rate_aligned = risk_free_rate.reindex(returns.index)
    if isinstance(returns, pd.Series):
        return returns - risk_free_rate_aligned
    return returns.sub(risk_free_rate_aligned, axis=0)


def load_excess_returns(through=settings.research_price_cutoff, apply_corrections=True):
    """Daily excess returns of the 503 securities (missing after the factor cutoff)."""
    return excess_returns(load_daily_returns(through=through, apply_corrections=apply_corrections), load_factors(through=through)["rf"])


def load_spy_excess_returns(through=settings.research_price_cutoff, apply_corrections=True):
    """SPY excess return: the market for ordinary and downside beta."""
    return excess_returns(load_spy_returns(through=through), load_factors(through=through)["rf"]).rename("spy_excess_return")


def load_smh_excess_returns(through=settings.research_price_cutoff, apply_corrections=True):
    """SMH excess return: the SMH factor."""
    return excess_returns(load_smh_returns(through=through), load_factors(through=through)["rf"]).rename("smh_excess_return")


# ---------- market cap (licensed FMP data) ----------

def clean_market_cap(market_cap_raw, adjusted_close, rules):
    """Same-security market cap by ticker (notebook 02's rule) and the masks behind it.

    A value counts only on a day the security has a usable adjusted close, and never before market_cap_valid_from.
    Returns (market_cap, market_cap_rejected, market_cap_missing_while_priced).
    """
    assert market_cap_raw.index.isin(adjusted_close.index).all()  # no market-cap date outside the price sessions
    market_cap_vendor = market_cap_raw.reindex(adjusted_close.index)
    market_cap_share_basis_rejected = pd.DataFrame(False, index=market_cap_vendor.index, columns=market_cap_vendor.columns)
    for ticker, rule in rules[rules.index.isin(market_cap_vendor.columns)].iterrows():
        if pd.notna(rule["market_cap_valid_from"]):
            market_cap_share_basis_rejected[ticker] = market_cap_vendor.index < pd.Timestamp(rule["market_cap_valid_from"])
    market_cap = market_cap_vendor.where(adjusted_close.notna()).mask(market_cap_share_basis_rejected)
    market_cap_rejected = market_cap_vendor.notna() & market_cap.isna()            # predecessor or wrong-basis values removed
    market_cap_missing_while_priced = adjusted_close.notna() & market_cap.isna()   # excludes a formation from size-dependent analysis, never the returns
    return market_cap, market_cap_rejected, market_cap_missing_while_priced


def issuer_market_cap_from_tickers(market_cap, cik_by_ticker, snapshot_weights):
    """One market cap per issuer (columns are CIKs, ascending): the class with the larger snapshot weight carries it.

    FMP prices the issuer's full share count at each class's own price, so classes are never summed.
    """
    issuer_market_cap_columns = {}
    for cik, tickers_of_issuer in cik_by_ticker.groupby(cik_by_ticker):
        class_tickers = list(tickers_of_issuer.index)
        issuer_figure_ticker = snapshot_weights[class_tickers].idxmax()
        issuer_market_cap_columns[cik] = market_cap[issuer_figure_ticker]
    issuer_market_cap = pd.DataFrame(issuer_market_cap_columns).rename_axis(columns="cik")
    assert issuer_market_cap.shape[1] == cik_by_ticker.nunique()
    return issuer_market_cap


def _market_cap_raw():
    market_cap_raw = _verified_csv(market_cap_raw_path(), index_column="date")
    assert tuple(market_cap_raw.columns) == load_universe_tickers()
    return market_cap_raw


def load_market_cap_same_security(through=settings.research_price_cutoff, apply_corrections=True, nonpositive_as_missing=True):
    """Ticker-level FMP market cap (USD) on same-security days with a usable price.

    nonpositive_as_missing=True (the study default) sets zero or negative values missing for size measurement
    (HONA, 2026-06-26); the return on that day is kept. The descriptive processed table keeps them (False).
    A price quarantined by the ledger also removes that day's market cap, by the usable-price rule.
    """
    price_panels = load_price_panels(through=settings.price_vintage_last_session, apply_corrections=apply_corrections)
    market_cap, _, _ = clean_market_cap(_market_cap_raw(), price_panels.adjusted_close, same_security.same_security_rules())
    if nonpositive_as_missing:
        market_cap = market_cap.where(market_cap > 0)
    return market_cap[market_cap.index <= _through_timestamp(through)]


def load_issuer_market_cap(through=settings.research_price_cutoff, apply_corrections=True, nonpositive_as_missing=True):
    """One FMP market cap per issuer (columns are SEC CIKs), from the class with the larger snapshot weight."""
    market_cap = load_market_cap_same_security(through=through, apply_corrections=apply_corrections, nonpositive_as_missing=nonpositive_as_missing)
    identifiers = load_identifiers()
    return issuer_market_cap_from_tickers(market_cap, identifiers["cik"], identifiers["snapshot_weight"])
