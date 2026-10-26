"""Regenerate every processed table notebooks/02_prices.ipynb writes, using the shared loader.

These tables are descriptive and cover the whole frozen vintage (through 2026-10-12), as they did before the loader
existed; the study loader itself defaults to the 2026-09-30 research cutoff. Each writer returns the table it wrote
and leaves a manifest beside it naming every input and the source code that built it.

Tables (in data/processed unless output_folder says otherwise)
- same_security_rules_2026-09-30.csv
- snapshot_portfolio_daily_returns_through_<vintage>.csv
- spy_daily_returns_through_<vintage>.csv, smh_daily_returns_through_<vintage>.csv
- portfolio_exclusions_through_<vintage>.csv
- ticker_data_flags_through_<vintage>.csv (extended with market_cap_nonpositive_days and quarantined_return_days)
- market_cap_same_security_through_<vintage>.csv, issuer_market_cap_through_<vintage>.csv (licensed: CSVs git-ignored)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import corrections, loader, manifests, same_security
from quiet_index_500.foundation.manifests import write_manifest

default_built_by = "notebooks/02_prices.ipynb"
minimum_sessions_for_one_year = 252  # threshold for the under-one-year flag, carried from notebook 02
large_move_threshold = settings.extreme_return_threshold  # a daily simple return beyond +-50% is flagged for review, not removed
source_modules_used = [settings, loader, corrections, same_security, manifests]  # build itself is added below, once it exists


def _source_modules():
    import quiet_index_500.foundation.build as build_module
    return source_modules_used + [build_module]


def _ledger_inputs(apply_corrections):
    """The ledger file as an input, when corrections are on and the file exists."""
    ledger_path = corrections.extreme_event_ledger_path()
    return [ledger_path] if apply_corrections and ledger_path.exists() else []


def _vintage_table_path(output_folder, stem, through):
    return Path(output_folder) / f"{stem}_through_{pd.Timestamp(through).strftime('%Y-%m-%d')}.csv"


# ---------- pure calculations ----------

def snapshot_portfolio_daily_returns(daily_simple_returns, snapshot_weights):
    """Daily return of the snapshot held at constant weights, renormalized each day over the names with a return.

    Returns a date-indexed frame (first session dropped: no returns yet) with snapshot_portfolio_return,
    weight_with_return (share of snapshot weight covered) and names_with_return.
    """
    has_return_today = daily_simple_returns.notna()
    weighted_return_sum = (daily_simple_returns.fillna(0) * snapshot_weights).sum(axis=1)  # a missing return adds nothing to the numerator
    weight_with_return = (has_return_today * snapshot_weights).sum(axis=1)                 # the share of snapshot weight that traded that day
    snapshot_portfolio_return = weighted_return_sum / weight_with_return                    # renormalize over the names with a return (Sai, 13 Oct 2026)
    portfolio_daily_returns = pd.DataFrame({
        "snapshot_portfolio_return": snapshot_portfolio_return,
        "weight_with_return": weight_with_return,
        "names_with_return": has_return_today.sum(axis=1),
    }).iloc[1:]  # the first session has no previous price, so no returns
    assert portfolio_daily_returns["snapshot_portfolio_return"].notna().all()
    assert (portfolio_daily_returns["weight_with_return"] <= 1 + 1e-12).all()
    assert (portfolio_daily_returns["snapshot_portfolio_return"] <= daily_simple_returns.iloc[1:].max(axis=1)).all()  # a weighted average lies between the day's extremes
    assert (portfolio_daily_returns["snapshot_portfolio_return"] >= daily_simple_returns.iloc[1:].min(axis=1)).all()
    return portfolio_daily_returns


def portfolio_exclusions(daily_simple_returns, price_panels, snapshot_weights):
    """One row per (date, ticker) with no return after the first session, with the reason it is missing.

    Reasons, checked in this order: before_same_security_start, after_history_end, zero_volume_day, quarantined,
    before_first_price, first_price_day, after_last_price, otherwise vendor_gap.
    """
    adjusted_close = price_panels.adjusted_close
    first_usable_price_date = adjusted_close.apply(lambda prices: prices.first_valid_index())
    last_usable_price_date = adjusted_close.apply(lambda prices: prices.last_valid_index())
    zero_volume_today_or_yesterday = price_panels.zero_volume_days | price_panels.zero_volume_days.shift(1, fill_value=False)  # a return needs two prices

    missing_return_grid = daily_simple_returns.iloc[1:].isna()  # the first session has no returns for anyone and is not in the portfolio table
    missing_return_pairs = missing_return_grid.stack()
    missing_return_pairs = missing_return_pairs[missing_return_pairs].index  # (date, ticker) pairs with no return

    exclusion_dates = missing_return_pairs.get_level_values("date")
    exclusion_tickers = missing_return_pairs.get_level_values(1)
    first_date_for_row = first_usable_price_date.reindex(exclusion_tickers).to_numpy()
    last_date_for_row = last_usable_price_date.reindex(exclusion_tickers).to_numpy()
    before_start_for_row = price_panels.before_same_security_start.stack().reindex(missing_return_pairs).to_numpy()
    after_end_for_row = price_panels.after_history_end.stack().reindex(missing_return_pairs).to_numpy()
    zero_volume_for_row = zero_volume_today_or_yesterday.stack().reindex(missing_return_pairs).to_numpy()
    quarantined_for_row = price_panels.quarantined_returns.stack().reindex(missing_return_pairs).to_numpy()

    exclusion_reason = np.select(
        [before_start_for_row,                         # before the same security existed (predecessor or pre-listing data excluded)
         after_end_for_row,                            # after the end of the recorded history window
         zero_volume_for_row,                          # a price was masked by the zero-volume rule
         quarantined_for_row,                          # the extreme-event ledger quarantined this return or a price it needs
         exclusion_dates < first_date_for_row,         # before the first usable price
         exclusion_dates == first_date_for_row,        # first usable price, no prior price to compare with
         exclusion_dates > last_date_for_row],         # after the last usable price
        ["before_same_security_start", "after_history_end", "zero_volume_day", "quarantined", "before_first_price", "first_price_day", "after_last_price"],
        default="vendor_gap",                          # inside the trading span but a price is missing in the vendor data
    )
    return pd.DataFrame({
        "date": exclusion_dates,
        "ticker": exclusion_tickers,
        "snapshot_weight": snapshot_weights.reindex(exclusion_tickers).to_numpy(),
        "reason": exclusion_reason,
    })


def ticker_data_flags(daily_simple_returns, price_panels, snapshot_weights, rules, exclusions,
                      market_cap, market_cap_rejected, market_cap_missing_while_priced, last_session):
    """Per-ticker data flags (notebook 02's table) plus market_cap_nonpositive_days and quarantined_return_days."""
    adjusted_close = price_panels.adjusted_close
    returns_after_first_session = daily_simple_returns.iloc[1:]
    flags = pd.DataFrame({
        "snapshot_weight": snapshot_weights,
        "price_source": rules["price_source"].reindex(snapshot_weights.index).fillna("yahoo"),
        "same_security_history_start": pd.to_datetime(rules["history_start"]).reindex(snapshot_weights.index),
        "history_end": pd.to_datetime(rules["history_end"]).reindex(snapshot_weights.index),
        "first_usable_price_date": adjusted_close.apply(lambda prices: prices.first_valid_index()),
        "last_usable_price_date": adjusted_close.apply(lambda prices: prices.last_valid_index()),
        "sessions_with_return": returns_after_first_session.notna().sum(),
        "days_before_same_security_start": price_panels.before_same_security_start.sum(),
        "zero_volume_days_masked": price_panels.zero_volume_days.sum(),
        "vendor_gap_days": exclusions[exclusions["reason"] == "vendor_gap"].groupby("ticker").size().reindex(snapshot_weights.index, fill_value=0),
        "large_daily_moves": (returns_after_first_session.abs() > large_move_threshold).sum(),
        "market_cap_days_rejected": market_cap_rejected.sum(),
        "market_cap_days_missing_while_priced": market_cap_missing_while_priced.sum(),
    }).rename_axis("ticker")
    flags["under_one_year_history"] = flags["sessions_with_return"] < minimum_sessions_for_one_year
    flags["ends_before_last_session"] = flags["last_usable_price_date"] < pd.Timestamp(last_session)
    flags["size_dependent_exclusion_days"] = flags["market_cap_days_missing_while_priced"]  # formations on these days drop the name from size-dependent analysis only
    market_cap_nonpositive_days = (market_cap <= 0).sum().reindex(snapshot_weights.index, fill_value=0)  # kept in the descriptive table, missing for size (HONA 2026-06-26)
    quarantined_return_days = price_panels.quarantined_returns.iloc[1:].sum().reindex(snapshot_weights.index, fill_value=0)
    flags["any_flag"] = (
        flags.index.isin(rules.index)
        | (flags["zero_volume_days_masked"] > 0) | (flags["vendor_gap_days"] > 0) | (flags["large_daily_moves"] > 0)
        | (flags["market_cap_days_rejected"] > 0) | (flags["market_cap_days_missing_while_priced"] > 0)
        | flags["under_one_year_history"] | flags["ends_before_last_session"]
        | (market_cap_nonpositive_days > 0) | (quarantined_return_days > 0)
    )
    flags["market_cap_nonpositive_days"] = market_cap_nonpositive_days  # added by the foundation, after the original columns
    flags["quarantined_return_days"] = quarantined_return_days
    return flags


# ---------- table writers ----------

def build_same_security_rules_table(output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Write same_security_rules_2026-09-30.csv from same_security.py and return the rules."""
    rules = same_security.same_security_rules()
    same_security_rules_path = Path(output_folder) / f"same_security_rules_{settings.holdings_date}.csv"
    rules.to_csv(same_security_rules_path)
    write_manifest(same_security_rules_path, [], built_by, source_modules=[same_security],
                   note="Same-security decisions (Sai, 13 Oct 2026): history window, price source and market-cap validity per exception, with the evidence.")
    return rules


def build_snapshot_portfolio_daily_returns(through=settings.price_vintage_last_session, apply_corrections=True,
                                           output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Write the constant-weight snapshot portfolio's daily returns (renormalized over names with a return)."""
    daily_simple_returns = loader.load_daily_returns(through=through, apply_corrections=apply_corrections)
    snapshot_weights = loader.load_snapshot_weights()
    portfolio_daily_returns = snapshot_portfolio_daily_returns(daily_simple_returns, snapshot_weights)
    table_path = _vintage_table_path(output_folder, "snapshot_portfolio_daily_returns", through)
    portfolio_daily_returns.to_csv(table_path)
    write_manifest(
        table_path,
        input_paths=[loader.universe_tickers_path(), loader.universe_weights_path(), loader.adjusted_close_path(), loader.split_adjusted_close_path(), loader.volume_path(), loader.supplement_path(),
                     Path(output_folder) / f"same_security_rules_{settings.holdings_date}.csv"] + _ledger_inputs(apply_corrections),
        built_by=built_by, source_modules=_source_modules(),
        note="Daily simple total return of the 30 Sep 2026 IVV snapshot held at constant weights. Each day the weights of names with a return "
             "are rescaled to sum to 1; weight_with_return records the share of snapshot weight covered. Zero-volume days are missing.",
        extra_fields={"corrections_applied": bool(_ledger_inputs(apply_corrections))},
    )
    return portfolio_daily_returns


def _build_etf_daily_returns(load_function, prices_path, stem, return_name, note, through, output_folder, built_by):
    etf_daily_returns = load_function(through=through).iloc[1:].to_frame()  # the first session has no previous price
    assert etf_daily_returns[return_name].notna().all()
    table_path = _vintage_table_path(output_folder, stem, through)
    etf_daily_returns.to_csv(table_path)
    write_manifest(table_path, input_paths=[prices_path, loader.adjusted_close_path(), loader.split_adjusted_close_path(), loader.volume_path(), loader.supplement_path(), loader.universe_tickers_path()], built_by=built_by, source_modules=_source_modules(), note=note)
    return etf_daily_returns


def build_spy_daily_returns(through=settings.price_vintage_last_session, output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Write SPY daily simple total returns over the sessions of the stock pull."""
    return _build_etf_daily_returns(loader.load_spy_returns, loader.spy_prices_path(), "spy_daily_returns", "spy_return",
                                    "Daily simple total return of SPY from Yahoo's adjusted close, over the sessions of the stock pull.",
                                    through, output_folder, built_by)


def build_smh_daily_returns(through=settings.price_vintage_last_session, output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Write SMH daily simple total returns over the sessions of the stock pull."""
    return _build_etf_daily_returns(loader.load_smh_returns, loader.smh_prices_path(), "smh_daily_returns", "smh_return",
                                    "Daily simple total return of SMH from Yahoo's adjusted close, over the sessions of the stock pull.",
                                    through, output_folder, built_by)


def build_market_cap_tables(through=settings.price_vintage_last_session, apply_corrections=True,
                            output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Write ticker-level same-security market cap and issuer market cap (licensed; CSVs git-ignored).

    Nonpositive values are kept in these descriptive tables and counted in the flags table; the study loader sets them
    missing for size. Returns (market_cap, issuer_market_cap, multi_class_issuers).
    """
    price_panels = loader.load_price_panels(through=through, apply_corrections=apply_corrections)
    market_cap = loader.load_market_cap_same_security(through=through, apply_corrections=apply_corrections, nonpositive_as_missing=False)
    identifiers = loader.load_identifiers()
    issuer_market_cap = loader.issuer_market_cap_from_tickers(market_cap, identifiers["cik"], identifiers["snapshot_weight"])
    assert market_cap.index.equals(price_panels.adjusted_close.index)

    multi_class_rows = []
    for cik, tickers_of_issuer in identifiers["cik"].groupby(identifiers["cik"]):
        class_tickers = list(tickers_of_issuer.index)
        if len(class_tickers) > 1:
            both_classes = market_cap[class_tickers].dropna()
            multi_class_rows.append({"cik": cik, "classes": ", ".join(class_tickers),
                                     "issuer_figure_from": identifiers.loc[class_tickers, "snapshot_weight"].idxmax(),
                                     "share_of_days_classes_identical": ((both_classes.max(axis=1) / both_classes.min(axis=1) - 1).abs() < 1e-9).mean()})
    multi_class_issuers = pd.DataFrame(multi_class_rows).set_index("cik")

    market_cap_same_security_path = _vintage_table_path(output_folder, "market_cap_same_security", through)
    issuer_market_cap_path = _vintage_table_path(output_folder, "issuer_market_cap", through)
    market_cap.to_csv(market_cap_same_security_path)
    issuer_market_cap.to_csv(issuer_market_cap_path)
    market_cap_inputs = ([loader.universe_tickers_path(), loader.market_cap_raw_path(), loader.adjusted_close_path(), loader.split_adjusted_close_path(), loader.volume_path(), loader.supplement_path(),
                          loader.ivv_equity_holdings_path(), loader.universe_weights_path(),
                          Path(output_folder) / f"same_security_rules_{settings.holdings_date}.csv"] + _ledger_inputs(apply_corrections))
    write_manifest(market_cap_same_security_path, market_cap_inputs, built_by, source_modules=_source_modules(),
        note="FMP market cap by ticker, kept only on days the same security has a usable price; WBD before 11 Apr 2022 rejected (wrong share basis). "
             "Derived from licensed vendor data: the CSV is git-ignored; this manifest is committed.",
        extra_fields={"nonpositive_values_kept": "Zero or negative values are kept here (HONA 2026-06-26) and counted in the flags table; "
                                                 "the study loader sets them missing for size measurement."})
    write_manifest(issuer_market_cap_path, market_cap_inputs, built_by, source_modules=_source_modules(),
        note="One market cap per issuer (columns are SEC CIKs). Multi-class issuers use the class with the larger snapshot weight "
             "(GOOGL, FOXA, NWSA); FMP prices the issuer's full share count at each class's own price, so classes are never summed. "
             "Derived from licensed vendor data: the CSV is git-ignored; this manifest is committed.",
        extra_fields={"nonpositive_values_kept": "Zero or negative values are kept here and set missing by the study loader for size measurement."})
    return market_cap, issuer_market_cap, multi_class_issuers


def build_portfolio_exclusions_and_flags(through=settings.price_vintage_last_session, apply_corrections=True,
                                         output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Write the per-day portfolio exclusion log and the per-ticker flag table; returns (exclusions, flags)."""
    price_panels = loader.load_price_panels(through=through, apply_corrections=apply_corrections)
    daily_simple_returns = loader.daily_simple_returns_from_prices(price_panels.adjusted_close, price_panels.quarantined_returns)
    snapshot_weights = loader.load_snapshot_weights()
    rules = same_security.same_security_rules()
    exclusions = portfolio_exclusions(daily_simple_returns, price_panels, snapshot_weights)

    portfolio_daily_returns = snapshot_portfolio_daily_returns(daily_simple_returns, snapshot_weights)
    excluded_weight_by_date = exclusions.groupby("date")["snapshot_weight"].sum().reindex(portfolio_daily_returns.index, fill_value=0)
    assert ((excluded_weight_by_date + portfolio_daily_returns["weight_with_return"] - 1).abs() < 1e-12).all()  # included plus excluded weight is the whole snapshot, every day

    market_cap, market_cap_rejected, market_cap_missing_while_priced = loader.clean_market_cap(
        loader._market_cap_raw(), loader.load_price_panels(through=settings.price_vintage_last_session, apply_corrections=apply_corrections).adjusted_close, rules)
    within_cutoff = market_cap.index <= pd.Timestamp(through)
    flags = ticker_data_flags(daily_simple_returns, price_panels, snapshot_weights, rules, exclusions,
                              market_cap[within_cutoff], market_cap_rejected[within_cutoff], market_cap_missing_while_priced[within_cutoff], through)

    portfolio_exclusions_path = _vintage_table_path(output_folder, "portfolio_exclusions", through)
    ticker_data_flags_path = _vintage_table_path(output_folder, "ticker_data_flags", through)
    exclusions.to_csv(portfolio_exclusions_path, index=False)
    flags.to_csv(ticker_data_flags_path)
    flag_inputs = ([loader.universe_tickers_path(), loader.universe_weights_path(), loader.adjusted_close_path(), loader.split_adjusted_close_path(), loader.volume_path(), loader.supplement_path(), loader.market_cap_raw_path(),
                    Path(output_folder) / f"same_security_rules_{settings.holdings_date}.csv"] + _ledger_inputs(apply_corrections))
    write_manifest(portfolio_exclusions_path, flag_inputs, built_by, source_modules=_source_modules(),
        note="One row per date and ticker left out of the snapshot portfolio renormalization. Reasons, checked in this order: before_same_security_start, "
             "after_history_end, zero_volume_day, quarantined, before_first_price, first_price_day, after_last_price, otherwise vendor_gap.")
    write_manifest(ticker_data_flags_path, flag_inputs, built_by, source_modules=_source_modules(),
        note=f"Per-ticker data flags. under_one_year_history means fewer than {minimum_sessions_for_one_year} returns; large_daily_moves counts daily "
             f"simple returns beyond +-{large_move_threshold:.0%}, flagged for review, not removed. size_dependent_exclusion_days counts priced days "
             "with no usable market cap. market_cap_nonpositive_days counts days with a zero or negative market cap (missing for size measurement, "
             "return kept); quarantined_return_days counts returns made missing by the extreme-event ledger.")
    return exclusions, flags


def build_all_processed_tables(through=settings.price_vintage_last_session, apply_corrections=True,
                               output_folder=settings.processed_data_folder, built_by=default_built_by):
    """Regenerate every table notebook 02 writes, in dependency order; returns the written paths."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    build_same_security_rules_table(output_folder, built_by)
    build_snapshot_portfolio_daily_returns(through, apply_corrections, output_folder, built_by)
    build_spy_daily_returns(through, output_folder, built_by)
    build_smh_daily_returns(through, output_folder, built_by)
    build_market_cap_tables(through, apply_corrections, output_folder, built_by)
    build_portfolio_exclusions_and_flags(through, apply_corrections, output_folder, built_by)
    return sorted(output_folder.glob("*.csv"))
