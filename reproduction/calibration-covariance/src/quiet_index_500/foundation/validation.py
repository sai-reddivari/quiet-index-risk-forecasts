"""Foundation validation: the checks behind notebooks/03b_foundation_validation.ipynb, kept here so the notebook holds none.

Each function returns a table; the notebook displays it and saves it under results/foundation/ (or
results/licensed/foundation/ when the values come from FMP market cap) with a manifest.

What is checked (docs/STUDY_SPEC.md section 3)
- Reproduction: with corrections off, the build reproduces the processed tables that existed before the shared loader
  (by sha256), except the two flag columns the foundation appended to ticker_data_flags.
- Correction effects: every value that changes between the corrections-off and corrections-on versions of the
  processed tables and of the loader's daily return panel.
- Correction sensitivity: corrections on versus off for the snapshot portfolio return series and, for every
  back-adjusted security, the 504-session volatility and 12-1 momentum at each formation whose window includes the event.
- Panel shapes, cutoffs, eligibility counts per formation, the event ledger against the loader, nonpositive market cap
  handling, manifest completeness and the git ignore rules for licensed and raw files.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import build, calendar, corrections, eligibility, loader, manifests

validation_notebook = "notebooks/03b_foundation_validation.ipynb"
results_area = "foundation"

# the tables notebook 02 writes, with what their rows and columns are, so a changed cell can be named by ticker and date
processed_table_layouts = {
    f"same_security_rules_{settings.holdings_date}": {"row": "ticker", "column": "field"},
    f"snapshot_portfolio_daily_returns_through_{settings.price_vintage_last_session}": {"row": "date", "column": "field"},
    f"spy_daily_returns_through_{settings.price_vintage_last_session}": {"row": "date", "column": "field"},
    f"smh_daily_returns_through_{settings.price_vintage_last_session}": {"row": "date", "column": "field"},
    f"portfolio_exclusions_through_{settings.price_vintage_last_session}": {"row": "date_and_ticker", "column": "field"},
    f"ticker_data_flags_through_{settings.price_vintage_last_session}": {"row": "ticker", "column": "field"},
    f"market_cap_same_security_through_{settings.price_vintage_last_session}": {"row": "date", "column": "ticker"},
    f"issuer_market_cap_through_{settings.price_vintage_last_session}": {"row": "date", "column": "cik"},
}
licensed_table_stems = {stem for stem in processed_table_layouts if "market_cap" in stem}  # FMP-derived: values never leave results/licensed
appended_flag_columns = ("market_cap_nonpositive_days", "quarantined_return_days")  # added by the foundation after the original columns
change_columns = ["table", "ticker", "date", "column", "before", "after"]
rounding_noise_tolerance = 1e-12  # a change this small in a return or weight is the last binary digit, not a correction


# ---------- saving ----------

def save_results_table(table, file_name, note, input_paths=(), derived_from_licensed_data=False, index=False, extra_fields=None):
    """Write a validation table to results/foundation/ (or results/licensed/foundation/) with its manifest; returns the path."""
    table_path = manifests.results_table_path(results_area, file_name, derived_from_licensed_data)
    table.to_csv(table_path, index=index)
    from quiet_index_500.research_io import study_input_paths
    dependencies = list(dict.fromkeys([*input_paths, *study_input_paths(include_market_cap=True)]))
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    manifests.write_manifest(table_path, dependencies, validation_notebook, note=note,
                             source_modules=[settings, *source_paths],
                             extra_fields=extra_fields)
    return table_path


# ---------- reproduction and correction effects ----------

def strip_trailing_csv_fields(csv_bytes, field_count):
    """The CSV bytes with the last field_count comma-separated fields removed from every line (no quoted commas assumed)."""
    stripped_lines = [line.rsplit(b",", field_count)[0] for line in csv_bytes.split(b"\n")]
    return b"\n".join(stripped_lines)


def read_pre_refactor_hashes(hashes_path):
    """The sha256 list saved before the loader rewrote data/processed ('<sha256>  <path>' per line) as a frame."""
    rows = []
    for line in Path(hashes_path).read_text().splitlines():
        if line.strip():
            sha256_text, path_text = line.split(maxsplit=1)
            rows.append({"file": path_text.strip(), "sha256_pre_refactor": sha256_text})
    return pd.DataFrame(rows, columns=["file", "sha256_pre_refactor"])


def build_corrections_off_tables(output_folder):
    """Rebuild every table notebook 02 writes with corrections off into output_folder (a scratch folder); returns the paths."""
    return build.build_all_processed_tables(through=settings.price_vintage_last_session, apply_corrections=False,
                                            output_folder=output_folder, built_by=f"{validation_notebook} (corrections-off rebuild)")


def reproduction_check(pre_refactor_hashes, rebuilt_folder, processed_folder=None):
    """Compare each pre-refactor CSV hash with its reproduction: the corrections-off rebuild for tables notebook 02 writes,
    the file on disk for the tables it does not write. ticker_data_flags is also compared with its two appended columns
    stripped, which must reproduce the old bytes exactly.
    """
    processed_folder = settings.processed_data_folder if processed_folder is None else Path(processed_folder)
    rows = []
    for record in pre_refactor_hashes.itertuples():
        file_name = Path(record.file).name
        if not file_name.endswith(".csv"):
            continue  # manifests carry build times and new fields by design; the tables are what must reproduce
        stem = file_name[:-len(".csv")]
        written_by_notebook_02 = stem in processed_table_layouts
        reproduced_path = (Path(rebuilt_folder) if written_by_notebook_02 else processed_folder) / file_name
        reproduced_bytes = reproduced_path.read_bytes()
        reproduced_sha256 = manifests.sha256_of_file(reproduced_path)
        row = {"file": record.file, "sha256_pre_refactor": record.sha256_pre_refactor, "sha256_reproduced": reproduced_sha256,
               "reproduced_from": "corrections-off rebuild" if written_by_notebook_02 else "unchanged file on disk",
               "identical": reproduced_sha256 == record.sha256_pre_refactor,
               "sha256_without_appended_columns": "", "identical_without_appended_columns": reproduced_sha256 == record.sha256_pre_refactor}
        if stem.startswith("ticker_data_flags"):  # the foundation appended two columns; every other byte must be unchanged
            stripped_sha256 = hashlib.sha256(strip_trailing_csv_fields(reproduced_bytes, len(appended_flag_columns))).hexdigest()
            row["sha256_without_appended_columns"] = stripped_sha256
            row["identical_without_appended_columns"] = stripped_sha256 == record.sha256_pre_refactor
        rows.append(row)
    return pd.DataFrame(rows)


def _read_processed_table(table_path, layout):
    """A processed CSV as a frame whose index is the row label (date, ticker, cik or (date, ticker))."""
    if layout["row"] == "date_and_ticker":
        return pd.read_csv(table_path, index_col=["date", "ticker"], dtype=str)
    return pd.read_csv(table_path, index_col=0, dtype=str)  # text comparison: a change is any change in the written value


def table_cell_changes(before, after, table_name, layout):
    """Every cell whose written value differs between two versions of one table, named by ticker, date and column.

    Both frames hold text (as written to CSV). A row or column present in only one version is reported with the
    missing side as '<absent>'. Returns a frame with columns table, ticker, date, column, before, after.
    """
    all_rows = before.index.union(after.index, sort=False)
    all_columns = before.columns.union(after.columns, sort=False)
    before_aligned = before.fillna("").reindex(index=all_rows, columns=all_columns).fillna("<absent>")  # an empty cell is blank; only alignment leaves gaps
    after_aligned = after.fillna("").reindex(index=all_rows, columns=all_columns).fillna("<absent>")
    changed_cells = (before_aligned != after_aligned).stack()
    changed_cells = changed_cells[changed_cells]
    rows = []
    for row_label, column_label in changed_cells.index:
        ticker, date = "", ""
        if layout["row"] == "date":
            date = row_label
        elif layout["row"] == "date_and_ticker":
            date, ticker = row_label
        else:
            ticker = row_label
        if layout["column"] in ("ticker", "cik"):
            ticker = column_label
        rows.append({"table": table_name, "ticker": str(ticker), "date": str(date)[:10], "column": "" if layout["column"] in ("ticker", "cik") else column_label,
                     "before": before_aligned.at[row_label, column_label], "after": after_aligned.at[row_label, column_label]})
    return pd.DataFrame(rows, columns=change_columns)


def processed_table_changes(before_folder, after_folder):
    """table_cell_changes for every table notebook 02 writes, before_folder against after_folder."""
    all_changes = [table_cell_changes(_read_processed_table(Path(before_folder) / f"{stem}.csv", layout),
                                      _read_processed_table(Path(after_folder) / f"{stem}.csv", layout), stem, layout)
                   for stem, layout in processed_table_layouts.items()]
    return pd.concat(all_changes, ignore_index=True)


def loader_return_changes(through=settings.research_price_cutoff):
    """Daily simple returns that differ between corrections off and on (loader panel), as change rows."""
    returns_off = loader.load_daily_returns(through=through, apply_corrections=False)
    returns_on = loader.load_daily_returns(through=through, apply_corrections=True)
    differs = ~((returns_off == returns_on) | (returns_off.isna() & returns_on.isna()))  # nan on both sides is no change
    changed_cells = differs.stack()
    changed_cells = changed_cells[changed_cells]
    return pd.DataFrame([{"table": "loader:daily_simple_returns", "ticker": ticker, "date": date.strftime("%Y-%m-%d"), "column": "return",
                          "before": repr(float(returns_off.at[date, ticker])), "after": repr(float(returns_on.at[date, ticker]))}
                         for date, ticker in changed_cells.index], columns=change_columns)


def corrections_off_comparison(pre_refactor_hashes, processed_folder=None):
    """Rebuild the processed tables with corrections off in a temporary folder and compare.

    Returns (reproduction, correction_effects): the reproduction check against the pre-refactor hashes, and every value
    that differs between the corrections-off rebuild and the corrections-on tables in processed_folder, plus the loader's
    daily returns. Licensed market cap tables are compared but their values are never copied into the effects table:
    a changed licensed cell would appear with before and after withheld.
    """
    processed_folder = settings.processed_data_folder if processed_folder is None else Path(processed_folder)
    with tempfile.TemporaryDirectory() as scratch_folder:  # the rebuild exists only for the comparison
        build_corrections_off_tables(scratch_folder)
        reproduction = reproduction_check(pre_refactor_hashes, scratch_folder)  # files notebook 02 does not write are read from data/processed
        table_changes = processed_table_changes(scratch_folder, processed_folder)
    correction_effects = classify_changes(pd.concat([table_changes, loader_return_changes()], ignore_index=True))
    licensed_rows = correction_effects["table"].isin(licensed_table_stems)
    correction_effects.loc[licensed_rows, ["before", "after", "absolute_difference"]] = np.nan  # licensed values never leave results/licensed
    return reproduction, correction_effects


def classify_changes(changes, rounding_tolerance=rounding_noise_tolerance):
    """Add absolute_difference and kind to change rows: 'floating_point_rounding' when both values are numbers within
    rounding_tolerance of each other (rescaling two prices by one factor can move their ratio in the last binary digit),
    otherwise 'correction'.
    """
    before_numbers = pd.to_numeric(changes["before"], errors="coerce")
    after_numbers = pd.to_numeric(changes["after"], errors="coerce")
    absolute_difference = (after_numbers - before_numbers).abs()
    is_rounding = absolute_difference.notna() & (absolute_difference <= rounding_tolerance)
    return changes.assign(absolute_difference=absolute_difference, kind=np.where(is_rounding, "floating_point_rounding", "correction"))


# ---------- correction sensitivity ----------

def annualised_mean_and_volatility(daily_returns):
    """(mean x 252, sample standard deviation x sqrt(252)) of a daily return series, missing days ignored."""
    return (float(daily_returns.mean() * settings.annualisation_sessions),
            float(daily_returns.std(ddof=1) * np.sqrt(settings.annualisation_sessions)))


def twelve_minus_one_momentum(adjusted_close, formation_position):
    """P(t-21) / P(t-252) - 1 at session position t of one security's adjusted close, and the valid daily returns between.

    Returns (momentum, valid_returns_between_endpoints); momentum is nan when either endpoint is missing. This is the
    plain formula for the correction sensitivity only; the study's momentum, with its eligibility rules, is Agent B's.
    """
    recent_position = formation_position - settings.momentum_recent_lag_sessions
    start_position = formation_position - settings.momentum_start_lag_sessions
    if start_position < 0:
        return np.nan, 0
    recent_price = adjusted_close.iloc[recent_position]
    start_price = adjusted_close.iloc[start_position]
    returns_between = adjusted_close.iloc[start_position:recent_position + 1].pct_change(fill_method=None).iloc[1:]
    momentum = recent_price / start_price - 1 if pd.notna(recent_price) and pd.notna(start_price) else np.nan
    return float(momentum), int(returns_between.notna().sum())


def snapshot_portfolio_sensitivity(through=settings.research_price_cutoff):
    """Snapshot portfolio return series, corrections on versus off: annualised mean and volatility and each changed day."""
    snapshot_weights = loader.load_snapshot_weights()
    portfolio_off = build.snapshot_portfolio_daily_returns(loader.load_daily_returns(through=through, apply_corrections=False), snapshot_weights)
    portfolio_on = build.snapshot_portfolio_daily_returns(loader.load_daily_returns(through=through, apply_corrections=True), snapshot_weights)
    returns_off = portfolio_off["snapshot_portfolio_return"]
    returns_on = portfolio_on["snapshot_portfolio_return"]
    first_date, last_date = returns_off.index.min().strftime("%Y-%m-%d"), returns_off.index.max().strftime("%Y-%m-%d")
    mean_off, volatility_off = annualised_mean_and_volatility(returns_off)
    mean_on, volatility_on = annualised_mean_and_volatility(returns_on)
    rows = [
        {"subject": "snapshot_portfolio", "measure": "annualised_mean", "date": "", "window_first_session": first_date, "window_last_session": last_date,
         "corrections_off": mean_off, "corrections_on": mean_on},
        {"subject": "snapshot_portfolio", "measure": "annualised_volatility", "date": "", "window_first_session": first_date, "window_last_session": last_date,
         "corrections_off": volatility_off, "corrections_on": volatility_on},
    ]
    for changed_date in returns_off.index[returns_off != returns_on]:  # the affected dates
        rows.append({"subject": "snapshot_portfolio", "measure": "daily_return", "date": changed_date.strftime("%Y-%m-%d"),
                     "window_first_session": "", "window_last_session": "",
                     "corrections_off": float(returns_off[changed_date]), "corrections_on": float(returns_on[changed_date])})
    return pd.DataFrame(rows)


def back_adjustment_sensitivity(ledger=None, window_sessions=settings.main_window_sessions):
    """For each back-adjusted ledger row, the security's annualised volatility over the window_sessions window ending on
    each formation date, and its 12-1 momentum, corrections off and on, at every formation whose volatility window or
    momentum window [t-252, t] includes the event date. Volatility is of daily simple total returns (a price-only
    measure, so the window ends on the formation date itself).
    """
    ledger = corrections.read_extreme_event_ledger() if ledger is None else ledger
    back_adjusted_rows = ledger[ledger["treatment"] == "back_adjust_before_date"]
    panels_off = loader.load_price_panels(apply_corrections=False)
    panels_on = loader.load_price_panels(apply_corrections=True)
    returns_off = loader.daily_simple_returns_from_prices(panels_off.adjusted_close, panels_off.quarantined_returns)
    returns_on = loader.daily_simple_returns_from_prices(panels_on.adjusted_close, panels_on.quarantined_returns)
    sessions = panels_on.adjusted_close.index
    rows = []
    for ledger_row in back_adjusted_rows.itertuples():
        ticker, event_date = ledger_row.ticker, pd.Timestamp(ledger_row.event_date)
        for formation_date in calendar.formation_dates(sessions):
            formation_position = sessions.get_loc(formation_date)
            volatility_window = calendar.estimation_window(sessions, formation_date, window_sessions)
            momentum_window = sessions[max(0, formation_position - settings.momentum_start_lag_sessions): formation_position + 1]
            in_volatility_window = volatility_window.min() <= event_date <= volatility_window.max()
            in_momentum_window = momentum_window.min() <= event_date <= momentum_window.max()
            if not (in_volatility_window or in_momentum_window):
                continue
            _, volatility_off = annualised_mean_and_volatility(returns_off[ticker].reindex(volatility_window))
            _, volatility_on = annualised_mean_and_volatility(returns_on[ticker].reindex(volatility_window))
            momentum_off, valid_returns_off = twelve_minus_one_momentum(panels_off.adjusted_close[ticker], formation_position)
            momentum_on, valid_returns_on = twelve_minus_one_momentum(panels_on.adjusted_close[ticker], formation_position)
            common = {"subject": ticker, "date": formation_date.strftime("%Y-%m-%d"), "event_date": event_date.strftime("%Y-%m-%d")}
            rows.append({**common, "measure": f"annualised_volatility_{window_sessions}", "window_first_session": volatility_window.min().strftime("%Y-%m-%d"),
                         "window_last_session": volatility_window.max().strftime("%Y-%m-%d"), "event_in_window": bool(in_volatility_window),
                         "corrections_off": volatility_off, "corrections_on": volatility_on})
            rows.append({**common, "measure": "momentum_12_1", "window_first_session": momentum_window.min().strftime("%Y-%m-%d"),
                         "window_last_session": momentum_window.max().strftime("%Y-%m-%d"), "event_in_window": bool(in_momentum_window),
                         "corrections_off": momentum_off, "corrections_on": momentum_on,
                         "valid_returns_between_endpoints": valid_returns_on})
    return pd.DataFrame(rows)


def correction_sensitivity():
    """snapshot_portfolio_sensitivity and back_adjustment_sensitivity in one long table, with the differences."""
    sensitivity = pd.concat([snapshot_portfolio_sensitivity(), back_adjustment_sensitivity()], ignore_index=True)
    sensitivity["difference"] = sensitivity["corrections_on"] - sensitivity["corrections_off"]
    sensitivity["kind"] = np.select([sensitivity["difference"] == 0, sensitivity["difference"].abs() <= rounding_noise_tolerance],
                                    ["unchanged", "floating_point_rounding"], default="correction")
    leading_columns = ["subject", "measure", "date", "event_date", "event_in_window", "window_first_session", "window_last_session",
                       "corrections_off", "corrections_on", "difference", "kind"]
    return sensitivity.reindex(columns=leading_columns + [column for column in sensitivity.columns if column not in leading_columns])


# ---------- panels, cutoffs, calendar and eligibility ----------

def _shape_row(table_name, frame):
    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    return {"table": table_name, "rows": frame.shape[0], "columns": frame.shape[1],
            "first_date": frame.index.min().strftime("%Y-%m-%d"), "last_date": frame.index.max().strftime("%Y-%m-%d"),
            "non_missing_values": int(frame.notna().to_numpy().sum())}


def panel_shapes_and_date_ranges(through=settings.research_price_cutoff):
    """Shape, first and last date and non-missing count of every panel the loader serves at the research cutoff."""
    price_panels = loader.load_price_panels(through=through)
    return pd.DataFrame([
        _shape_row("adjusted_close (corrected)", price_panels.adjusted_close),
        _shape_row("split_adjusted_close", price_panels.split_adjusted_close),
        _shape_row("volume", price_panels.volume),
        _shape_row("daily_simple_returns", loader.load_daily_returns(through=through)),
        _shape_row("excess_returns", loader.load_excess_returns(through=through)),
        _shape_row("factors (ff5 + mom + rf)", loader.load_factors(through=through)),
        _shape_row("spy_returns", loader.load_spy_returns(through=through)),
        _shape_row("smh_returns", loader.load_smh_returns(through=through)),
        _shape_row("market_cap_same_security (shape only)", loader.load_market_cap_same_security(through=through)),
        _shape_row("issuer_market_cap (shape only)", loader.load_issuer_market_cap(through=through)),
    ])


def study_cutoffs():
    """The study's dates: the frozen vintage, the research price cutoff, the last loaded session and the factor cutoff from the data."""
    sessions = calendar.trading_sessions()
    return pd.DataFrame([
        {"item": "price_vintage_last_session", "value": settings.price_vintage_last_session, "source": "settings (the frozen pull, named explicitly)"},
        {"item": "research_price_cutoff", "value": settings.research_price_cutoff, "source": "settings"},
        {"item": "last_session_served_by_loader", "value": sessions.max().strftime("%Y-%m-%d"), "source": "loader (sessions on or before the research cutoff)"},
        {"item": "first_session_served_by_loader", "value": sessions.min().strftime("%Y-%m-%d"), "source": "loader"},
        {"item": "factor_cutoff", "value": loader.factor_data_last_date().strftime("%Y-%m-%d"), "source": "read from the factor file (factor_data_last_date)"},
        {"item": "first_formation_date", "value": settings.first_formation_date, "source": "settings"},
    ])


def study_formation_calendar():
    """calendar.formation_calendar on the loader's sessions with the factor cutoff read from the data."""
    return calendar.formation_calendar(calendar.trading_sessions(), loader.factor_data_last_date())


def eligibility_counts_by_formation(window_sessions, minimum_observations):
    """Per formation, how many securities are eligible for a window_sessions window ending on the factor measurement
    cutoff with at least minimum_observations complete joint observations of the excess return and mkt_rf, with the
    count for each exclusion reason, the quarantined days the windows hold, and the eligible representative securities.
    """
    price_panels = loader.load_price_panels()
    excess_returns = loader.excess_returns(loader.daily_simple_returns_from_prices(price_panels.adjusted_close, price_panels.quarantined_returns),
                                           loader.load_factors()["rf"])
    market_factor = loader.load_factors()["mkt_rf"]
    factor_cutoff = loader.factor_data_last_date()
    is_representative = loader.load_identifiers()["is_representative_security"]
    sessions = price_panels.adjusted_close.index
    rows = []
    for formation_date in calendar.formation_dates(sessions):
        cutoff = calendar.measurement_cutoff(formation_date, factor_cutoff)
        window = calendar.estimation_window(sessions, cutoff, window_sessions)
        window_table = eligibility.panel_window_eligibility(excess_returns, window, minimum_observations, market_factor,
                                                           price_panels.before_same_security_start, price_panels.quarantined_returns)
        reason_counts = window_table["reason"].value_counts()
        rows.append({"formation_date": formation_date.strftime("%Y-%m-%d"), "measurement_cutoff": cutoff.strftime("%Y-%m-%d"),
                     "window_sessions": window_sessions, "minimum_observations": minimum_observations,
                     "window_first_session": window.min().strftime("%Y-%m-%d"), "window_last_session": window.max().strftime("%Y-%m-%d"),
                     "securities": len(window_table), "eligible": int(window_table["eligible"].sum()),
                     "eligible_representative_securities": int((window_table["eligible"] & is_representative.reindex(window_table.index)).sum()),
                     **{f"reason_{reason}": int(reason_counts.get(reason, 0)) for reason in eligibility.eligibility_reasons if reason != eligibility.reason_eligible},
                     "quarantined_returns_in_windows": int(window_table["quarantined_in_window"].sum())})
    return pd.DataFrame(rows)


# ---------- ledger, market cap, manifests and ignore rules ----------

def event_ledger_with_returns(threshold=settings.extreme_return_threshold):
    """The ledger joined to the loader's return on each event date with corrections off and on, and a check that every
    return beyond the threshold (corrections off, through the research cutoff) has a ledger row.

    ledger_return_matches_loader compares the ledger's corrected_return with the corrections-on loader return to 1e-10.
    """
    ledger = corrections.read_extreme_event_ledger()
    returns_off = loader.load_daily_returns(apply_corrections=False)
    returns_on = loader.load_daily_returns(apply_corrections=True)
    extreme_returns = returns_off.stack()
    extreme_returns = extreme_returns[extreme_returns.abs() > threshold]
    ledger_keys = set(zip(ledger["ticker"], pd.to_datetime(ledger["event_date"])))
    table = ledger.copy()
    table["return_corrections_off"] = [float(returns_off.at[pd.Timestamp(date), ticker]) for ticker, date in zip(ledger["ticker"], ledger["event_date"])]
    table["return_corrections_on"] = [float(returns_on.at[pd.Timestamp(date), ticker]) for ticker, date in zip(ledger["ticker"], ledger["event_date"])]
    table["ledger_return_matches_loader"] = (table["corrected_return"] - table["return_corrections_on"]).abs() <= corrections.corrected_return_tolerance
    table["beyond_threshold_corrections_off"] = table["return_corrections_off"].abs() > threshold
    table["beyond_threshold_corrections_on"] = table["return_corrections_on"].abs() > threshold
    unledgered = [(ticker, date) for date, ticker in extreme_returns.index if (ticker, date) not in ledger_keys]
    if unledgered:
        raise ValueError(f"returns beyond +-{threshold:.0%} with no ledger row: {unledgered}")
    return table


def nonpositive_market_cap_handling(through=settings.price_vintage_last_session):
    """Every (ticker, date) whose raw FMP market cap is zero or negative: how each layer treats it, and the return that day.

    Licensed: the table holds no cap values, only how they are treated, but it is still saved under results/licensed.
    """
    market_cap_raw = loader._market_cap_raw()
    market_cap_raw = market_cap_raw[market_cap_raw.index <= pd.Timestamp(through)]
    nonpositive_cells = (market_cap_raw <= 0).stack()
    nonpositive_cells = nonpositive_cells[nonpositive_cells]
    descriptive_market_cap = loader.load_market_cap_same_security(through=through, nonpositive_as_missing=False)
    study_market_cap = loader.load_market_cap_same_security(through=through, nonpositive_as_missing=True)
    daily_simple_returns = loader.load_daily_returns(through=through)
    flags_path = settings.processed_data_folder / f"ticker_data_flags_through_{settings.price_vintage_last_session}.csv"
    flags = pd.read_csv(flags_path, index_col="ticker")
    rows = []
    for date, ticker in nonpositive_cells.index:
        rows.append({"ticker": ticker, "date": date.strftime("%Y-%m-%d"),
                     "raw_value_nonpositive": True,
                     "descriptive_table_keeps_nonpositive_value": bool(descriptive_market_cap.at[date, ticker] <= 0),
                     "study_loader_value_missing": bool(pd.isna(study_market_cap.at[date, ticker])),
                     "daily_return_kept": bool(pd.notna(daily_simple_returns.at[date, ticker])),
                     "daily_return": float(daily_simple_returns.at[date, ticker]),
                     "flags_table_market_cap_nonpositive_days": int(flags.at[ticker, "market_cap_nonpositive_days"])})
    return pd.DataFrame(rows)


def manifest_completeness(folders=None, skip_tables=()):
    """For every CSV under the folders: whether its manifest exists, the table's sha256 matches, every input exists and
    matches its recorded sha256, and the recorded source code still matches (required for acceptance).

    skip_tables names files to leave out, such as the table this check is about to be saved as.
    """
    folders = [settings.processed_data_folder, settings.corrections_folder, settings.results_folder] if folders is None else folders
    skipped_paths = {Path(table_path).resolve() for table_path in skip_tables}
    rows = []
    for folder in folders:
        for table_path in sorted(Path(folder).rglob("*.csv")):
            if table_path.resolve() in skipped_paths:
                continue
            manifest_path = manifests.manifest_path_for(table_path)
            row = {"table": manifests.repository_relative_path(table_path), "manifest_exists": manifest_path.exists(),
                   "table_sha256_matches": False, "inputs": 0, "inputs_all_exist": False, "inputs_sha256_all_match": False,
                   "mismatched_or_missing_inputs": "", "source_code_files": 0, "source_code_sha256_all_match": True}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                input_entries = manifest.get("inputs", [])
                problems = []
                for entry in input_entries:
                    input_path = settings.repository_root_folder / entry["file"] if not Path(entry["file"]).is_absolute() else Path(entry["file"])
                    if not input_path.exists():
                        problems.append(f"{entry['file']} (missing)")
                    elif manifests.sha256_of_file(input_path) != entry["sha256"]:
                        problems.append(f"{entry['file']} (changed)")
                source_entries = manifest.get("source_code", [])
                source_matches = all((settings.repository_root_folder / entry["file"]).exists()
                                     and manifests.sha256_of_file(settings.repository_root_folder / entry["file"]) == entry["sha256"]
                                     for entry in source_entries)
                row.update({"table_sha256_matches": manifest.get("sha256") == manifests.sha256_of_file(table_path),
                            "inputs": len(input_entries), "inputs_all_exist": not any(problem.endswith("(missing)") for problem in problems),
                            "inputs_sha256_all_match": not problems, "mismatched_or_missing_inputs": "; ".join(problems),
                            "source_code_files": len(source_entries), "source_code_sha256_all_match": source_matches})
            row["complete"] = (row["manifest_exists"] and row["table_sha256_matches"]
                               and row["inputs_sha256_all_match"] and row["source_code_sha256_all_match"])
            rows.append(row)
    return pd.DataFrame(rows)


def git_ignore_listing(paths_expected_ignored, paths_expected_tracked):
    """git check-ignore for each path: whether it is ignored, by which rule, and whether that is what was expected.

    Uses --no-index so a path is judged by the rules alone, whether or not it is tracked.
    """
    all_paths = list(paths_expected_ignored) + list(paths_expected_tracked)
    completed = subprocess.run(["git", "check-ignore", "--no-index", "--verbose", "--non-matching", *all_paths],
                               cwd=settings.repository_root_folder, capture_output=True, text=True)
    rule_by_path = {}
    for line in completed.stdout.splitlines():  # '<source>:<line>:<pattern>\t<path>', or '::\t<path>' when nothing matches
        rule_text, path_text = line.split("\t", 1)
        rule_by_path[path_text] = rule_text
    rows = []
    for path_text in all_paths:
        rule_text = rule_by_path.get(path_text, "::")
        ignored = rule_text != "::" and not rule_text.split(":", 2)[2].startswith("!")
        expected_ignored = path_text in paths_expected_ignored
        rows.append({"path": path_text, "expected_ignored": expected_ignored, "ignored": ignored,
                     "matched_rule": "" if rule_text == "::" else rule_text, "as_expected": ignored == expected_ignored})
    return pd.DataFrame(rows)
