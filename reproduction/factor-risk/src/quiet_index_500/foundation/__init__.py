"""Shared data foundation (docs/STUDY_SPEC.md section 3): the one loader every analysis uses.

Loader functions take through= (default settings.research_price_cutoff, 2026-09-30) and apply_corrections=
(default True, which applies data/corrections/extreme_event_ledger_2026-09-30.csv when it exists).
"""

from quiet_index_500.foundation.build import (
    build_all_processed_tables,
    build_market_cap_tables,
    build_portfolio_exclusions_and_flags,
    build_same_security_rules_table,
    build_smh_daily_returns,
    build_snapshot_portfolio_daily_returns,
    build_spy_daily_returns,
    portfolio_exclusions,
    snapshot_portfolio_daily_returns,
    ticker_data_flags,
)
from quiet_index_500.foundation.calendar import (
    estimation_window,
    following_quarter_complete,
    following_quarter_sessions,
    formation_calendar,
    formation_dates,
    measurement_cutoff,
    outcome_block,
    outcome_quarter,
    trading_sessions,
)
from quiet_index_500.foundation.corrections import (
    CorrectedAdjustedClose,
    apply_extreme_event_ledger,
    extreme_event_ledger_path,
    ledger_columns,
    read_extreme_event_ledger,
)
from quiet_index_500.foundation.eligibility import complete_joint_observations, panel_window_eligibility, window_eligibility
from quiet_index_500.foundation.inventory import inventory_leftover_files
from quiet_index_500.foundation.loader import (
    PricePanels,
    clean_market_cap,
    clean_price_panels,
    daily_simple_returns_from_prices,
    excess_returns,
    factor_data_last_date,
    issuer_market_cap_from_tickers,
    load_daily_returns,
    load_excess_returns,
    load_factors,
    load_identifiers,
    load_issuer_market_cap,
    load_market_cap_same_security,
    load_price_panels,
    load_smh_excess_returns,
    load_smh_returns,
    load_snapshot_weights,
    load_spy_excess_returns,
    load_spy_returns,
    load_universe_tickers,
    representative_security_by_ticker,
)
from quiet_index_500.foundation.manifests import (
    licensed_results_path,
    results_table_path,
    sha256_of_file,
    verify_file_against_manifest,
    write_manifest,
)
from quiet_index_500.foundation.same_security import same_security_rules
from quiet_index_500.foundation.validation import (
    annualised_mean_and_volatility,
    back_adjustment_sensitivity,
    correction_sensitivity,
    corrections_off_comparison,
    eligibility_counts_by_formation,
    event_ledger_with_returns,
    git_ignore_listing,
    loader_return_changes,
    manifest_completeness,
    nonpositive_market_cap_handling,
    panel_shapes_and_date_ranges,
    processed_table_changes,
    read_pre_refactor_hashes,
    reproduction_check,
    save_results_table,
    snapshot_portfolio_sensitivity,
    strip_trailing_csv_fields,
    study_cutoffs,
    study_formation_calendar,
    table_cell_changes,
    twelve_minus_one_momentum,
)
