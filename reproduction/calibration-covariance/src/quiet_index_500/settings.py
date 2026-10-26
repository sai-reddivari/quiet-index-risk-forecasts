"""Study settings: every choice a person made lives here once. See docs/STUDY_SPEC.md.

Dates set by an external publisher (the factor cutoff) are read from the data, not set here.
Owned by the orchestrator; agents read these values and never redefine them.
"""

from pathlib import Path

repository_root_folder = Path(__file__).resolve().parents[2]  # src/quiet_index_500/settings.py -> repository root
raw_data_folder = repository_root_folder / "data" / "raw"
processed_data_folder = repository_root_folder / "data" / "processed"
corrections_folder = repository_root_folder / "data" / "corrections"
results_folder = repository_root_folder / "results"
licensed_results_folder = results_folder / "licensed"  # FMP-derived tables: CSVs git-ignored, manifests committed

# snapshot and vintages, chosen explicitly (never "newest matching file")
holdings_date = "2026-09-30"
price_vintage_last_session = "2026-10-12"  # the frozen Yahoo pull; the 2026-10-13 leftovers are outside the study
price_history_start = "2015-01-01"
research_price_cutoff = "2026-09-30"  # no price after this date is used
ff5_vintage = "202608"  # CRSP vintage of the French files
ff5_mom_factors_file = "ff5_mom_factors_daily_2015-01-02_2026-08-31.csv"

# calendar
first_formation_date = "2017-12-29"  # its following quarter, 2018 Q1, starts the main evaluation
development_outcome_quarters = ("2018Q1", "2022Q4")
validation_outcome_quarters = ("2023Q1", "2024Q4")
confirmation_outcome_quarters = ("2025Q1", "2026Q2")

# estimation windows and minimum observations
main_window_sessions = 504
main_window_minimum_observations = 400
sensitivity_window_sessions = 252
sensitivity_window_minimum_observations = 200  # orchestrator-set: same 400/504 ratio
downside_minimum_negative_market_days_main = 100
downside_minimum_negative_market_days_sensitivity = 50  # orchestrator-set
worst_decile_quantile = 0.10  # supplementary downside conditioning

# own momentum (12-1), sessions on the trading calendar
momentum_recent_lag_sessions = 21
momentum_start_lag_sessions = 252
momentum_minimum_valid_returns_between_endpoints = 200  # orchestrator-set, of the 231 between the endpoints

# liquidity
dollar_volume_window_sessions = 60
dollar_volume_minimum_observations = 40
amihud_window_sessions = 252
amihud_minimum_observations = 200
amihud_scale_dollars = 1_000_000  # Amihud reported per USD 1 million traded

# forward outcomes and inference
forward_outcome_minimum_valid_days = 40  # orchestrator-set
industry_group_minimum_companies = 5  # orchestrator-set: smaller groups pooled into a per-sector remainder
newey_west_lags = 4  # orchestrator-set, in formation quarters
bootstrap_block_length = 4  # orchestrator-set, in formation quarters
bootstrap_replications = 2000
bootstrap_random_seed = 20261014
forecast_minimum_training_quarters = 8  # two years before an expanding-window forecast
mechanical_overlap_subindustry_code = 45301020  # semiconductors; supplementary sample exclusion
event_convention_sensitivity_tickers = ("DHR", "KDP", "BKR", "GEN", "OXY", "TKO", "ECHO")

# events and conventions
extreme_return_threshold = 0.50  # daily simple returns beyond +-50% go to the event ledger
annualisation_sessions = 252
representative_security_rule = "largest snapshot weight within the issuer, fixed for the study"
