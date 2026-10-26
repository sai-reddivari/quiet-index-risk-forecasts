"""Choices fixed before the Step 0.5 expansion estimates were inspected."""

from quiet_index_500 import settings as foundation

factor_names = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']
root = foundation.repository_root_folder
result_root = foundation.results_folder / 'expansions'
source_date = '2026-10-15'
variants = {'main': (504, True), 'window252': (252, True), 'equity_shock': (504, False)}
bootstrap_draws = foundation.bootstrap_replications
bootstrap_block = foundation.bootstrap_block_length
bootstrap_seed = foundation.bootstrap_random_seed
minimum_future_days = foundation.forward_outcome_minimum_valid_days
initial_training_end = '2022-12-31'
primary_names = ['RI1_calibration', 'RI2_activity', 'H2_asymmetry', 'H3_liquidity', 'H7_rates', 'H4_lag']
common_predictors = ['log_market_cap', *[f'ff_{f}' for f in factor_names], 'rate_beta', 'lambda', 'own_momentum']
calibrated_predictors = [*common_predictors, 'log_trailing_variance', 'log_recent_ratio']
conditional_predictors = [*calibrated_predictors, 'log_dollar_volume_60', 'log_amihud_252',
                          'deterioration', 'abs_lambda', 'spy_beta', 'spy_downside_amplification']


def treasury_source(series):
    return foundation.raw_data_folder / f'treasury_{series}_source_retrieved_{source_date}.html'


def minimum_observations(window):
    return foundation.main_window_minimum_observations if window == 504 else foundation.sensitivity_window_minimum_observations


def minimum_sign_observations(window):
    return 100 if window == 504 else 50
