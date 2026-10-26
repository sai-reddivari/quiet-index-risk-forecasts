"""Bound the DHR distribution decision in the first affected formation estimates.

The primary close-valued correction stays untouched. Alternative and original vendor
returns are diagnostic scenarios, applied only in memory to DHR's one event return.
"""

import sys

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.factor_exposures import estimate_factor_models, standard_factor_names
from quiet_index_500.foundation import calendar, loader
from quiet_index_500.research_io import save_table


def distribution_return(previous_parent_close, parent_close, distributed_shares, distributed_price):
    """One-session economic wealth return including the marked distribution."""
    return (parent_close + distributed_shares * distributed_price) / previous_parent_close - 1


def event_sensitivity_table():
    """Both study windows, only formations whose window actually contains the DHR event."""
    event_date = pd.Timestamp('2016-07-05')
    corrected = loader.load_daily_returns()
    uncorrected = loader.load_daily_returns(apply_corrections=False)
    factors = loader.load_factors()
    common_factors = factors[list(standard_factor_names)].join(loader.load_smh_excess_returns().rename('smh_excess'))
    scenarios = {
        'primary_fortive_first_close': corrected.at[event_date, 'DHR'],
        'original_vendor_double_adjustment': uncorrected.at[event_date, 'DHR'],
        'form_8937_average_valuation': distribution_return(102.32, 80.40, .5, 49.11),
    }
    rows = []
    for formation in calendar.formation_dates(corrected.index):
        cutoff = calendar.measurement_cutoff(formation, loader.factor_data_last_date())
        for length, minimum in [(settings.main_window_sessions, settings.main_window_minimum_observations),
                                (settings.sensitivity_window_sessions, settings.sensitivity_window_minimum_observations)]:
            window = calendar.estimation_window(corrected.index, cutoff, length)
            if event_date not in window:
                continue
            for name, event_return in scenarios.items():
                returns = corrected['DHR'].reindex(window).copy()
                returns.loc[event_date] = event_return
                excess = returns - factors.rf.reindex(window)
                estimate = estimate_factor_models(excess, common_factors.reindex(window), minimum)
                baseline = estimate['models']['ff5_mom']
                augmented = estimate['models']['ff5_mom_smh']
                rows.append({'ticker': 'DHR', 'event_date': event_date, 'formation_date': formation,
                             'measurement_cutoff': cutoff, 'window_sessions': length, 'scenario': name,
                             'event_return': event_return, 'observations': baseline['nobs'],
                             'total_volatility': returns.std(ddof=1)*np.sqrt(settings.annualisation_sessions),
                             'residual_volatility': np.sqrt(baseline['residual_variance']*settings.annualisation_sessions),
                             'm3_mkt_rf': baseline['coefficient'][1],
                             'm4_smh_excess': augmented['coefficient'][-1],
                             'incremental_r_squared': augmented['r_squared']-baseline['r_squared']})
    return pd.DataFrame(rows)


def build_event_sensitivity():
    """Write the compact scenario table with exact data, ledger and code fingerprints."""
    from quiet_index_500 import factor_exposures
    table = event_sensitivity_table()
    return save_table(table, 'review', 'event_sensitivity.csv', source_modules=[sys.modules[__name__], factor_exposures],
                      inputs=[settings.repository_root_folder/'data/corrections/extreme_event_evidence.md',
                              settings.repository_root_folder/'docs/reviews/foundation_events.md'],
                      note='DHR 2016-07-05 only. Primary close-valued correction versus original vendor error and Form 8937 average valuation; no raw data changed.')


if __name__ == '__main__':
    print(build_event_sensitivity())
