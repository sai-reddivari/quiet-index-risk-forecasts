"""Notebook 04: frozen official Treasury observations and equity-session alignment."""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_index_500.foundation import calendar, loader
from quiet_index_500.foundation.manifests import verify_file_against_manifest
from . import common, settings as cfg


def source_levels(series):
    path = cfg.treasury_source(series)
    verify_file_against_manifest(path)
    rows = re.findall(r'#(\d{4}-\d{2}-\d{2})\|\s*([.\d-]+)', path.read_text())
    frame = pd.DataFrame(rows, columns=['date', series])
    frame['date'] = pd.to_datetime(frame.date)
    frame[series] = pd.to_numeric(frame[series], errors='coerce')
    if frame.date.duplicated().any():
        raise ValueError('Duplicate Treasury source dates')
    return frame.set_index('date').sort_index().loc['2014-12-30':cfg.foundation.research_price_cutoff]


def align_yield_changes(levels, sessions):
    """Missing consecutive equity-session quotes remain missing, including the next day."""
    aligned = levels.reindex(pd.DatetimeIndex(sessions))
    changes = 100 * aligned.diff()
    return pd.DataFrame({'date': aligned.index, 'yield_percent': aligned.to_numpy(),
                         'previous_equity_session': pd.Series(aligned.index, index=aligned.index).shift(1).to_numpy(),
                         'previous_yield_percent': aligned.shift(1).to_numpy(),
                         'rate_change_bp': changes.to_numpy(),
                         'current_yield_missing': aligned.isna().to_numpy(),
                         'previous_yield_missing': aligned.shift(1).isna().to_numpy()})


def daily_rates():
    path = cfg.result_root / '04' / 'aligned_rates.csv'
    if not path.exists():
        run()
    return common.read('04', 'aligned_rates', ['date', 'previous_equity_session']).set_index('date')


def run():
    sessions = common.inputs()['sessions']
    # The foundation starts on 2015-01-02; 2014-12-31 was its preceding equity session.
    extended = pd.DatetimeIndex([pd.Timestamp('2014-12-31'), *sessions])
    aligned = align_yield_changes(source_levels('DGS10').DGS10, extended)
    aligned = aligned[aligned.date.isin(sessions)].copy()
    aligned_path = common.save(aligned, '04', 'aligned_rates', modules=[sys.modules[__name__]])
    rows = []
    for window in (504, 252):
        for formation in calendar.formation_dates(sessions):
            cutoff = min(formation, common.inputs()['factor_cutoff'])
            dates = calendar.estimation_window(sessions, cutoff, window)
            sample = aligned.set_index('date').reindex(dates)
            joint = sample.rate_change_bp.notna() & common.inputs()['factors'].reindex(dates)[cfg.factor_names].notna().all(axis=1)
            counts = common.inputs()['excess'].reindex(dates).where(joint, axis=0).notna().sum()
            eligible = counts >= cfg.minimum_observations(window)
            ids = common.inputs()['identifiers']
            rows.append({'formation_date': formation, 'measurement_cutoff': cutoff, 'window': window,
                         'window_sessions': len(dates), 'valid_rate_days': sample.rate_change_bp.notna().sum(),
                         'missing_current_quotes': sample.current_yield_missing.sum(), 'common_factor_rate_days': joint.sum(),
                         'eligible_securities': eligible.sum(), 'represented_snapshot_weight': ids.snapshot_weight[eligible].sum()})
    coverage = pd.DataFrame(rows)
    coverage_path = common.save(coverage, '04', 'coverage', modules=[sys.modules[__name__]])
    sources = pd.DataFrame([{'series': 'DGS10', 'source': str(cfg.treasury_source('DGS10').relative_to(cfg.root)),
                             'source_url': 'https://fred.stlouisfed.org/data/DGS10', 'units': 'percent per annum',
                             'transformed_units': 'basis points per consecutive equity session',
                             'first_research_date': aligned.date.min(), 'last_research_date': aligned.date.max(),
                             'retrieval_date': cfg.source_date, 'publication_time': 'H.15 release posted 4:15 p.m. ET; retrospective series vintage'}])
    common.save(sources, '04', 'sources', modules=[sys.modules[__name__]])
    common.save(aligned[aligned.rate_change_bp.isna()], '04', 'calendar_exclusions', modules=[sys.modules[__name__]])
    common.save(aligned.head(10), '04', 'worked_observations', modules=[sys.modules[__name__]])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(aligned.date, aligned.yield_percent); axes[0].set_ylabel('10-year yield (%)')
    axes[1].plot(aligned.date, aligned.rate_change_bp, linewidth=.6); axes[1].set_ylabel('Daily change (bp)')
    axes[2].scatter(aligned.date, aligned.rate_change_bp.isna().astype(int), s=3); axes[2].set_ylabel('Missing change')
    fig.suptitle('Treasury input: levels, changes and calendar exclusions')
    common.figure(fig, '04', 'treasury_alignment', [aligned_path])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for window, group in coverage.groupby('window'):
        axes[0].plot(group.formation_date, group.valid_rate_days, label=f'{window} sessions')
        axes[1].plot(group.formation_date, group.represented_snapshot_weight, label=f'{window} sessions')
    axes[0].set_ylabel('Valid rate days'); axes[1].set_ylabel('Eligible snapshot weight'); axes[0].legend()
    common.figure(fig, '04', 'rate_coverage', [coverage_path])
    return coverage
