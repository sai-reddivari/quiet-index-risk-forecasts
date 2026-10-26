"""Numerical primitives, provenance and plotting shared by the component analyses."""

from functools import lru_cache
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_index_500 import research_io
from quiet_index_500.foundation import calendar, loader
from quiet_index_500.foundation.manifests import sha256_of_file, verify_file_against_manifest
from . import settings as cfg


def ols(x, y):
    """Least squares with an intercept supplied by the caller; no normal equations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('OLS received missing or nonfinite observations')
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError('Rank-deficient daily regression')
    return coefficients, y - x @ coefficients


def design(frame):
    values = np.asarray(frame, float)
    return np.column_stack([np.ones(len(values)), values])


def positive_log(value):
    return float(np.log(value)) if np.isfinite(value) and value > 0 else np.nan


def variance(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.var(values, ddof=1)) if len(values) > 1 else np.nan


def qlike(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if np.any(actual <= 0) or np.any(forecast <= 0):
        raise ValueError('QLIKE requires strictly positive variances')
    log_ratio = np.log(actual) - np.log(forecast)
    return np.expm1(log_ratio) - log_ratio


def historical_block(formation):
    return {'development': 'development', 'validation': 'historical_validation',
            'confirmation': 'historical_late', 'incomplete': 'incomplete'}[calendar.outcome_block(formation)]


def security_window_valid(prices, dates, ticker):
    return not (prices.before_same_security_start.loc[dates, ticker].any() or prices.after_history_end.loc[dates, ticker].any())


@lru_cache(maxsize=1)
def inputs():
    """All equity inputs use the fingerprint-verified accepted loader."""
    returns = loader.load_daily_returns()
    factors = loader.load_factors().reindex(returns.index)
    return {'returns': returns, 'excess': returns.sub(factors.rf, axis=0),
            'factors': factors, 'spy': loader.load_spy_excess_returns().reindex(returns.index),
            'smh': loader.load_smh_excess_returns().reindex(returns.index),
            'prices': loader.load_price_panels(), 'identifiers': loader.load_identifiers(),
            'caps': loader.load_issuer_market_cap(), 'sessions': returns.index,
            'factor_cutoff': loader.factor_data_last_date()}


def output_folder(component):
    path = cfg.result_root / component
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(table, component, name, *, licensed=False, extra_inputs=(), modules=()):
    """Save tabular evidence with the accepted research manifest conventions."""
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    if table.shape[1] == 0:
        table = pd.DataFrame(columns=['status'])
    modules = [sys.modules[__name__], cfg, *modules]
    upstream_modules = {'04': ['rates'], '05': ['rates', 'measurements'],
                        '06': ['rates', 'measurements', 'forecasts'],
                        '07': ['rates', 'measurements', 'forecasts', 'conditional'],
                        '08': ['rates', 'measurements', 'covariance'],
                        '09': ['rates', 'timing'], '10': ['synthesis']}[component]
    modules += [Path(__file__).parent / (name + '.py') for name in upstream_modules]
    dependencies = list(extra_inputs)
    if component in ('05', '06', '07', '08', '09'):
        dependencies.append(cfg.result_root / '04' / 'aligned_rates.csv')
    if component in ('06', '07', '08'):
        dependencies += sorted((cfg.result_root / '05').glob('panel_*.csv'))
    if component == '10' and licensed:
        dependencies += [path for folder in ['04', '05', '06', '07', '08', '09']
                         for path in sorted((cfg.result_root / folder).glob('*.csv'))]
    # Keep one discoverable evidence tree; the manifest retains FMP provenance.
    path = research_io.save_table(table, f'expansions/{component}', name + '.csv', licensed=False,
                                 inputs=[cfg.treasury_source('DGS10'), *dependencies,
                                         *([loader.market_cap_raw_path()] if licensed else [])],
                                 source_modules=modules,
                                 note=f'Step 0.5; results-informed historical study; FMP-derived={licensed}; interpretation reserved for staged review.')
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['licensed_market_cap_dependency'] = licensed
    manifest['historical_evaluation_not_fresh_confirmation'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def read(component, name, dates=()):
    path = cfg.result_root / component / (name + '.csv')
    verify_file_against_manifest(path)
    return pd.read_csv(path, parse_dates=list(dates))


def figure(fig, component, name, tables=()):
    path = output_folder(component) / (name + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    manifest = {'file': str(path.relative_to(cfg.root)), 'sha256': sha256_of_file(path),
                'pdf_sha256': sha256_of_file(path.with_suffix('.pdf')),
                'source_tables': [{'file': str(Path(p).relative_to(cfg.root)), 'sha256': sha256_of_file(p)} for p in tables]}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def block_interval(values, dates=None):
    """Circular four-quarter moving blocks, conditional on saved quarterly effects."""
    values = np.asarray(values, float)
    if dates is not None:
        periods = pd.PeriodIndex(pd.to_datetime(dates), freq='Q')
        if not periods.is_unique:
            raise ValueError('Quarterly effects must be unique')
        values = pd.Series(values, index=periods).reindex(pd.period_range(periods.min(), periods.max(), freq='Q')).to_numpy()
    count = int(np.isfinite(values).sum())
    if not count:
        return {'effect': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_two_sided': np.nan, 'quarters': 0}
    rng = np.random.default_rng(cfg.bootstrap_seed)
    starts = rng.integers(0, len(values), size=(cfg.bootstrap_draws, int(np.ceil(len(values) / cfg.bootstrap_block))))
    indices = ((starts[:, :, None] + np.arange(cfg.bootstrap_block)) % len(values)).reshape(cfg.bootstrap_draws, -1)[:, :len(values)]
    sampled = values[indices]
    accepted = np.isfinite(sampled).any(axis=1)
    boot = np.nanmean(sampled[accepted], axis=1)
    effect = np.nanmean(values)
    p_value = (1 + np.sum(np.abs(boot - effect) >= abs(effect))) / (len(boot) + 1)
    return {'effect': effect, 'ci_low': np.quantile(boot, .025), 'ci_high': np.quantile(boot, .975),
            'p_two_sided': p_value, 'quarters': count, 'calendar_quarters': len(values),
            'missing_quarters': len(values) - count}


def summarize_effects(quarters, grouping=('comparison', 'variant', 'controls')):
    rows = []
    for keys, group in quarters.groupby(list(grouping), dropna=False, sort=True):
        base = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for block, part in group.groupby('block', sort=True):
            part = part.sort_values('formation_date')
            row = {**base, 'block': block, **block_interval(part.improvement, part.formation_date)}
            if 'baseline_loss' in part:
                row.update(baseline_loss=part.baseline_loss.mean(), extension_loss=part.extension_loss.mean(),
                           relative_improvement=1 - part.extension_loss.mean() / part.baseline_loss.mean())
            rows.append(row)
    return pd.DataFrame(rows)
