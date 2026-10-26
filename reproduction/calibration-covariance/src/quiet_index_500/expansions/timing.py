"""Notebook 09: matched one-equity-session lag conditional explanation."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_index_500.foundation import calendar
from . import common, rates, settings as cfg


def fit_lag_pair(y, f, z, s, previous_f, previous_z, previous_s, *, conditioned=True, asymmetric=True,
                 minimum=400, minimum_sign=100):
    raw = np.column_stack([y, f, z, s, previous_f, previous_z, previous_s])
    mask = np.isfinite(raw).all(axis=1)
    if mask.sum() < minimum:
        return {'eligible': False, 'reason': 'insufficient_after_lag', 'n': int(mask.sum())}
    yy, ff, zz, ss = y[mask], f[mask], z[mask], s[mask]
    pff, pzz, pss = previous_f[mask], previous_z[mask], previous_s[mask]
    project = np.column_stack([ff, zz]) if conditioned else ff
    projection, u = common.ols(common.design(project), ss)
    previous_project = np.column_stack([pff, pzz]) if conditioned else pff
    previous_u = pss - common.design(previous_project) @ projection
    signs = (int((u < 0).sum()), int((u >= 0).sum()))
    if asymmetric and min(signs) < minimum_sign:
        return {'eligible': False, 'reason': 'insufficient_current_signs', 'n': len(yy)}
    xx = np.column_stack([ff, zz, u])
    if asymmetric:
        xx = np.column_stack([xx, u * (u < 0)])
    base = common.design(xx)
    extended = np.column_stack([base, previous_u])
    b0, e0 = common.ols(base, yy)
    b1, e1 = common.ols(extended, yy)
    return {'eligible': True, 'reason': 'eligible', 'n': len(yy), 'projection': projection,
            'baseline_beta': b0, 'extension_beta': b1, 'negative_days': signs[0], 'nonnegative_days': signs[1],
            'in_sample_mse_change': np.mean(e0 ** 2) - np.mean(e1 ** 2)}


def evaluate_lag_pair(fit, y, f, z, s, previous_f, previous_z, previous_s, *, conditioned=True, asymmetric=True):
    raw = np.column_stack([y, f, z, s, previous_f, previous_z, previous_s])
    mask = np.isfinite(raw).all(axis=1)
    yy, ff, zz = y[mask], f[mask], z[mask]
    current = np.column_stack([ff, zz]) if conditioned else ff
    previous = np.column_stack([previous_f[mask], previous_z[mask]]) if conditioned else previous_f[mask]
    u = s[mask] - common.design(current) @ fit['projection']
    previous_u = previous_s[mask] - common.design(previous) @ fit['projection']
    xx = np.column_stack([ff, zz, u])
    if asymmetric:
        xx = np.column_stack([xx, u * (u < 0)])
    baseline = common.design(xx)
    e0, e1 = yy - baseline @ fit['baseline_beta'], yy - np.column_stack([baseline, previous_u]) @ fit['extension_beta']
    return {'future_n': len(yy), 'baseline_loss': np.mean(e0 ** 2) if len(yy) else np.nan,
            'extension_loss': np.mean(e1 ** 2) if len(yy) else np.nan, 'mask': mask,
            'baseline_errors': e0, 'extension_errors': e1}


def run(variant='main', progress=False):
    data = common.inputs()
    window, conditioned = cfg.variants[variant]
    f = data['factors'][cfg.factor_names]
    z, s = rates.daily_rates().rate_change_bp.reindex(data['sessions']), data['smh']
    # Shift on the complete equity calendar before any stock-specific rows are omitted.
    previous_f, previous_z, previous_s = f.shift(1), z.shift(1), s.shift(1)
    representatives = data['identifiers'].query('is_representative_security').index
    rows, eligibility, daily_rows = [], [], []
    for formation in calendar.formation_dates(data['sessions']):
        if not calendar.following_quarter_complete(formation, data['factor_cutoff']):
            continue
        dates = calendar.estimation_window(data['sessions'], min(formation, data['factor_cutoff']), window)
        future = calendar.following_quarter_sessions(data['sessions'], formation)
        def arrays(index, ticker):
            return (data['excess'].loc[index, ticker].to_numpy(), f.loc[index].to_numpy(), z.loc[index].to_numpy(),
                    s.loc[index].to_numpy(), previous_f.loc[index].to_numpy(), previous_z.loc[index].to_numpy(), previous_s.loc[index].to_numpy())
        for ticker in representatives:
            for asymmetric in (True, False):
                comparison = 'H4_lag' if asymmetric else 'linear_lag_supporting'
                if not common.security_window_valid(data['prices'], dates, ticker):
                    eligibility.append({'formation_date': formation, 'ticker': ticker, 'variant': variant,
                                        'comparison': comparison, 'block': common.historical_block(formation),
                                        'reason': 'window_crosses_same_security_boundary', 'training_n': 0, 'future_n': 0})
                    continue
                fit = fit_lag_pair(*arrays(dates, ticker), conditioned=conditioned, asymmetric=asymmetric,
                                   minimum=cfg.minimum_observations(window), minimum_sign=cfg.minimum_sign_observations(window))
                meta = {'formation_date': formation, 'ticker': ticker, 'variant': variant,
                        'comparison': comparison, 'block': common.historical_block(formation)}
                if not fit['eligible']:
                    eligibility.append({**meta, 'reason': fit['reason'], 'training_n': fit['n'], 'future_n': 0})
                    continue
                result = evaluate_lag_pair(fit, *arrays(future, ticker), conditioned=conditioned, asymmetric=asymmetric)
                accepted = result['future_n'] >= cfg.minimum_future_days
                eligibility.append({**meta, 'reason': 'eligible' if accepted else 'insufficient_future_days',
                                    'training_n': fit['n'], 'future_n': result['future_n']})
                if accepted:
                    rows.append({**meta, 'baseline_loss': result['baseline_loss'], 'extension_loss': result['extension_loss'],
                                 'improvement': result['baseline_loss'] - result['extension_loss'],
                                 'lag_coefficient': fit['extension_beta'][-1], 'training_n': fit['n'],
                                 'negative_current_u_days': fit['negative_days'], 'nonnegative_current_u_days': fit['nonnegative_days'],
                                 **{f'projection_{k}': b for k, b in enumerate(fit['projection'])},
                                 **{f'baseline_coef_{k}': b for k, b in enumerate(fit['baseline_beta'])},
                                 **{f'extension_coef_{k}': b for k, b in enumerate(fit['extension_beta'])},
                                 'future_n': result['future_n'], 'in_sample_mse_change': fit['in_sample_mse_change'],
                                 'snapshot_weight': data['identifiers'].at[ticker, 'issuer_snapshot_weight']})
        if progress:
            print('lag', variant, formation.date(), flush=True)
    company = pd.DataFrame(rows)
    quarters = company.groupby(['formation_date', 'block', 'variant', 'comparison'], as_index=False).agg(
        baseline_loss=('baseline_loss', 'mean'), extension_loss=('extension_loss', 'mean'), improvement=('improvement', 'mean'),
        companies=('ticker', 'size'), represented_snapshot_weight=('snapshot_weight', 'sum'), lag_coefficient=('lag_coefficient', 'mean'),
        training_n=('training_n', 'mean'), future_n=('future_n', 'mean'))
    quarters['controls'] = 'contemporaneous'
    for name, table in [('companies', company), ('quarters', quarters), ('eligibility', pd.DataFrame(eligibility))]:
        common.save(table, '09', name + '_' + variant, modules=[sys.modules[__name__]])
    summary = common.summarize_effects(quarters)
    path = common.save(summary, '09', 'summary_' + variant, modules=[sys.modules[__name__]])
    if variant == 'main':
        alignment = pd.DataFrame({'date': data['sessions'], 'previous_equity_session': pd.Series(data['sessions']).shift(1),
                                   'current_z_valid': z.notna().to_numpy(), 'previous_z_valid': previous_z.notna().to_numpy()})
        common.save(alignment, '09', 'lag_calendar', modules=[sys.modules[__name__]])
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for name, group in quarters.groupby('comparison'):
            axes[0].plot(group.formation_date, group.improvement, label=name)
            axes[1].plot(group.formation_date, group.lag_coefficient, label=name)
        axes[0].axhline(0, color='black', linewidth=.7); axes[0].set_ylabel('Conditional daily MSE improvement')
        axes[1].set_ylabel('Mean lag coefficient'); axes[0].legend()
        common.figure(fig, '09', 'lag_errors_and_coefficients', [path])
    return summary
