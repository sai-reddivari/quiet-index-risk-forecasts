"""Notebook 07: semiconductor asymmetry and liquidity interaction forecasts."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_index_500.foundation import calendar
from . import common, forecasts, measurements, rates, settings as cfg


def estimate_gamma(y, ff, z, smh, rate_conditioned, minimum, minimum_sign):
    mask = np.isfinite(y) & np.isfinite(ff).all(axis=1) & np.isfinite(z) & np.isfinite(smh)
    if mask.sum() < minimum:
        return {'gamma': np.nan, 'asymmetry_n': int(mask.sum()), 'asymmetry_reason': 'insufficient_total_observations'}
    f, zz, s, yy = ff[mask], z[mask], smh[mask], y[mask]
    projection_inputs = np.column_stack([f, zz]) if rate_conditioned else f
    _, u = common.ols(common.design(projection_inputs), s)
    negative, nonnegative = int((u < 0).sum()), int((u >= 0).sum())
    result = {'asymmetry_n': len(u), 'negative_u_days': negative, 'nonnegative_u_days': nonnegative}
    if min(negative, nonnegative) < minimum_sign:
        return {**result, 'gamma': np.nan, 'asymmetry_reason': 'insufficient_sign_observations'}
    x = common.design(np.column_stack([f, zz, u, u * (u < 0)]))
    beta, errors = common.ols(x, yy)
    covariance = np.linalg.pinv(x.T @ x) * (errors @ errors / (len(u) - x.shape[1]))
    return {**result, 'gamma': beta[-1], 'positive_u_slope': beta[-2], 'negative_u_slope': beta[-2] + beta[-1],
            'gamma_conditional_ols_se': np.sqrt(max(covariance[-1, -1], 0)), 'asymmetry_reason': 'eligible'}


def features(variant='main', progress=False):
    data = common.inputs()
    window, conditioned = cfg.variants[variant]
    panel = measurements.read_panel(variant)
    z = rates.daily_rates().rate_change_bp
    rows = []
    for formation, companies in panel.groupby('formation_date', sort=True):
        dates = calendar.estimation_window(data['sessions'], min(formation, data['factor_cutoff']), window)
        ff = data['factors'].reindex(dates)[cfg.factor_names].to_numpy()
        zz, ss = z.reindex(dates).to_numpy(), data['smh'].reindex(dates).to_numpy()
        ys = data['excess'].reindex(dates)
        for ticker in companies.ticker:
            if not common.security_window_valid(data['prices'], dates, ticker):
                rows.append({'formation_date': formation, 'ticker': ticker, 'variant': variant,
                             'gamma': np.nan, 'asymmetry_n': 0, 'asymmetry_reason': 'window_crosses_same_security_boundary'})
                continue
            row = estimate_gamma(ys[ticker].to_numpy(), ff, zz, ss, conditioned,
                                 cfg.minimum_observations(window), cfg.minimum_sign_observations(window))
            rows.append({'formation_date': formation, 'ticker': ticker, 'variant': variant, **row})
        if progress:
            print('gamma', variant, formation.date(), flush=True)
    return pd.DataFrame(rows)


def run(variant='main', controls='sector', progress=False):
    feature_path = cfg.result_root / '07' / f'asymmetry_{variant}.csv'
    if controls == 'sector' or not feature_path.exists():
        gamma = features(variant, progress)
        common.save(gamma, '07', f'asymmetry_{variant}', modules=[sys.modules[__name__]])
    else:
        gamma = common.read('07', f'asymmetry_{variant}', ['formation_date'])
    panel = measurements.read_panel(variant).merge(gamma, on=['formation_date', 'ticker', 'variant'], how='left', validate='one_to_one')
    base = cfg.conditional_predictors
    recipes = {'B': {'columns': base}, 'B_gamma': {'columns': [*base, 'gamma']},
               'B_interaction': {'columns': base, 'interaction': True},
               'B_joint': {'columns': [*base, 'gamma'], 'interaction': True}}
    comparisons = [('H2_asymmetry', 'B', 'B_gamma'), ('H3_liquidity', 'B', 'B_interaction'),
                   ('joint_increment', 'B', 'B_joint'), ('gamma_after_interaction', 'B_interaction', 'B_joint'),
                   ('interaction_after_gamma', 'B_gamma', 'B_joint')]
    result = forecasts.evaluate_models(panel, recipes, comparisons, variant=variant, controls=controls)
    suffix = variant + '_' + controls
    paths = {}
    for label, table in result.items():
        paths[label] = common.save(table, '07', label + '_' + suffix, licensed=True, modules=[sys.modules[__name__]])
    summary = common.summarize_effects(result['quarters'])
    common.save(summary, '07', 'summary_' + suffix, licensed=True, modules=[sys.modules[__name__]])
    counts = gamma.groupby('asymmetry_reason', as_index=False).agg(company_formations=('ticker', 'size'))
    common.save(counts, '07', 'sign_eligibility_' + variant, modules=[sys.modules[__name__]])
    if variant == 'main' and controls == 'sector':
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(gamma.gamma.dropna(), bins=60); axes[0].set_xlabel('Semiconductor downside amplification gamma')
        slopes = gamma.groupby('formation_date')[['positive_u_slope', 'negative_u_slope']].median()
        slopes.plot(ax=axes[1]); axes[1].set_ylabel('Cross-company median slope')
        common.figure(fig, '07', 'asymmetry_measurements')
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for name in ['H2_asymmetry', 'H3_liquidity', 'joint_increment']:
            group = result['quarters'].query('comparison == @name')
            ax.plot(group.formation_date, group.improvement, marker='o', label=name)
        ax.axhline(0, color='black', linewidth=.7); ax.set_ylabel('Variance QLIKE improvement'); ax.legend()
        common.figure(fig, '07', 'conditional_forecast_comparisons', [paths['quarters']])
    return summary
