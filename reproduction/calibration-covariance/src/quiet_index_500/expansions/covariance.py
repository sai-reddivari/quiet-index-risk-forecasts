"""Notebook 08: matched portfolio covariance, rates and cross-industry accounting."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_index_500.foundation import calendar
from . import common, measurements, rates, settings as cfg


def residual_portfolio_variance(residuals, weights, groups, treatment):
    if treatment == 'diagonal':
        return float(np.var(residuals, axis=0, ddof=1) @ (weights ** 2))
    return sum(common.variance(residuals[:, groups == group] @ weights[groups == group]) for group in np.unique(groups))


def predict_four_models(y, f, z, s, weights, groups):
    """A common n-by-security response; annualized variance uses n minus 1 everywhere."""
    weights, groups = np.asarray(weights, float), np.asarray(groups)
    if not np.isclose(weights.sum(), 1):
        raise ValueError('Formation portfolio weights must sum to one')
    matrices = {'F': f, 'FS': np.column_stack([f, s]), 'FZ': np.column_stack([f, z]),
                'FZS': np.column_stack([f, z, s])}
    fits, rows = {}, []
    for model, inputs in matrices.items():
        xx = common.design(inputs)
        beta, residual = common.ols(xx, y)
        fits[model] = {'beta': beta, 'residual': residual}
        factor_eigenvalue = np.linalg.eigvalsh(np.atleast_2d(np.cov(inputs, rowvar=False, ddof=1))).min()
        residual_eigenvalue = min(np.linalg.eigvalsh(np.atleast_2d(np.cov(residual[:, groups == group], rowvar=False, ddof=1))).min()
                                 for group in np.unique(groups))
        if min(factor_eigenvalue, residual_eigenvalue) < -1e-10:
            raise AssertionError('A sample covariance component is not numerically positive semidefinite')
        factor_variance = common.variance(xx @ beta @ weights)
        for treatment in ['industry_block', 'diagonal']:
            rv = residual_portfolio_variance(residual, weights, groups, treatment)
            rows.append({'model': model, 'residual_treatment': treatment, 'predicted_variance': 252 * (factor_variance + rv),
                         'factor_variance': 252 * factor_variance, 'residual_variance': 252 * rv,
                         'minimum_factor_covariance_eigenvalue': factor_eigenvalue,
                         'minimum_residual_block_eigenvalue': residual_eigenvalue,
                         'training_n': len(y), 'covariance_denominator': len(y) - 1})
    identities, contributions = [], []
    for base, extended, projection_inputs in [('F', 'FS', f), ('FZ', 'FZS', np.column_stack([f, z]))]:
        projection, u = common.ols(common.design(projection_inputs), s)
        lam = fits[extended]['beta'][-1]
        grouped = {str(group): float(weights[groups == group] @ lam[groups == group]) for group in np.unique(groups)}
        var_u = common.variance(u)
        for treatment in ['industry_block', 'diagonal']:
            subtraction = sum(g ** 2 for g in grouped.values()) if treatment == 'industry_block' else np.sum((weights * lam) ** 2)
            analytical = 252 * var_u * ((weights @ lam) ** 2 - subtraction)
            direct = next(r['predicted_variance'] for r in rows if r['model'] == extended and r['residual_treatment'] == treatment) - next(r['predicted_variance'] for r in rows if r['model'] == base and r['residual_treatment'] == treatment)
            identities.append({'baseline_model': base, 'extension_model': extended, 'residual_treatment': treatment,
                               'direct_variance_change': direct, 'analytical_variance_change': analytical,
                               'identity_error': abs(direct - analytical), 'training_u_variance': 252 * var_u,
                               'portfolio_loading': weights @ lam})
        for group, loading in grouped.items():
            contributions.append({'baseline_model': base, 'industry_group': group, 'weighted_loading': loading,
                                  'variance_contribution': 252 * var_u * loading * (sum(grouped.values()) - loading)})
        fits[extended].update(projection=projection, u_variance=var_u, group_loadings=grouped)
    return pd.DataFrame(rows), fits, identities, contributions


def diagnose_future(fits, y, f, z, s, weights, groups):
    """Hindsight measurements use a labeled subset; they never replace the scored forecast."""
    rows = []
    for base, extended, inputs in [('F', 'FS', f), ('FZ', 'FZS', np.column_stack([f, z]))]:
        fit = fits[extended]
        u = s - common.design(inputs) @ fit['projection']
        beta_future, _ = common.ols(common.design(np.column_stack([inputs, s])), y)
        future_errors = y - common.design(np.column_stack([inputs, s])) @ fit['beta']
        future_cross_residual = common.variance(future_errors @ weights) - residual_portfolio_variance(future_errors, weights, groups, 'industry_block')
        frozen_loading = fit['beta'][-1]
        g = np.array(list(fit['group_loadings'].values()))
        rows.append({'baseline_model': base, 'diagnostic_future_n': len(y),
                     'training_u_variance': 252 * fit['u_variance'], 'future_u_variance': 252 * common.variance(u),
                     'frozen_portfolio_loading': weights @ frozen_loading,
                     'future_refitted_portfolio_loading': weights @ beta_future[-1],
                     'future_cross_group_frozen_residual_covariance': 252 * future_cross_residual,
                     'frozen_loading_adjustment_at_future_u_variance': 252 * common.variance(u) * ((g.sum()) ** 2 - np.sum(g ** 2)),
                     'hindsight_only': True})
    return rows


def run(variant='main', progress=False):
    data, panel = common.inputs(), measurements.read_panel(variant, companies=False)
    window = cfg.variants[variant][0]
    z = rates.daily_rates().rate_change_bp.reindex(data['sessions'])
    f, s = data['factors'][cfg.factor_names], data['smh']
    evaluations, identity_rows, contribution_rows, diagnostics, memberships, exclusions = [], [], [], [], [], []
    for formation, securities in panel.groupby('formation_date', sort=True):
        if not calendar.following_quarter_complete(formation, data['factor_cutoff']):
            continue
        eligible = securities[securities.eligible_exposure].sort_values(['lambda', 'ticker']).copy()
        half = (len(eligible) + 1) // 2
        portfolios = {'snapshot': eligible, 'equal': eligible, 'low_smh': eligible.iloc[:half], 'high_smh': eligible.iloc[half:]}
        dates = calendar.estimation_window(data['sessions'], min(formation, data['factor_cutoff']), window)
        future_dates = calendar.following_quarter_sessions(data['sessions'], formation)
        ff, zz, ss = f.reindex(dates).to_numpy(), z.reindex(dates).to_numpy(), s.reindex(dates).to_numpy()
        future_ff, future_zz, future_ss = f.reindex(future_dates).to_numpy(), z.reindex(future_dates).to_numpy(), s.reindex(future_dates).to_numpy()
        for portfolio, selected in portfolios.items():
            names = selected.ticker.tolist()
            weights = np.ones(len(names)) if portfolio == 'equal' else selected.snapshot_weight.to_numpy(float)
            weights = weights / weights.sum()
            groups = selected.industry_group.astype(str).to_numpy()
            y = data['excess'].reindex(dates)[names].to_numpy()
            mask = np.isfinite(y).all(axis=1) & np.isfinite(np.column_stack([ff, zz, ss])).all(axis=1)
            future_y = data['excess'].reindex(future_dates)[names].to_numpy()
            future_mask = np.isfinite(future_y).all(axis=1)
            meta = {'formation_date': formation, 'variant': variant, 'portfolio': portfolio, 'block': common.historical_block(formation)}
            for ticker, weight in zip(names, weights):
                memberships.append({**meta, 'ticker': ticker, 'weight': weight, 'group': str(data['identifiers'].at[ticker, 'gics_industry_group_code'])})
            reason = 'eligible'
            if mask.sum() < cfg.minimum_observations(window):
                reason = 'insufficient_common_training_dates'
            elif future_mask.sum() < cfg.minimum_future_days:
                reason = 'insufficient_common_future_dates'
            exclusions.append({**meta, 'reason': reason, 'holdings': len(names), 'training_n': mask.sum(),
                               'future_n': future_mask.sum(), 'represented_snapshot_weight': selected.snapshot_weight.sum(),
                               'omitted_training_dates': '|'.join(dates[~mask].strftime('%Y-%m-%d')),
                               'omitted_future_dates': '|'.join(future_dates[~future_mask].strftime('%Y-%m-%d'))})
            if reason != 'eligible':
                continue
            realized = 252 * common.variance(future_y[future_mask] @ weights)
            predictions, fits, identities, contributions = predict_four_models(y[mask], ff[mask], zz[mask], ss[mask], weights, groups)
            predictions = predictions.assign(**meta, realized_variance=realized, future_n=int(future_mask.sum()),
                                             represented_snapshot_weight=selected.snapshot_weight.sum(), holdings=len(names))
            predictions['qlike'] = common.qlike(realized, predictions.predicted_variance)
            predictions['predicted_realized_ratio'] = predictions.predicted_variance / realized
            evaluations.extend(predictions.to_dict('records'))
            identity_rows.extend([{**meta, **row} for row in identities])
            contribution_rows.extend([{**meta, **row} for row in contributions])
            diagnostic_mask = future_mask & np.isfinite(np.column_stack([future_ff, future_zz, future_ss])).all(axis=1)
            if diagnostic_mask.sum() >= cfg.minimum_future_days:
                values = diagnose_future(fits, future_y[diagnostic_mask], future_ff[diagnostic_mask], future_zz[diagnostic_mask],
                                         future_ss[diagnostic_mask], weights, groups)
                diagnostics.extend([{**meta, **row} for row in values])
        if progress:
            print('covariance', variant, formation.date(), flush=True)
    scores = pd.DataFrame(evaluations)
    quarters = []
    for keys, group in scores.groupby(['formation_date', 'block', 'portfolio', 'residual_treatment']):
        formation, block, portfolio, treatment = keys
        losses = group.set_index('model').qlike
        ge, gr = losses.F - losses.FS, losses.FZ - losses.FZS
        quarters.append({'formation_date': formation, 'block': block, 'portfolio': portfolio, 'residual_treatment': treatment,
                         'variant': variant, 'controls': treatment, 'comparison': 'H7_rates', 'improvement': ge - gr,
                         'G_equity': ge, 'G_rates': gr, 'rate_baseline_improvement': losses.F - losses.FZ,
                         'material_attenuation': bool(ge > 0 and gr <= .5 * ge),
                         'original_advantage_present': bool(ge > 0)})
    quarters = pd.DataFrame(quarters)
    summary = common.summarize_effects(quarters, ('comparison', 'variant', 'portfolio', 'residual_treatment'))
    gain_summary = quarters.groupby(['comparison', 'variant', 'portfolio', 'residual_treatment', 'block'], as_index=False).agg(
        mean_G_equity=('G_equity', 'mean'), mean_G_rates=('G_rates', 'mean'),
        mean_rate_baseline_improvement=('rate_baseline_improvement', 'mean'))
    summary = summary.merge(gain_summary, on=['comparison', 'variant', 'portfolio', 'residual_treatment', 'block'], validate='one_to_one')
    summary['original_advantage_present'] = summary.mean_G_equity > 0
    summary['material_attenuation'] = (summary.mean_G_equity > 0) & (summary.mean_G_rates <= .5 * summary.mean_G_equity)
    tables = {'evaluations': scores, 'quarters': quarters, 'summary': summary, 'identity_checks': pd.DataFrame(identity_rows),
              'group_contributions': pd.DataFrame(contribution_rows), 'hindsight_diagnostics': pd.DataFrame(diagnostics),
              'memberships': pd.DataFrame(memberships), 'eligibility': pd.DataFrame(exclusions)}
    paths = {}
    for name, table in tables.items():
        paths[name] = common.save(table, '08', name + '_' + variant, modules=[sys.modules[__name__]])
    if variant == 'main':
        chosen = scores.query("portfolio == 'snapshot' and residual_treatment == 'industry_block' and block != 'development'")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for model, group in chosen.groupby('model'):
            axes[0].plot(group.formation_date, group.predicted_realized_ratio, marker='o', label=model)
            axes[1].plot(group.formation_date, group.qlike, label=model)
        axes[0].axhline(1, color='black', linewidth=.7); axes[0].set_yscale('log'); axes[0].set_ylabel('Predicted / realized variance')
        axes[1].set_ylabel('Variance QLIKE'); axes[0].legend(ncol=4)
        common.figure(fig, '08', 'broad_portfolio_forecast_accuracy', [paths['evaluations']])
        contributions = tables['group_contributions'].query("baseline_model == 'FZ' and block == 'historical_late'")
        pivot = contributions.groupby(['portfolio', 'industry_group']).variance_contribution.mean().unstack(0)
        fig, ax = plt.subplots(figsize=(12, 5)); pivot.plot.bar(ax=ax); ax.axhline(0, color='black', linewidth=.7)
        ax.set_ylabel('Mean signed annualized cross-group contribution'); ax.set_xlabel('GICS industry group')
        common.figure(fig, '08', 'signed_industry_covariance_contributions', [paths['group_contributions']])
        diag = tables['hindsight_diagnostics'].query("portfolio == 'snapshot' and baseline_model == 'FZ' and block != 'development'")
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for col in ['training_u_variance', 'future_u_variance']:
            axes[0].plot(diag.formation_date, diag[col], label=col)
        for col in ['frozen_portfolio_loading', 'future_refitted_portfolio_loading']:
            axes[1].plot(diag.formation_date, diag[col], label=col)
        axes[2].plot(diag.formation_date, diag.future_cross_group_frozen_residual_covariance)
        axes[0].set_ylabel('SMH residual variance'); axes[1].set_ylabel('Portfolio loading'); axes[2].set_ylabel('Future residual cross-group covariance')
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); fig.suptitle('Hindsight diagnostics: excluded from scored formation forecasts')
        common.figure(fig, '08', 'covariance_error_diagnostics', [paths['hindsight_diagnostics']])
    return summary
