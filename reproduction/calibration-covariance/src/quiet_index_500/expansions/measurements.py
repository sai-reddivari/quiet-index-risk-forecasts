"""Notebook 05: formation measurements and fixed-coefficient future risk targets."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quiet_index_500.foundation import calendar
from . import common, rates, settings as cfg


def _positive_mean(series, minimum):
    values = series.dropna()
    # An unchanged stock price produces a valid zero daily Amihud ratio.
    mean = values.mean() if len(values) >= minimum and (values >= 0).all() else np.nan
    return mean if mean > 0 else np.nan


def _positive_median(series, minimum):
    values = series.dropna()
    return values.median() if len(values) >= minimum and (values > 0).all() else np.nan


def activity_features(dollar_volume, amihud):
    """Input rows are the trailing 252 calendar sessions, not compressed valid rows."""
    recent_dv, prior_dv = dollar_volume.iloc[-60:], dollar_volume.iloc[:-60]
    recent_a, prior_a = amihud.iloc[-60:], amihud.iloc[:-60]
    a60, a252 = _positive_mean(recent_a, 40), _positive_mean(amihud, 200)
    dv60, dv252 = _positive_median(recent_dv, 40), _positive_median(dollar_volume, 200)
    ap, dvp = _positive_mean(prior_a, 150), _positive_median(prior_dv, 150)
    return {'log_dollar_volume_60': common.positive_log(dv60), 'log_dollar_volume_252': common.positive_log(dv252),
            'log_amihud_60': common.positive_log(a60), 'log_amihud_252': common.positive_log(a252),
            'deterioration': common.positive_log(a60 / ap) if ap > 0 else np.nan,
            'own_history_activity': common.positive_log(dv60 / dvp) if dvp > 0 else np.nan,
            'recent_activity_days': recent_dv.count(), 'prior_activity_days': prior_dv.count(),
            'recent_amihud_days': recent_a.count(), 'prior_amihud_days': prior_a.count()}


def fit_exposure_models(y, standard, rate, semiconductor):
    """All nested fits use the same finite mask; return the exact FWL check."""
    values = np.column_stack([standard, rate, semiconductor])
    mask = np.isfinite(y) & np.isfinite(values).all(axis=1)
    yy, ff, zz, ss = y[mask], standard[mask], rate[mask], semiconductor[mask]
    models = {}
    for name, xx in [('F', ff), ('FS', np.column_stack([ff, ss])),
                     ('FZ', np.column_stack([ff, zz])), ('FZS', np.column_stack([ff, zz, ss]))]:
        beta, residual = common.ols(common.design(xx), yy)
        models[name] = (beta, residual)
    projection_rate, u_rate = common.ols(common.design(np.column_stack([ff, zz])), ss)
    projection_equity, u_equity = common.ols(common.design(ff), ss)
    fwl_rate = np.dot(u_rate, models['FZ'][1]) / np.dot(u_rate, u_rate)
    fwl_equity = np.dot(u_equity, models['F'][1]) / np.dot(u_equity, u_equity)
    return {'models': models, 'mask': mask, 'projection_rate': projection_rate,
            'projection_equity': projection_equity, 'u_rate': u_rate, 'u_equity': u_equity,
            'fwl_error': max(abs(fwl_rate - models['FZS'][0][-1]), abs(fwl_equity - models['FS'][0][-1]))}


def build_variant(variant, progress=False):
    data = common.inputs()
    sessions, ids = data['sessions'], data['identifiers']
    window, rate_conditioned = cfg.variants[variant]
    z = rates.daily_rates().rate_change_bp.reindex(sessions)
    factors = data['factors'][cfg.factor_names]
    prices = data['prices']
    dollar_volume = prices.split_adjusted_close * prices.volume
    amihud = data['returns'].abs() / dollar_volume * cfg.foundation.amihud_scale_dollars
    rows = []
    for number, formation in enumerate(calendar.formation_dates(sessions)):
        cutoff = min(formation, data['factor_cutoff'])
        dates = calendar.estimation_window(sessions, cutoff, window)
        activity_dates = calendar.estimation_window(sessions, formation, 252)
        future_dates = calendar.following_quarter_sessions(sessions, formation)
        complete = calendar.following_quarter_complete(formation, data['factor_cutoff'])
        ff = factors.reindex(dates).to_numpy()
        zz, ss = z.reindex(dates).to_numpy(), data['smh'].reindex(dates).to_numpy()
        future_f = common.design(factors.reindex(future_dates).to_numpy())
        historical_y = data['excess'].reindex(dates)
        future_y_panel = data['excess'].reindex(future_dates)
        historical_dollar_volume = dollar_volume.reindex(activity_dates)
        historical_amihud = amihud.reindex(activity_dates)
        pos = sessions.get_loc(formation)
        for ticker, ident in ids.iterrows():
            y = historical_y[ticker].to_numpy()
            standard_mask = np.isfinite(y) & np.isfinite(ff).all(axis=1)
            joint_mask = standard_mask & np.isfinite(zz) & np.isfinite(ss)
            row = {'variant': variant, 'window': window, 'rate_conditioned': rate_conditioned,
                   'formation_date': formation, 'factor_measurement_cutoff': cutoff, 'ticker': ticker,
                   'cik': ident.cik, 'is_representative': bool(ident.is_representative_security),
                   'snapshot_weight': ident.snapshot_weight, 'issuer_snapshot_weight': ident.issuer_snapshot_weight,
                   'sector': str(ident.gics_sector_code), 'industry_group': str(ident.gics_industry_group_code),
                   'sector_name': ident.gics_sector, 'industry_group_name': ident.gics_industry_group,
                   'outcome_quarter': str(calendar.outcome_quarter(formation)), 'block': common.historical_block(formation),
                   'outcome_end': calendar.outcome_quarter(formation).end_time.normalize(),
                   'outcome_last_session': future_dates.max() if len(future_dates) else pd.NaT,
                   'standard_n': int(standard_mask.sum()), 'joint_n': int(joint_mask.sum()),
                   'target_days': 0, 'future_variance': np.nan, 'eligible_exposure': False,
                   'exclusion_reason': ''}
            cap = data['caps'].at[formation, ident.cik]
            row['log_market_cap'] = common.positive_log(cap)
            row.update(activity_features(historical_dollar_volume[ticker], historical_amihud[ticker]))
            row['own_momentum'] = np.nan
            if pos >= 252:
                history = prices.adjusted_close[ticker].iloc[pos - 252:pos + 1]
                between_returns = data['returns'][ticker].iloc[pos - 251:pos - 20]
                if history.iloc[0] > 0 and history.iloc[-22] > 0 and between_returns.count() >= 200:
                    # Invalid predecessor periods are already missing in the accepted price panel.
                    row['own_momentum'] = history.iloc[-22] / history.iloc[0] - 1
            if not common.security_window_valid(prices, dates, ticker):
                row['exclusion_reason'] = 'window_crosses_same_security_boundary'
                rows.append(row)
                continue
            if standard_mask.sum() < cfg.minimum_observations(window):
                row['exclusion_reason'] = 'insufficient_standard_factor_observations'
                rows.append(row)
                continue
            beta, residual = common.ols(common.design(ff[standard_mask]), y[standard_mask])
            row['ff_alpha'] = beta[0]
            row.update({f'ff_{f}': b for f, b in zip(cfg.factor_names, beta[1:])})
            trailing_v = common.variance(residual) * 252
            row['original_trailing_volatility'] = np.sqrt(np.sum(residual ** 2) / (len(residual) - len(beta)) * 252)
            historical_residual = pd.Series(np.nan, index=dates)
            historical_residual.loc[standard_mask] = residual
            recent = historical_residual.iloc[-60:].dropna()
            recent_v = common.variance(recent) * 252 if len(recent) >= 40 else np.nan
            row.update(trailing_variance=trailing_v, trailing_volatility=np.sqrt(trailing_v),
                       log_trailing_variance=common.positive_log(trailing_v), recent_variance=recent_v,
                       log_recent_ratio=common.positive_log(recent_v / trailing_v))
            original_mask = standard_mask & np.isfinite(ss)
            if original_mask.sum() >= cfg.minimum_observations(window):
                original_beta, _ = common.ols(common.design(np.column_stack([ff[original_mask], ss[original_mask]])), y[original_mask])
                row['original_lambda'] = original_beta[-1]
            if complete and len(future_dates):
                future_y = future_y_panel[ticker].to_numpy()
                future_e = future_y - future_f @ beta
                row['target_days'] = int(np.isfinite(future_e).sum())
                if row['target_days'] >= cfg.minimum_future_days:
                    row['future_variance'] = common.variance(future_e) * 252
            spy = data['spy'].reindex(dates).to_numpy()
            market_mask = np.isfinite(spy) & np.isfinite(y)
            negative = market_mask & (spy < 0)
            row['market_n'], row['market_negative_n'] = int(market_mask.sum()), int(negative.sum())
            if market_mask.sum() >= cfg.minimum_observations(window) and negative.sum() >= cfg.minimum_sign_observations(window):
                market_beta, _ = common.ols(common.design(spy[market_mask, None]), y[market_mask])
                down_beta, _ = common.ols(common.design(spy[negative, None]), y[negative])
                row['spy_beta'], row['spy_downside_amplification'] = market_beta[1], down_beta[1] - market_beta[1]
            if joint_mask.sum() < cfg.minimum_observations(window):
                row['exclusion_reason'] = 'insufficient_matched_factor_rate_smh_observations'
                rows.append(row)
                continue
            fits = fit_exposure_models(y, ff, zz, ss)
            selected = 'FZS' if rate_conditioned else 'FS'
            row['lambda'] = fits['models'][selected][0][-1]
            row['abs_lambda'] = abs(row['lambda'])
            row['rate_beta'] = fits['models']['FZS'][0][-2]
            row['smh_variance_component'] = 252 * row['lambda'] ** 2 * common.variance(fits['u_rate' if rate_conditioned else 'u_equity'])
            row['fwl_error'] = fits['fwl_error']
            projection = fits['projection_rate' if rate_conditioned else 'projection_equity']
            for k, value in enumerate(projection):
                row[f'projection_{k}'] = value
            for model, (coefs, errors) in fits['models'].items():
                row.update({f'{model}_coef_{k}': value for k, value in enumerate(coefs)})
                row[f'{model}_residual_variance'] = common.variance(errors) * 252
                row[f'{model}_r2'] = 1 - np.sum(errors ** 2) / np.sum((y[joint_mask] - y[joint_mask].mean()) ** 2)
            row['eligible_exposure'] = True
            rows.append(row)
        if progress:
            print(f'{variant}: {number + 1}/36 formations ({formation.date()})', flush=True)
    panel = pd.DataFrame(rows)
    panel['log_future_variance'] = np.log(panel.future_variance.where(panel.future_variance > 0))
    return panel


def read_panel(variant='main', companies=True):
    panel = common.read('05', f'panel_{variant}', ['formation_date', 'factor_measurement_cutoff', 'outcome_end', 'outcome_last_session'])
    return panel[panel.is_representative].copy() if companies else panel


def run(variant='main', progress=False):
    panel = build_variant(variant, progress)
    source = [sys.modules[__name__]]
    common.save(panel, '05', f'panel_{variant}', licensed=True, modules=source)
    coverage = panel.groupby(['formation_date', 'block'], as_index=False).agg(
        securities=('ticker', 'size'), eligible_exposures=('eligible_exposure', 'sum'),
        valid_targets=('future_variance', 'count'), max_fwl_error=('fwl_error', 'max'))
    coverage['variant'] = variant
    common.save(coverage, '05', f'coverage_{variant}', licensed=True, modules=source)
    if variant == 'main':
        company = panel[panel.is_representative]
        cols = ['log_market_cap', 'lambda', 'rate_beta', 'log_dollar_volume_60', 'log_amihud_252', 'deterioration', 'log_recent_ratio']
        correlations = company[cols].corr(method='spearman')
        cp = common.save(correlations.rename_axis('characteristic').reset_index(), '05', 'correlations', licensed=True, modules=source)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(company['lambda'].dropna(), bins=60); axes[0].set_xlabel('Rate-aware signed SMH loading'); axes[0].set_ylabel('Company-formations')
        axes[1].plot(coverage.formation_date, coverage.eligible_exposures, label='Eligible exposures')
        axes[1].plot(coverage.formation_date, coverage.valid_targets, label='Valid future targets'); axes[1].legend(); axes[1].tick_params(axis='x', rotation=30)
        common.figure(fig, '05', 'exposures_and_coverage')
        fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(cols)), cols, rotation=65, ha='right'); ax.set_yticks(range(len(cols)), cols); fig.colorbar(im, ax=ax, label='Pooled descriptive Spearman correlation')
        common.figure(fig, '05', 'characteristic_overlap', [cp])
    return coverage
