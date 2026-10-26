"""Notebook 06: chronological weighted risk forecasts, calibration and activity."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common, measurements, settings as cfg


def fit_predict(training, evaluation, columns, controls='sector', *, interaction=False,
                equal_quarters=True, smear=True):
    """Train-only transformations; quarter weights apply to fit, scaling and smearing."""
    x = training[columns].to_numpy(float)
    test = evaluation[columns].to_numpy(float)
    weights = 1 / training.groupby('formation_date').formation_date.transform('size').to_numpy(float) if equal_quarters else np.ones(len(training))
    weights /= weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0))
    scale[scale < 1e-12] = 1
    x, test = (x - mean) / scale, (test - mean) / scale
    terms = list(columns)
    if interaction:
        a, b = columns.index('abs_lambda'), columns.index('deterioration')
        x, test = np.column_stack([x, x[:, a] * x[:, b]]), np.column_stack([test, test[:, a] * test[:, b]])
        terms.append('abs_lambda_x_deterioration')
    train_sector, test_sector = training.sector.astype(str), evaluation.sector.astype(str)
    if controls == 'industry_group':
        counts = training.groupby('industry_group').cik.nunique()
        allowed = set(counts[counts >= 5].index.astype(str))
        train_group, test_group = training.industry_group.astype(str), evaluation.industry_group.astype(str)
        train_labels = ('industry_' + train_group).where(train_group.isin(allowed), 'sector_remainder_' + train_sector)
        test_labels = ('industry_' + test_group).where(test_group.isin(allowed), 'sector_remainder_' + test_sector)
    else:
        train_labels, test_labels = 'sector_' + train_sector, 'sector_' + test_sector
    categories = sorted(train_labels.unique())
    x = np.column_stack([np.ones(len(x)), x, *[train_labels.eq(c).to_numpy(float) for c in categories[1:]]])
    test = np.column_stack([np.ones(len(test)), test, *[test_labels.eq(c).to_numpy(float) for c in categories[1:]]])
    y = np.log(training.future_variance.to_numpy(float))
    beta, _, rank, _ = np.linalg.lstsq(x * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)
    residual = y - x @ beta
    smear_factor = np.sum(weights * np.exp(residual)) if smear else 1.
    log_prediction = test @ beta + np.log(smear_factor)
    predicted = np.exp(log_prediction)
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError('Invalid positive-variance forecast; no silent clipping allowed')
    coefficients = pd.DataFrame({'term': ['intercept', *terms, *['control:' + c for c in categories[1:]]],
                                  'coefficient': beta})
    coefficients['training_mean'] = [np.nan, *mean, *([np.nan] * (len(terms) - len(columns) + len(categories) - 1))]
    coefficients['training_scale'] = [np.nan, *scale, *([np.nan] * (len(terms) - len(columns) + len(categories) - 1))]
    return {'predicted': predicted, 'coefficients': coefficients, 'rank': rank, 'parameters': x.shape[1],
            'unseen_categories': int((~test_labels.isin(categories)).sum()), 'smearing_factor': smear_factor}


def score_pair(actual, baseline, extension):
    low, high = common.qlike(actual, baseline).mean(), common.qlike(actual, extension).mean()
    a = .5 * np.log(np.asarray(actual))
    b, e = a - .5 * np.log(baseline), a - .5 * np.log(extension)
    return {'baseline_loss': low, 'extension_loss': high, 'improvement': low - high,
            'baseline_logvol_mse': np.mean(b ** 2), 'extension_logvol_mse': np.mean(e ** 2),
            'baseline_mean_error_squared': b.mean() ** 2, 'extension_mean_error_squared': e.mean() ** 2,
            'baseline_centered_error': np.mean((b - b.mean()) ** 2), 'extension_centered_error': np.mean((e - e.mean()) ** 2),
            'baseline_spearman': pd.Series(a).corr(pd.Series(baseline), method='spearman'),
            'extension_spearman': pd.Series(a).corr(pd.Series(extension), method='spearman')}


def evaluate_models(panel, recipes, comparisons, *, variant='main', controls='sector',
                    common_columns=None, original_bridge=False):
    """Fit a recipe set on one intersection sample; do not select a winning recipe."""
    required = list(dict.fromkeys(common_columns or [c for recipe in recipes.values() for c in recipe['columns']]))
    sample = panel.dropna(subset=[*required, 'future_variance', 'sector', 'industry_group']).copy()
    sample = sample[sample.future_variance > 0]
    predictions, quarters, coefficients, eligibility = [], [], [], []
    for formation, test in sample.groupby('formation_date', sort=True):
        if not original_bridge and test.block.iloc[0] == 'development':
            continue
        if test.block.iloc[0] == 'incomplete':
            continue
        training = sample[(sample.formation_date < formation) & (sample.outcome_last_session <= formation)]
        minimum_quarters = 8 if original_bridge else 20
        if training.formation_date.nunique() < minimum_quarters:
            continue
        fits = {}
        for label, recipe in recipes.items():
            fit = fit_predict(training, test, recipe['columns'], controls,
                              interaction=recipe.get('interaction', False),
                              equal_quarters=not original_bridge, smear=not original_bridge)
            fits[label] = fit
            coefs = fit['coefficients'].assign(model=label, formation_date=formation, variant=variant, controls=controls)
            coefficients.append(coefs)
            pred = test[['formation_date', 'outcome_quarter', 'block', 'ticker', 'cik', 'future_variance']].copy()
            pred['model'], pred['predicted_variance'], pred['variant'], pred['controls'] = label, fit['predicted'], variant, controls
            predictions.append(pred)
        possible = panel[panel.formation_date.eq(formation)]
        eligibility.append({'formation_date': formation, 'variant': variant, 'controls': controls,
                            'companies': len(test), 'possible_companies': len(possible),
                            'represented_snapshot_weight': test.issuer_snapshot_weight.sum(),
                            'training_quarters': training.formation_date.nunique(), 'training_rows': len(training),
                            'latest_training_outcome_end': training.outcome_end.max(),
                            'latest_training_outcome_last_session': training.outcome_last_session.max(),
                            'latest_training_formation': training.formation_date.max()})
        for name, baseline, extension in comparisons:
            row = {'comparison': name, 'variant': variant, 'controls': controls,
                   'formation_date': formation, 'block': test.block.iloc[0],
                   'baseline_model': baseline, 'extension_model': extension, 'companies': len(test),
                   'represented_snapshot_weight': test.issuer_snapshot_weight.sum(),
                   'training_quarters': training.formation_date.nunique(),
                   'unseen_control_rows': fits[extension]['unseen_categories'],
                   **score_pair(test.future_variance.to_numpy(), fits[baseline]['predicted'], fits[extension]['predicted'])}
            quarters.append(row)
    return {'quarters': pd.DataFrame(quarters), 'predictions': pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
            'coefficients': pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
            'eligibility': pd.DataFrame(eligibility)}


def recipes():
    base, calibrated = cfg.common_predictors, cfg.calibrated_predictors
    return {'C0': {'columns': [*base, 'trailing_volatility']},
            'C_log': {'columns': [*base, 'log_trailing_variance']}, 'C': {'columns': calibrated},
            'C_activity60': {'columns': [*calibrated, 'log_dollar_volume_60']},
            'C_activity252': {'columns': [*calibrated, 'log_dollar_volume_252']},
            'C_amihud60': {'columns': [*calibrated, 'log_amihud_60']},
            'C_amihud252': {'columns': [*calibrated, 'log_amihud_252']},
            'C_own_activity': {'columns': [*calibrated, 'own_history_activity']}}


comparisons = [('RI1_calibration', 'C0', 'C'), ('calibration_log_shape', 'C0', 'C_log'),
               ('calibration_recency', 'C_log', 'C'), ('RI2_activity', 'C', 'C_activity60'),
               ('activity252', 'C', 'C_activity252'), ('amihud60', 'C', 'C_amihud60'),
               ('amihud252', 'C', 'C_amihud252'), ('own_history_activity', 'C', 'C_own_activity')]


def original_bridge(panel):
    """Narrow original-style comparison; target and FF6 fits preserve the original masks."""
    base = ['log_market_cap', *[f'ff_{f}' for f in cfg.factor_names], 'own_momentum', 'original_trailing_volatility']
    semi = [*base, 'original_lambda']
    definitions = {'old_styles': {'columns': base}, 'old_smh': {'columns': semi},
                   'old_activity': {'columns': [*semi, 'log_dollar_volume_60']},
                   'old_amihud': {'columns': [*semi, 'log_amihud_252']}}
    comparisons = [('old_semiconductor', 'old_styles', 'old_smh'), ('old_dollar_volume', 'old_smh', 'old_activity'),
                   ('old_amihud', 'old_smh', 'old_amihud')]
    tables = []
    for comparison, baseline, extension in comparisons:
        result = evaluate_models(panel, {baseline: definitions[baseline], extension: definitions[extension]},
                                 [(comparison, baseline, extension)], variant='original_bridge', original_bridge=True)
        tables.append(result['quarters'])
    quarters = pd.concat(tables, ignore_index=True)
    quarters['old_relative_mse_improvement'] = 1 - quarters.extension_logvol_mse / quarters.baseline_logvol_mse
    return quarters


def run(variant='main', controls='sector'):
    panel = measurements.read_panel(variant)
    result = evaluate_models(panel, recipes(), comparisons, variant=variant, controls=controls)
    suffix = variant + '_' + controls
    paths = {}
    for label, table in result.items():
        paths[label] = common.save(table, '06', label + '_' + suffix, licensed=True, modules=[sys.modules[__name__]])
    summary = common.summarize_effects(result['quarters'])
    summary_path = common.save(summary, '06', 'summary_' + suffix, licensed=True, modules=[sys.modules[__name__]])
    if variant == 'main' and controls == 'sector':
        bridge = original_bridge(panel)
        common.save(bridge, '06', 'original_bridge_quarters', licensed=True, modules=[sys.modules[__name__]])
        bridge_summary = bridge.groupby(['comparison', 'block'], as_index=False).agg(
            mean_original_relative_mse_improvement=('old_relative_mse_improvement', 'mean'), quarters=('formation_date', 'nunique'))
        common.save(bridge_summary, '06', 'original_bridge_summary', licensed=True, modules=[sys.modules[__name__]])
        q = result['quarters']
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for name in ['RI1_calibration', 'RI2_activity']:
            part = q[q.comparison.eq(name)]
            axes[0].plot(part.formation_date, part.improvement, marker='o', label=name)
            axes[1].plot(part.formation_date, part.baseline_mean_error_squared - part.extension_mean_error_squared, label=name + ' mean')
            axes[1].plot(part.formation_date, part.baseline_centered_error - part.extension_centered_error, linestyle='--', label=name + ' centered')
        for ax in axes: ax.axhline(0, color='black', linewidth=.7); ax.legend(fontsize=8)
        axes[0].set_ylabel('Variance QLIKE improvement'); axes[1].set_ylabel('Log-volatility MSE component improvement')
        common.figure(fig, '06', 'calibration_and_error_components', [paths['quarters']])
        chosen = summary[summary.comparison.isin(['RI2_activity', 'activity252', 'amihud60', 'amihud252', 'own_history_activity'])]
        fig, ax = plt.subplots(figsize=(11, 5))
        for j, (block, part) in enumerate(chosen.groupby('block')):
            ax.bar(np.arange(len(part)) + j * .35, part.effect, width=.35, label=block)
        ax.set_xticks(np.arange(len(part)) + .175, part.comparison, rotation=25, ha='right'); ax.axhline(0, color='black', linewidth=.7)
        ax.set_ylabel('Mean variance QLIKE improvement'); ax.legend()
        common.figure(fig, '06', 'activity_horizon_comparisons', [summary_path])
    return summary
