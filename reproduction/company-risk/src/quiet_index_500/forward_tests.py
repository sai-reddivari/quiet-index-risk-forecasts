"""Fixed-coefficient future risk outcomes and matched chronological cross-sectional tests.

Formation-level associations are separate from expanding pooled forecasts. All paired model
comparisons share their observations, and forecast scaling and category maps use training data
only. Inference resamples whole quarterly scores, preserving each underlying cross-section.
"""

import sys

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import calendar, loader
from quiet_index_500.research_io import save_table, make_notebook


standard_style_columns = ["m3_smb", "m3_hml", "m3_rmw", "m3_cma", "m3_mom"]
baseline_columns = ["m3_mkt_rf", "own_momentum", "trailing_residual_volatility"]
outcome_columns = ["log_forward_total_volatility", "log_forward_residual_volatility"]
factor_names = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


def forward_outcomes(company_panel, daily_returns, factors):
    """Future-quarter sample volatility, with formation model-3 coefficients held fixed."""
    panel = company_panel.copy()
    panel["formation_date"] = pd.to_datetime(panel.formation_date)
    records = []
    for formation, companies in panel.groupby("formation_date", sort=True):
        if not calendar.following_quarter_complete(formation, factors.index.max()):
            continue
        sessions = calendar.following_quarter_sessions(daily_returns.index, formation)
        future_returns = daily_returns.reindex(sessions)
        future_factors = factors.reindex(sessions)
        design = np.column_stack([np.ones(len(sessions)), future_factors[factor_names].to_numpy()])
        for _, company in companies.iterrows():
            ticker_returns = future_returns[company.ticker]
            coefficients = company[["m3_alpha", *[f"m3_{factor}" for factor in factor_names]]].to_numpy(dtype=float)
            fixed_prediction = pd.Series(design @ coefficients, index=sessions)
            residual = ticker_returns - future_factors.rf - fixed_prediction
            total_days, residual_days = ticker_returns.notna().sum(), residual.notna().sum()
            total_vol = ticker_returns.std(ddof=1) * np.sqrt(settings.annualisation_sessions) if total_days >= settings.forward_outcome_minimum_valid_days else np.nan
            residual_vol = residual.std(ddof=1) * np.sqrt(settings.annualisation_sessions) if residual_days >= settings.forward_outcome_minimum_valid_days else np.nan
            records.append({"formation_date": formation, "ticker": company.ticker, "cik": company.cik,
                            "outcome_quarter": str(calendar.outcome_quarter(formation)),
                            "outcome_last_session": sessions.max(),
                            "outcome_end": calendar.outcome_quarter(formation).end_time.normalize(),
                            "outcome_block": calendar.outcome_block(formation),
                            "outcome_sessions": len(sessions), "forward_total_days": int(total_days),
                            "forward_residual_days": int(residual_days),
                            "forward_total_volatility": total_vol, "forward_residual_volatility": residual_vol,
                            "log_forward_total_volatility": np.log(total_vol) if total_vol > 0 else np.nan,
                            "log_forward_residual_volatility": np.log(residual_vol) if residual_vol > 0 else np.nan})
    return pd.DataFrame(records)


def comparison_specifications(window_sessions):
    """Predefined paired models; exploration is labelled and never chosen from outcomes."""
    base = ["log_market_cap", *baseline_columns]
    styles = [*base, *standard_style_columns]
    semiconductor = [*styles, "m4_smh_excess"]
    specifications = [
        ("standard_styles", base, styles, "sector", "all", False),
        ("semiconductor", styles, semiconductor, "sector", "all", False),
        ("semiconductor_industry", styles, semiconductor, "industry", "all", False),
        ("liquidity", semiconductor, [*semiconductor, "log_amihud_252_per_million"], "sector", "all", False),
        ("size_interaction", semiconductor, [*semiconductor, "size_smh_interaction"], "sector", "all", False),
        ("semiconductor_no_size", [*baseline_columns, *standard_style_columns],
         [*baseline_columns, *standard_style_columns, "m4_smh_excess"], "sector", "all", False),
    ]
    if window_sessions == settings.main_window_sessions:
        specifications.extend([
            ("semiconductor_nonsemiconductors", styles, semiconductor, "sector", "nonsemiconductors", True),
            ("semiconductor_exclude_events", styles, semiconductor, "sector", "exclude_events", True),
            ("dollar_volume", semiconductor, [*semiconductor, "log_median_dollar_volume_60"], "sector", "all", True),
            ("semiconductor_variance_component", [*baseline_columns, *standard_style_columns],
             [*baseline_columns, *standard_style_columns, "semiconductor_variance_component"], "sector", "all", True),
        ])
    return [{"comparison": name, "restricted_columns": restricted, "extended_columns": extended,
             "controls": controls, "sample": sample, "exploratory": exploratory}
            for name, restricted, extended, controls, sample, exploratory in specifications]


def _category_labels(data, control, allowed_industries=None):
    sector = data.gics_sector_code.astype(str)
    if control == "sector":
        return "sector_" + sector
    industry = data.gics_industry_group_code.astype(str)
    return ("industry_" + industry).where(industry.isin(allowed_industries), "sector_remainder_" + sector)


def fit_predict_linear(training, evaluation, numeric_columns, outcome, controls):
    """OLS with scaling and category construction fitted only on the training frame.

    The returns include coefficient rows on the standardized scale and the number of test
    rows whose control category was absent from training. Such rows use the omitted-category
    intercept rather than an invented category coefficient and are counted explicitly.
    """
    mean = training[numeric_columns].mean()
    scale = training[numeric_columns].std(ddof=0).replace(0, 1)
    allowed = None
    if controls == "industry":
        counts = training.groupby("gics_industry_group_code").cik.nunique()
        allowed = set(counts[counts >= settings.industry_group_minimum_companies].index.astype(str))
    train_labels = _category_labels(training, controls, allowed)
    test_labels = _category_labels(evaluation, controls, allowed)
    categories = sorted(train_labels.unique())
    def design(frame, labels):
        continuous = ((frame[numeric_columns] - mean) / scale).to_numpy(dtype=float)
        dummies = np.column_stack([labels.eq(category).to_numpy(dtype=float) for category in categories[1:]]) if len(categories) > 1 else np.empty((len(frame), 0))
        return np.column_stack([np.ones(len(frame)), continuous, dummies])
    train_design, test_design = design(training, train_labels), design(evaluation, test_labels)
    response = training[outcome].to_numpy(dtype=float)
    # the normal equations are small; a symmetric pseudoinverse also retains any sparse redundant controls
    gram = train_design.T @ train_design
    inverse = np.linalg.pinv(gram, hermitian=True)
    coefficients = inverse @ (train_design.T @ response)
    fitted = train_design @ coefficients
    residual = response - fitted
    rank = np.linalg.matrix_rank(gram)
    variance = residual @ residual / (len(training) - rank) if len(training) > rank else np.nan
    standard_error = np.sqrt(np.maximum(np.diag(inverse) * variance, 0))
    terms = ["const", *numeric_columns, *[f"control:{category}" for category in categories[1:]]]
    coefficient_rows = pd.DataFrame({"term": terms, "coefficient": coefficients, "conditional_ols_se": standard_error,
                                    "training_mean": [0, *mean.tolist(), *([0] * (len(categories) - 1))],
                                    "training_scale": [1, *scale.tolist(), *([1] * (len(categories) - 1))]})
    return {"prediction": test_design @ coefficients, "coefficients": coefficient_rows,
            "rank": int(rank), "parameters": train_design.shape[1],
            "unseen_control_rows": int((~test_labels.isin(categories)).sum()),
            "pooled_industry_rows": int(train_labels.str.startswith("sector_remainder_").sum()),
            "pooled_industry_groups": int(training.loc[train_labels.str.startswith("sector_remainder_"), "gics_industry_group_code"].nunique())}


def _metrics(actual, predicted):
    errors = actual - predicted
    total_variation = np.sum((actual - actual.mean()) ** 2)
    return {"mse": np.mean(errors ** 2), "mae": np.mean(np.abs(errors)),
            "r_squared": 1 - np.sum(errors ** 2) / total_variation if total_variation > 0 else np.nan,
            "spearman": pd.Series(actual).corr(pd.Series(predicted), method="spearman")}


def evaluate_comparison(data, specification, outcome):
    """Paired per-quarter association and expanding forecast fits on identical samples."""
    source = data.copy()
    if specification["sample"] == "nonsemiconductors":
        source = source[source.gics_sub_industry_code.ne(settings.mechanical_overlap_subindustry_code)]
    elif specification["sample"] == "exclude_events":
        source = source[~source.ticker.isin(settings.event_convention_sensitivity_tickers)]
    columns = specification["extended_columns"]
    sample = source.dropna(subset=[outcome, *columns, "gics_sector_code", "gics_industry_group_code"])
    quarterly_rows, coefficients, predictions = [], [], []
    for formation, test in sample.groupby("formation_date", sort=True):
        possible = source[source.formation_date.eq(formation)]
        historical = sample[(sample.outcome_last_session <= formation) & (sample.formation_date < formation)]
        training_quarters = historical.formation_date.nunique()
        for analysis in ["association", "forecast"]:
            training = test if analysis == "association" else historical
            if len(test) <= len(columns) + 12 or (analysis == "forecast" and training_quarters < settings.forecast_minimum_training_quarters):
                continue
            restricted = fit_predict_linear(training, test, specification["restricted_columns"], outcome, specification["controls"])
            extended = fit_predict_linear(training, test, columns, outcome, specification["controls"])
            actual = test[outcome].to_numpy(dtype=float)
            low_metrics, high_metrics = _metrics(actual, restricted["prediction"]), _metrics(actual, extended["prediction"])
            common = {"formation_date": formation, "outcome_block": test.outcome_block.iloc[0],
                      "analysis": analysis, "comparison": specification["comparison"], "outcome": outcome,
                      "controls": specification["controls"], "sample": specification["sample"],
                      "exploratory": specification["exploratory"]}
            quarterly_rows.append({**common, "companies": len(test), "possible_companies": len(possible),
                                   "excluded_missing": len(possible) - len(test),
                                   "represented_snapshot_weight": test.issuer_snapshot_weight.sum(),
                                   "training_rows": len(training), "training_quarters": 1 if analysis == "association" else training_quarters,
                                   "latest_training_outcome_end": training.outcome_end.max(),
                                   "latest_training_outcome_last_session": training.outcome_last_session.max(),
                                   "restricted_rank": restricted["rank"], "extended_rank": extended["rank"],
                                   "extended_parameters": extended["parameters"],
                                   "unseen_control_rows": extended["unseen_control_rows"],
                                   "pooled_industry_rows": extended["pooled_industry_rows"],
                                   "pooled_industry_groups": extended["pooled_industry_groups"],
                                   **{f"restricted_{name}": value for name, value in low_metrics.items()},
                                   **{f"extended_{name}": value for name, value in high_metrics.items()},
                                   "mse_improvement": low_metrics["mse"] - high_metrics["mse"],
                                   "relative_mse_improvement": 1 - high_metrics["mse"] / low_metrics["mse"],
                                   "r_squared_improvement": high_metrics["r_squared"] - low_metrics["r_squared"]})
            for model, fit in [("restricted", restricted), ("extended", extended)]:
                coefficients.append(fit["coefficients"].assign(**common, model=model))
            if analysis == "forecast":
                prediction_table = test[["formation_date", "ticker", "cik", "outcome_block"]].copy()
                prediction_table["comparison"] = specification["comparison"]
                prediction_table["outcome"] = outcome
                prediction_table["actual"] = actual
                prediction_table["restricted_prediction"] = restricted["prediction"]
                prediction_table["extended_prediction"] = extended["prediction"]
                predictions.append(prediction_table)
    return {"quarters": pd.DataFrame(quarterly_rows),
            "coefficients": pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
            "predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()}


def _block_resample_indices(starts, block_length, count, circular=False):
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(len(starts), -1)[:, :count]
    return indices % count if circular else indices


def time_dependence_inference(values, seed=settings.bootstrap_random_seed):
    """Mean, Newey-West uncertainty and moving-block bootstrap over ordered quarter scores."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        return {"quarters": 0}
    average = values.mean()
    centered = values - average
    long_run_variance = centered @ centered / count
    lags = min(settings.newey_west_lags, count - 1)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        long_run_variance += 2 * weight * (centered[lag:] @ centered[:-lag]) / count
    standard_error = np.sqrt(max(long_run_variance, 0) / count)
    block_length = min(settings.bootstrap_block_length, count)
    generator = np.random.default_rng(seed)
    block_starts = generator.integers(0, count - block_length + 1,
                                     size=(settings.bootstrap_replications, int(np.ceil(count / block_length))))
    indices = _block_resample_indices(block_starts, block_length, count)
    bootstrapped_means = values[indices].mean(axis=1)
    circular_generator = np.random.default_rng(seed)
    circular_starts = circular_generator.integers(0, count,
                         size=(settings.bootstrap_replications, int(np.ceil(count / block_length))))
    circular_means = values[_block_resample_indices(circular_starts, block_length, count, circular=True)].mean(axis=1)
    return {"quarters": count, "mean": average, "newey_west_se": standard_error,
            "newey_west_t": average / standard_error if standard_error > 0 else np.nan,
            "bootstrap_low": np.quantile(bootstrapped_means, 0.025),
            "bootstrap_high": np.quantile(bootstrapped_means, 0.975),
            "circular_bootstrap_low": np.quantile(circular_means, 0.025),
            "circular_bootstrap_high": np.quantile(circular_means, 0.975),
            "positive_quarter_share": np.mean(values > 0), "newey_west_lags": lags,
            "bootstrap_block_length": block_length, "bootstrap_replications": settings.bootstrap_replications,
            "inference_conditions": "fixed estimated exposures; common quarter shocks preserved"}


def summarize_comparisons(quarterly):
    """Inference by development, validation, confirmation and pooled evaluation blocks."""
    output = []
    for keys, group in quarterly.groupby(["window_sessions", "analysis", "comparison", "outcome", "exploratory"], sort=True):
        common = dict(zip(["window_sessions", "analysis", "comparison", "outcome", "exploratory"], keys))
        blocks = [("all", group), *list(group.groupby("outcome_block", sort=True))]
        for block, rows in blocks:
            rows = rows.sort_values("formation_date")
            for metric in ["mse_improvement", "relative_mse_improvement", "r_squared_improvement"]:
                output.append({**common, "outcome_block": block, "metric": metric,
                               "mean_companies": rows.companies.mean(),
                               **time_dependence_inference(rows[metric])})
    return pd.DataFrame(output)


def summarize_added_coefficients(coefficients):
    """Quarterly mean association slopes for added terms on the within-quarter SD scale."""
    selected = coefficients[coefficients.analysis.eq("association") & coefficients.model.eq("extended")]
    term_by_comparison = {
        "semiconductor": ["m4_smh_excess"], "semiconductor_industry": ["m4_smh_excess"],
        "semiconductor_no_size": ["m4_smh_excess"], "semiconductor_nonsemiconductors": ["m4_smh_excess"],
        "semiconductor_exclude_events": ["m4_smh_excess"], "liquidity": ["log_amihud_252_per_million"],
        "dollar_volume": ["log_median_dollar_volume_60"], "size_interaction": ["size_smh_interaction"],
        "semiconductor_variance_component": ["semiconductor_variance_component"],
        "standard_styles": standard_style_columns,
    }
    output = []
    for keys, group in selected.groupby(["window_sessions", "comparison", "outcome", "term", "exploratory"], sort=True):
        common = dict(zip(["window_sessions", "comparison", "outcome", "term", "exploratory"], keys))
        if common["term"] not in term_by_comparison[common["comparison"]]:
            continue
        for block, rows in [("all", group), *list(group.groupby("outcome_block", sort=True))]:
            output.append({**common, "outcome_block": block,
                           "scale": "log volatility change per within-quarter characteristic SD, conditional on other columns",
                           **time_dependence_inference(rows.sort_values("formation_date").coefficient)})
    return pd.DataFrame(output)


def build_forward_tests():
    """Run the frozen forward tests, saving all statistics behind the findings."""
    returns, factors = loader.load_daily_returns(), loader.load_factors()
    all_quarters, all_coefficients, paths = [], [], {}
    freeze_path = settings.repository_root_folder / "docs/TEST_FREEZE.md"
    for window in [settings.main_window_sessions, settings.sensitivity_window_sessions]:
        panel_path = settings.licensed_results_folder / "panel" / f"company_panel_{window}.csv"
        panel = pd.read_csv(panel_path, parse_dates=["formation_date"])
        outcomes = forward_outcomes(panel, returns, factors)
        paths[f"outcomes_{window}"] = save_table(outcomes, "forward_tests", f"outcomes_{window}.csv",
                                                     inputs=[panel_path, freeze_path], source_modules=[sys.modules[__name__]],
                                                     note="Future residuals use fixed formation FF5+momentum coefficients, no future refit; complete quarters only.")
        data = panel.merge(outcomes, on=["formation_date", "ticker", "cik"], validate="one_to_one")
        data["size_smh_interaction"] = data.log_market_cap * data.m4_smh_excess
        data["semiconductor_variance_component"] = data.m4_smh_excess ** 2 * data.smh_residual_volatility ** 2
        quarter_tables, coefficient_tables = [], []
        for outcome in outcome_columns:
            outcome_predictions = []
            for specification in comparison_specifications(window):
                result = evaluate_comparison(data, specification, outcome)
                quarter_tables.append(result["quarters"])
                coefficient_tables.append(result["coefficients"])
                outcome_predictions.append(result["predictions"])
            predictions = pd.concat(outcome_predictions, ignore_index=True).assign(window_sessions=window)
            suffix = outcome.removeprefix("log_forward_")
            paths[f"predictions_{window}_{suffix}"] = save_table(predictions, "forward_tests", f"predictions_{window}_{suffix}.csv",
                    licensed=True, inputs=[panel_path, paths[f"outcomes_{window}"], freeze_path], source_modules=[sys.modules[__name__]],
                    note="Chronological paired predictions; all numeric scaling and control category maps fitted on earlier completed quarters only.")
        quarterly = pd.concat(quarter_tables, ignore_index=True).assign(window_sessions=window)
        coefficients = pd.concat(coefficient_tables, ignore_index=True).assign(window_sessions=window)
        for name, table in [("quarters", quarterly), ("coefficients", coefficients)]:
            paths[f"{name}_{window}"] = save_table(table, "forward_tests", f"{name}_{window}.csv", licensed=True,
                    inputs=[panel_path, paths[f"outcomes_{window}"], freeze_path], source_modules=[sys.modules[__name__]],
                    note="Paired samples match exactly; association is separate from forecasting; company-level representatives only.")
        all_quarters.append(quarterly)
        all_coefficients.append(coefficients)
    summary = summarize_comparisons(pd.concat(all_quarters, ignore_index=True))
    paths["inference"] = save_table(summary, "forward_tests", "paired_inference.csv", licensed=True,
                                   inputs=[paths["quarters_504"], paths["quarters_252"], freeze_path], source_modules=[sys.modules[__name__]],
                                   note="Whole-quarter score inference conditional on fixed exposures; six confirmation quarters do not justify broad significance claims.")
    paths["coefficient_inference"] = save_table(summarize_added_coefficients(pd.concat(all_coefficients, ignore_index=True)),
                    "forward_tests", "coefficient_inference.csv", licensed=True,
                    inputs=[paths["coefficients_504"], paths["coefficients_252"], freeze_path], source_modules=[sys.modules[__name__]],
                    note="Association slopes on standardized feature scale; quarter NW/block uncertainty conditional on estimated exposures. Association fit gains are mechanically nonnegative.")
    make_notebook(settings.repository_root_folder / "notebooks/08_forward_risk_tests.ipynb", "Forward risk tests",
                  "Fixed formation coefficients define future residual volatility. Association uses observed future outcomes; expanding forecasts use only earlier completed quarters. Paired comparisons share their eligible companies. Inference preserves quarter-level shocks and conditions on estimated exposures. See docs/findings/forward_tests.md for interpretation and the six-quarter confirmation limit.",
                  ["from quiet_index_500.forward_tests import notebook_summary\nnotebook_summary()"])
    return paths


def notebook_summary():
    """Display the principal confirmation comparison from saved inference tables."""
    table = pd.read_csv(settings.licensed_results_folder / "forward_tests/paired_inference.csv")
    return table[table.comparison.isin(["semiconductor", "semiconductor_industry", "semiconductor_no_size"])
                 & table.outcome_block.eq("confirmation") & table.metric.eq("relative_mse_improvement")]


if __name__ == "__main__":
    print(build_forward_tests(), flush=True)
