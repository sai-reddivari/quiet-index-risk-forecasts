"""Formation-fixed portfolio variance evaluation with paired model observations.

Every portfolio is selected before inspecting its future returns. Both factor
models are refitted to the same common complete training dates for every holding
in that portfolio. Factor and residual covariance use denominator n-1, so adding a
factor cannot gain or lose variance solely through a changed degrees correction.
"""

import sys

import numpy as np
import pandas as pd

from quiet_index_500 import factor_exposures, settings
from quiet_index_500.factor_exposures import standard_factor_names
from quiet_index_500.foundation import calendar, loader
from quiet_index_500.research_io import save_table


def formation_portfolios(exposures, identifiers, size_rows):
    """Return predetermined portfolio membership and normalized fixed target weights.

    Securities are factor-eligible at formation; both share classes are retained.
    Size membership is assigned once per issuer using the saved company quartile,
    then mapped to all its eligible share classes. Size portfolios and SMH halves
    are snapshot-weighted; the equal portfolio is equal-security-weighted.
    """
    eligible = exposures.loc[exposures["eligible"]].set_index("ticker")
    members = identifiers.loc[eligible.index].copy()
    members["smh_loading"] = eligible["m4_smh_excess"]
    quartile_by_cik = size_rows.set_index("cik")["size_quartile"]
    if not quartile_by_cik.index.is_unique:
        raise ValueError("size rows must contain one formation and one row per issuer")
    members["size_quartile"] = members["cik"].map(quartile_by_cik)
    median_loading = members["smh_loading"].median()
    assignments = {"snapshot": members.index, "equal": members.index,
                   "high_smh": members.index[members["smh_loading"] > median_loading],
                   "low_smh": members.index[members["smh_loading"] <= median_loading]}
    assignments.update({f"size_q{quartile}": members.index[members["size_quartile"] == quartile] for quartile in range(1, 5)})
    rows = []
    for portfolio, tickers in assignments.items():
        selected = members.loc[tickers]
        unnormalized = pd.Series(1.0, index=tickers) if portfolio == "equal" else selected["snapshot_weight"]
        weights = unnormalized / unnormalized.sum()
        for ticker in tickers:
            rows.append({"portfolio": portfolio, "ticker": ticker, "cik": int(selected.at[ticker, "cik"]),
                         "weight": float(weights[ticker]), "snapshot_weight": float(selected.at[ticker, "snapshot_weight"]),
                         "gics_industry_group_code": selected.at[ticker, "gics_industry_group_code"],
                         "formation_smh_loading": float(selected.at[ticker, "smh_loading"]),
                         "smh_split_median": float(median_loading) if portfolio.endswith("smh") else np.nan,
                         "licensed_size_dependency": portfolio.startswith("size_")})
    return pd.DataFrame(rows)


def predict_portfolio_variances(training_excess_returns, training_factors, weights, industry_groups, minimum_observations):
    """Fit the two models on one common matrix and return annualized predictions.

    The caller supplies fixed weights indexed by ticker. Missing dates are omitted
    jointly; no holding is removed. The result contains the exact training dates,
    training omission diagnostics and predictions for both residual treatments.
    """
    stocks = training_excess_returns.reindex(columns=weights.index)
    factor_names = [*standard_factor_names, "smh_excess"]
    factors = training_factors.reindex(stocks.index)[factor_names]
    common = stocks.notna().all(axis=1) & factors.notna().all(axis=1)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("fixed portfolio weights must sum to one")
    omissions = pd.DataFrame({"date": stocks.index[~common],
                              "missing_security_count": stocks.isna().sum(axis=1).loc[~common].to_numpy(),
                              "missing_factor_count": factors.isna().sum(axis=1).loc[~common].to_numpy()})
    observation_dates = stocks.index[common]
    if len(observation_dates) < minimum_observations:
        return {"eligible": False, "reason": "insufficient_common_training_dates", "observations": len(observation_dates),
                "observation_dates": observation_dates, "omissions": omissions, "predictions": pd.DataFrame()}
    response = stocks.loc[common].to_numpy(dtype=float)
    ordered_weights = weights.to_numpy(dtype=float)
    industry = industry_groups.reindex(weights.index)
    if industry.isna().any():
        raise ValueError("industry labels required for the fixed holdings")
    predictions = []
    for model, names in (("ff5_mom", standard_factor_names), ("ff5_mom_smh", factor_names)):
        design = np.column_stack([np.ones(len(response)), factors.loc[common, list(names)].to_numpy(dtype=float)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        if rank != design.shape[1]:
            raise ValueError("rank-deficient covariance factor design")
        fitted = design @ coefficients
        residual = response - fitted
        fitted_portfolio = fitted @ ordered_weights
        factor_variance = np.var(fitted_portfolio, ddof=1)
        diagonal_residual_variance = float((ordered_weights ** 2) @ np.var(residual, axis=0, ddof=1))
        industry_residual_variance = 0.0
        for group in industry.unique():
            selection = industry.to_numpy() == group
            weighted_group_residual = residual[:, selection] @ ordered_weights[selection]
            industry_residual_variance += float(np.var(weighted_group_residual, ddof=1))
        for residual_treatment, residual_variance in (("diagonal", diagonal_residual_variance), ("industry_block", industry_residual_variance)):
            predictions.append({"model": model, "residual_treatment": residual_treatment,
                                "predicted_variance": (factor_variance + residual_variance) * settings.annualisation_sessions,
                                "factor_variance": factor_variance * settings.annualisation_sessions,
                                "residual_variance": residual_variance * settings.annualisation_sessions,
                                "training_observations": len(response), "factor_design_rank": rank,
                                "covariance_denominator": len(response) - 1})
    return {"eligible": True, "reason": "eligible", "observations": len(response), "observation_dates": observation_dates,
            "omissions": omissions, "predictions": pd.DataFrame(predictions)}


def realized_portfolio_variance(future_excess_returns, future_factors, weights, minimum_observations):
    """Realized annualized variance with fixed weights and no missing-quote reweighting."""
    stocks = future_excess_returns.reindex(columns=weights.index)
    factors = future_factors.reindex(stocks.index)[[*standard_factor_names, "smh_excess"]]
    common = stocks.notna().all(axis=1) & factors.notna().all(axis=1)
    omissions = pd.DataFrame({"date": stocks.index[~common],
                              "missing_security_count": stocks.isna().sum(axis=1).loc[~common].to_numpy(),
                              "missing_factor_count": factors.isna().sum(axis=1).loc[~common].to_numpy()})
    portfolio_returns = stocks.loc[common] @ weights
    eligible = len(portfolio_returns) >= minimum_observations
    return {"eligible": eligible, "reason": "eligible" if eligible else "insufficient_common_future_dates",
            "observations": len(portfolio_returns),
            "realized_variance": float(portfolio_returns.var(ddof=1) * settings.annualisation_sessions) if eligible else np.nan,
            "returns": portfolio_returns, "omissions": omissions}


def variance_losses(predicted_variance, realized_variance):
    """Scale-free variance diagnostics; QLIKE is r/p-log(r/p)-1, lower is better."""
    if predicted_variance <= 0 or realized_variance <= 0:
        raise ValueError("variance loss requires positive predicted and realized variance")
    bias_ratio = predicted_variance / realized_variance
    inverse_ratio = 1 / bias_ratio
    log_ratio = float(np.log(bias_ratio))
    return {"bias_ratio": bias_ratio, "log_variance_ratio": log_ratio,
            "absolute_log_variance_ratio": abs(log_ratio),
            "qlike": inverse_ratio - np.log(inverse_ratio) - 1,
            "underestimated": predicted_variance < realized_variance}


def _block_resample_indices(starts, block_length, calendar_count, circular=False):
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(len(starts), -1)[:, :calendar_count]
    return indices % calendar_count if circular else indices


def _valid_resample_means(dense_values, indices):
    samples = dense_values[indices]
    finite_counts = np.isfinite(samples).sum(axis=1)
    accepted = finite_counts > 0
    means = np.nansum(samples[accepted], axis=1) / finite_counts[accepted]
    if len(means) == 0:
        raise ValueError("all bootstrap samples contain only missing quarters")
    return means, int((~accepted).sum())


def quarterly_mean_uncertainty(values, *, quarters=None, replications=settings.bootstrap_replications):
    """Quarter-level NW and moving-block mean uncertainty, conditional on estimates.

    Resampling units are whole quarters, preserving common shocks inside each
    portfolio. The bootstrap does not reestimate first-stage loadings or memberships.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("nonempty finite quarterly differences required")
    count = len(values)
    mean = float(values.mean())
    if quarters is None:
        dense_values = values
    else:
        periods = pd.PeriodIndex(pd.to_datetime(quarters), freq="Q")
        if not periods.is_unique or len(periods) != count:
            raise ValueError("one value per unique calendar quarter required")
        dense_values = pd.Series(values, index=periods).reindex(pd.period_range(periods.min(), periods.max(), freq="Q")).to_numpy()
    calendar_count = len(dense_values)
    centered = np.where(np.isfinite(dense_values), dense_values - mean, 0.0)  # missing quarters contribute no pair, never collapse a lag
    lag_count = min(settings.newey_west_lags, calendar_count - 1)
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, lag_count + 1):
        autocovariance = float(centered[lag:] @ centered[:-lag] / count)
        long_run_variance += 2 * (1 - lag / (lag_count + 1)) * autocovariance
    standard_error = np.sqrt(max(long_run_variance, 0) / count)
    block_length = min(settings.bootstrap_block_length, calendar_count)
    generator = np.random.default_rng(settings.bootstrap_random_seed)
    blocks_needed = int(np.ceil(calendar_count / block_length))
    starts = generator.integers(0, calendar_count - block_length + 1, size=(replications, blocks_needed))
    indices = _block_resample_indices(starts, block_length, calendar_count)
    bootstrap_means, all_missing = _valid_resample_means(dense_values, indices)
    circular_generator = np.random.default_rng(settings.bootstrap_random_seed)
    circular_starts = circular_generator.integers(0, calendar_count, size=(replications, blocks_needed))
    circular_indices = _block_resample_indices(circular_starts, block_length, calendar_count, circular=True)
    circular_means, circular_all_missing = _valid_resample_means(dense_values, circular_indices)
    return {"quarters": count, "calendar_quarters": calendar_count, "missing_quarters": calendar_count - count,
            "mean_difference": mean, "newey_west_standard_error": standard_error,
            "newey_west_lags": lag_count, "normal_interval_low": mean - 1.96 * standard_error,
            "normal_interval_high": mean + 1.96 * standard_error,
            "bootstrap_interval_low": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_interval_high": float(np.quantile(bootstrap_means, 0.975)),
            "bootstrap_block_length": block_length, "bootstrap_replications": replications,
            "bootstrap_valid_replications": len(bootstrap_means), "bootstrap_all_missing_replications": all_missing,
            "circular_bootstrap_interval_low": float(np.quantile(circular_means, 0.025)),
            "circular_bootstrap_interval_high": float(np.quantile(circular_means, 0.975)),
            "circular_bootstrap_valid_replications": len(circular_means),
            "circular_bootstrap_all_missing_replications": circular_all_missing}


def summarize_covariance_results(evaluations):
    """Period summaries and paired SMH-minus-baseline loss differences by portfolio."""
    valid = evaluations.loc[evaluations["evaluation_eligible"]].copy()
    summaries, differences, inference = [], [], []
    group_columns = ["window_sessions", "portfolio", "residual_treatment"]
    for keys, group in valid.groupby(group_columns, sort=True):
        identity = dict(zip(group_columns, keys))
        for block in ("all", "development", "validation", "confirmation"):
            period = group if block == "all" else group.loc[group["outcome_block"] == block]
            if period.empty:
                continue
            for model, model_rows in period.groupby("model"):
                summaries.append({**identity, "outcome_block": block, "model": model, "quarters": len(model_rows),
                                  "mean_bias_ratio": model_rows["bias_ratio"].mean(),
                                  "mean_log_variance_ratio": model_rows["log_variance_ratio"].mean(),
                                  "mean_absolute_log_variance_ratio": model_rows["absolute_log_variance_ratio"].mean(),
                                  "mean_qlike": model_rows["qlike"].mean(),
                                  "share_underestimated": model_rows["underestimated"].mean(),
                                  "mean_training_observations": model_rows["training_observations"].mean(),
                                  "mean_future_observations": model_rows["future_observations"].mean()})
            for metric in ("qlike", "absolute_log_variance_ratio", "log_variance_ratio"):
                paired = period.pivot(index="formation_date", columns="model", values=metric).dropna()
                if paired.empty:
                    continue
                differences_by_date = paired["ff5_mom_smh"] - paired["ff5_mom"]
                if block == "all":
                    for date, difference in differences_by_date.items():
                        differences.append({**identity, "formation_date": date, "metric": metric, "smh_minus_baseline": difference})
                inference.append({**identity, "outcome_block": block, "metric": metric,
                                  **quarterly_mean_uncertainty(differences_by_date.to_numpy(), quarters=differences_by_date.index)})
    return {"period_summary": pd.DataFrame(summaries), "paired_quarterly_differences": pd.DataFrame(differences),
            "paired_inference": pd.DataFrame(inference)}


def build_covariance_evaluation():
    """Run the frozen portfolio comparisons for both windows and save exact evidence."""
    freeze_path = settings.repository_root_folder / "docs/TEST_FREEZE.md"
    if not freeze_path.exists():
        raise RuntimeError("freeze the tests before evaluating outcomes")
    identifiers = loader.load_identifiers()
    price_panels = loader.load_price_panels()
    daily_returns = loader.daily_simple_returns_from_prices(price_panels.adjusted_close, price_panels.quarantined_returns)
    factors = loader.load_factors()
    excess_returns = loader.excess_returns(daily_returns, factors["rf"])
    common_factors = factors[list(standard_factor_names)].join(loader.load_smh_excess_returns().rename("smh_excess"))
    size_path = settings.licensed_results_folder / "size_liquidity/size.csv"
    size = pd.read_csv(size_path, parse_dates=["formation_date"])
    sessions = daily_returns.index
    factor_cutoff = loader.factor_data_last_date()
    evaluation_rows, membership_rows, omission_rows, realized_rows, training_rows = [], [], [], [], []
    exposure_paths = []
    for window_sessions, minimum in ((settings.main_window_sessions, settings.main_window_minimum_observations),
                                     (settings.sensitivity_window_sessions, settings.sensitivity_window_minimum_observations)):
        exposure_path = settings.results_folder / "factor_exposures" / f"exposures_{window_sessions}.csv"
        exposure_paths.append(exposure_path)
        exposures = pd.read_csv(exposure_path, parse_dates=["formation_date"])
        for formation_date, formation_exposures in exposures.groupby("formation_date", sort=True):
            cutoff = calendar.measurement_cutoff(formation_date, factor_cutoff)
            training_window = calendar.estimation_window(sessions, cutoff, window_sessions)
            future_window = calendar.following_quarter_sessions(sessions, formation_date)
            complete_outcome = calendar.following_quarter_complete(formation_date, factor_cutoff)
            memberships = formation_portfolios(formation_exposures, identifiers, size.loc[size["formation_date"] == formation_date])
            memberships["formation_date"] = formation_date
            memberships["window_sessions"] = window_sessions
            membership_rows.append(memberships)
            for portfolio, member_rows in memberships.groupby("portfolio", sort=True):
                weights = member_rows.set_index("ticker")["weight"]
                industry = identifiers["gics_industry_group_code"]
                common = {"formation_date": formation_date, "measurement_cutoff": cutoff, "window_sessions": window_sessions,
                          "portfolio": portfolio, "holdings": len(weights), "issuers": member_rows["cik"].nunique(),
                          "represented_snapshot_weight": member_rows["snapshot_weight"].sum(),
                          "outcome_quarter": str(calendar.outcome_quarter(formation_date)), "outcome_block": calendar.outcome_block(formation_date),
                          "outcome_complete": complete_outcome, "licensed_size_dependency": portfolio.startswith("size_")}
                predicted = predict_portfolio_variances(excess_returns.reindex(training_window), common_factors, weights, industry, minimum)
                training_rows.append({**common, "training_window_sessions": len(training_window), "training_observations": predicted["observations"],
                                      "training_omitted_dates": len(predicted["omissions"]), "training_eligible": predicted["eligible"],
                                      "training_reason": predicted["reason"], "minimum_observations": minimum,
                                      "window_first_session": training_window.min(), "window_last_session": training_window.max()})
                for row in predicted["omissions"].to_dict("records"):
                    omission_rows.append({**common, "phase": "training", **row})
                future = realized_portfolio_variance(excess_returns.reindex(future_window), common_factors, weights,
                                                     settings.forward_outcome_minimum_valid_days) if complete_outcome else None
                if future is not None:
                    for row in future["omissions"].to_dict("records"):
                        omission_rows.append({**common, "phase": "future", **row})
                    for date, value in future["returns"].items():
                        realized_rows.append({**common, "date": date, "portfolio_excess_return": value})
                evaluation_eligible = predicted["eligible"] and complete_outcome and future["eligible"]
                reason = (predicted["reason"] if not predicted["eligible"] else "incomplete_factor_outcome" if not complete_outcome
                          else future["reason"])
                prediction_rows = predicted["predictions"].to_dict("records") if predicted["eligible"] else [
                    {"model": model, "residual_treatment": treatment, "predicted_variance": np.nan,
                     "training_observations": predicted["observations"]}
                    for model in ("ff5_mom", "ff5_mom_smh") for treatment in ("diagonal", "industry_block")]
                for prediction in prediction_rows:
                    row = {**common, **prediction, "evaluation_eligible": bool(evaluation_eligible), "evaluation_reason": reason,
                           "training_omitted_dates": len(predicted["omissions"]), "future_sessions": len(future_window),
                           "future_observations": future["observations"] if future is not None else 0,
                           "future_omitted_dates": len(future["omissions"]) if future is not None else len(future_window),
                           "realized_variance": future["realized_variance"] if future is not None else np.nan}
                    if evaluation_eligible:
                        row.update(variance_losses(row["predicted_variance"], row["realized_variance"]))
                    evaluation_rows.append(row)
            print(f"covariance: {window_sessions} sessions, formation {formation_date.date()}", flush=True)
    tables = {"quarterly_evaluation": pd.DataFrame(evaluation_rows), "membership": pd.concat(membership_rows, ignore_index=True),
              "date_omissions": pd.DataFrame(omission_rows), "realized_daily_returns": pd.DataFrame(realized_rows),
              "training_coverage": pd.DataFrame(training_rows)}
    tables.update(summarize_covariance_results(tables["quarterly_evaluation"]))
    outputs = {}
    for name, table in tables.items():
        size_rows = table["portfolio"].str.startswith("size_")
        for licensed, selected in ((False, table.loc[~size_rows]), (True, table.loc[size_rows])):
            inputs = [freeze_path, *exposure_paths, *([size_path] if licensed else [])]
            path = save_table(selected, "covariance_evaluation", f"{name}.csv", licensed=licensed, inputs=inputs,
                              source_modules=[sys.modules[__name__], factor_exposures],
                              note="Formation-fixed holdings and target weights; both models refit on the same portfolio-specific complete training dates. "
                                   "Factor and residual covariances use n-1. Future common dates exclude any missing holding or factor; no holding reweighting. "
                                   "QLIKE is realized/predicted-log(realized/predicted)-1. Quarter uncertainty conditions on memberships and estimated loadings.")
            outputs[f"{'size_' if licensed else ''}{name}"] = path
    return outputs


def saved_covariance_summary():
    """Read public and size-dependent period summaries for the executed notebook."""
    return pd.concat([pd.read_csv(folder / "covariance_evaluation/period_summary.csv")
                      for folder in (settings.results_folder, settings.licensed_results_folder)], ignore_index=True)


if __name__ == "__main__":
    build_covariance_evaluation()
