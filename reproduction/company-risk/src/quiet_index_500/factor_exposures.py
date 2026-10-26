"""Quarterly security factor estimates on one common observation mask per security.

Standard errors are classical OLS conditional diagnostics, not dependence-robust
research inference. Every model includes an intercept. Residual risk uses SSE /
(n - rank), while residual variance share uses SSE / centered total sum of squares.
The four nested models share exactly the same stock, factor and SMH observations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import calendar, eligibility, loader
from quiet_index_500.research_io import save_table

standard_factor_names = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
model_factor_names = {
    "market": ("mkt_rf",),
    "market_smh": ("mkt_rf", "smh_excess"),
    "ff5_mom": standard_factor_names,
    "ff5_mom_smh": (*standard_factor_names, "smh_excess"),
}
model_prefixes = {"market": "m1", "market_smh": "m2", "ff5_mom": "m3", "ff5_mom_smh": "m4"}


def _ols(response, design):
    coefficient, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("rank-deficient factor design")
    residual = response - design @ coefficient
    residual_sum_squares = float(residual @ residual)
    residual_degrees_freedom = len(response) - rank
    if residual_degrees_freedom <= 0:
        raise ValueError("insufficient residual degrees of freedom")
    residual_variance = residual_sum_squares / residual_degrees_freedom
    coefficient_covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_error = np.sqrt(np.maximum(np.diag(coefficient_covariance), 0))
    centered_response = response - response.mean()
    total_sum_squares = float(centered_response @ centered_response)
    r_squared = 1 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    adjusted_r_squared = 1 - (1 - r_squared) * (len(response) - 1) / residual_degrees_freedom
    return {"coefficient": coefficient, "standard_error": standard_error, "residual": residual,
            "nobs": len(response), "rank": rank, "residual_degrees_freedom": residual_degrees_freedom,
            "residual_variance": residual_variance, "residual_sample_variance": residual_sum_squares / (len(response) - 1),
            "r_squared": r_squared, "adjusted_r_squared": adjusted_r_squared,
            "residual_share_of_variance": 1 - r_squared}


def estimate_factor_models(security_excess_returns, factors_with_smh, minimum_observations):
    """Fit all models for one security on identical dates; return fits and FWL diagnostics.

    Inputs are a date-indexed decimal excess-return Series and a date-indexed frame
    holding FF5, mom and smh_excess. The caller imposes the shared security-history
    eligibility boundary before invoking this numerical estimator.
    """
    required_names = list(model_factor_names["ff5_mom_smh"])
    joint = factors_with_smh[required_names].reindex(security_excess_returns.index).copy()
    joint.insert(0, "security", security_excess_returns)
    joint = joint.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joint) < minimum_observations:
        raise ValueError("insufficient complete common observations")
    response = joint["security"].to_numpy(dtype=float)
    fits = {}
    for model, names in model_factor_names.items():
        design = np.column_stack([np.ones(len(joint)), joint[list(names)].to_numpy(dtype=float)])
        fits[model] = _ols(response, design)
        fits[model]["terms"] = ("const", *names)
    standard_design = np.column_stack([np.ones(len(joint)), joint[list(standard_factor_names)].to_numpy(dtype=float)])
    semiconductor_fit = _ols(joint["smh_excess"].to_numpy(dtype=float), standard_design)
    semiconductor_residual = semiconductor_fit["residual"]
    semiconductor_residual_sum_squares = float(semiconductor_residual @ semiconductor_residual)
    fwl_loading = float(semiconductor_residual @ fits["ff5_mom"]["residual"] / semiconductor_residual_sum_squares)
    joint_loading = float(fits["ff5_mom_smh"]["coefficient"][-1])
    return {"models": fits, "observation_dates": joint.index,
            "fwl_loading": fwl_loading, "fwl_difference": joint_loading - fwl_loading,
            "smh_factor_r_squared": semiconductor_fit["r_squared"],
            "smh_residual_volatility": np.sqrt(semiconductor_fit["residual_variance"] * settings.annualisation_sessions),
            "smh_residual_variance": semiconductor_fit["residual_variance"]}


def apply_formation_coefficients(coefficients, factor_returns, model="ff5_mom"):
    """Apply one formation's saved long coefficients to later decimal factor returns.

    Returns a date-by-ticker fitted excess-return frame. No fitting occurs. The
    coefficient table must contain exactly one formation and at most one window;
    callers select the intended vintage explicitly. Missing required factors yield
    missing predictions. Subtract the result from future security excess returns
    to obtain the fixed-coefficient residuals used by forward evaluation.
    """
    if model not in model_factor_names:
        raise ValueError(f"unknown model {model}")
    selected = coefficients.loc[coefficients["model"] == model].copy()
    if selected.empty:
        return pd.DataFrame(index=factor_returns.index)
    for key in ("formation_date", "window_sessions"):
        if key in selected and selected[key].nunique(dropna=False) != 1:
            raise ValueError(f"select exactly one {key} before applying coefficients")
    terms = ["const", *model_factor_names[model]]
    coefficient_matrix = selected.pivot(index="term", columns="ticker", values="coefficient").reindex(terms)
    if coefficient_matrix.isna().any().any():
        raise ValueError("incomplete formation coefficient vector")
    factor_design = factor_returns[list(model_factor_names[model])].copy()
    factor_design.insert(0, "const", 1.0)
    values = factor_design.to_numpy(dtype=float) @ coefficient_matrix.to_numpy(dtype=float)
    return pd.DataFrame(values, index=factor_returns.index, columns=coefficient_matrix.columns)


def estimate_window_panel(window_sessions, minimum_observations, *, apply_corrections=True):
    """Estimate every security at all formations for one specified window.

    Returns exposures (including ineligible rows), reusable long coefficients,
    per-model diagnostics and a compact formation summary. All data enter through
    the shared loader, with min(formation, factor cutoff) applied explicitly.
    """
    price_panels = loader.load_price_panels(apply_corrections=apply_corrections)
    returns = loader.daily_simple_returns_from_prices(price_panels.adjusted_close, price_panels.quarantined_returns)
    factors = loader.load_factors()
    excess_returns = loader.excess_returns(returns, factors["rf"])
    common_factors = factors[list(standard_factor_names)].join(loader.load_smh_excess_returns().rename("smh_excess"))
    identifiers = loader.load_identifiers()
    factor_cutoff = loader.factor_data_last_date()
    sessions = price_panels.adjusted_close.index
    exposure_rows, coefficient_rows, diagnostic_rows = [], [], []
    for formation_date in calendar.formation_dates(sessions):
        cutoff = calendar.measurement_cutoff(formation_date, factor_cutoff)
        window = calendar.estimation_window(sessions, cutoff, window_sessions)
        eligibility_table = eligibility.panel_window_eligibility(
            excess_returns, window, minimum_observations, common_factors,
            price_panels.before_same_security_start, price_panels.quarantined_returns)
        window_factors = common_factors.reindex(window)
        for ticker, eligibility_row in eligibility_table.iterrows():
            common = {"formation_date": formation_date, "ticker": ticker, "window_sessions": window_sessions}
            exposure = {**common, "cik": int(identifiers.at[ticker, "cik"]),
                        "is_representative_security": bool(identifiers.at[ticker, "is_representative_security"]),
                        "snapshot_weight": float(identifiers.at[ticker, "snapshot_weight"]),
                        "measurement_cutoff": cutoff, "window_first_session": window.min(), "window_last_session": window.max(),
                        "minimum_observations": minimum_observations, **eligibility_row.to_dict()}
            if not eligibility_row["eligible"]:
                exposure_rows.append(exposure)
                continue
            estimation = estimate_factor_models(excess_returns[ticker].reindex(window), window_factors, minimum_observations)
            if len(estimation["observation_dates"]) != eligibility_row["observations"]:
                raise AssertionError("regression and shared eligibility masks disagree")
            if abs(estimation["fwl_difference"]) > 1e-9:
                raise AssertionError("joint and same-mask residualized semiconductor estimates disagree")
            for model, fit in estimation["models"].items():
                prefix = model_prefixes[model]
                for term, coefficient, standard_error in zip(fit["terms"], fit["coefficient"], fit["standard_error"]):
                    exposure[f"{prefix}_{'alpha' if term == 'const' else term}"] = float(coefficient)
                    coefficient_rows.append({**common, "measurement_cutoff": cutoff, "model": model, "term": term,
                                             "coefficient": float(coefficient), "standard_error": float(standard_error), "nobs": fit["nobs"]})
                diagnostic_rows.append({**common, "measurement_cutoff": cutoff, "model": model,
                                        **{name: fit[name] for name in ("nobs", "rank", "residual_degrees_freedom", "r_squared", "adjusted_r_squared",
                                                                      "residual_variance", "residual_sample_variance", "residual_share_of_variance")}})
                exposure[f"{prefix}_r_squared"] = fit["r_squared"]
                exposure[f"{prefix}_adjusted_r_squared"] = fit["adjusted_r_squared"]
                exposure[f"{prefix}_residual_variance"] = fit["residual_variance"]
            baseline = estimation["models"]["ff5_mom"]
            exposure.update({"trailing_residual_volatility": np.sqrt(baseline["residual_variance"] * settings.annualisation_sessions),
                             "residual_share_of_variance": baseline["residual_share_of_variance"],
                             "smh_factor_r_squared": estimation["smh_factor_r_squared"],
                             "smh_residual_volatility": estimation["smh_residual_volatility"],
                             "smh_residual_variance": estimation["smh_residual_variance"],
                             "fwl_loading": estimation["fwl_loading"], "fwl_difference": estimation["fwl_difference"],
                             "incremental_r_squared": estimation["models"]["ff5_mom_smh"]["r_squared"] - baseline["r_squared"]})
            exposure_rows.append(exposure)
    exposures = pd.DataFrame(exposure_rows).sort_values(["ticker", "formation_date"])
    loading_columns = [f"{model_prefixes[model]}_{factor}" for model, names in model_factor_names.items() for factor in names]
    for column in loading_columns:
        exposures[f"{column}_change"] = exposures.groupby("ticker", sort=False)[column].diff()  # no bridge across an ineligible formation
    exposures = exposures.sort_values(["formation_date", "ticker"]).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summaries = []
    for formation_date, rows in exposures.groupby("formation_date", sort=True):
        eligible_rows = rows.loc[rows["eligible"]]
        summaries.append({"formation_date": formation_date, "window_sessions": window_sessions,
                          "measurement_cutoff": rows["measurement_cutoff"].iloc[0], "securities": len(rows), "eligible": len(eligible_rows),
                          "eligible_representatives": int(eligible_rows["is_representative_security"].sum()),
                          "represented_snapshot_weight": eligible_rows["snapshot_weight"].sum(),
                          "median_smh_loading": eligible_rows["m4_smh_excess"].median(),
                          "median_incremental_r_squared": eligible_rows["incremental_r_squared"].median(),
                          "median_smh_factor_r_squared": eligible_rows["smh_factor_r_squared"].median(),
                          "median_smh_residual_volatility": eligible_rows["smh_residual_volatility"].median(),
                          "maximum_absolute_fwl_difference": eligible_rows["fwl_difference"].abs().max()})
    return {"exposures": exposures, "coefficients": coefficients, "diagnostics": diagnostics, "formation_summary": pd.DataFrame(summaries)}


def build_factor_exposures():
    """Save both specified windows; return a mapping of output names to saved paths."""
    paths = {}
    for window, minimum in ((settings.main_window_sessions, settings.main_window_minimum_observations),
                            (settings.sensitivity_window_sessions, settings.sensitivity_window_minimum_observations)):
        tables = estimate_window_panel(window, minimum)
        for name, table in tables.items():
            path = save_table(table, "factor_exposures", f"{name}_{window}.csv", source_modules=[sys.modules[__name__]],
                              note="Four intercept OLS models on identical per-security observations. Standard errors are classical conditional diagnostics; "
                                   "residual variance uses SSE/(n-rank). Factor estimates stop at min(formation date, observed factor cutoff).")
            paths[f"{name}_{window}"] = path
        print(f"factor exposures: saved {window}-session window, {int(tables['exposures']['eligible'].sum())} eligible security-formations", flush=True)
    return paths


def saved_factor_summary():
    """Read the saved compact summary for notebook display without rerunning estimation."""
    return pd.concat([pd.read_csv(settings.results_folder / "factor_exposures" / f"formation_summary_{window}.csv")
                      for window in (settings.main_window_sessions, settings.sensitivity_window_sessions)], ignore_index=True)


if __name__ == "__main__":
    build_factor_exposures()
