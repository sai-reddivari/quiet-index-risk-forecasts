"""Ordinary and conditional SPY betas, with separately dated own 12-1 momentum.

All regressions include an intercept. HC3 standard errors describe conditional estimation
uncertainty, not uncertainty from selecting a universe or a downside state. The amplification
standard error uses aligned slope influences from both regressions, preserving their dependence.
"""

import sys

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import calendar, eligibility, loader
from quiet_index_500.research_io import save_table


def _slope_fit(stock, market):
    centered_market = market - market.mean()
    sum_squared_market = centered_market @ centered_market
    if len(stock) < 3 or sum_squared_market <= np.finfo(float).eps:
        return None
    slope = centered_market @ (stock - stock.mean()) / sum_squared_market
    intercept = stock.mean() - slope * market.mean()
    residuals = stock - intercept - slope * market
    leverage = 1 / len(stock) + centered_market ** 2 / sum_squared_market
    if np.any(1 - leverage <= 1e-12):
        return None
    slope_influence = centered_market * residuals / ((1 - leverage) * sum_squared_market)
    return {"beta": slope, "intercept": intercept,
            "beta_se": np.sqrt(slope_influence @ slope_influence), "influence": slope_influence}


def beta_estimates(stock_excess, spy_excess, minimum_negative_days):
    """Fit all-day, negative-SPY and worst-decile regressions on matched nonmissing days.

    The caller applies the shared window eligibility first. Missing observations in either
    input are removed jointly; the downside and decile selections use SPY excess returns.
    """
    joint = pd.concat([stock_excess.rename("stock"), spy_excess.rename("spy")], axis=1).dropna()
    stock = joint.stock.to_numpy(dtype=float)
    market = joint.spy.to_numpy(dtype=float)
    result = {"observations": len(joint), "negative_market_observations": int((market < 0).sum()),
              "worst_decile_observations": 0, "ordinary_eligible": False,
              "downside_eligible": False, "worst_decile_eligible": False,
              "ordinary_reason": "degenerate_market", "downside_reason": "insufficient_negative_market_days"}
    for name in ["ordinary", "downside", "worst_decile"]:
        for statistic in ["beta", "intercept", "beta_se"]:
            result[f"{name}_{statistic}"] = np.nan
    result["downside_amplification"] = result["downside_amplification_se"] = np.nan
    if len(joint) < 3:
        return result
    ordinary_fit = _slope_fit(stock, market)
    if ordinary_fit is None:
        return result
    result.update(ordinary_eligible=True, ordinary_reason="eligible")
    for statistic in ["beta", "intercept", "beta_se"]:
        result[f"ordinary_{statistic}"] = ordinary_fit[statistic]
    downside_mask = market < 0
    decile_threshold = np.quantile(market, settings.worst_decile_quantile)
    decile_mask = market <= decile_threshold
    result["worst_decile_observations"] = int(decile_mask.sum())
    result["worst_decile_market_threshold"] = decile_threshold
    for name, mask, minimum in [("downside", downside_mask, minimum_negative_days),
                                ("worst_decile", decile_mask, 3)]:
        if mask.sum() < minimum:
            continue
        fit = _slope_fit(stock[mask], market[mask])
        if fit is None:
            if name == "downside":
                result["downside_reason"] = "degenerate_downside_market"
            continue
        result[f"{name}_eligible"] = True
        for statistic in ["beta", "intercept", "beta_se"]:
            result[f"{name}_{statistic}"] = fit[statistic]
        if name == "downside":
            result["downside_reason"] = "eligible"
            result["downside_amplification"] = fit["beta"] - ordinary_fit["beta"]
            downside_influence = np.zeros(len(joint))
            downside_influence[mask] = fit["influence"]
            difference_influence = downside_influence - ordinary_fit["influence"]
            result["downside_amplification_se"] = np.sqrt(difference_influence @ difference_influence)
    return result


def momentum_at_formation(price_panels, daily_returns, formation_date):
    """Own momentum by ticker from exact session endpoints, retaining all ineligible rows.

    At least 200 of the 231 daily returns strictly after the older endpoint and through the
    newer endpoint must be valid. A quarantined interval return disqualifies this price ratio:
    otherwise an excluded move would re-enter through the two endpoint prices.
    """
    prices = price_panels.adjusted_close
    sessions = prices.index
    end_position = sessions.searchsorted(pd.Timestamp(formation_date), side="right") - 1
    older_position = end_position - settings.momentum_start_lag_sessions
    newer_position = end_position - settings.momentum_recent_lag_sessions
    output = pd.DataFrame(index=prices.columns).rename_axis("ticker")
    output["formation_date"] = pd.Timestamp(formation_date)
    output["price_measurement_cutoff"] = sessions[end_position] if end_position >= 0 else pd.NaT
    output["momentum_start_date"] = pd.NaT
    output["momentum_end_date"] = pd.NaT
    output["momentum_observations"] = 0
    output["momentum_quarantined_returns"] = 0
    output["own_momentum"] = np.nan
    output["momentum_eligible"] = False
    output["momentum_reason"] = "insufficient_endpoint_history"
    if older_position < 0:
        return output.reset_index()
    older_date, newer_date = sessions[older_position], sessions[newer_position]
    interval = sessions[older_position + 1:newer_position + 1]
    history = sessions[older_position:end_position + 1]
    old_price, new_price = prices.loc[older_date], prices.loc[newer_date]
    quarantined = price_panels.quarantined_returns.reindex(interval).sum()
    valid_returns = daily_returns.reindex(interval).mask(price_panels.quarantined_returns.reindex(interval)).notna().sum()
    before_start = price_panels.before_same_security_start.reindex(history).any()
    endpoint_valid = old_price.gt(0) & new_price.gt(0)
    reason = np.select([before_start, ~endpoint_valid, quarantined.gt(0),
                        valid_returns.lt(settings.momentum_minimum_valid_returns_between_endpoints)],
                       ["before_same_security_start", "missing_endpoint", "quarantined_interval_return",
                        "insufficient_intervening_returns"], default="eligible")
    output["momentum_start_date"] = older_date
    output["momentum_end_date"] = newer_date
    output["momentum_observations"] = valid_returns
    output["momentum_quarantined_returns"] = quarantined
    output["momentum_reason"] = reason
    output["momentum_eligible"] = output.momentum_reason.eq("eligible")
    output["own_momentum"] = (new_price / old_price - 1).where(output.momentum_eligible)
    return output.reset_index()


def build_downside_momentum_tables(apply_corrections=True):
    """Build security-level characteristics for every formation and both estimation windows."""
    panels = loader.load_price_panels(apply_corrections=apply_corrections)
    returns = loader.load_daily_returns(apply_corrections=apply_corrections)
    excess = loader.load_excess_returns(apply_corrections=apply_corrections)
    spy = loader.load_spy_excess_returns(apply_corrections=apply_corrections)
    factor_cutoff = loader.factor_data_last_date()
    formations = calendar.formation_dates(returns.index)
    windows = [(settings.main_window_sessions, settings.main_window_minimum_observations,
                settings.downside_minimum_negative_market_days_main),
               (settings.sensitivity_window_sessions, settings.sensitivity_window_minimum_observations,
                settings.downside_minimum_negative_market_days_sensitivity)]
    beta_rows, momentum_tables = [], []
    for formation in formations:
        cutoff = calendar.measurement_cutoff(formation, factor_cutoff)
        momentum_tables.append(momentum_at_formation(panels, returns, formation))
        for window_sessions, minimum_observations, minimum_negative in windows:
            window = calendar.estimation_window(returns.index, cutoff, window_sessions)
            eligible = eligibility.panel_window_eligibility(excess, window, minimum_observations, spy,
                                                            panels.before_same_security_start, panels.quarantined_returns)
            stock_in_window, spy_in_window = excess.reindex(window), spy.reindex(window)
            negative_counts = stock_in_window.notna().mul(spy_in_window.lt(0), axis=0).sum()
            for ticker, status in eligible.iterrows():
                row = {"ticker": ticker, "formation_date": formation, "window_sessions": window_sessions,
                       "factor_measurement_cutoff": cutoff, "estimation_first_session": window.min(),
                       "estimation_last_session": window.max(), "quarantined_in_window": status.quarantined_in_window,
                       "observations": status.observations, "ordinary_eligible": False, "downside_eligible": False,
                       "worst_decile_eligible": False, "ordinary_reason": status.reason,
                       "downside_reason": status.reason, "negative_market_observations": negative_counts[ticker],
                       "worst_decile_observations": 0}
                if status.eligible:
                    row.update(beta_estimates(stock_in_window[ticker], spy_in_window, minimum_negative))
                beta_rows.append(row)
    beta_table = pd.DataFrame(beta_rows)
    momentum_table = pd.concat(momentum_tables, ignore_index=True)
    characteristics = beta_table.merge(momentum_table, on=["ticker", "formation_date"], validate="many_to_one")
    coverage = characteristics.groupby(["formation_date", "window_sessions"], as_index=False).agg(
        securities=("ticker", "size"), ordinary_eligible=("ordinary_eligible", "sum"),
        downside_eligible=("downside_eligible", "sum"), momentum_eligible=("momentum_eligible", "sum"),
        median_ordinary_beta=("ordinary_beta", "median"), median_downside_beta=("downside_beta", "median"),
        median_downside_amplification=("downside_amplification", "median"),
        median_own_momentum=("own_momentum", "median"))
    return {"downside_momentum": characteristics, "own_momentum": momentum_table, "coverage": coverage}


def write_downside_momentum_outputs(tables=None):
    """Write characteristics, own momentum and coverage with the complete shared provenance."""
    tables = build_downside_momentum_tables() if tables is None else tables
    return {name: save_table(table, "downside_momentum", f"{name}.csv", source_modules=[sys.modules[__name__]],
                             note="Betas use SPY excess returns and HC3 standard errors; own momentum uses separate formation-date prices.")
            for name, table in tables.items()}
