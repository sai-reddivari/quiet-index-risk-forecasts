"""Eligibility of an estimation window, with the reason when it fails, so every table can carry exclusions.

A window is eligible when it has at least the minimum number of complete joint observations (sessions on which every
required series has a value) and it contains no session before the security's same-security start
(docs/STUDY_SPEC.md section 4). A quarantined return is one missing observation: it never enters the count, but it
does not by itself disqualify the window. Each result reports how many quarantined returns the window holds, so the
exclusion stays visible (Sai, 14 Oct 2026, following the master plan's "quarantine the affected return").

Reasons, checked in this order (the first that applies is reported)
- before_same_security_start: the window reaches back before the security's same-security history.
- no_prices: the security's own series has no value anywhere in the window.
- insufficient_observations: fewer complete joint observations than the minimum (quarantined days count as missing).
- eligible: none of the above.
"""

import numpy as np
import pandas as pd

reason_eligible = "eligible"
eligibility_reasons = ("before_same_security_start", "no_prices", "insufficient_observations", reason_eligible)


def complete_joint_observations(required_series, window):
    """Number of sessions in the window on which every column of required_series has a value."""
    required_in_window = required_series.reindex(pd.DatetimeIndex(window))
    if isinstance(required_in_window, pd.Series):
        return int(required_in_window.notna().sum())
    return int(required_in_window.notna().all(axis=1).sum())


def window_eligibility(security_series, window, minimum_observations, other_required_series=None,
                       before_same_security_start=None, quarantined_returns=None):
    """Eligibility of one security's window: a dict with observations, quarantined_in_window, eligible and reason.

    security_series is the security's own series (its returns, quarantined days already missing); other_required_series
    holds the other series the estimate needs on the same days (factors, SPY); the two boolean masks are the security's
    columns from PricePanels.
    """
    window = pd.DatetimeIndex(window)
    required_series = security_series.to_frame("security")
    if other_required_series is not None:
        required_series = required_series.join(other_required_series.reindex(required_series.index), how="left")
    quarantined_in_window = (int(quarantined_returns.reindex(window, fill_value=False).sum())
                             if quarantined_returns is not None else 0)
    if quarantined_returns is not None:  # a quarantined return never counts, even if a caller passes an unmasked series
        required_series = required_series.mask(quarantined_returns.reindex(required_series.index, fill_value=False), axis=0)
    observations = complete_joint_observations(required_series, window)
    reaches_before_start = before_same_security_start is not None and bool(before_same_security_start.reindex(window, fill_value=False).any())
    has_no_values = not required_series["security"].reindex(window).notna().any()
    if reaches_before_start:
        reason = "before_same_security_start"
    elif has_no_values:
        reason = "no_prices"
    elif observations < minimum_observations:
        reason = "insufficient_observations"
    else:
        reason = reason_eligible
    return {"observations": observations, "quarantined_in_window": quarantined_in_window,
            "eligible": reason == reason_eligible, "reason": reason}


def panel_window_eligibility(security_panel, window, minimum_observations, other_required_series=None,
                             before_same_security_start=None, quarantined_returns=None):
    """window_eligibility for every column of a panel at once: a ticker-indexed frame with observations,
    quarantined_in_window, eligible, reason."""
    window = pd.DatetimeIndex(window)
    panel_in_window = security_panel.reindex(window)
    quarantined_in_window_grid = (quarantined_returns.reindex(index=window, columns=security_panel.columns, fill_value=False).to_numpy()
                                  if quarantined_returns is not None else np.zeros(panel_in_window.shape, dtype=bool))
    security_has_value = panel_in_window.notna().to_numpy() & ~quarantined_in_window_grid  # a quarantined return is missing
    other_series_complete = np.ones(len(window), dtype=bool)
    if other_required_series is not None:
        other_in_window = other_required_series.reindex(window)
        other_series_complete = (other_in_window.notna().all(axis=1) if isinstance(other_in_window, pd.DataFrame) else other_in_window.notna()).to_numpy()
    observations = (security_has_value & other_series_complete[:, None]).sum(axis=0)
    reaches_before_start = (before_same_security_start.reindex(index=window, columns=security_panel.columns, fill_value=False).to_numpy().any(axis=0)
                            if before_same_security_start is not None else np.zeros(security_panel.shape[1], dtype=bool))
    has_no_values = ~security_has_value.any(axis=0)
    reason = np.select(
        [reaches_before_start, has_no_values, observations < minimum_observations],
        ["before_same_security_start", "no_prices", "insufficient_observations"],
        default=reason_eligible,
    )
    return pd.DataFrame({"observations": observations, "quarantined_in_window": quarantined_in_window_grid.sum(axis=0),
                         "eligible": reason == reason_eligible, "reason": reason},
                        index=security_panel.columns).rename_axis("ticker")
