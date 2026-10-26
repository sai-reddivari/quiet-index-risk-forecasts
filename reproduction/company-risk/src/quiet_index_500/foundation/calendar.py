"""Study calendar: trading sessions, quarterly formation dates, measurement cutoffs, windows and outcome quarters.

Sessions come from the price index, never from an exchange calendar library, so the calendar is the data's calendar.
A formation date is the final session of a calendar quarter. Its outcome quarter is the next calendar quarter, which
falls in one of three evaluation blocks (development, validation, confirmation) or is labelled incomplete.

Completeness of a quarter is judged on calendar dates: a quarter is complete for a data source when the source's last
date is on or after the quarter's last calendar day. Both study cutoffs (factors 2026-08-31, prices 2026-09-30) are
calendar month ends, so this is exact here.
"""

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import loader


def trading_sessions(through=settings.research_price_cutoff, apply_corrections=True):
    """Every session of the price panel on or before the cutoff, as a DatetimeIndex."""
    return loader.load_price_panels(through=through, apply_corrections=apply_corrections).adjusted_close.index


def _calendar_quarter_is_complete(quarter, data_last_date):
    """A calendar quarter is complete when the data run to its last calendar day or later."""
    return quarter.end_time.normalize() <= pd.Timestamp(data_last_date)


def formation_dates(sessions, first_formation_date=settings.first_formation_date, last_date=settings.research_price_cutoff):
    """The last session of each complete calendar quarter, from first_formation_date to last_date (a DatetimeIndex)."""
    sessions = pd.DatetimeIndex(sessions)
    sessions_in_range = sessions[(sessions >= pd.Timestamp(first_formation_date)) & (sessions <= pd.Timestamp(last_date))]
    last_session_by_quarter = pd.Series(sessions_in_range, index=sessions_in_range.to_period("Q")).groupby(level=0).max()
    complete_quarters = [quarter for quarter in last_session_by_quarter.index if _calendar_quarter_is_complete(quarter, last_date)]  # a partial quarter has no formation yet
    return pd.DatetimeIndex(last_session_by_quarter[complete_quarters].to_numpy(), name="formation_date")


def measurement_cutoff(formation_date, factor_cutoff):
    """Cutoff for factor-dependent estimates: min(formation date, factor cutoff)."""
    return min(pd.Timestamp(formation_date), pd.Timestamp(factor_cutoff))


def estimation_window(sessions, cutoff, window_sessions):
    """The window_sessions sessions ending on the last session on or before cutoff, inclusive.

    Fewer sessions are returned when the history is shorter; eligibility counts then decide whether that is enough.
    """
    sessions = pd.DatetimeIndex(sessions)
    last_position = sessions.searchsorted(pd.Timestamp(cutoff), side="right") - 1  # the last session not after the cutoff
    if last_position < 0:
        return sessions[:0]
    return sessions[max(0, last_position - window_sessions + 1): last_position + 1]


def outcome_quarter(formation_date):
    """The calendar quarter after the formation date's quarter (a pandas Period)."""
    return pd.Timestamp(formation_date).to_period("Q") + 1


def following_quarter_sessions(sessions, formation_date):
    """Sessions of the outcome quarter that are present in sessions (empty if none are loaded yet)."""
    sessions = pd.DatetimeIndex(sessions)
    quarter = outcome_quarter(formation_date)
    return sessions[(sessions >= quarter.start_time) & (sessions <= quarter.end_time)]


def outcome_block(formation_date):
    """development, validation or confirmation by the outcome quarter (docs/STUDY_SPEC.md section 2), else incomplete.

    Outcome quarters after the confirmation block (2026 Q3 and later) are incomplete for the evaluation. An outcome
    quarter before the development block means a formation date earlier than the study allows, which is an error.
    """
    quarter = outcome_quarter(formation_date)
    for block_name, (first_quarter, last_quarter) in [("development", settings.development_outcome_quarters),
                                                      ("validation", settings.validation_outcome_quarters),
                                                      ("confirmation", settings.confirmation_outcome_quarters)]:
        if pd.Period(first_quarter, "Q") <= quarter <= pd.Period(last_quarter, "Q"):
            return block_name
    if quarter < pd.Period(settings.development_outcome_quarters[0], "Q"):
        raise ValueError(f"formation {pd.Timestamp(formation_date).date()} is before the first formation {settings.first_formation_date}")
    return "incomplete"


def following_quarter_complete(formation_date, data_last_date):
    """Whether the whole outcome quarter is covered by data ending on data_last_date."""
    return _calendar_quarter_is_complete(outcome_quarter(formation_date), data_last_date)


def formation_calendar(sessions, factor_cutoff, price_cutoff=settings.research_price_cutoff):
    """One row per formation date: outcome quarter and block, both measurement cutoffs, outcome completeness and session range."""
    sessions = pd.DatetimeIndex(sessions)
    rows = []
    for formation_date in formation_dates(sessions, last_date=price_cutoff):
        outcome_sessions = following_quarter_sessions(sessions[sessions <= pd.Timestamp(price_cutoff)], formation_date)
        rows.append({
            "formation_date": formation_date,
            "outcome_quarter": str(outcome_quarter(formation_date)),
            "outcome_block": outcome_block(formation_date),
            "factor_measurement_cutoff": measurement_cutoff(formation_date, factor_cutoff),
            "price_measurement_cutoff": formation_date,  # size, liquidity and price-only measures use the formation date itself
            "outcome_complete_for_factor_results": following_quarter_complete(formation_date, factor_cutoff),
            "outcome_complete_for_price_results": following_quarter_complete(formation_date, price_cutoff),
            "outcome_first_session": outcome_sessions.min() if len(outcome_sessions) else pd.NaT,
            "outcome_last_session": outcome_sessions.max() if len(outcome_sessions) else pd.NaT,
            "outcome_sessions": len(outcome_sessions),
        })
    return pd.DataFrame(rows).set_index("formation_date")
