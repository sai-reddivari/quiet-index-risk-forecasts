"""Extreme-event corrections: an evidenced ledger applied to the cleaned adjusted-close panel, never to raw files.

The ledger lives at data/corrections/extreme_event_ledger_2026-09-30.csv. If the file does not exist, no correction is
applied. Each row is one decision about one ticker; a ticker may have several rows.

Ledger columns (CSV, one header row; dates as YYYY-MM-DD; blank where a column does not apply)
- ticker: the universe ticker the row applies to.
- event_date: the session of the daily return under review (the return from the previous session's price to this one).
- classification: one of verified_market_move, verified_vendor_error, corporate_action_discrepancy, unresolved.
- treatment: one of retain, back_adjust_before_date, replace_price, quarantine_price, quarantine_return.
- factor: back_adjust_before_date only. Every adjusted close strictly before event_date is multiplied by it. If blank,
  it is solved from corrected_return so that the event_date return equals corrected_return.
- price_date: replace_price only. The session whose adjusted close is replaced.
- price_value: replace_price only. The replacement adjusted close, quoted on the basis of the downloaded panel.
- quarantine_dates: quarantine_price and quarantine_return. Semicolon-separated sessions; blank means event_date alone.
  For quarantine_price these are price dates (the price is set missing, so the returns on that session and the next
  are both missing); for quarantine_return they are return dates (that return is missing; prices are untouched).
- corrected_return: the evidenced daily simple return on event_date. Required when back_adjust_before_date has no
  factor; whenever it is given for back_adjust_before_date or replace_price, the corrected panel must reproduce it.
- evidence_reference: where the evidence is recorded (a document section, filing or note). Required for every row.

Which treatments each classification allows (docs/STUDY_SPEC.md section 3)
- verified_market_move: retain (no change).
- verified_vendor_error: replace_price or back_adjust_before_date (an evidenced correction).
- corporate_action_discrepancy: back_adjust_before_date or replace_price (reconstruct the distribution-inclusive return).
- unresolved: quarantine_price or quarantine_return (disclosed, never filled).

Order of application, so the result does not depend on row order: replacements first (their values are on the
downloaded basis), then back adjustments in date order, then quarantines. Missing sessions are never bridged: a
return always needs the price of the session immediately before it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quiet_index_500 import settings

ledger_file_name = f"extreme_event_ledger_{settings.holdings_date}.csv"
ledger_columns = ("ticker", "event_date", "classification", "treatment", "factor", "price_date", "price_value",
                  "quarantine_dates", "corrected_return", "evidence_reference")
treatments_allowed_by_classification = {
    "verified_market_move": {"retain"},
    "verified_vendor_error": {"replace_price", "back_adjust_before_date"},
    "corporate_action_discrepancy": {"back_adjust_before_date", "replace_price"},
    "unresolved": {"quarantine_price", "quarantine_return"},
}
corrected_return_tolerance = 1e-10  # how closely a corrected panel must reproduce an evidenced return


def extreme_event_ledger_path():
    """data/corrections/extreme_event_ledger_2026-09-30.csv (it may not exist)."""
    return settings.corrections_folder / ledger_file_name


def empty_extreme_event_ledger():
    """A ledger with the documented columns and no rows: what an absent ledger means."""
    return pd.DataFrame({column_name: pd.Series(dtype="object") for column_name in ledger_columns})


def read_extreme_event_ledger(ledger_path=None):
    """Read and validate the ledger; an absent file returns an empty ledger (no corrections)."""
    ledger_path = extreme_event_ledger_path() if ledger_path is None else ledger_path
    if not ledger_path.exists():
        return empty_extreme_event_ledger()
    ledger = pd.read_csv(ledger_path, dtype={"ticker": str, "classification": str, "treatment": str,
                                             "quarantine_dates": str, "evidence_reference": str})
    validate_extreme_event_ledger(ledger)
    return ledger


def validate_extreme_event_ledger(ledger):
    """Raise ValueError if a column is missing, a value is outside the documented sets, or a required field is blank."""
    missing_columns = [column_name for column_name in ledger_columns if column_name not in ledger.columns]
    if missing_columns:
        raise ValueError(f"ledger is missing columns {missing_columns}")
    for row_number, ledger_row in ledger.iterrows():
        row_label = f"ledger row {row_number} ({ledger_row['ticker']}, {ledger_row['event_date']})"
        if ledger_row["classification"] not in treatments_allowed_by_classification:
            raise ValueError(f"{row_label}: unknown classification {ledger_row['classification']!r}")
        if ledger_row["treatment"] not in treatments_allowed_by_classification[ledger_row["classification"]]:
            raise ValueError(f"{row_label}: treatment {ledger_row['treatment']!r} is not allowed for {ledger_row['classification']}")
        if pd.isna(ledger_row["evidence_reference"]) or not str(ledger_row["evidence_reference"]).strip():
            raise ValueError(f"{row_label}: evidence_reference is required")
        if pd.isna(ledger_row["event_date"]):
            raise ValueError(f"{row_label}: event_date is required")
        if ledger_row["treatment"] == "back_adjust_before_date" and pd.isna(ledger_row["factor"]) and pd.isna(ledger_row["corrected_return"]):
            raise ValueError(f"{row_label}: back_adjust_before_date needs a factor or a corrected_return")
        if ledger_row["treatment"] == "back_adjust_before_date" and pd.notna(ledger_row["factor"]) and not float(ledger_row["factor"]) > 0:
            raise ValueError(f"{row_label}: factor must be positive")
        if ledger_row["treatment"] == "replace_price" and (pd.isna(ledger_row["price_date"]) or pd.isna(ledger_row["price_value"])):
            raise ValueError(f"{row_label}: replace_price needs price_date and price_value")
        if ledger_row["treatment"] == "replace_price" and not float(ledger_row["price_value"]) > 0:
            raise ValueError(f"{row_label}: price_value must be positive")


def _quarantine_sessions(ledger_row):
    """The sessions a quarantine row names: quarantine_dates split on ';', or event_date when blank."""
    if pd.isna(ledger_row["quarantine_dates"]) or not str(ledger_row["quarantine_dates"]).strip():
        return [pd.Timestamp(ledger_row["event_date"])]
    return [pd.Timestamp(date_text.strip()) for date_text in str(ledger_row["quarantine_dates"]).split(";") if date_text.strip()]


def _require_session(sessions, session_date, row_label):
    """Raise unless the date is a session of the panel."""
    if session_date not in sessions:
        raise ValueError(f"{row_label}: {session_date.date()} is not a session of the price panel")


@dataclass
class CorrectedAdjustedClose:
    """What applying the ledger produces: the corrected panel, the quarantine masks and a log of what was done."""
    adjusted_close: pd.DataFrame
    quarantined_prices: pd.DataFrame
    quarantined_returns: pd.DataFrame
    applied_corrections: pd.DataFrame = field(default_factory=pd.DataFrame)


def apply_extreme_event_ledger(adjusted_close, ledger):
    """Apply the ledger to a cleaned adjusted-close panel (date index, one column per ticker) and return the result.

    The input is not modified. quarantined_prices marks prices set missing; quarantined_returns marks every return made
    missing by a quarantine: the named return dates, and for each quarantined price its own session and the next one.
    """
    corrected_adjusted_close = adjusted_close.copy()
    sessions = corrected_adjusted_close.index
    quarantined_prices = pd.DataFrame(False, index=sessions, columns=corrected_adjusted_close.columns)
    quarantined_returns = quarantined_prices.copy()
    applied_log_rows = []
    if len(ledger) == 0:
        return CorrectedAdjustedClose(corrected_adjusted_close, quarantined_prices, quarantined_returns, pd.DataFrame(applied_log_rows))

    validate_extreme_event_ledger(ledger)
    treatment_order = {"replace_price": 0, "back_adjust_before_date": 1, "quarantine_price": 2, "quarantine_return": 3, "retain": 4}
    ordered_ledger = ledger.assign(
        treatment_rank=ledger["treatment"].map(treatment_order),
        event_timestamp=pd.to_datetime(ledger["event_date"]),
    ).sort_values(["treatment_rank", "event_timestamp"], kind="stable")

    for row_number, ledger_row in ordered_ledger.iterrows():
        ticker = ledger_row["ticker"]
        event_date = ledger_row["event_timestamp"]
        row_label = f"ledger row {row_number} ({ticker}, {event_date.date()})"
        if ticker not in corrected_adjusted_close.columns:
            raise ValueError(f"{row_label}: ticker is not in the price panel")
        if event_date > sessions.max():  # an event after the loaded window has nothing to act on here
            continue
        _require_session(sessions, event_date, row_label)
        treatment = ledger_row["treatment"]

        if treatment == "replace_price":
            price_date = pd.Timestamp(ledger_row["price_date"])
            _require_session(sessions, price_date, row_label)
            price_before_replacement = corrected_adjusted_close.at[price_date, ticker]
            corrected_adjusted_close.at[price_date, ticker] = float(ledger_row["price_value"])  # on the downloaded basis, before any back adjustment
            applied_log_rows.append({"ticker": ticker, "event_date": event_date, "treatment": treatment, "detail": f"{price_date.date()}: {price_before_replacement} -> {float(ledger_row['price_value'])}"})

        elif treatment == "back_adjust_before_date":
            event_position = sessions.get_loc(event_date)
            if event_position == 0:
                raise ValueError(f"{row_label}: no session before event_date to adjust")
            if pd.notna(ledger_row["factor"]):
                back_adjustment_factor = float(ledger_row["factor"])
            else:  # solve the factor so that P_t / (factor * P_t-1) - 1 equals the evidenced return
                price_on_event_date = corrected_adjusted_close.iat[event_position, corrected_adjusted_close.columns.get_loc(ticker)]
                price_on_previous_session = corrected_adjusted_close.iat[event_position - 1, corrected_adjusted_close.columns.get_loc(ticker)]
                if pd.isna(price_on_event_date) or pd.isna(price_on_previous_session):
                    raise ValueError(f"{row_label}: a factor cannot be solved without prices on the event session and the session before it")
                back_adjustment_factor = price_on_event_date / ((1 + float(ledger_row["corrected_return"])) * price_on_previous_session)
            sessions_before_event = sessions < event_date
            corrected_adjusted_close.loc[sessions_before_event, ticker] = corrected_adjusted_close.loc[sessions_before_event, ticker] * back_adjustment_factor
            applied_log_rows.append({"ticker": ticker, "event_date": event_date, "treatment": treatment, "detail": f"prices before {event_date.date()} x {back_adjustment_factor:.10g}"})

        elif treatment == "quarantine_price":
            for price_date in _quarantine_sessions(ledger_row):
                _require_session(sessions, price_date, row_label)
                price_position = sessions.get_loc(price_date)
                corrected_adjusted_close.at[price_date, ticker] = np.nan
                quarantined_prices.at[price_date, ticker] = True
                quarantined_returns.iat[price_position, quarantined_returns.columns.get_loc(ticker)] = True  # the return into the suspect price
                if price_position + 1 < len(sessions):
                    quarantined_returns.iat[price_position + 1, quarantined_returns.columns.get_loc(ticker)] = True  # and the return out of it
                applied_log_rows.append({"ticker": ticker, "event_date": event_date, "treatment": treatment, "detail": f"price {price_date.date()} set missing"})

        elif treatment == "quarantine_return":
            for return_date in _quarantine_sessions(ledger_row):
                _require_session(sessions, return_date, row_label)
                quarantined_returns.at[return_date, ticker] = True
                applied_log_rows.append({"ticker": ticker, "event_date": event_date, "treatment": treatment, "detail": f"return {return_date.date()} set missing"})

        else:  # retain: a verified market move stays as downloaded
            applied_log_rows.append({"ticker": ticker, "event_date": event_date, "treatment": treatment, "detail": "unchanged"})

    # every evidenced return given for a price correction must come out of the corrected panel exactly
    for row_number, ledger_row in ordered_ledger.iterrows():
        if ledger_row["treatment"] not in ("back_adjust_before_date", "replace_price") or pd.isna(ledger_row["corrected_return"]):
            continue
        event_date = ledger_row["event_timestamp"]
        if event_date > sessions.max():
            continue
        event_position = sessions.get_loc(event_date)
        ticker_prices = corrected_adjusted_close[ledger_row["ticker"]]
        reproduced_return = ticker_prices.iloc[event_position] / ticker_prices.iloc[event_position - 1] - 1  # the session immediately before, never a bridge
        if not abs(reproduced_return - float(ledger_row["corrected_return"])) <= corrected_return_tolerance:
            raise ValueError(f"ledger row {row_number} ({ledger_row['ticker']}, {event_date.date()}): corrected panel gives {reproduced_return}, "
                             f"ledger says {ledger_row['corrected_return']}")

    return CorrectedAdjustedClose(corrected_adjusted_close, quarantined_prices, quarantined_returns, pd.DataFrame(applied_log_rows))
