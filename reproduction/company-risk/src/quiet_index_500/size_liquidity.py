"""Historical issuer size and security liquidity on the shared cleaned foundation."""

import sys

import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import calendar, loader
from quiet_index_500.foundation.eligibility import panel_window_eligibility
from quiet_index_500.research_io import make_notebook, save_table


def daily_dollar_volume(split_adjusted_close, volume):
    """Close times compatible split-adjusted share volume; nonpositive inputs are missing.

    This is a daily dollar-turnover proxy, not the sum of each execution's dollar value.
    The split convention is checked in split_basis_checks and documented in the findings.
    """
    return (split_adjusted_close * volume).where((split_adjusted_close > 0) & (volume > 0))


def measure_liquidity(price_panels, returns, formations):
    """503-security liquidity rows; absence of market cap never changes this sample."""
    dollar_volume = daily_dollar_volume(price_panels.split_adjusted_close, price_panels.volume)
    amihud_daily = returns.abs().div(dollar_volume) * settings.amihud_scale_dollars
    rows = []
    for formation in formations:
        dollar_window = calendar.estimation_window(dollar_volume.index, formation, settings.dollar_volume_window_sessions)
        amihud_window = calendar.estimation_window(dollar_volume.index, formation, settings.amihud_window_sessions)
        dollar_eligibility = panel_window_eligibility(
            dollar_volume, dollar_window, settings.dollar_volume_minimum_observations,
            before_same_security_start=price_panels.before_same_security_start)
        amihud_eligibility = panel_window_eligibility(
            amihud_daily, amihud_window, settings.amihud_minimum_observations,
            before_same_security_start=price_panels.before_same_security_start,
            quarantined_returns=price_panels.quarantined_returns)
        median_dollar_volume = dollar_volume.loc[dollar_window].median().where(dollar_eligibility['eligible'])
        mean_amihud = amihud_daily.loc[amihud_window].mean().where(amihud_eligibility['eligible'])
        frame = pd.DataFrame({
            'median_dollar_volume_60': median_dollar_volume,
            'log_median_dollar_volume_60': np.log(median_dollar_volume.where(median_dollar_volume > 0)),
            'amihud_252_per_million': mean_amihud,
            'log_amihud_252_per_million': np.log(mean_amihud.where(mean_amihud > 0)),
            'dollar_volume_observations': dollar_eligibility['observations'],
            'dollar_volume_eligible': dollar_eligibility['eligible'],
            'dollar_volume_reason': dollar_eligibility['reason'],
            'amihud_observations': amihud_eligibility['observations'],
            'amihud_eligible': amihud_eligibility['eligible'],
            'amihud_reason': amihud_eligibility['reason'],
            'amihud_quarantined_in_window': amihud_eligibility['quarantined_in_window'],
        }).rename_axis('ticker').reset_index()
        frame['formation_date'] = pd.Timestamp(formation)
        frame['liquidity_measurement_cutoff'] = pd.Timestamp(formation)
        frame['dollar_volume_window_start'] = dollar_window.min()
        frame['amihud_window_start'] = amihud_window.min()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def size_at_formation(issuer_cap, representative_identifiers, formation):
    """Positive issuer cap once per CIK; four equal-count bins, ties ordered by CIK.

    A missing value on the formation date remains missing. Size percentile uses average
    ranks, so exact ties have equal percentiles even when a quartile boundary splits them.
    """
    frame = representative_identifiers.copy().sort_values('cik').reset_index(drop=True)
    caps = issuer_cap.reindex(index=[pd.Timestamp(formation)]).iloc[0]
    frame['issuer_market_cap'] = frame['cik'].map(caps).where(lambda values: values > 0)
    frame['log_market_cap'] = np.log(frame['issuer_market_cap'])
    frame['size_percentile'] = frame['issuer_market_cap'].rank(method='average', pct=True)
    frame['size_quartile'] = pd.Series(pd.NA, index=frame.index, dtype='Int64')
    positive_order = frame.dropna(subset=['issuer_market_cap']).sort_values(['issuer_market_cap', 'cik']).index
    for quartile, members in enumerate(np.array_split(positive_order.to_numpy(), 4), start=1):
        frame.loc[members, 'size_quartile'] = quartile
    frame['size_eligible'] = frame['issuer_market_cap'].notna()
    frame['size_reason'] = np.where(frame['size_eligible'], 'eligible', 'no_positive_same_security_cap_on_formation')
    frame['formation_date'] = pd.Timestamp(formation)
    frame['size_measurement_cutoff'] = pd.Timestamp(formation)
    return frame


def split_basis_checks(price_panels):
    """Actual frozen pre/post split diagnostics, not a claim to audit every split.

    Median volumes in twenty sessions before/after a split should stay on comparable
    share units. Unadjusted earlier volume would systematically be smaller by the split
    ratio. Real trading intensity can change, so both distances remain visible.
    """
    events = [
        ('NVDA', '2024-06-10', 10., 'https://investor.nvidia.com/files/doc_downloads/2024/06/nvidia-2024-stock-split_faq_investors.pdf'),
        ('AAPL', '2020-08-31', 4., 'https://investor.apple.com/dividend-history/default.aspx'),
        ('AMZN', '2022-06-06', 20., 'https://ir.aboutamazon.com/faqs/default.aspx'),
    ]
    rows = []
    for ticker, date, ratio, source in events:
        position = price_panels.volume.index.get_loc(pd.Timestamp(date))
        prior = price_panels.volume.iloc[position-20:position][ticker]
        following = price_panels.volume.iloc[position:position+20][ticker]
        volume_ratio = prior.median()/following.median()
        rows.append({'ticker': ticker, 'split_date': pd.Timestamp(date), 'split_ratio': ratio,
                     'previous_close': price_panels.split_adjusted_close.iloc[position-1][ticker],
                     'ex_date_close': price_panels.split_adjusted_close.iloc[position][ticker],
                     'previous_volume': price_panels.volume.iloc[position-1][ticker],
                     'ex_date_volume': price_panels.volume.iloc[position][ticker],
                     'pre_post_median_volume_ratio': volume_ratio,
                     'log_distance_to_adjusted_volume': abs(np.log(volume_ratio)),
                     'log_distance_to_unadjusted_volume': abs(np.log(volume_ratio*ratio)),
                     'consistent_with_adjusted_volume': abs(np.log(volume_ratio)) < abs(np.log(volume_ratio*ratio)),
                     'source': source})
    return pd.DataFrame(rows)


def build_size_liquidity():
    """Return liquidity, licensed company size/boundaries, coverage and split diagnostics."""
    prices = loader.load_price_panels()
    returns = loader.load_daily_returns()
    identifiers = loader.load_identifiers().reset_index()
    formations = calendar.formation_dates(prices.adjusted_close.index)
    liquidity = measure_liquidity(prices, returns, formations).merge(identifiers, on='ticker', validate='many_to_one')
    representatives = identifiers.loc[identifiers['is_representative_security']].copy()
    company_weights = identifiers.groupby('cik')['snapshot_weight'].sum()
    representatives['snapshot_company_weight'] = representatives['cik'].map(company_weights)
    issuer_cap = loader.load_issuer_market_cap()
    size = pd.concat([size_at_formation(issuer_cap, representatives, formation) for formation in formations], ignore_index=True)
    boundaries = size.dropna(subset=['size_quartile']).groupby(['formation_date', 'size_quartile']).agg(
        companies=('cik', 'size'), minimum_market_cap=('issuer_market_cap', 'min'),
        maximum_market_cap=('issuer_market_cap', 'max')).reset_index()
    coverage = liquidity.groupby('formation_date').agg(
        securities=('ticker', 'size'), dollar_volume_eligible=('dollar_volume_eligible', 'sum'),
        amihud_eligible=('amihud_eligible', 'sum')).reset_index()
    for measure in ['dollar_volume', 'amihud']:
        weight = liquidity['snapshot_weight'].where(liquidity[f'{measure}_eligible'], 0)
        coverage[f'{measure}_represented_snapshot_weight'] = weight.groupby(liquidity['formation_date']).sum().to_numpy()
    assert not liquidity.duplicated(['formation_date', 'ticker']).any()
    assert not size.duplicated(['formation_date', 'cik']).any()
    assert len(liquidity) == 503 * len(formations) and len(size) == 500 * len(formations)
    return {'liquidity': liquidity, 'size': size, 'quartile_boundaries': boundaries,
            'liquidity_coverage': coverage, 'split_basis_checks': split_basis_checks(prices)}


def build_and_save():
    """Persist all measurements with licensed-cap provenance retained."""
    tables = build_size_liquidity()
    for name, table in tables.items():
        save_table(table, 'size_liquidity', f'{name}.csv', licensed=name in ('size', 'quartile_boundaries'),
                   source_modules=[sys.modules[__name__]],
                   note='Size uses one positive issuer cap on the formation date. Liquidity uses cleaned compatible close and volume; split diagnostics are spot checks.')
    return tables


def create_notebook():
    """Notebook displays availability and split checks, with size provenance explicit."""
    return make_notebook(settings.repository_root_folder/'notebooks/06_size_liquidity.ipynb',
        'Historical company size and security liquidity',
        'Company size is measured once per issuer from positive same-security cap on the formation date. '
        'The FMP-derived size panel and quartile dollar boundaries are under results/licensed/size_liquidity (the user authorized committing these derived tables). '
        'This notebook displays only liquidity coverage and split-basis diagnostics. '
        'Median close-times-volume measures turnover; Amihud measures absolute vendor-adjusted return per USD 1 million traded. '
        'No size observation is carried forward and no missing cap removes a liquidity observation.',
        ['from quiet_index_500.size_liquidity import build_and_save\ntables = build_and_save()\ndisplay(tables["liquidity_coverage"])',
         'display(tables["split_basis_checks"])'])


if __name__ == '__main__':
    built = build_and_save()
    create_notebook()
    print({name: len(table) for name, table in built.items()})
