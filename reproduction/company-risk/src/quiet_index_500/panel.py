"""Company-by-formation integration, with measure-specific missingness preserved."""

import sys
import numpy as np
import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.foundation import load_identifiers, load_price_panels, formation_dates
from quiet_index_500.research_io import save_table, make_notebook

characteristics = ["log_market_cap", "log_median_dollar_volume_60", "log_amihud_252_per_million",
                   "ordinary_beta", "downside_beta", "downside_amplification", "own_momentum",
                   "m3_mkt_rf", "m3_smb", "m3_hml", "m3_rmw", "m3_cma", "m3_mom",
                   "m4_smh_excess", "trailing_residual_volatility"]


def join_company_panel(identifiers, dates, factor_table, downside_table, liquidity_table, size_table):
    """Unique left joins keep every representative company even when a measure is absent."""
    representatives = identifiers[identifiers.is_representative_security].reset_index()
    base = representatives.merge(pd.DataFrame({"formation_date": pd.to_datetime(dates)}), how="cross")
    factor_table = factor_table.rename(columns={"eligible": "factor_eligible", "reason": "factor_reason",
                                                "observations": "factor_observations",
                                                "measurement_cutoff": "factor_measurement_cutoff"})
    for table in [factor_table, downside_table, liquidity_table, size_table]:
        table = table.copy()
        table["formation_date"] = pd.to_datetime(table.formation_date)
        key = ["formation_date", "ticker"]
        if table.duplicated(key).any():
            raise ValueError("measurement join is not unique by formation and ticker")
        keep = key + [column for column in table if column not in base and column not in key]
        base = base.merge(table[keep], on=key, how="left", validate="one_to_one")
    assert not base.duplicated(["formation_date", "cik"]).any()
    return base.sort_values(["formation_date", "cik"]).reset_index(drop=True)


def describe_panel(panel):
    """Save sample counts with each distribution, correlation and weighted summary."""
    distributions, correlations, summaries, coverage = [], [], [], []
    for formation, rows in panel.groupby("formation_date", sort=True):
        for name in characteristics:
            valid = rows[name].dropna()
            distributions.append({"formation_date": formation, "characteristic": name,
                                  "companies": len(valid), "mean": valid.mean(), "std": valid.std(),
                                  **{f"p{int(q * 100):02d}": valid.quantile(q) for q in [.05, .25, .5, .75, .95]}})
            sample = rows[rows[name].notna()]
            coverage.append({"formation_date": formation, "characteristic": name, "companies": len(sample),
                             "represented_snapshot_weight": sample.issuer_snapshot_weight.sum()})
        for position, first in enumerate(characteristics):
            for second in characteristics[position + 1:]:
                joint = rows[[first, second]].dropna()
                correlations.append({"formation_date": formation, "first": first, "second": second,
                                     "companies": len(joint), "spearman": joint.corr(method="spearman").iloc[0, 1]})
        groupings = [("all", "all", rows)]
        for grouping in ["gics_sector", "size_quartile"]:
            groupings.extend((grouping, label, group) for label, group in rows.groupby(grouping, dropna=True))
        for grouping, label, group in groupings:
            for name in characteristics:
                valid = group[group[name].notna()]
                cap_valid = valid[valid.issuer_market_cap.gt(0)]
                summaries.append({"formation_date": formation, "grouping": grouping, "group": label,
                                  "characteristic": name, "companies": len(valid),
                                  "equal_company_mean": valid[name].mean(),
                                  "cap_weighted_companies": len(cap_valid),
                                  "cap_weighted_mean": np.average(cap_valid[name], weights=cap_valid.issuer_market_cap) if len(cap_valid) else np.nan})
    return {"distributions": pd.DataFrame(distributions), "correlations": pd.DataFrame(correlations),
            "summaries": pd.DataFrame(summaries), "coverage": pd.DataFrame(coverage)}


def loading_stability(panel):
    """Consecutive-quarter company loading persistence; missing estimates never bridge quarters."""
    loadings = panel.pivot(index="formation_date", columns="cik", values="m4_smh_excess").sort_index()
    rows = []
    for position in range(1, len(loadings)):
        joint = pd.concat([loadings.iloc[position - 1].rename("previous"), loadings.iloc[position].rename("current")], axis=1).dropna()
        rows.append({"formation_date": loadings.index[position], "previous_formation": loadings.index[position - 1],
                     "companies": len(joint), "spearman": joint.corr(method="spearman").iloc[0, 1],
                     "median_absolute_loading_change": (joint.current - joint.previous).abs().median()})
    return pd.DataFrame(rows)


def build_characteristic_panel():
    """Integrate both estimation windows and write descriptive evidence plus figures."""
    root = settings.results_folder
    identifiers = load_identifiers()
    dates = formation_dates(load_price_panels().adjusted_close.index)
    downside_path = root / "downside_momentum/downside_momentum.csv"
    liquidity_path = root / "size_liquidity/liquidity.csv"
    size_path = settings.licensed_results_folder / "size_liquidity/size.csv"
    downside, liquidity, size = [pd.read_csv(path) for path in [downside_path, liquidity_path, size_path]]
    paths = {}
    for window in [settings.main_window_sessions, settings.sensitivity_window_sessions]:
        factor_path = root / f"factor_exposures/exposures_{window}.csv"
        factors = pd.read_csv(factor_path)
        joined = join_company_panel(identifiers, dates, factors, downside[downside.window_sessions.eq(window)], liquidity, size)
        joined["window_sessions"] = window
        inputs = [factor_path, downside_path, liquidity_path, size_path]
        paths[f"panel_{window}"] = save_table(joined, "panel", f"company_panel_{window}.csv", licensed=True,
                                                inputs=inputs, source_modules=[sys.modules[__name__]],
                                                note="500 fixed representative issuers at every formation; partial rows retained. FMP-derived size fields.")
        descriptions = describe_panel(joined)
        descriptions["loading_stability"] = loading_stability(joined)
        for name, table in descriptions.items():
            paths[f"{name}_{window}"] = save_table(table, "panel", f"{name}_{window}.csv", licensed=True,
                                                       inputs=[paths[f"panel_{window}"]], source_modules=[sys.modules[__name__]])
        ledger_columns = [c for c in joined if c in ["formation_date", "ticker", "cik", "window_sessions"]
                          or "eligible" in c or "reason" in c or "observations" in c or "cutoff" in c]
        paths[f"eligibility_{window}"] = save_table(joined[ledger_columns], "panel", f"eligibility_{window}.csv", licensed=True,
                                                        inputs=[paths[f"panel_{window}"]], source_modules=[sys.modules[__name__]])
    _plot_panel(pd.read_csv(paths["panel_504"]), pd.read_csv(paths["correlations_504"]))
    make_notebook(settings.repository_root_folder / "notebooks/07_characteristic_panel.ipynb", "Company characteristics",
                  "The fixed snapshot has 500 issuers. Each measurement retains its own eligible sample and cutoff. "
                  "Size and capitalization-weighted summaries use FMP data; GICS is retrospective. See the saved coverage and eligibility tables.",
                  ["from quiet_index_500.panel import notebook_summary\nnotebook_summary()",
                   "from IPython.display import Image, display\nfrom quiet_index_500 import settings\ndisplay(Image(filename=str(settings.results_folder / 'panel/loading_stability.png')))\ndisplay(Image(filename=str(settings.results_folder / 'panel/characteristic_correlations.png')))"])
    return paths


def _plot_panel(panel, correlations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = settings.results_folder / "panel"
    destination.mkdir(exist_ok=True)
    dates = pd.to_datetime(panel.formation_date)
    panel = panel.assign(formation_date=dates)
    quantiles = panel.groupby("formation_date").m4_smh_excess.quantile([.1, .5, .9]).unstack()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axis = axes[0]
    axis.fill_between(quantiles.index, quantiles[.1], quantiles[.9], alpha=.2, label="10th–90th percentile")
    axis.plot(quantiles.index, quantiles[.5], label="Median")
    axis.set(title="Additional SMH loading across companies", ylabel="FF5 + momentum conditional SMH loading")
    axis.legend()
    stability = loading_stability(panel)
    axes[1].plot(pd.to_datetime(stability.formation_date), stability.spearman, color="tab:green")
    axes[1].set(title="Consecutive-formation loading rank stability", ylabel="Spearman correlation", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(destination / "loading_stability.png", dpi=160)
    plt.close(fig)
    averaged = correlations.groupby(["first", "second"]).spearman.mean()
    grid = pd.DataFrame(np.eye(len(characteristics)), index=characteristics, columns=characteristics)
    for (first, second), value in averaged.items():
        grid.loc[first, second] = grid.loc[second, first] = value
    fig, axis = plt.subplots(figsize=(11, 9))
    heatmap = axis.imshow(grid, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(grid)), grid.columns, rotation=75, ha="right", fontsize=8)
    axis.set_yticks(range(len(grid)), grid.index, fontsize=8)
    axis.set_title("Mean within-formation Spearman correlation (FMP size included)")
    fig.colorbar(heatmap, ax=axis)
    fig.tight_layout()
    fig.savefig(destination / "characteristic_correlations.png", dpi=160)
    plt.close(fig)


def notebook_summary():
    """Read saved coverage and descriptive tables for the notebook."""
    root = settings.licensed_results_folder / "panel"
    coverage = pd.read_csv(root / "coverage_504.csv")
    return coverage.groupby("characteristic").agg(minimum_companies=("companies", "min"),
                                                   maximum_companies=("companies", "max"),
                                                   mean_snapshot_weight=("represented_snapshot_weight", "mean"))


if __name__ == "__main__":
    print(build_characteristic_panel())
