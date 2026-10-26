"""Rebuild publication figures and reading tables from saved study results, offline.

This presentation layer does not estimate models or load raw vendor data. Each
figure has its own CSV and manifest. Run after quiet_index_500.reproduce.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from quiet_index_500 import settings
from quiet_index_500.foundation.manifests import write_manifest, sha256_of_file

ROOT = settings.repository_root_folder
OUT = ROOT / "results/publication"
PERIODS = ["development", "validation", "confirmation"]
COLORS = ["#176B87", "#D16B31", "#467B57", "#845A97"]


def read(relative):
    return pd.read_csv(ROOT / "results" / relative)


def save_data(name, frame, sources, note):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    frame.to_csv(path, index=False, float_format="%.12g")
    write_manifest(path, [ROOT / "results" / s for s in sources],
                   "quiet_index_500.publication", source_modules=[sys.modules[__name__]], note=note)
    return path


def save_figure(fig, name, subtitle):
    import matplotlib.pyplot as plt
    fig.text(.06, .015, subtitle, fontsize=9, color="#53616b", va="bottom")
    fig.tight_layout(rect=(0, .08, 1, .98))
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def build_publication():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.titlesize": 15, "axes.titleweight": "bold",
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.labelcolor": "#253544", "text.color": "#253544",
                        "axes.prop_cycle": plt.cycler(color=COLORS), "svg.fonttype": "none"})

    # 1. Report companies and represented snapshot weight, not just panel rows.
    coverage = read("licensed/panel/coverage_504.csv")
    labels = {"m4_smh_excess": "Factor exposure", "log_market_cap": "Company size",
              "log_amihud_252_per_million": "Amihud illiquidity", "own_momentum": "Own momentum"}
    data = coverage[coverage.characteristic.isin(labels)].copy()
    save_data("01_coverage", data, ["licensed/panel/coverage_504.csv"],
              "Company-level valid measurements, 504-session panel; snapshot weight is retrospective, not historical market coverage.")
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7), sharex=True)
    for name, label in labels.items():
        rows = data[data.characteristic.eq(name)]
        dates = pd.to_datetime(rows.formation_date)
        axes[0].plot(dates, rows.companies, label=label, linewidth=2)
        axes[1].plot(dates, rows.represented_snapshot_weight * 100, linewidth=2)
    axes[0].axhline(500, color="#8b979e", ls=":", label="500-company snapshot")
    axes[0].set(title="The company panel is complete; each measure has its own sample", ylabel="Companies with a value", ylim=(0, 520))
    axes[0].legend(ncol=2, fontsize=9, loc="lower right")
    axes[1].set(ylabel="Represented snapshot weight (%)", ylim=(0, 103), xlabel="Formation date")
    save_figure(fig, "01_coverage", "36 formations; one representative per issuer. September 2026 factor measurements stop at August 31.\nSource: Notebook 07, combining 04–06. Later regressions apply their own matched-sample rules.")

    # 2. Distribution and persistence answer distinct questions.
    panel = read("licensed/panel/company_panel_504.csv")
    quantiles = panel.groupby("formation_date").m4_smh_excess.quantile([.1, .5, .9]).unstack()
    quantiles.columns = ["p10", "p50", "p90"]
    quantiles = quantiles.reset_index()
    stability = read("licensed/panel/loading_stability_504.csv")
    data = quantiles.merge(stability, on="formation_date", how="left", validate="one_to_one")
    data["measurement_cutoff"] = pd.to_datetime(data.formation_date).clip(upper=pd.Timestamp("2026-08-31"))
    save_data("02_exposure_stability", data, ["licensed/panel/company_panel_504.csv", "licensed/panel/loading_stability_504.csv"],
              "Equal-company loading quantiles and consecutive-formation Spearman rank correlation; persistence is not forecast evidence.")
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7), sharex=True)
    dates = pd.to_datetime(data.formation_date)
    axes[0].fill_between(dates, data.p10, data.p90, color=COLORS[0], alpha=.16, label="10th–90th percentile")
    axes[0].plot(dates, data.p50, linewidth=2, label="Median")
    axes[0].axhline(0, color="#8b979e", linewidth=.8)
    axes[0].set(title="Semiconductor loading varies across companies", ylabel="Conditional SMH loading")
    axes[0].legend(fontsize=9)
    axes[1].plot(dates, data.spearman, color=COLORS[2], linewidth=2)
    axes[1].set(ylabel="Consecutive-quarter rank correlation", ylim=(0, 1), xlabel="Formation date")
    save_figure(fig, "02_exposure_stability", "504-session estimates; paired companies for rank correlation. A stable exposure is not necessarily a useful forecast feature.\nSource: Notebook 07 using Notebook 04. Final factor cutoff: August 31, 2026.")

    # 3. A compact relationship map; save pairwise counts rather than imply one common sample.
    labels = {"log_market_cap": "Size (log cap)", "log_median_dollar_volume_60": "Dollar volume (log)",
              "log_amihud_252_per_million": "Amihud (log)", "ordinary_beta": "Ordinary beta",
              "downside_beta": "Downside beta", "own_momentum": "Own momentum", "m4_smh_excess": "SMH loading"}
    correlations = read("licensed/panel/correlations_504.csv")
    pairs = correlations[correlations["first"].isin(labels) & correlations["second"].isin(labels)]
    data = pairs.groupby(["first", "second"]).agg(mean_spearman=("spearman", "mean"),
                formations=("spearman", "count"), minimum_companies=("companies", "min"), maximum_companies=("companies", "max")).reset_index()
    save_data("03_characteristic_overlap", data, ["licensed/panel/correlations_504.csv"],
              "Equal-formation mean pairwise Spearman correlations; company samples differ by pair and formation. FMP-derived size included.")
    grid = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for row in data.itertuples():
        grid.loc[row.first, row.second] = grid.loc[row.second, row.first] = row.mean_spearman
    fig, ax = plt.subplots(figsize=(10.4, 8))
    im = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels.values(), rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)), labels.values(), fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = grid.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if abs(v)>.65 else "#253544")
    ax.set_title("Related characteristics are not interchangeable", pad=16)
    fig.colorbar(im, ax=ax, shrink=.7, label="Mean Spearman correlation")
    save_figure(fig, "03_characteristic_overlap", "36 equally weighted formations; pairwise eligible company samples; descriptive correlations, not causal effects.\nSource: Notebook 07, combining factor, downside, momentum, size and liquidity measurements.")

    # 4. Use corrected circular intervals explicitly; do not use old bootstrap_*_percent fields.
    forecasts = read("licensed/report/forecast_summary.csv")
    labels = {"semiconductor": "Signed SMH · sector controls", "semiconductor_industry": "Signed SMH · industry groups",
              "liquidity": "+ Amihud · separate extension", "dollar_volume": "+ Dollar volume · exploratory"}
    data = forecasts[(forecasts.window_sessions.eq(504)) & forecasts.outcome.eq("log_forward_residual_volatility")
                     & forecasts.comparison.isin(labels) & forecasts.outcome_block.isin(PERIODS[1:])].copy()
    for col in ["mean", "circular_bootstrap_low", "circular_bootstrap_high"]:
        data[col + "_pct"] = 100 * data[col]
    save_data("04_forward_risk", data, ["licensed/report/forecast_summary.csv"],
              "Mean quarterly percentage reduction in log-volatility MSE; 95% circular quarter-block intervals conditional on first-stage exposures. Liquidity comparisons add to the SMH model, not the styles-only baseline.")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5), sharex=True, sharey=True)
    for ax, period in zip(axes, PERIODS[1:]):
        rows = data[data.outcome_block.eq(period)].set_index("comparison").reindex(labels)
        for i, (_, r) in enumerate(rows.iterrows()):
            ax.plot([r.circular_bootstrap_low_pct, r.circular_bootstrap_high_pct], [i, i], lw=2, color=COLORS[i])
            ax.scatter(r.mean_pct, i, s=55, color=COLORS[i], zorder=3)
            ax.annotate(f"{r.mean_pct:+.2f}%", (r.mean_pct, i), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, color="#8b979e", ls="--")
        ax.set_title(f"{period.title()} · {int(rows.quarters.iloc[0])} quarters")
        ax.set_xlabel("Reduction in log-risk forecast MSE (%)")
        ax.set_yticks(range(len(labels)), labels.values(), fontsize=10)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_ylim(3.55, -.55)
    save_figure(fig, "04_forward_risk", "Positive = lower forecast error. Dots: quarter-equal means; lines: conditional 95% circular block intervals.\nNotebook 08: matched companies within each comparison. Only six confirmation quarters; exploratory results remain provisional.")

    # 5. Absolute losses retained in CSV; relative loss change has a different aggregation from Figure 4.
    covariance = read("licensed/report/covariance_summary.csv")
    labels = {"snapshot": "Snapshot weighted", "equal": "Equal security weighted", "high_smh": "Higher SMH loading half", "low_smh": "Lower SMH loading half"}
    data = covariance[covariance.window_sessions.eq(504) & covariance.outcome_block.eq("confirmation") & covariance.portfolio.isin(labels)].copy()
    save_data("05_portfolio_comparison", data, ["licensed/report/covariance_summary.csv"],
              "100*(1-mean QLIKE augmented/mean QLIKE baseline), six confirmation quarters per portfolio. Positive is better; portfolios overlap and are not independent tests.")
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for i, (treatment, label) in enumerate([("diagonal", "Diagonal residuals"), ("industry_block", "Within-industry residuals")]):
        rows = data[data.residual_treatment.eq(treatment)].set_index("portfolio").reindex(labels)
        bars = ax.barh(np.arange(4)+(i-.5)*.32, rows.qlike_improvement_percent, height=.30, label=label, color=COLORS[i])
        ax.bar_label(bars, fmt="%+.2f%%", padding=4, fontsize=9)
    ax.axvline(0, color="#8b979e", linewidth=.8)
    ax.set_yticks(range(4), labels.values())
    ax.invert_yaxis()
    ax.set(xlabel="Reduction in period mean QLIKE (%)", title="Portfolio benefit depends on the residual-covariance baseline", xlim=(-6, 17))
    ax.legend(loc="lower right", fontsize=9)
    save_figure(fig, "05_portfolio_comparison", "2025 Q1–2026 Q2; 504-session estimation; fixed holdings and daily target weights. Positive = lower variance loss.\nNotebook 09. Six quarters per bar; point estimates only. Snapshot paired-difference intervals include zero under both treatments.")

    # 6. Variance levels explain what the loss comparisons are measuring.
    evaluation = read("covariance_evaluation/quarterly_evaluation.csv")
    data = evaluation[evaluation.window_sessions.eq(504) & evaluation.portfolio.eq("snapshot")].copy()
    save_data("06_variance_calibration", data, ["covariance_evaluation/quarterly_evaluation.csv"],
              "Annualized variance in decimal-return-squared units. Ineligible quarters remain in the table and plot as gaps; outcome quarters after 2026Q2 excluded from chart.")
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.3), sharex=True)
    for ax, treatment, title in zip(axes, ["diagonal", "industry_block"], ["Diagonal residual covariance", "Within-industry residual covariance"]):
        rows = data[data.residual_treatment.eq(treatment) & data.outcome_complete].copy()
        rows["date"] = pd.PeriodIndex(rows.outcome_quarter, freq="Q").to_timestamp(how="end")
        for j, (model, label) in enumerate([("ff5_mom", "FF5 + momentum"), ("ff5_mom_smh", "+ SMH")]):
            r = rows[rows.model.eq(model)].sort_values("date")
            ax.plot(r.date, r.predicted_variance.where(r.evaluation_eligible), label=label, color=COLORS[j], lw=1.8)
        r = rows[rows.model.eq("ff5_mom")].sort_values("date")
        ax.plot(r.date, r.realized_variance.where(r.evaluation_eligible), color="#253544", label="Realized", lw=1.5, ls="--")
        ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-30"), color=COLORS[0], alpha=.06)
        ax.set(title=title, ylabel="Annualized variance (decimal²)", ylim=(0, None))
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].set_xlabel("Following-quarter end")
    save_figure(fig, "06_variance_calibration", "Notebook 09: snapshot-weighted portfolio; 33 eligible outcome quarters. Gap = failed common training sample.\nShading marks six confirmation quarters. Realized variance is a noisy outcome measure; these are not return forecasts.")

    # 7. An event correction illustrates why the foundation matters to later estimates.
    data = read("review/event_sensitivity.csv")
    save_data("07_event_sensitivity", data, ["review/event_sensitivity.csv"],
              "DHR residual-volatility sensitivity; only three 504-session windows include the 2016 event. Alternative valuation changes no primary inputs.")
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    labels = {"original_vendor_double_adjustment": "Original vendor error", "primary_fortive_first_close": "Accepted correction", "form_8937_average_valuation": "Alternative distribution valuation"}
    for i, (scenario, label) in enumerate(labels.items()):
        r = data[data.scenario.eq(scenario)].sort_values("formation_date")
        ax.bar(np.arange(len(r))+(i-1)*.24, r.residual_volatility*100, width=.23, label=label, color=COLORS[i])
    ax.set_xticks(range(3), ["Dec 2017", "Mar 2018", "Jun 2018"])
    ax.set(ylabel="Annualized FF5 + momentum residual volatility (%)", xlabel="Formation date", title="One erroneous return can dominate a company's risk estimate")
    ax.legend(fontsize=9)
    save_figure(fig, "07_event_sensitivity", "DHR only; three affected 504-session windows. The two corrected series nearly overlap.\nFoundation correction → Notebook 04 estimates. Event scenarios change one return in memory, never raw data.")

    # Exact comparison samples and uncertainty used by the publication.
    quarters = read("licensed/forward_tests/quarters_504.csv")
    forward = quarters[quarters.analysis.eq("forecast") & quarters.outcome.eq("log_forward_residual_volatility")]
    counts = forward.groupby(["comparison", "outcome_block"]).agg(minimum_companies=("companies", "min"), maximum_companies=("companies", "max"),
                    first_formation=("formation_date", "min"), last_formation=("formation_date", "max")).reset_index()
    table = forecasts[forecasts.window_sessions.eq(504) & forecasts.outcome.eq("log_forward_residual_volatility") & forecasts.outcome_block.ne("all")].merge(counts, on=["comparison", "outcome_block"], validate="one_to_one")
    save_data("forecast_evidence", table, ["licensed/report/forecast_summary.csv", "licensed/forward_tests/quarters_504.csv"], "All main-window residual-risk forecast comparisons, with exact company-count ranges and conditional uncertainty.")
    ci = read("covariance_evaluation/paired_inference.csv")
    ci = ci[ci.window_sessions.eq(504) & ci.portfolio.eq("snapshot") & ci.metric.eq("qlike")]
    save_data("covariance_uncertainty", ci, ["covariance_evaluation/paired_inference.csv"], "Paired SMH-minus-baseline QLIKE; negative favors SMH. Use circular interval columns; six confirmation quarters.")
    render_report()
    files = sorted(p for p in OUT.iterdir() if p.suffix in {".png", ".svg", ".csv"})
    (OUT / "publication_manifest.json").write_text(json.dumps({"generator": "src/quiet_index_500/publication.py", "files": [
        {"file": p.relative_to(ROOT).as_posix(), "sha256": sha256_of_file(p)} for p in files]}, indent=2)+"\n")
    return OUT


def render_report():
    """Static local HTML; uses relative image links and no remote scripts/fonts."""
    source = ROOT / "docs/RESEARCH_REPORT.md"
    if not source.exists():
        return
    import mistune
    markdown = mistune.create_markdown(plugins=["table", "strikethrough", "url"])
    body = markdown(source.read_text())
    css = """body{font-family:system-ui,-apple-system,sans-serif;color:#253544;background:#f4f6f7;margin:0;line-height:1.65}
    main{max-width:1000px;margin:40px auto;padding:52px;background:white;border-top:6px solid #176B87}
    h1{font-size:2.5em;line-height:1.15;letter-spacing:-.035em}h2{margin-top:2em;line-height:1.25}h3{margin-top:1.6em}
    a{color:#176B87}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:.9em;width:100%;display:block;overflow:auto}
    th,td{border-bottom:1px solid #dce3e7;padding:10px;text-align:left}th{background:#edf3f5}blockquote{border-left:3px solid #176B87;margin-left:0;padding-left:20px;color:#53616b}
    code{font-family:ui-monospace,monospace;font-size:.85em}pre{overflow:auto;background:#f4f6f7;padding:16px}p,li{max-width:90ch}
    @media(max-width:700px){main{margin:0;padding:24px}h1{font-size:2em}}@media print{body{background:white}main{margin:0;padding:0;border:0}h2{break-after:avoid}img,table{break-inside:avoid}}"""
    (ROOT / "docs/RESEARCH_REPORT.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Semiconductor exposure and subsequent risk</title><style>{css}</style><main>{body}</main></html>')


if __name__ == "__main__":
    print(build_publication())
