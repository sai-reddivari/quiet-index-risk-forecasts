"""Compact report evidence and scientific figures from saved evaluation tables."""

import sys

import pandas as pd

from quiet_index_500 import settings
from quiet_index_500.research_io import save_table


def build_report_tables_and_figures():
    """Select and label the statistics cited by the final research report."""
    forward_folder = settings.licensed_results_folder / "forward_tests"
    inference_path = forward_folder / "paired_inference.csv"
    coefficient_path = forward_folder / "coefficient_inference.csv"
    inference = pd.read_csv(inference_path)
    forecasts = inference[inference.analysis.eq("forecast") & inference.metric.eq("relative_mse_improvement")].copy()
    for column in ["mean", "bootstrap_low", "bootstrap_high"]:
        forecasts[f"{column}_percent"] = forecasts[column] * 100
    forecasts["positive_means"] = "SMH/extension reduces MSE relative to its paired restricted model"
    paths = {"forecast_summary": save_table(forecasts, "report", "forecast_summary.csv", licensed=True,
              inputs=[inference_path], source_modules=[sys.modules[__name__]],
              note="Quarter-equal percentage MSE improvement; negative is worse. Intervals conditional on first-stage estimates; confirmation has only six quarters.")}
    coefficients = pd.read_csv(coefficient_path)
    paths["association_summary"] = save_table(coefficients, "report", "association_summary.csv", licensed=True,
                                               inputs=[coefficient_path], source_modules=[sys.modules[__name__]])
    covariance_paths = [folder / "covariance_evaluation/period_summary.csv" for folder in [settings.results_folder, settings.licensed_results_folder]]
    covariance = pd.concat([pd.read_csv(path) for path in covariance_paths], ignore_index=True)
    key = ["window_sessions", "portfolio", "residual_treatment", "outcome_block"]
    baseline = covariance[covariance.model.eq("ff5_mom")].drop(columns="model")
    augmented = covariance[covariance.model.eq("ff5_mom_smh")].drop(columns="model")
    paired = baseline.merge(augmented, on=key, suffixes=("_baseline", "_smh"), validate="one_to_one")
    paired["qlike_improvement_percent"] = 100 * (1 - paired.mean_qlike_smh / paired.mean_qlike_baseline)
    paired["positive_means"] = "SMH reduces mean QLIKE relative to baseline"
    paths["covariance_summary"] = save_table(paired, "report", "covariance_summary.csv", licensed=True,
                                               inputs=covariance_paths, source_modules=[sys.modules[__name__]])
    _figures(forecasts, paired)
    return paths


def _figures(forecasts, covariance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    destination = settings.results_folder / "report"
    destination.mkdir(exist_ok=True)
    periods = ["development", "validation", "confirmation"]
    labels = {"semiconductor": "Signed SMH, sector controls", "semiconductor_industry": "Signed SMH, industry controls",
              "semiconductor_variance_component": "SMH variance component (exploratory)"}
    fig, axis = plt.subplots(figsize=(10, 5))
    subset = forecasts[forecasts.window_sessions.eq(504) & forecasts.outcome.eq("log_forward_residual_volatility")]
    for index, (name, label) in enumerate(labels.items()):
        values = subset[subset.comparison.eq(name)].set_index("outcome_block").reindex(periods)
        axis.bar(np.arange(3) + (index - 1) * .24, values.mean_percent, width=.24, label=label)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(np.arange(3), ["Development\n12 forecast quarters", "Validation\n8 quarters", "Confirmation\n6 quarters"])
    axis.set(ylabel="Mean quarterly reduction in log-risk MSE (%)", title="SMH does not consistently improve residual-risk forecasts")
    axis.legend(fontsize=8)
    fig.text(.01, .01, "504-session exposures; expanding training; matched companies. Positive is better. Six confirmation quarters limit inference.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(destination / "forward_residual_risk.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    subset = covariance[covariance.window_sessions.eq(504) & covariance.portfolio.eq("snapshot")]
    for index, (name, label) in enumerate([("diagonal", "Diagonal residual covariance"), ("industry_block", "Within-industry residual covariance")]):
        values = subset[subset.residual_treatment.eq(name)].set_index("outcome_block").reindex(periods)
        axis.bar(np.arange(3) + (index - .5) * .32, values.qlike_improvement_percent, width=.32, label=label)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(np.arange(3), ["Development\n19 eligible quarters", "Validation\n8 quarters", "Confirmation\n6 quarters"])
    axis.set(ylabel="Reduction in mean QLIKE (%)", title="Snapshot portfolio: SMH benefit depends on residual treatment")
    axis.legend(fontsize=8)
    fig.text(.01, .01, "Same fixed holdings, common training/future dates, and covariance denominator. Positive is better; FMP size not used for this portfolio.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(destination / "snapshot_covariance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    print(build_report_tables_and_figures())
