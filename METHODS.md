# Methods and evidence

This package supports a reading route and computational reproduction of selected comparisons from an executed historical investigation. The tables below are exact copies of saved scientific evidence. The selected reading notebooks retain calculation code and saved outputs, with adapted explanatory text and navigation. Separate scientific study folders preserve the executable layouts and required prepared inputs.

## Population and timing

The universe contains 503 securities representing 500 issuers from a September 2026 IVV holdings snapshot. Company models use a fixed representative security for each issuer; portfolio models retain eligible security holdings. Applying contemporary membership, classifications and weights to earlier periods introduces survivorship and hindsight limitations.

The original and revised company forecasts cover 26 quarters: 12 in 2020–22, eight in 2023–24 and six in 2025–2026 Q2. The intermediate company study scores the latter 14 quarters. The label `confirmation` in saved tables is historical: these periods have since been inspected and reused.

The main formation window is 504 trading sessions, with a 252-session sensitivity. Model fitting uses earlier completed outcomes, and paired model comparisons use matched eligible observations. The original and revised principal comparisons contain 12,511 issuer-quarters; the later blocks contain 3,892 and 2,945. Matching evaluation sets does not equate training weights, feature definitions or model specifications across studies.

## Prepared recent-risk and liquidity inputs

The revised company comparisons start from [saved formation measurements](reproduction/company-risk/extensions/company_risk/data/processed/formation_measurements.csv). The definitions below were checked against the complete revised measurement generator at scientific revision `4fa7d9f07a427931fd86854c780a589d8499aca0`, including its final eligibility corrections. All logarithms are natural; returns are daily decimals. Annualization uses 252 sessions.

For factor-dependent measurements, the cutoff is the earlier of formation and the available factor-data endpoint. The principal 504-session fit requires at least **400 jointly valid dates** for company excess returns, the six conventional factors and SMH excess returns; the 252-session sensitivity requires 200. Conventional OLS includes an intercept, rejects rank-deficient designs and uses the same matched dates as the semiconductor measurement. Liquidity windows end at formation and do not require factor observations on each liquidity date.

| Forecast input | Calculation and transformation | Window and minimum coverage | Units before the logarithm |
|---|---|---|---|
| Recent risk, `R60` | Apply the long-window conventional coefficients to recent excess returns; take `log(sqrt(252 × sample variance of residuals))`, with denominator `n_recent − 1`. Coefficients are not re-estimated on the recent window. | Last 60 sessions through the factor cutoff, intersected with the long fit's matched dates; at least **40** valid residuals. Requires an eligible long-window fit. | Annualized residual volatility in decimal-return units. |
| Main illiquidity, `L_amihud252` | For each valid date, calculate `abs(total return) / dollar volume × 1e6`; take the arithmetic mean, then its logarithm. | Last **252** sessions through formation; at least **200** valid daily ratios. | Absolute decimal return per **USD million** traded. |
| Main trading activity, `L_dollar_volume60` | Dollar volume is compatible split-adjusted close × share volume. Take the median of valid daily dollar volumes, then its logarithm. | Last **60** sessions through formation; at least **40** valid dollar-volume observations. | **USD per trading session**. |
| Long-window risk in C, `long_residual_vol_annual` | Stored as `long_log_residual_vol = log(sqrt(252 × SSE/(n_joint − 7)))`; the forecasting code exponentiates it and uses **raw annualized volatility**. Seven parameters are the intercept and six conventional slopes. | Principal **504** sessions / at least **400** jointly valid dates; sensitivity **252** / at least **200**. | Annualized residual volatility in decimal-return units; the forecast input is not logged. |

The matched-horizon sensitivities use the same definitions for `L_amihud60` (60 sessions / 40 observations) and `L_dollar_volume252` (252 / 200). Recent-risk sample variance and the long-risk residual degrees-of-freedom convention are intentionally different. Consequently, adding R changes both horizon and functional form; it does not isolate a pure recency effect.

**Missing and excluded observations:** use the corrected, same-security price/return panels. Dollar volume requires positive price and volume; Amihud also requires a valid daily total return. Missing daily values are dropped within the stated window and counted, never filled or bridged. A feature remains missing if coverage is insufficient or the aggregate to be logged is nonpositive. Factor-dependent features are cleared when the formation window crosses a same-security start; liquidity features are cleared when their own horizon crosses that boundary. Final saved reason columns report eligibility after these corrections; a raw observation count alone does not prove eligibility. Comparison-specific union-feature rules then exclude any row missing a required input or fixed outcome, with the same sample used for every model in that comparison.

The [measurement manifest](reproduction/company-risk/extensions/company_risk/data/processed/formation_measurements.manifest.json) identifies `02_formation_measurements.ipynb` and its code hash. **That revised upstream notebook is not included in this selected public source subset.** The manifest preserves provenance; it is not an executable raw-data rebuild. The public company reproduction starts with prepared measurements and outcomes, then performs chronological training, forecasting and scoring. The [comparison manifest](reproduction/company-risk/extensions/company_risk/specification/comparison_manifest.json) selects horizons and model combinations; the calculations above supply the feature definitions. See the [reproduction boundary](REPRODUCE.md#full-raw-to-measurement-rebuild-boundary).

## Metrics and signs

**Company MSE:** for each quarter, average squared log-volatility prediction errors over eligible companies. The primary improvement is `100 × (baseline MSE − extended MSE) / baseline MSE`, then averaged equally over quarters. This differs from the percentage reduction in period-average MSE. Positive is favorable.

**Calibration decomposition:** quarterly MSE equals squared mean error plus mean squared deviations from that mean. Divide changes in both components by the same baseline quarterly MSE before averaging their percentage-point contributions. Centered errors are not a ranking measure. These are retrospective diagnostics; no future mean-error correction enters a forecast.

**Portfolio QLIKE:** the reported increment is the equal-quarter mean of baseline loss minus extended loss. Positive is favorable. The portfolio numbers in the synthesis are absolute loss differences, not percentages. The intermediate company study also uses variance QLIKE; a common loss name does not make its outcome or aggregation interchangeable with the portfolio endpoint.

**Uncertainty:** use the corrected circular-block columns, not older noncircular columns retained in source tables. The revised intervals use four-quarter blocks and 2,000 resamples, conditioning on measured features and fitted forecasts. Six late quarters provide limited independent information.

## Claim-to-evidence map

| Claim | Exact table and selection |
|---|---|
| Recent risk 10.057%; liquidity alone 4.504%; liquidity after recent risk 1.148% | [company_inference.csv](evidence/company_inference.csv): `run_id=calibration_main`, `block_label=all`; pairs C→C+R, C→C+L, C+R→B; `mean_percentage_gain`, `conditional_low/high` |
| Liquidity calibration 0.927 pp and centered-error 0.220 pp | Same table, C+R→B: `mean_calibration_percentage_gain`, `mean_dispersion_percentage_gain` |
| Fixed-horizon liquidity 0.716% / −0.366% | Same table, `calibration_matchedL252` / `calibration_matchedL60`, `block_label=all`, C+R→B |
| Original signed-SMH +1.043% / −0.363% | [aggregate_comparisons.csv](evidence/aggregate_comparisons.csv): `original_504_semiconductor`, `block=validation/confirmation`, `mean_quarterly_gain_pct` |
| Revised signed-SMH +0.406% / −0.461% | Same table: `revised_semiconductor_main`, same block/metric; also company inference, B→B+S |
| Semiconductor subgroup benefits and industry-control reversals | [group_comparisons.csv](evidence/group_comparisons.csv): `group=Semiconductors & Semiconductor Equipment`, `block=confirmation`; `original_504_semiconductor`, `revised_semiconductor_main`, `original_504_semiconductor_industry`, `revised_semiconductor_industry_group`; `mean_quarterly_gain_pct`, `favorable_quarters` |
| Subgroup uncertainty and family-adjusted diagnostics | [group_uncertainty.csv](evidence/group_uncertainty.csv): primary runs, `signflip_two_sided_p`, `holm_p`, `family_size`. Independent/symmetric-quarter sign-flip assumptions do not allow arbitrary serial dependence. |
| Portfolio sign changes across residual restrictions | [portfolio_contrasts.csv](evidence/portfolio_contrasts.csv): `outcome_block=confirmation`, `contrast=smh_without_treasury`, `residual_treatment=industry_block/diagonal`; `mean_difference`, `circular_bootstrap_interval_low/high` |
| Treasury controls barely change portfolio increments | Same table: `contrast=smh_with_treasury`, otherwise identical selection |

The tables retain all their saved rows, including adverse comparisons and sensitivities. Source commits, original paths and SHA-256 fingerprints are in [evidence provenance](evidence/provenance.json). Commit identities record origin; the reading route does not require access to the private research repository.

## How the calculations relate

The [factor-risk source](reproduction/factor-risk/) supplies the initial company and portfolio tests. The [calibration-covariance source](reproduction/calibration-covariance/) investigates calibration, trading activity, rate timing and covariance. The [company-risk source](reproduction/company-risk/) compares C, C+R, C+L and B before testing signed semiconductor loading, variance composition and asymmetry. Its Treasury application is a separate portfolio experiment. Historical study labels and exact revisions remain in [provenance](reproduction/provenance.json).

The integration reconciles outcomes, samples, scoring and covariance accounting; it is not a fourth independent experiment. The subgroup diagnostic uses saved forecasts and does not refit the underlying models. Its [standalone public route](SUBGROUPS.md) exports and rescores the four displayed 504-session sector/industry comparisons, checks the paired observations and reconciles them to the existing evidence tables. The [Python walkthrough](FORECAST_WALKTHROUGH.md) separately reconstructs one revised formation from saved numeric coefficients; it recovers omitted sector offsets from training-only OLS normal equations and checks all companies at that date.

The original/revised main log-volatility predictions are unsmeared. Intermediate variance predictions use a training-only retransformation correction. Scores on different transformed scales must not be silently substituted. Revised recent-risk addition also changes functional form as well as horizon.

Covariance identities require matched observation masks, intercept OLS and consistent sample covariance denominators. Under those conditions, an industry-block residual model preserves individual variances and within-industry covariance when the factor span is expanded. Those identities check accounting, not forecasting performance.

## Reproduction scope

The selected notebooks at `notebooks/` are **reading editions**. The executable layouts, calculation sources and required prepared inputs are under `reproduction/`. The [runner and instructions](REPRODUCE.md) execute supported comparisons in a separate output directory so the shipped evidence remains unchanged.

Prepared-measurement reproduction refits and scores forecasts from frozen measurements; it does not rebuild those measurements from prices. Raw-input-dependent runs additionally require the checksum-verified private vendor snapshot. The package includes the intermediate study's frozen FRED source files. No equivalence between FMP tenor yields and FRED series/vintages is asserted.

The [reproduction record](REPRODUCE.md) distinguishes checks performed on this package from historical executions. Existing checks and bounded methods reviews support narrower claims than full independent replication. Notebook narrative has been adapted for reading; interpretation remains a research draft. No fresh market-data collection is required.

Source records contain a historical training-exception ledger with incorrect labels outside overlapping evaluation periods. This package does not distribute or rely on that faulty ledger. The supported later-period comparison concerns one excluded CTVA training observation at 2020-12-31, repeated across 14 later evaluation formations; it does not generalize to earlier non-overlapping experiments.

[Overview](README.md) · [Findings](FINDINGS.md) · [Supporting notebooks](notebooks/README.md)
