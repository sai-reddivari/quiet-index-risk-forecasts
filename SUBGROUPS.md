# Where signed SMH helped—and where it did not

This is a historical diagnostic of saved company-risk forecasts. It supports descriptive subgroup benefits, with insufficient evidence to identify reliable beneficiaries on new observations. It adds no fitted models or market data. Positive gains mean lower squared error in **log annualized conventional-factor residual volatility**; they are not changes in realized volatility or investment returns.

The compact input contains four **504-session** comparisons across the same 26 outcome quarters, 2020Q1–2026Q2: 12 development, eight validation and six late/confirmation quarters. Original and revised forecasts remain separate experiments. Each has 12,511 matched issuer-quarters, including 2,945 late rows; their saved targets agree within rounding. The existing aggregate evidence includes additional runs that this compact export does not reproduce.

| Saved run | Benchmark controls | Late whole-population gain | Favorable late quarters |
|---|---|---:|---:|
| `original_504_semiconductor` | Original benchmark, sector indicators | −0.363% | 2/6 |
| `original_504_semiconductor_industry` | Original benchmark, industry-group indicators | −0.159% | 0/6 |
| `revised_semiconductor_main` | Revised B=C+R+L, sector indicators | −0.461% | 0/6 |
| `revised_semiconductor_industry_group` | Revised B=C+R+L, industry-group indicators | −0.136% | 1/6 |

These are means of quarterly percentage reductions in MSE. [All quarters and summaries](evidence/subgroups/derived/all_summary.csv) retain losses, counts, favorable/adverse signs and a separately labeled ratio of period-average losses.

## Benefits depend on the benchmark

The 19 semiconductor-group firms have all 114 late issuer-quarters. Against original sector controls, their MSE falls from 0.141116 to 0.123919: **+11.548% mean quarterly gain**, with five favorable quarters. The reduction in period-average loss is a different statistic, **12.187%**. The adverse semiconductor quarter, 2025Q2, remains included. Revised sector controls also show a benefit, +5.160%, in five of six quarters. Both gains reverse under the corresponding industry-group controls.

| Semiconductor group | Development gain | Validation gain | Late gain | Favorable late quarters |
|---|---:|---:|---:|---:|
| Original, sector controls | −2.436% | +1.125% | +11.548% | 5/6 |
| Original, industry-group controls | −0.037% | −0.831% | −1.644% | 2/6 |
| Revised, sector controls | +2.202% | +0.589% | +5.160% | 5/6 |
| Revised, industry-group controls | −0.247% | −0.533% | −0.704% | 3/6 |

Controls enter the forecasting regression before prediction. Subgroup reporting groups its subsequent errors. Sector controls distinguish broad categories; industry-group controls supply finer membership information, including semiconductor membership. Reporting semiconductor errors therefore remains informative under either benchmark. Each source model refits all coefficients, so the difference between specifications does not isolate a causal membership channel or the SMH term alone. The supplied retrospective GICS **industry groups** are not finer subindustries or reconstructed point-in-time classifications.

The original late sector results contain consistent gains and consistent deterioration:

| Sector | Mean quarterly gain | Favorable quarters | Contribution to population headline |
|---|---:|---:|---:|
| Health Care | +3.396% | 6/6 | +0.456 pp |
| Materials | +3.366% | 6/6 | +0.142 pp |
| Consumer Staples | +1.921% | 6/6 | +0.123 pp |
| Financials | −3.066% | 0/6 | −0.417 pp |
| Information Technology | −3.961% | 0/6 | −0.847 pp |

These illustrative rows retain adverse evidence. [Every sector and industry group in every quarter](evidence/subgroups/derived/group_quarters.csv) and [all period summaries](evidence/subgroups/derived/group_summary.csv) include all 11 sectors and 25 industry groups in all four runs. Revised late Health Care and Materials gains are smaller, +1.095% and +0.934%. Neither a favorable group mean nor a count of benefiting firms measures ranking accuracy. Calibration/centered-error columns are an accounting identity, not an adjustment available to a future forecast.

## Companies, missing coverage and weighting

Among 494 firms with late coverage, 260 have lower original mean loss and 242 lower revised mean loss. Original company loss reductions include MU +24.190% and CVNA +22.646%, alongside SMCI −22.168% and ORCL −12.480%. These are retrospective examples, not a screened recommendation list. [Company summaries](evidence/subgroups/derived/company_summary.csv.gz) retain every fixed-universe issuer, all four runs and all periods; [late company-quarter rows](evidence/subgroups/derived/late_company_quarters.csv.gz) retain all six quarters, including absent observations and adverse errors.

Each quarter weights its matched companies equally; period group/headline summaries weight quarters equally. If b_q and e_q are quarterly mean losses, the headline is `mean_q[100*(b_q-e_q)/b_q]`. The separately labeled period-loss ratio is `100*(mean_q b_q-mean_q e_q)/mean_q b_q`. Those two percentages can differ in sign. Company `ratio_loss_gain_pct` is the ratio of that firm's available-period mean losses; single-company quarterly percentages are not averaged, because a nearly zero benchmark error can make them unstable.

For group g, the additive headline contribution is `mean_q[100*sum_i_in_g(loss_b-loss_e)/(n_q*b_q)]`. All sectors sum to the population headline; all industry groups separately do too. A group's own percentage does not add. Semiconductor and Information Technology contributions overlap and cannot be added together. The scorer independently checks group and company contribution reconciliation. Missing observations contribute no term to this accounting; they do not receive imputed zero errors in company means.

The [fixed universe](evidence/subgroups/universe.csv) contains 500 issuers. [Coverage flags](evidence/subgroups/derived/coverage.csv.gz) retain all 500×26 possibilities per run. Late cross-sections contain 489–494 observed firms, leaving 55 of 3,000 possible rows missing per run: 489 companies have six observations; PPL has one, RDDT two, SOLV two, VLTO four and GEV two. SW, SNDK, PSKY, Q, FDXF and HONA have none. Their company losses are undefined, not zero. [Quarter coverage](evidence/subgroups/derived/coverage_quarters.csv) and group missing counts make these exclusions visible. Saved eligibility rules and the universe were not reselected.

## Conditional uncertainty is limited

The [uncertainty table](evidence/subgroups/derived/group_uncertainty.csv) uses six late **quarterly percentage gains**. The code enumerates all **64** whole-quarter sign vectors for exact two-sided sign flips. It also enumerates all **36** ordered pairs of circular four-quarter block starts, concatenates two blocks, truncates to six observations and reports linearly interpolated 2.5th/97.5th percentiles. The adjacent two-quarter sensitivity flips three two-quarter means, giving eight sign vectors.

Original/revised sector-controlled semiconductor circular intervals are approximately [6.53%, 16.57%] and [1.72%, 8.60%]. Their raw sign-flip p-values are 0.0625 and 0.125; both become 1 after Holm adjustment across 25 industry groups. The three original sectors improving in 6/6 quarters have raw p=0.03125 and Holm p=0.34375 across 11 sectors. Adjustments are separate for each run and family. No supported sector/industry result passes 5% after its family adjustment; this does not control every inspected choice across studies and periods.

Sign flips assume independent, symmetric quarter effects; adjacent-block flips need independence/symmetry between blocks. Neither is valid under arbitrary serial dependence. Circular wraparound and only six shared market episodes can produce misleadingly tight intervals. All calculations condition on fixed saved forecasts and propagate no fitting uncertainty. Enumeration removes simulation noise, not these limitations. Non-rejection is not proof of no effect, and favorable conditional intervals do not establish new-period reliability. The periods were already inspected; the experiments share outcomes and are not independent confirmations.

## Reproduce from the compact public input

From the package root, with numpy and pandas installed:

```sh
python analysis/subgroup_diagnostics.py --output /tmp/quiet-index-subgroup-check
```

Use a fresh/empty output directory. `--package-root PATH` supports running the script from elsewhere. The scorer reads [paired forecasts](evidence/subgroups/paired_forecasts.csv.gz), the 500-firm universe and [input provenance](evidence/subgroups/input_provenance.json). It needs no private Git repository, vendor data or network access. It verifies input/reference hashes, unique pairs, finite values, labels, chronology, counts and shared targets; recomputes losses, summaries, coverage, contributions and uncertainty; and compares only the four covered runs against the package's existing aggregate/group/uncertainty evidence. The saved [verification](evidence/subgroups/derived/verification.json) records versions, hashes, counts and numerical discrepancies. Ranking-specific columns in existing evidence are outside this scorer.

For maintainers with canonical snapshots, `analysis/prepare_subgroup_inputs.py --original ORIGINAL_ROOT --revised REVISED_ROOT --supplement REVIEW_ROOT --git-dir HISTORY_GIT --output EXPORT_DIR` regenerates the two compact CSV inputs and their provenance. It verifies every consumed file against canonical Git bytes: original `14c5ba1333c63c6e12c116d468d60434b6aaedcf`, revised `4fa7d9f07a427931fd86854c780a589d8499aca0`, and supplement `a7cd4db5c39d18b30a2efd707a265d3f23b54615`. Original predictions are already paired and join formation/CIK/ticker labels one-to-one; revised B/B+S rows pair one-to-one on run/sample/formation/outcome-quarter/issuer and cross-check targets, weights, dates, blocks and formation classifications. This establishes saved-artifact provenance and numerical reproducibility, not independent reconstruction of the underlying models.


[Overview](README.md) · [Findings](FINDINGS.md) · [Python forecast walkthrough](FORECAST_WALKTHROUGH.md) · [Reproduction](REPRODUCE.md)
