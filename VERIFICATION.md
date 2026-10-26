# Package verification

## Documentation clarification — 26 October 2026

The research/development attribution was updated from the author's account, including his private-branch rewrites of two named functions. The [prepared-feature definitions](METHODS.md#prepared-recent-risk-and-liquidity-inputs) were checked against the canonical revised measurement notebook, including its final corrections. Its bytes match scientific revision `4fa7d9f07a427931fd86854c780a589d8499aca0`, and its concatenated code-cell SHA-256 matches the shipped measurement manifest (`3957d545be42bb2326d8145978316da1f82847c9054e9b6d02e4a65dcabeb374`). The public source-subset boundary is now explicit.

These edits change documentation only. All scientific files, Python scripts, notebooks and evidence tables retain their pre-edit bytes; local reading links and whitespace were checked. No calculation, model fit or data collection was repeated for this clarification. The earlier numerical verification below retains its original scope.

## Publication walkthrough and subgroup support — 26 October 2026

The new [Python walkthrough](FORECAST_WALKTHROUGH.md) reconstructed **978 saved predictions** (B and B+S for 489 companies at the first late formation), with maximum absolute difference **2.054 × 10⁻¹⁴**. Training cutoffs, eligibility, preprocessing, training-only retransformation and both quarterly MSEs passed. Saved numeric coefficients were held fixed; the combined sector offsets were recovered from training-only OLS normal equations. No new regression was fitted.

The [subgroup route](SUBGROUPS.md) rescored **50,044 matched pairs** across four original/revised 504-session comparisons and all 26 quarters. It retains **52,000 coverage rows**, including missing company-quarters. Sector and company contributions reconcile to the whole population. All nine regenerated data outputs are byte-identical to the shipped outputs. Reference checks cover 16 aggregate rows, 576 group rows and 144 uncertainty rows; their maximum absolute numerical discrepancies are respectively **3.842 × 10⁻¹⁴**, **4.015 × 10⁻¹²** and **1.348 × 10⁻¹²**. An independent review also reproduced the compact inputs directly from canonical source bytes and checked the narrative's adverse evidence and uncertainty limits.

All **269 scientific source/input files** retain their recorded hashes. The **seven reading notebooks**, including their **64 code cells and saved outputs**, are byte-identical to the preserved pre-review package. Reader-facing local links and Git whitespace checks passed. [Machine-readable publication review](evidence/publication_review.json) records the scope and fingerprints.

These are focused checks of the additions. The eight notebook routes and 94-table comparison below were verified on 20 October; they were not rerun for this prose and saved-forecast addition. No market data were collected. Numerical agreement does not create new holdout evidence, reliable subgroup winners or credit-risk findings. The separate extension feasibility assessment remains a private planning document; no extension was implemented.

## Topic-based layout verification — 20 October 2026

After renaming the source directories and reading notebooks by research topic, all **eight supported notebook executions** completed in fresh output directories: factor-risk 08–09, calibration-covariance 06/08, company-risk 03–05 and the portfolio/Treasury calculation 06. All **94 regenerated CSV payloads** were byte-identical to the previously verified runs, including decompressed prediction tables; the maximum numerical difference was **0**. This preserves the earlier agreement with the pinned scientific sources documented below, including the small calibration/covariance differences from those sources.

All **269 scientific source/input files** retained their recorded hashes, and all **64 code cells** in the seven reading notebooks retained their code, metadata, execution counts and saved outputs. Reader-facing links and renamed paths passed the navigation audit. [Machine-readable layout verification](evidence/navigation_verification.json) records the execution routes and per-table comparisons. [Source provenance](reproduction/provenance.json) maps the topic names to the historical study identities used in the earlier verification record below.

Only package organization, navigation and explanatory text changed. These reruns do not extend the historical evaluation or the raw-to-measurement reproduction boundary.

## Scientific verification record

Verification date: **19 October 2026 (America/Chicago)**. UTC audit records fall on 20 October. The execution environment used **Python 3.14.8** and the dependency versions recorded in [requirements-lock.txt](requirements-lock.txt). [Machine-readable results](evidence/package_verification.json) record the checks and exact numerical scope.

## Shipped files and runner safeguards

All **269 shipped scientific files** match their recorded SHA-256 hashes: 95 original, 85 intermediate and 89 revised files. The runner's prepared-company check and nine-file original frozen-overlay check passed. Requests to execute an acquisition notebook or reuse an existing output directory were refused.

These integrity checks establish the identity of packaged files. They do not validate the underlying estimates.

## Factor-based company and portfolio forecasts

The original forward-test and covariance builders completed, followed by notebooks **08–09**, using the nine checksum-verified private equity/capitalization inputs and shipped upstream tables. All **26 comparison CSVs**, totaling **678,094 data rows**, are byte-identical to source revision `14c5ba1333c63c6e12c116d468d60434b6aaedcf`. This covers the original forward-test and covariance output directories, including both formation windows. Maximum numerical difference is **0**.

## Forecast calibration and covariance diagnostics

Intermediate notebooks **06 and 08** completed using the private frozen equity/capitalization overlay, shipped prepared upstream inputs and frozen FRED source observations. All **46 output CSVs**, totaling **404,468 data rows**, have the same columns, row counts and categorical values as source revision `eab47102acb38d483271752934981277f5c90bec`. Ten tables are byte-identical. Across the remaining tables, the maximum absolute numerical difference is **1.000000082740371×10⁻¹¹**; no text fields differ. The machine-readable record gives each table's numerical difference and output hash.

These small floating-point differences do not change the reported rounded results. Exact-byte reproduction is therefore established for the original outputs and revised CSV payloads described here, while the intermediate outputs agree to the reported numerical precision.

## Company forecasts with recent risk and liquidity

Revised notebooks **03–05** completed in an isolated study copy from the shipped prepared measurements, outcomes and comparison specification, without private vendor inputs. All **15 regenerated scientific artifacts** match source revision `4fa7d9f07a427931fd86854c780a589d8499aca0`: 12 plain CSVs are byte-identical, and the three compressed prediction CSVs have byte-identical decompressed contents. Gzip container bytes differ; the scientific CSV payloads do not. Maximum numerical difference is **0**.

The checks cover calibration, semiconductor representation and downside asymmetry, including fitted preprocessing, predictions, quarterly scores, paired effects and all 136 inference rows. Prediction tables contain 200,176 calibration, 225,498 semiconductor and 75,216 asymmetry rows. Against the separately consolidated [public inference table](evidence/company_inference.csv), all 136 comparison keys match; the maximum difference across 1,768 numeric cells is **1.7763568394002505×10⁻¹⁵**, from floating-point serialization. No text fields differ. The headline mean quarterly percentage MSE gains remain **10.057%** for C → C+R and **1.148%** for C+R → B. Revised signed semiconductor exposure remains **+0.406%** in validation and **−0.461%** late.

This run verifies model fitting and scoring from frozen prepared inputs. It does not reconstruct the company measurements from raw prices or establish a new evaluation sample.

## Portfolio risk with Treasury controls

Revised notebook **06, Portfolio Treasury Application**, completed in a new isolated study copy using ten checksum-verified private frozen inputs. Every regenerated portfolio CSV was **byte-identical** to the saved table at source revision `4fa7d9f07a427931fd86854c780a589d8499aca0`:

| Table | Rows | Maximum absolute numerical difference |
|---|---:|---:|
| `contrast_summary.csv` | 24 | 0 |
| `covariance_accounting.csv` | 186 | 0 |
| `date_ledger.csv` | 19,270 | 0; categorical date/sample records |
| `forecasts.csv` | 240 | 0 |
| `original_bridge.csv` | 132 | 0 |
| `quarterly_contrasts.csv` | 60 | 0 |
| `samples.csv` | 34 | 0 |

The regenerated contrast table also matches [the public portfolio evidence](evidence/portfolio_contrasts.csv) byte for byte: all 24 comparison keys and 432 numeric cells agree, with maximum absolute difference **0**. The completion record agrees with the pinned source, including 3,170 Treasury observations, 2,880 valid equity-session changes, 132 original bridge predictions and maximum bridge discrepancy **8.6014528832834×10⁻¹⁴**, below the calculation's **10⁻¹⁰** tolerance.

The late-period mean QLIKE improvement before Treasury remains **−0.01692731587683094** for industry-block residual covariance and **+0.004553115262359612** for diagonal residual covariance. Their circular 95% intervals remain **[−0.04508306028950049, +0.011228428535838608]** and **[−0.0022479841612602267, +0.01135421468597945]**. Both include zero. These are absolute loss improvements, not percentages.

## Scope of the result

These runs verify the selected original, intermediate and revised calculations against frozen historical evidence. They do not establish a full raw-to-measurement rebuild of all three studies: some routes consume frozen upstream panels or measurements. Exact or close numerical agreement does not create independent empirical replication, new holdout periods, economic causation or evidence of investment returns. The uncertainty remains conditional on fitted features/exposures and the reused historical quarters.
