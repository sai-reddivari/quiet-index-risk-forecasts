# Supporting research notebooks

Choose a question below after reading the [overview](../README.md) or [findings](../FINDINGS.md). Each reading edition includes the methods, calculation code and saved outputs for that question.

| Research question | Reading notebook |
|---|---|
| Does semiconductor exposure improve company-risk forecasts against the initial factor benchmark? | [Company risk: factor benchmark](company-risk-factor-benchmark.ipynb) |
| How does semiconductor exposure affect portfolio covariance forecasts? | [Portfolio covariance](portfolio-covariance.ipynb) |
| Do recent information and trading activity improve calibration? | [Forecast calibration and trading activity](forecast-calibration-and-trading-activity.ipynb) |
| Do interest rates and residual covariance assumptions change the portfolio result? | [Interest rates and covariance](interest-rates-and-covariance.ipynb) |
| How much does liquidity add after recent risk is included? | [Recent risk and liquidity](recent-risk-and-liquidity.ipynb) |
| Does semiconductor exposure add value beyond the strengthened company benchmark? | [Semiconductor exposure](semiconductor-exposure.ipynb) |
| Does the portfolio result survive Treasury controls and alternative covariance assumptions? | [Portfolio risk and Treasury controls](portfolio-risk-and-treasury-controls.ipynb) |

For a single end-to-end example before exploring the notebooks, read [Follow one company forecast in Python](../FORECAST_WALKTHROUGH.md). It reconstructs a revised forecast from saved inputs and coefficients, then checks its losses. [Subgroup support](../SUBGROUPS.md) provides a separate saved-forecast diagnostic.

## Reading and running

These are reading editions: code cells, execution counts and saved outputs are preserved from their source snapshots. Explanatory text and navigation have been adapted. No research was rerun to produce these editions; separate verification runs use the executable copies.

For execution, use the [reproduction guide](../REPRODUCE.md) and [source directory map](../reproduction/README.md). The company-risk comparisons can refit models from bundled prepared measurements. Portfolio and other selected comparisons also require authorized access to the frozen private vendor inputs. The runner creates a new output directory and leaves these reading editions untouched.

Company residual-risk forecasts and portfolio covariance forecasts are separate endpoints. The calibration investigation also uses a variance loss that is not interchangeable with the main company log-volatility MSE. Read each notebook's target, sample and scoring definition before comparing results.

## Sources and limitations

The notebooks trace the development from the first factor-based forecasts through calibration diagnostics to a company benchmark containing recent risk and liquidity. These are successive parts of one historical investigation, not independent replications. The fixed contemporary population, reused evaluation periods and short late sample limit inference; saved intervals condition on fitted inputs and forecasts.

The [calibration and covariance method reference](../method-reference/calibration-covariance/README.md) supplies the numerical implementations called by those notebooks. [Notebook and image provenance](provenance.json) retains exact source paths, commits and code-cell hashes; the [name mapping](../reproduction/provenance.json) records the former package names.
