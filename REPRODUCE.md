# Reproduce the comparisons

The package includes executable study sources, frozen prepared measurements and saved results. The runner verifies shipped file hashes, copies one study to a **new output directory**, and executes selected notebooks there. It preserves the distributed files. Execution uses the Python interpreter that launches the runner, a temporary kernel specification, one BLAS thread and a noninteractive plotting backend.

See the [source directory map](reproduction/README.md) for the research questions and the [provenance mapping](reproduction/provenance.json) for historical study identities.

## Environment

Use Python 3.11 or newer. From this package directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

The lockfile records the installed dependency versions in the preparation environment. Platform and numerical-library differences can still affect byte-for-byte results. The runner supplies the selected copy's `src` through `PYTHONPATH`; no editable package installation is needed. Each source's `pyproject.toml` also records its dependency declarations.

## Company comparisons from prepared measurements

The company-risk notebooks consume frozen formation measurements, subsequent outcomes and the comparison specification. These inputs allow model fitting, scoring and uncertainty calculations to be repeated without vendor credentials or raw price files. This starts at the **prepared-measurement stage**: it does not independently reconstruct those measurements from market data. The revised upstream generator, `02_formation_measurements.ipynb`, is identified in the shipped measurement manifests but is **not included** in the selected public source subset. The [prepared-feature definitions](METHODS.md#prepared-recent-risk-and-liquidity-inputs) explain its principal calculations and final eligibility rules.

```sh
.venv/bin/python reproduce.py --study company-risk --notebooks 03 04 05 --check
.venv/bin/python reproduce.py --study company-risk --notebooks 03 04 05 --output /absolute/path/to/new-company-run
```

Notebook 03 executes the liquidity/recent-risk comparisons; 04 executes the semiconductor representation comparisons; 05 executes the downside-asymmetry comparisons. Other notebook selectors work only when that exact notebook is shipped. The runner executes selectors in the supplied order. Outputs and executed notebooks remain under the new directory; `reproduction-run.json` records completion or failure. A successful execution alone does not establish exact numerical agreement: compare its scientific tables with the shipped results and their source definitions.

## Inspect saved forecasts without refitting

Two smaller routes need only the shipped inputs and the same environment:

```sh
python analysis/forecast_walkthrough.py --output runs/forecast-walkthrough
python analysis/subgroup_diagnostics.py --output runs/subgroup-diagnostics
```

Use a new output directory for each run. The [company walkthrough](FORECAST_WALKTHROUGH.md) applies saved numeric slopes and reconstructs omitted sector offsets from training-only OLS normal equations; it verifies all 978 predictions and both quarterly MSEs at the displayed formation. It does not refit slopes or reconstruct raw-price measurements. Its [input provenance](evidence/walkthrough/provenance.json) records the source-table selections and checksums.

The [subgroup route](SUBGROUPS.md) rescores and aggregates 50,044 saved paired forecasts across four 504-session comparisons. It checks company identities, targets, coverage, contributions and conditional group uncertainty against the existing evidence tables. It writes company/quarter/group tables and a verification record. It does not reproduce the additional 252-session runs or ranking-specific columns retained in the broader evidence tables. Neither route creates new forecast evidence.

## Private frozen inputs and portfolio calculations

The factor-risk and calibration-covariance notebooks, plus the company-risk portfolio notebook, require the frozen equity/capitalization overlay. The company-risk portfolio notebook also requires the frozen FMP Treasury JSON. Those raw vendor inputs remain outside the public package. Existing project policy permits the included derived research tables; source manifests identify the raw inputs but do not grant redistribution rights.

Readers with authorized access can obtain the existing companion repository, pinned to the documented input revision:

```sh
git clone https://github.com/sai-reddivari/quiet-index-500-data.git /absolute/path/to/private-inputs
git -C /absolute/path/to/private-inputs checkout --detach 85fde50477c51c9851b4a2c03be2e793a66196a0
```

The bundle must contain `manifest.json` and `frozen/`, with snapshot `equity-2026-10-12_treasury-2026-10-16`. No market-data API key is needed. The runner verifies every selected SHA-256 and byte count before copying or executing; it overlays nine files for factor-risk/calibration-covariance or ten for company-risk, and verifies the copies again. The companion commit establishes the documented bundle revision; checksums establish file identity, not scientific validity or provider equivalence.

```sh
.venv/bin/python reproduce.py --study company-risk --notebooks 06 --input-backup /absolute/path/to/private-inputs --check
.venv/bin/python reproduce.py --study company-risk --notebooks 06 --input-backup /absolute/path/to/private-inputs --output /absolute/path/to/new-portfolio-run
```

The same runner selects the factor-risk and calibration/covariance comparisons:

```sh
.venv/bin/python reproduce.py --study factor-risk --notebooks 08 09 --input-backup /absolute/path/to/private-inputs --output /absolute/path/to/new-factor-risk-run

.venv/bin/python reproduce.py --study calibration-covariance --notebooks 06 08 --input-backup /absolute/path/to/private-inputs --output /absolute/path/to/new-calibration-covariance-run
```

The calibration-covariance source additionally uses the shipped frozen DGS2/DGS10 FRED source HTML and manifests. FMP Treasury yields and FRED series are distinct recorded sources; restoring both does not prove methodological or vintage equivalence. October 13 price leftovers must not replace the October 12 frozen equity files.

The factor-risk notebooks define their calculation functions and then display saved summaries. For factor-risk 08 and 09, the runner first invokes `build_forward_tests()` or `build_covariance_evaluation()` in the disposable copy, then executes the notebook displays. These calculation steps appear separately in `reproduction-run.json`; the shipped notebooks are preserved.

## Full raw-to-measurement rebuild boundary

The selected notebook routes above mix raw-data calculations with explicitly saved upstream measurements, depending on the notebook. A full rebuild would also require upstream generators outside this selected subset, including the revised `02_formation_measurements.ipynb`, and the exact frozen inputs. The package preserves selected executable comparison sources, prepared inputs and their provenance; it does not bundle every upstream calculation needed for a complete raw-data rebuild. Do not interpret a successful selected-notebook run as a verified end-to-end rebuild of all three studies.

Acquisition notebooks are not enabled by this runner. It does not download replacement market data. An in-kernel guard blocks ordinary nonlocal Python socket acquisition; this is not a general operating-system sandbox. The guard is inserted as an additional cell only in executed copies; the shipped notebook code is preserved. Credentials, authenticated URLs, virtual environments and private development records are unnecessary inputs and must not be added to a shared run.

All runs reuse inspected historical periods and a contemporary population applied retrospectively. They do not create fresh out-of-sample confirmation, an independent replication, or an investable backtest.

The [package verification record](VERIFICATION.md) documents completed runs for factor-risk 08–09, calibration-covariance 06/08, company-risk 03–05 and its portfolio notebook 06, including their agreement with frozen evidence. It distinguishes executed checks from the full-pipeline boundary.
