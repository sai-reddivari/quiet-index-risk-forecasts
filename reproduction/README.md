# Executable research sources

The directories are named for the questions they support. Each contains selected executable comparison sources, required prepared inputs and saved results. These are subsets of the pinned studies, not complete upstream source snapshots. In particular, the revised `02_formation_measurements.ipynb` is identified in the measurement manifests but is not bundled; public company reproduction begins with its saved prepared outputs. Internal source filenames and table identifiers are retained so the calculations remain traceable.

| Directory | Purpose | Supported notebook selectors |
|---|---|---|
| [factor-risk](factor-risk/) | Initial company-risk forecasts and portfolio covariance comparisons | 08: company forecasts; 09: portfolio covariance |
| [calibration-covariance](calibration-covariance/) | Forecast calibration, trading activity, interest rates and covariance restrictions | 06: calibration/activity; 08: rates/covariance |
| [company-risk](company-risk/) | Recent risk and liquidity benchmark; semiconductor and downside additions; portfolio Treasury controls | 03: recent risk/liquidity; 04: semiconductor exposure; 05: downside asymmetry; 06: portfolio Treasury controls |

Use the [reproduction guide](../REPRODUCE.md) to run these through `reproduce.py` in a new output directory. Source notebook numbers are execution selectors, not a single reading sequence. For explanation with saved outputs, use the [topic-based notebook index](../notebooks/README.md).

[Provenance](provenance.json) maps these names to the historical studies and pinned source revisions. The [manifest](manifest.json) records every scientific file's unchanged hash and its source-study identity. Historical run IDs and filenames inside the snapshots remain unchanged.
