# Calibration and covariance: scientific methods

The calibration and covariance reading notebooks preserve source calls into `quiet_index_500.expansions`; the numerical implementations are supplied here as text for inspection. The `.py.txt` extension keeps these references outside the importable package. All material comes from source commit `eab47102acb38d483271752934981277f5c90bec`.

| Reference | What to inspect |
|---|---|
| [forecasts.py](forecasts.py.txt) | Training-only scaling, quarter-weighted chronological fits, training smearing, matched losses and calibration recipes; complete source module |
| [covariance.py](covariance.py.txt) | Common-mask four-model covariance estimates, industry residual blocks, frozen exposures and future accounting; complete source module |
| [common.py scientific functions](common.py.txt) | OLS, sample variance, stable QLIKE, whole-quarter circular intervals and period summaries; verbatim selected functions |
| [measurements.py](measurements.py.txt) | Formation-time liquidity/activity definitions, partial exposure fits and fixed-coefficient future targets; complete source module |
| [rates.py](rates.py.txt) | Treasury observations and adjacent equity-session yield changes without filling missing quotes; complete source module |
| [settings.py](settings.py.txt) | Fixed feature recipes, study variants and links to inherited conventions; complete source module |

`common.py` excludes data loading, directory creation, evidence writing and figure/provenance helpers. The complete scientific modules retain their source imports and `run` functions so the calculations can be followed in context. These text references are for inspection. The separate executable source tree and runner are described in the [reproduction guide](../../REPRODUCE.md).

The [executable calibration and covariance source](../../reproduction/calibration-covariance/src/quiet_index_500/expansions/forecasts.py) includes the scientific foundation loader, calendar, manifest helpers, root settings and research I/O. Frozen prepared panels and Treasury source observations are bundled in that isolated study tree. The authorized private input restore supplies raw vendor prices and capitalization where selected computations require them. Complete-module `run` functions fit models and write artifacts in disposable run directories. Provider credentials and private process records are excluded. See the reproduction guide for concrete commands and the tested scope.

The fixed source conventions use 504/400 main and 252/200 sensitivity windows/minimum observations, 252-session annualization, training-only transformations, and whole-quarter conditional uncertainty. The source settings retain inherited constants rather than inventing replacement values. Reading the code does not resolve data access or establish independent replication.

Return to the [selected notebook index](../../notebooks/README.md), [calibration/activity notebook](../../notebooks/forecast-calibration-and-trading-activity.ipynb), or [rates/covariance notebook](../../notebooks/interest-rates-and-covariance.ipynb).
