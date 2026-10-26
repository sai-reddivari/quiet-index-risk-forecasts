# What stronger benchmarks changed about semiconductor risk forecasts

**Sai Reddivari · Independent research informed by CQF studies · Research draft**

The strongest result is a change in how the question should be asked. Before attributing forecast improvements to a thematic exposure, test whether the benchmark adequately represents risk information already available at formation.

In these historical comparisons, recent risk materially improved the company benchmark. Liquidity added a smaller, specification-dependent increment. Semiconductor characteristics did not establish stable broad forecasting value beyond the strengthened benchmark, although meaningful subgroup benefits appeared under some controls. Portfolio results raised a different issue: how much historical covariance a restricted model retains.

## From exploration to a more demanding question

The original [S&P 100 quiet-index](https://github.com/sai-reddivari/quiet-index) explored risk concentration, volatility contributions and sensitivity to covariance estimators. It motivated a more deliberate investigation using a fixed September 2026 population of 503 securities representing 500 companies.

The first expanded study measured semiconductor exposure after conventional equity factors and tested two distinct outcomes: subsequent company residual volatility and portfolio variance. An intermediate study examined risk calibration, activity, rate timing and covariance restrictions. The revised company study then tested semiconductor characteristics after strengthening the benchmark with recent risk and liquidity. A later diagnostic asked whether broad averages concealed useful subgroup results.

These stages refine one research argument. They reuse securities and market episodes and therefore do not constitute repeated independent confirmation.

The economic motivation was broader: whether apparently different investments share dependence on the AI buildout. Conditional semiconductor co-movement was a measurable starting proxy. Testing it against a stronger equity-risk benchmark does not settle questions about customer concentration, data-center financing or debt collateral. Future work on those mechanisms needs its own outcome and data design; it is not an unreported result of this study.

The [worked Python example](FORECAST_WALKTHROUGH.md) connects formation measurements, completed training outcomes, preprocessing, model coefficients, forecasts and paired losses. The [subgroup methods and runnable diagnostic](SUBGROUPS.md) support the qualifications below.

## 1. Strengthening the company benchmark changed the comparison

The revised conventional benchmark, **C**, contains factor loadings, size, own momentum, long-window residual volatility, market/downside terms and sector controls. **R** adds 60-session log residual volatility. **L** contains 252-session Amihud illiquidity and 60-session dollar volume. The fuller benchmark is **B = C + R + L**.

The target is next-quarter log annualized residual volatility, with conventional factor coefficients estimated at formation and held fixed. It can include subsequent exposure changes and omitted common influences; it is not a pure measure of structural business risk.

| Company-benchmark comparison | Mean quarterly MSE improvement, all 26 quarters | Conditional 95% interval |
|---|---:|---:|
| Add recent risk: C → C+R | **10.057%** | **[7.191%, 13.449%]** |
| Add liquidity before recent risk: C → C+L | **4.504%** | **[2.224%, 6.871%]** |
| Add liquidity after recent risk: C+R → B | **1.148%** | **[0.298%, 1.924%]** |

Source: [company inference](evidence/company_inference.csv), `calibration_main`, `block_label=all`, 2020 Q1–2026 Q2.

These are averages of quarterly percentage reductions in **unsmeared log-volatility MSE**. They are neither QLIKE improvements nor reductions in companies' actual risk. Each comparison uses its own baseline denominator; the percentages should not be added.

The recent-risk addition changes both measurement horizon and functional form: C retains raw long-window volatility, while R enters in logs. It is therefore too strong to label all 10.057% a pure recency effect. The intermediate study separately found benefits from recent information after a log long-risk term was already present, but its different training, controls and score do not identify a component of the revised gain.

The remaining liquidity improvement also needs qualification. Its 1.148% comprises approximately **0.927 percentage points** of squared-mean-error calibration improvement and **0.220 points** of centered-error improvement; rounding accounts for the displayed difference. With both liquidity horizons set to 252 sessions, the gain is **0.716%**; with both set to 60 sessions, it is **−0.366%**. Thus the result supports conditional usefulness, not a universal liquidity advantage.

## 2. Semiconductor exposure did not supply a stable broad increment

Signed SMH exposure measures the direction and strength of historical conditional co-movement. A measured exposure need not improve a subsequent risk forecast.

| Signed semiconductor addition | 2023–24 MSE improvement | 2025–2026 Q2 MSE improvement |
|---|---:|---:|
| Initial factor benchmark, with sector controls | **+1.043%** | **−0.363%** |
| Benchmark B, including recent risk and liquidity | **+0.406%** | **−0.461%** |

Positive means lower MSE. Source: [aggregate subgroup-study reconciliation](evidence/aggregate_comparisons.csv), primary original and revised comparisons; revised values also appear in [company inference](evidence/company_inference.csv), `semiconductor_main`, B → B+S.

The main revised variance-composition and downside-asymmetry additions were also small or adverse. Some shorter-window sensitivities were favorable, so the conclusion is **insufficient evidence of stable broad incremental benefit**, not absence of semiconductor dependence or proof that such information can never help.

Nor do the revised results establish that recent risk causally “explained away” the original semiconductor relationship. The experiments differ in their fitted benchmarks. Matching observations and outcomes does not turn cross-study comparisons into a causal feature attribution.

### Broad averages conceal real, benchmark-dependent subgroup gains

For the 19 semiconductor companies, all 114 company-quarter observations are available in the six-quarter late block.

| Semiconductor-group comparison | Late mean quarterly MSE improvement | Favorable quarters |
|---|---:|---:|
| Initial factor benchmark, sector controls | **+11.548%** | 5/6 |
| Benchmark with recent risk and liquidity, sector controls | **+5.160%** | 5/6 |
| Initial factor benchmark, finer industry controls | **−1.644%** | 2/6 |
| Benchmark with recent risk and liquidity, finer industry controls | **−0.704%** | 3/6 |

Source: [complete group comparisons](evidence/group_comparisons.csv), 504-session formation window, Semiconductors & Semiconductor Equipment, late block.

These favorable sector-controlled results deserve to be retained. The finer controls ask whether continuous exposure adds information beyond industry membership; they change both refitted models and do not isolate a causal membership channel. The positive sector-controlled gains also mainly concern calibration, rather than demonstrated improvement in company ordering.

The subgroup diagnostic inspected many companies and groups on already-used periods. Under an independent/symmetric-quarter sign-flip assumption, no primary sector or industry group passed the corresponding Holm-adjusted 5% diagnostic. With only six late quarters, its resolution is coarse; non-rejection does not establish zero subgroup effects.

## 3. Portfolio results depend on covariance assumptions

The portfolio analysis does not consume the company-risk forecasts. It evaluates historical covariance estimates against next-quarter realized portfolio variance using **QLIKE**, a different loss.

For the revised portfolio application, the late-period SMH increments before Treasury controls are:

| Residual covariance treatment | Mean QLIKE improvement | Conditional 95% interval |
|---|---:|---:|
| Industry blocks | **−0.016927** | **[−0.045083, +0.011228]** |
| Diagonal | **+0.004553** | **[−0.002248, +0.011354]** |

Source: [portfolio contrasts](evidence/portfolio_contrasts.csv), `outcome_block=confirmation`, `contrast=smh_without_treasury`. These are **absolute baseline-minus-extension loss differences**, not percentages. Both intervals include zero. Adding Treasury controls changes the point estimates only slightly, to −0.016891 and +0.004522 respectively.

The covariance accounting explains what is being tested. Under matched dates, intercept OLS and consistent covariance denominators, adding a factor reallocates fitted and residual variation. With full within-industry residual blocks retained, total individual variances and within-industry covariances remain unchanged; the model changes cross-industry entries. With unrestricted residual covariance, the reconstruction equals historical sample covariance.

Consequently, the portfolio result tests a way of carrying estimated historical dependence forward. It does not establish improved individual variance forecasts. Exact covariance identities can hold while subsequent forecasts disappoint.

## What the research supports

The practical lesson is to define the forecast objective, strengthen the benchmark and retain counterevidence before interpreting a thematic characteristic. Lower prediction loss, better risk ordering and better portfolio covariance forecasts are different achievements.

The evidence is limited by a fixed contemporary universe and classifications applied retrospectively, historical data vintages, overlapping measurement windows, reused evaluation periods and only six late market episodes. Uncertainty conditions on estimated features and fitted forecasts; it does not rerun the entire pipeline. Full independent replication and a new untouched evaluation have not been established.

The project stands as independent risk research informed by my CQF studies, with no CQF assessment or endorsement. It supports a bounded historical conclusion and a transparent basis for further work.

[Supporting notebooks](notebooks/README.md) · [Methods and evidence map](METHODS.md) · [Return to overview](README.md)
