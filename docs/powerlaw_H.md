# The skew power law does not measure H

*Research note. Reproduce with `julia --project=. scripts/05_identify_H.jl`
(experiment A runs from committed results in about 30 s); the tests in
`test/test_study.jl` cover every claim marked †.*

## Question

The standard model-free evidence for rough volatility is the ATM skew term
structure. Under rough volatility, |∂σ/∂k| ∝ T^{H−½} as T → 0, so the
slope of log|skew| against log T, plus ½, is read as the Hurst exponent. On
SPX this gives H ≈ 0.1, the figure usually quoted. Our own 2026-07-02 chain
gives the same: **H_eff = 0.107** over 50 expiries
(`results/skew_powerlaw.csv`).

The estimator is exact only in the limit T → 0. Options are quoted at 1 day
to several years, and a model with mean reversion bends the skew term
structure at maturities around 1/κ. The question is therefore: if the
market *were* rough Heston with a known H, what would this estimator report
on the market's own expiry grid?

## Method

`powerlaw_bias_map` (`src/study.jl`) fixes (κ, θ, ξ, ρ, v0) and, for each
true H ∈ {0.05, …, 0.5}, does three things:

1. It prices the rough Heston ATM skew on the 50 expiries of the SPX chain.
2. It measures each skew exactly as `market_atm_skew` measures the market's:
   the least-squares slope of implied vol on log-moneyness over
   |k| ≤ max(0.02, 1.5·σ_atm·√T).
3. It fits the power law over the same four maturity windows used for the
   market.

The result is the effective H, **H_eff(H)**. `invert_powerlaw_H` reads the
market's H_eff back through that map.

The map is computed for two parameter sets:

- **SPX fit:** the rough Heston calibration to this chain
  (`results/rough_fit.csv`). Its parameters are κ = 5.93, θ = 0.052,
  ξ = 1.67, ρ = −0.69 and v0 = 0.0096.
- **Reference:** a moderate set with κ = 0.3, θ = 0.02, ξ = 0.3, ρ = −0.7 and
  v0 = 0.02, in the range used in the rough Heston literature.

## Results

### 1. Even on clean data with a known H, the estimator misses badly †

The test is a synthetic CBOE snapshot priced by rough Heston with **H = 0.10**
(κ = 1.5, θ = 0.04, ξ = 0.4, ρ = −0.7, v0 = 0.03). It has 7 expiries from
1 week to 1 year, 41 strikes each, and a 0.2% bid-ask spread.

| estimator | H |
|---|---|
| skew power law (T ≤ 0.25) | **−0.006** |
| rough Heston calibration (`identification_study`) | **0.0998 ± 0.0020** (fit RMSE 2.1 bp) |

Calibration recovers H to within its standard error. On the same quotes,
the power law reports a value that is not even rough.

### 2. The bias depends on the other parameters (`results/powerlaw_bias.csv`)

**Reference parameters.** Here H_eff is monotone in H but biased low by
0.05–0.15:

| true H | full | front T ≤ 0.16 | belly 0.03–1.04 | back T ≥ 0.5 |
|---|---|---|---|---|
| 0.05 | 0.001 | 0.014 | 0.001 | −0.019 |
| 0.10 | 0.017 | 0.031 | 0.018 | −0.006 |
| 0.20 | 0.070 | 0.106 | 0.070 | 0.022 |
| 0.30 | 0.149 | 0.222 | 0.150 | 0.055 |
| 0.50 | 0.365 | 0.483 | 0.377 | 0.141 |
| *SPX market* | *0.107* | *0.165* | *0.097* | *0.022* |
| **implied true H** | **0.247** | **0.251** | **0.233** | **0.199** |

Read through rough Heston with these parameters, the market's "H ≈ 0.1"
corresponds to a true **H ≈ 0.25**, and the full, front and belly windows
agree to within 0.02. The figure usually quoted as the roughness of SPX
volatility is, under this model, the estimator's bias acting on a much
smoother process.

**SPX-fit parameters.** Here H_eff is **flat in H**:

| true H | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|
| H_eff (full) | −0.040 | −0.040 | −0.040 | −0.037 | −0.027 | −0.008 |

With κ = 5.9, mean reversion takes over beyond T ≈ 0.17. Because
ξ/√v0 ≈ 17, the short end sits far outside the small-vol-of-vol regime where
the T^{H−½} asymptotic applies. The skew term structure carries almost no
information about H, which matches what calibration found: H drifts to its
bound and rough ties classical (`docs/rough_heston.md`).

No H reproduces the market's slope at these parameters. Every row gives
H_eff ≤ −0.008, against 0.107 for the market. So the structural misfit is in
**κ, not H**: a single mean-reversion speed fast enough to fit the level of
the surface makes the skew decay too quickly across maturities.

### 3. The estimator is consistent only in the short, narrow-window limit †

Restricting to T ≤ 0.008 and a window of 0.1·σ√T recovers H = 0.3
(H_eff = 0.302). For H = 0.1 it still gives only 0.078, because the
asymptotic regime shrinks as H falls. Quoted maturities and the usual
1.5·σ√T window (with a 2% floor) are far from this limit.

## What follows

1. **A skew-slope H is not a model parameter.** Before it is compared with,
   or used to set, the H of a pricing model, it should go through that
   model's H_eff map at the calibrated parameters. The shift is large
   (0.1 → 0.25 above) and depends on the parameters. When the map is flat,
   as at the SPX fit, the slope says nothing about H at all.
2. **Calibration identifies H when the model is right.** On the synthetic
   chain the price-based fit pins H to ±0.002. Calibration's failure on SPX
   is the model's failure, not the method's.
3. **What the SPX skew rejects is one-factor mean reversion.** A second
   variance factor, or a kernel with slower long-memory decay, is the
   natural next model. Lifted Heston (`src/lifted_heston.jl`) is a
   multi-factor form whose factors can be decoupled to test this.

## Relation to the literature and limits

- **Prior work.** Guyon and El Amrani (2023, *Risk*, "Does the term
  structure of the at-the-money skew really follow a power law?") show that
  the SPX ATM skew is not a power law at short maturities, and that
  power-law fits mostly reflect the maturity range chosen. This note is
  consistent with that.
- **What this note adds.** It gives a model-side quantification: the
  H_eff(H) map inside rough Heston itself; its dependence on κ and ξ/√v0;
  and a synthetic case where calibration recovers H and the power law does
  not.
- **Scope.** The SPX numbers come from one weekend snapshot.
  `scripts/05_identify_H.jl` (experiment B) runs `identification_study`
  under calendar and business time, with and without the skew-slope loss,
  for every CBOE file placed in `data/snapshots/`. The maps cover two
  parameter sets, not a calibrated distribution of them.
