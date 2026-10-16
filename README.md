# QuantJulia

[![CI](https://github.com/Yugam2508/QuantJulia/actions/workflows/ci.yml/badge.svg)](https://github.com/Yugam2508/QuantJulia/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://yugam2508.github.io/QuantJulia/dev/)

A differentiable option-pricing and volatility-calibration framework in Julia.

**Story (the destination):** a differentiable calibration engine, demonstrated on
classical Heston, extended to the rough-volatility frontier (rough Heston). The
same Fourier pricing / implied-vol / calibration machinery is written once and
reused across models — the only thing that changes between models is the
characteristic function.

> **Status: complete** (1,003 tests green on Julia 1.10 and current). Classical and rough Heston are both
> calibrated to a real SPX chain (11,858 OTM quotes, 50 expiries, 2026-07-02
> close), with a hand-rolled implicit fractional Riccati solver, AD gradients
> through everything including the Hurst exponent, and a five-gate validation
> stack anchored by exact α→1 classical recovery.
>
> **Headline results (the honest version):** the market's own ATM-skew term
> structure follows the rough power law with **H ≈ 0.107** (model-free fit,
> squarely in the literature range) — but vega-weighted calibration on this
> weekend snapshot does **not** select roughness: rough ties classical
> (134 vs 134 bps full-chain) and H runs to its bound, because front-end
> calendar-time distortions (non-monotone skews across daily expiries)
> dominate the loss and leave H unidentified. Full reading in
> [docs/rough_heston.md](docs/rough_heston.md).
>
> **Follow-up finding:** that 0.107 is not H. Inside rough Heston the skew
> power-law estimator is biased at quoted maturities — on a synthetic chain
> with true H = 0.10 it reports −0.006 while calibration recovers
> 0.0998 ± 0.002, and through the model's H_eff(H) map the SPX slope
> corresponds to H ≈ 0.25. At the SPX-calibrated parameters the map is flat
> and no H reproduces the market slope: the misfit is one-factor mean
> reversion, not roughness. See [docs/powerlaw_H.md](docs/powerlaw_H.md).

| | classical | rough |
|---|---|---|
| full-chain IV RMSE (11,858 quotes) | 133.6 bps | 134.1 bps |
| short-end IV RMSE (T ≤ 0.16) | 157.6 bps | 157.6 bps |
| calibrated H | — | 0.49 (at bound) |
| market skew power law | — | **H ≈ 0.107** (model-free) |
| calibration wall time (192 quotes) | 9 s | 68 s |

## Install

```julia
import Pkg
Pkg.add(url = "https://github.com/Yugam2508/QuantJulia")
```

Julia ≥ 1.10. See [CHANGELOG.md](CHANGELOG.md) for what changed between versions.

## Layout

```
src/
  QuantJulia.jl         # module and exports
  blackscholes.jl       # BSM price, Greeks, safeguarded-Newton implied vol
  heston.jl             # Heston CF, little-trap form
  fourier.jl            # model-agnostic Gil-Pelaez pricer (adaptive + batch)
  cos.jl                # COS pricer (Fang–Oosterlee) with a rotation recurrence
  fractional_riccati.jl # implicit fractional Adams solver
  rough_heston.jl       # El Euch–Rosenbaum rough Heston CF
  lifted_heston.jl      # Markovian multi-factor approximation of rough Heston
  jumps.jl              # Merton, Bates, Variance Gamma + generic CF calibration
  cboe.jl               # CBOE parser, filters, parity forwards, business time
  calibration.jl        # vega-weighted losses, LBFGS / Levenberg–Marquardt
  diagnostics.jl        # standard errors, correlations, condition numbers
  identification.jl     # skew-slope loss, joint multi-date rough fits
  study.jl              # H-identifiability study and power-law bias map
  greeks.jl             # model Greeks by automatic differentiation
  svi.jl                # SVI / SSVI surfaces and arbitrage checks
  localvol.jl           # Dupire local vol + local-vol Monte Carlo
  variance.jl           # variance swaps, replication, model VIX
  parallel.jl           # reproducible multithreaded Monte Carlo blocks
  montecarlo.jl         # rough Heston Monte Carlo (hybrid Volterra Euler)
  heston_mc.jl          # Heston Monte Carlo, Andersen QE
  rbergomi.jl           # rough Bergomi, hybrid scheme
  exotics.jl            # American (CRR, LSM), barriers, Asians
  marketdata.jl         # discount curves, cash dividends, forwards
scripts/                # 01–05: data prep, calibrations, the H study
test/                   # one file per source file + QuantLib cross-checks
validation/             # QuantLib 1.43 reference generators and CSVs
data/                   # market data (not committed — see data/README.md)
results/                # committed fit results
docs/                   # Documenter site, research notes, theory notes
```

## Monte Carlo is threaded and reproducible

Start Julia with `-t auto` and every simulator uses all cores (about 3.5× on
4 cores). Paths are generated in fixed blocks, each with its own random
stream derived from one seed, so `seed = 42` gives bit-identical results on 1
thread or 64:

```julia
heston_mc_price(100.0, 100.0, 0.03, 0.0, HestonParams(1.5, 0.04, 0.5, -0.7, 0.04), 1.0;
                npaths = 400_000, seed = 42)
```

## Running the tests

```julia
julia --project=.        # from the repo root
julia> ]                 # enter Pkg mode
(QuantJulia) pkg> test
```

or in one shot:

```
julia --project=. -e 'import Pkg; Pkg.test()'
```

Multithreaded: `JULIA_NUM_THREADS=4 julia --project=. -e 'import Pkg; Pkg.test()'`.
The suite checks that Monte Carlo results do not depend on the thread count.

## Reproducing everything

```
# 0. environment (Julia ≥ 1.10)
julia --project=. -e 'import Pkg; Pkg.instantiate()'

# 1. tests — 1,003 checks, ~5 min (add -t auto to use all cores)
julia --project=. -e 'import Pkg; Pkg.test()'

# 2. data — download the CBOE SPX chain per data/README.md, then:
julia --project=. scripts/01_prepare_data.jl          # parse, filter, implied vols
julia --project=. scripts/02_calibrate.jl             # classical Heston fit (~1 min)
julia --project=. scripts/03_calibrate_rough.jl       # rough vs classical duel (~2 min)
julia --project=. scripts/04_short_end_and_powerlaw.jl # short-end duel + market H (~6 min)
julia --project=. scripts/05_identify_H.jl            # power-law bias map (~30 s)
```

Outputs land in `results/*.csv` (committed, so the numbers in this README are
inspectable without rerunning). The market snapshot itself is not
redistributed; `data/README.md` documents the exact download.

## Validation

Every closed form, Greek and Fourier pricer is cross-checked against
[QuantLib](https://www.quantlib.org) 1.43 on each CI run (152 reference values;
machine-precision agreement for closed forms, ~1e-11 for Heston and Bates). See
[validation/README.md](validation/README.md).

## Documentation

API reference (built from the docstrings), the results write-up and the
per-stage theory notes are published at
[yugam2508.github.io/QuantJulia](https://yugam2508.github.io/QuantJulia/dev/).
To build locally:

```
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl        # → docs/build/index.html
```

## Design constraint carried from day one

Everything is written to stay **automatic-differentiation friendly** — functions
are generic in their numeric type so `ForwardDiff.Dual` can flow through prices
to yield Greeks and calibration gradients. This is why the rough-Heston
fractional Riccati solver is hand-rolled rather than pulled from a package.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

## Companion repo

[variance-risk-premium](https://github.com/Yugam2508/variance-risk-premium) —
this repo builds models *of* implied volatility; that one measures the premium
*inside* it: 36 years of VIX² vs realized variance, with honest accounting of
the crash months that pay for it.
