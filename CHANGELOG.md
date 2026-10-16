# Changelog

All notable changes to QuantJulia. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: a minor bump may break
the API).

## 0.2.0 — 2026-10-16

### Breaking

- **Monte Carlo streams changed.** Every simulator now runs its paths in
  parallel blocks of `MC_BLOCK` = 1024 paths. Each block has its own
  Xoshiro stream derived from one master seed, so the same seed or `rng`
  produces different paths than in 0.1. Prices agree statistically; exact
  values do not. `rng` is now used only to draw the master seed.

### Added

- **Reproducible multithreaded Monte Carlo** (`src/parallel.jl`). A new
  `seed` keyword is accepted by:
  - `simulate_heston`, `simulate_rough_heston`, `simulate_rbergomi`,
    `simulate_gbm_paths` and `simulate_heston_paths`;
  - `heston_mc_price`, `rough_heston_mc_price`, `rbergomi_mc_prices` and
    `local_vol_mc_prices`.

  A given seed gives bit-identical paths with any number of threads, and a
  shorter run is a prefix of a longer one. The speedup is about 3.1–3.6× on
  4 threads.
- **Research tools for the Hurst exponent H.** `identification_study`,
  `powerlaw_bias_map`, `invert_powerlaw_H` and `model_window_skew`, with
  `scripts/05_identify_H.jl` and the note `docs/powerlaw_H.md` (#21).
- **Market data.**
  - `FlatCurve` and `ZeroCurve` yield curves.
  - `DividendSchedule` with escrowed cash dividends.
  - `MarketData`, `forward`, `black_price`, `market_price`, and `crr_price`
    on curves (#20).
- **Validation.** Cross-validation against QuantLib 1.43 covering 152
  committed reference cases, re-checked on every CI run (#19, #20).
- **Documentation.** A user guide whose examples are executed at every docs
  build (#18), and calibration diagnostics: standard errors, correlations and
  the condition number (#17).
- **Path-dependent options.** American options (CRR and Longstaff–Schwartz),
  closed-form barriers with the BGK correction, and geometric and arithmetic
  Asians (#16).
- **Fourier pricing.** The COS pricer (#9), sped up 3–8× per smile with a
  rotation recurrence (#15).
- **Models.**
  - Merton, Bates and Variance Gamma jump models, plus generic calibration
    of any model defined by its characteristic function (#8).
  - Lifted Heston (#11).
  - Rough Bergomi by the hybrid scheme (#13).
- **Monte Carlo.** Heston by Andersen's QE scheme (#12), and a rough Heston
  Monte Carlo cross-check (#6).
- **Volatility surfaces.** SVI/SSVI surfaces (#5), and Dupire local
  volatility with a local-vol Monte Carlo (#14).
- **Variance.** Variance swaps, model-free replication and a model VIX (#10).
- **Calibration.** Batch pricing with Levenberg–Marquardt (#2), Greeks by
  automatic differentiation (#3), and tools for identifying H (#4).
- **Black–Scholes.** Gamma, theta and rho, plus a Feller-ratio helper (#1).
- **CI.** Runs on Julia 1 and the LTS release, and builds the Documenter site
  (#7).

### Changed

- `Random` is now a package dependency (it was test-only). Every stdlib
  dependency has a `[compat]` bound.
- The authors field is now `Yugam2508`.

## 0.1.0 — 2026-07-04

First release, covering stages 1–8 of the roadmap:

- Black–Scholes prices, Greeks and implied volatility.
- The Heston characteristic function in its stable ("little trap") form,
  with a Gil-Pelaez Fourier pricer whose AD gradients are tested.
- A CBOE options-chain parser and a Heston calibration to a real SPX chain.
- Rough Heston via an implicit fractional Riccati solver, compared with
  classical Heston on SPX.
