# QuantJulia

A differentiable option-pricing and volatility-calibration framework in Julia.

**Story (the destination):** a differentiable calibration engine, demonstrated on
classical Heston, extended to the rough-volatility frontier (rough Heston). The
same Fourier pricing / implied-vol / calibration machinery is written once and
reused across models — the only thing that changes between models is the
characteristic function.

> **Status: Stages 1–4 complete** (120 tests green). Classical Heston is
> calibrated to a real SPX chain (11,858 OTM quotes, 50 expiries, 2026-07-02):
> full-chain IV RMSE 130 bps, with the classic failure signature — 150–500 bps
> at short maturities vs ~30 bps in the belly — which is exactly the gap the
> rough-Heston extension (Stages 6–8) targets. See `docs/notes/` for the
> per-stage reasoning and `results/classical_fit.csv` for the fit.

## Layout

```
src/
  QuantJulia.jl      # module
  blackscholes.jl    # Stage 1 — BSM price, Greeks, implied vol        ✅
  heston.jl          # Stage 2 — Heston CF, little-trap form           ✅
  fourier.jl         # Stage 3 — model-agnostic Gil-Pelaez pricer      ✅
  cboe.jl            # Stage 4 — CBOE parser, filters, parity forwards ✅
  calibration.jl     # Stage 4 — vega-weighted loss + LBFGS/AD         ✅
  (rough_heston.jl)  # Stage 6+ — fractional Riccati solver + rough CF
scripts/
  01_prepare_data.jl # raw CBOE csv → filtered chain with implied vols
  02_calibrate.jl    # fit classical Heston, report per-expiry RMSE
test/
  runtests.jl        # + per-stage test files (120 tests)
data/                # market data (not committed — see data/README.md)
results/             # committed fit results
docs/
  roadmap.md         # the full plan, grounded to where the code actually is
  rough_heston_spec.md   # committed destination spec (Stages 6–8)
  notes/             # per-stage theory notes: derivations + the "why"
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

The tests are written first; you implement `src/blackscholes.jl` until they pass.

## Design constraint carried from day one

Everything is written to stay **automatic-differentiation friendly** — functions
are generic in their numeric type so `ForwardDiff.Dual` can flow through prices
to yield Greeks and calibration gradients. This is why the eventual rough-Heston
fractional ODE solver will be hand-rolled rather than pulled from a package.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).
