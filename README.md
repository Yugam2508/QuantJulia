# QuantJulia

A differentiable option-pricing and volatility-calibration framework in Julia.

**Story (the destination):** a differentiable calibration engine, demonstrated on
classical Heston, extended to the rough-volatility frontier (rough Heston). The
same Fourier pricing / implied-vol / calibration machinery is written once and
reused across models — the only thing that changes between models is the
characteristic function.

> **Status: Stages 1–3 complete** (94 tests green): Black-Scholes + implied-vol
> inversion, Heston characteristic function ("little trap" form), and a
> model-agnostic Gil-Pelaez Fourier pricer with ForwardDiff gradients verified
> end-to-end. Next: SPX data + calibration (Stage 4). See the roadmap.

## Layout

```
src/
  QuantJulia.jl      # module
  blackscholes.jl    # Stage 1 — BSM price, Greeks, implied vol        ✅
  heston.jl          # Stage 2 — Heston CF, little-trap form           ✅
  fourier.jl         # Stage 3 — model-agnostic Gil-Pelaez pricer      ✅
  (calibration.jl)   # Stage 4+ — loss + optimizer over the vol surface
  (rough_heston.jl)  # Stage 6+ — fractional Riccati solver + rough CF
test/
  runtests.jl
  test_blackscholes.jl
  test_heston.jl
  test_fourier.jl
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
