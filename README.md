# QuantJulia

A differentiable option-pricing and volatility-calibration framework in Julia.

**Story (the destination):** a differentiable calibration engine, demonstrated on
classical Heston, extended to the rough-volatility frontier (rough Heston). The
same Fourier pricing / implied-vol / calibration machinery is written once and
reused across models — the only thing that changes between models is the
characteristic function.

> **Status: in development.** Week 1 — Black-Scholes warm-up. The math functions
> are being implemented against a test suite; see the roadmap below.

## Layout

```
src/
  QuantJulia.jl      # module
  blackscholes.jl    # Week 1 — BSM price, Greeks, implied vol   ← current
  (heston.jl)        # Week 2+ — Heston characteristic function
  (fourier.jl)       # Week 2+ — Carr-Madan / Fourier pricer
  (calibration.jl)   # Week 4+ — loss + optimizer over the vol surface
  (rough_heston.jl)  # Week 6+ — fractional Riccati solver + rough CF
test/
  runtests.jl
  test_blackscholes.jl
docs/
  roadmap.md         # the full plan, grounded to where the code actually is
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
