# Validation against QuantLib

`quantlib_reference.py` prices a fixed grid of options with
[QuantLib](https://www.quantlib.org) 1.43 and writes `quantlib_reference.csv`.
`test/test_quantlib.jl` reprices every row with QuantJulia on each CI run.

| family | QuantLib engine | worst disagreement |
|---|---|---|
| Black-Scholes price, Δ, Γ, vega, Θ, ρ | `AnalyticEuropeanEngine` | ~1e-14 |
| Black-Scholes implied vol | `impliedVolatility` | 3e-11 |
| Heston (3 parameter sets incl. Feller-violating) | `AnalyticHestonEngine` | 1.6e-11 |
| Bates | `BatesEngine` | 6.8e-12 |
| Barrier, all 8 single-barrier types | `AnalyticBarrierEngine` | 1.8e-14 |
| Geometric Asian, discrete fixings | `AnalyticDiscreteGeometricAveragePriceAsianEngine` | 1.7e-14 |
| American put | `FdBlackScholesVanillaEngine` (2000×2000) vs our CRR (N = 5000) | 8e-4 (rel 1.4e-4) |

The American put compares two different approximations (lattice vs finite
differences), so its tolerance is 2e-3; every closed form agrees to machine
precision. Merton is not included: QuantLib's Python bindings do not expose
its `JumpDiffusionEngine` (QuantJulia checks Merton against its closed-form
Poisson series instead, and Bates covers the jump component here).

Regenerate:

```
python -m pip install QuantLib==1.43
python validation/quantlib_reference.py
```
