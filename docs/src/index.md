# QuantJulia

A differentiable option-pricing and volatility-calibration framework in Julia:
Black-Scholes → Heston → rough Heston, calibrated to a real SPX chain.

The same Fourier pricing / implied-vol / calibration machinery is written once
and reused across models; the only thing that changes between models is the
characteristic function. Every pricer is `ForwardDiff`-safe end to end, so
calibration gradients — including with respect to the Hurst exponent H — come
from automatic differentiation through the fractional Riccati solver.

## Quick start

```julia
using QuantJulia

# Black-Scholes price, Greeks, implied vol
c = bs_price(100.0, 105.0, 0.03, 0.01, 0.2, 0.5)
implied_vol(c, 100.0, 105.0, 0.03, 0.01, 0.5)

# Classical Heston by Fourier inversion
p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)   # κ, θ, ξ, ρ, v0
heston_price(100.0, 105.0, 0.03, 0.01, p, 0.5)

# Rough Heston (H = 0.1): CF from the implicit fractional Riccati solver,
# all strikes of an expiry priced from one set of CF evaluations
pr = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)
T = 0.25; F = 100.0 * exp(0.02T); disc = exp(-0.03T)
batch_call_prices(make_rough_cf(T, pr), F, disc, [95.0, 100.0, 105.0], T)
```

## Where to go next

- [API reference](api.md) — every exported function, with docstrings.
- [Results: rough vs classical](generated/rough_heston.md) — the SPX
  calibration and the honest reading of what it says about H.
- Theory notes — per-stage derivations and the reasoning behind the design.
