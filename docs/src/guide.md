# User guide

A tour of the package by task. Every block on this page is executed when the
documentation is built, so the outputs shown are real and the examples cannot
silently go stale.

The organizing idea: **a model is its characteristic function.** Each model
provides `ψ(u) = E[exp(iuX_T)]` for the de-drifted log return
``X_T = \log(S_T/F)`` (so ``ψ(-i) = 1``), and every pricer, Greek and
calibrator works from that alone.

```@setup guide
using QuantJulia, Random
```

## Black-Scholes, Greeks, implied volatility

```@example guide
S, K, r, q, σ, T = 100.0, 105.0, 0.03, 0.01, 0.2, 0.5
c = bs_price(S, K, r, q, σ, T)
(price = c,
 delta = bs_delta(S, K, r, q, σ, T), gamma = bs_gamma(S, K, r, q, σ, T),
 vega = bs_vega(S, K, r, q, σ, T), theta = bs_theta(S, K, r, q, σ, T),
 rho = bs_rho(S, K, r, q, σ, T),
 implied_vol = implied_vol(c, S, K, r, q, T))
```

## Heston by Fourier inversion

`heston_price` uses adaptive Gil-Pelaez quadrature. For a whole smile, the
batch pricer and the COS pricer share one set of CF evaluations across all
strikes of an expiry.

```@example guide
p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)        # κ, θ, ξ, ρ, v0
T = 0.5; F = S * exp((r - q) * T); disc = exp(-r * T)
ψ = u -> heston_cf(u, T, p)
Ks = [80.0, 90.0, 100.0, 110.0, 120.0]
batch = batch_call_prices(ψ, F, disc, Ks, T)
cos = cos_call_prices(ψ, F, disc, Ks, T)
ivs = [implied_vol(c, S, K, r, q, T) for (K, c) in zip(Ks, cos)]
(adaptive_100 = heston_price(S, 100.0, r, q, p, T), batch = batch, cos = cos, smile = ivs)
```

## Rough Heston and its Markovian lift

The rough CF comes from the implicit fractional Riccati solver; lifted Heston
approximates it with `n` ordinary Riccati ODEs and converges to it as `n`
grows.

```@example guide
pr = RoughHestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.1)   # … , H
T = 0.25; F = S * exp((r - q) * T); disc = exp(-r * T)
rough = batch_call_prices(make_rough_cf(T, pr; N = 128), F, disc, Ks, T)
lifted = batch_call_prices(make_lifted_cf(T, pr; n = 20), F, disc, Ks, T)
ivr = [implied_vol(c, S, K, r, q, T) for (K, c) in zip(Ks, rough)]
ivl = [implied_vol(c, S, K, r, q, T) for (K, c) in zip(Ks, lifted)]
(rough_smile = ivr, lifted_minus_rough_bps = round.(1e4 .* (ivl .- ivr); digits = 1))
```

## Jump models

Merton, Bates and Variance Gamma are just more characteristic functions;
Merton also has a closed-form series to check against.

```@example guide
pm = MertonParams(0.15, 0.8, -0.12, 0.10)                  # σ, λ, μJ, δJ
(fourier = price_from_cf(u -> merton_cf(u, 1.0, pm), S, K, r, q, 1.0),
 closed_form = merton_price(S, K, r, q, pm, 1.0),
 bates = price_from_cf(u -> bates_cf(u, 1.0, BatesParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.8, -0.12, 0.1)),
                       S, K, r, q, 1.0),
 variance_gamma = price_from_cf(u -> vg_cf(u, 1.0, VGParams(0.18, 0.25, -0.14)), S, K, r, q, 1.0))
```

## Calibration

Quotes are NamedTuples with fields `T, K, r, q, side, mid, iv, vega` (plus
`F`, which `prepare_chain` provides). Here a synthetic Heston chain is fitted
from a perturbed start by Levenberg–Marquardt with the batch pricer.

```@example guide
truth = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
quotes = map(Iterators.product((0.1, 0.5, 1.0), -2:0.5:2)) do (T, z)
    F = S * exp((r - q) * T); K = F * exp(z * 0.2 * sqrt(T)); side = K >= F ? :call : :put
    mid = heston_price(S, K, r, q, truth, T; call = side === :call)
    iv = implied_vol(mid, S, K, r, q, T; call = side === :call)
    (; T, K, r, q, side, mid, iv, vega = bs_vega(S, K, r, q, iv, T))
end |> vec
fit = calibrate_heston(quotes, S; p0 = HestonParams(1.2, 0.06, 0.8, -0.4, 0.06),
                       method = :lm, pricer = :batch)
(params = fit.params, rmse_bps = 1e4 * fit.rmse, iterations = fit.iterations)
```

Any CF model can be calibrated with `calibrate_cf_model`, given a map from an
unconstrained vector to the CF:

```@example guide
mk(x, T) = (pm = MertonParams(exp(x[1]), exp(x[2]), x[3], exp(x[4])); u -> merton_cf(u, T, pm))
res = calibrate_cf_model(mk, [log(0.2), log(0.5), -0.1, log(0.15)], quotes; S = S)
(rmse_bps = 1e4 * res.rmse, converged = res.converged)    # Merton fitted to a Heston chain
```

## Model Greeks by automatic differentiation

```@example guide
g = heston_greeks(S, K, r, q, truth, 0.5)
(delta = g.delta, gamma = g.gamma, vega = g.vega, theta = g.theta, dV_dρ = g.sens.ρ)
```

## Volatility surfaces: SVI, SSVI, local volatility

```@example guide
surface = fit_svi_surface([(; T = q.T, K = q.K, F = S * exp((r - q.q) * q.T), iv = q.iv) for q in quotes])
(svi_rmse_bps = 1e4 * surface.svi_rmse, ssvi_rmse_bps = 1e4 * surface.ssvi_rmse,
 butterfly_free = all(surface.butterfly_free), calendar_free = isempty(surface.calendar_violations))
```

An arbitrage-free SSVI surface defines a Dupire local volatility:

```@example guide
w = ssvi_surface(SSVIParams(-0.7, 1.1, 0.4), [0.25, 0.5, 1.0], [0.01, 0.02, 0.04])
[sqrt(dupire_local_variance(w, k, 0.5)) for k in (-0.2, 0.0, 0.2)]    # local vols at T = 0.5
```

## Variance swaps and the VIX

```@example guide
(heston_varswap_1y = variance_swap_strike(u -> heston_cf(u, 1.0, truth), 1.0),
 rough_varswap_1y = variance_swap_strike(make_rough_cf(1.0, pr), 1.0),
 heston_vix = model_vix(T -> (u -> heston_cf(u, T, truth))))
```

## Monte Carlo

Heston with Andersen's QE scheme (accurate with few steps even when the
Feller condition fails) and rough Bergomi by the hybrid scheme:

```@example guide
qe = heston_mc_price(S, K, r, q, truth, 0.5; nsteps = 16, npaths = 20_000, rng = MersenneTwister(1))
rb = rbergomi_mc_prices(S, Ks, r, q, RBergomiParams(0.04, 1.9, -0.9, 0.1), 0.5;
                        nsteps = 50, npaths = 20_000, rng = MersenneTwister(2))
(heston_qe = qe, heston_fourier = heston_price(S, K, r, q, truth, 0.5), rbergomi_calls = rb.prices)
```

## Market data

A CBOE delayed-quotes CSV goes through `read_cboe` and `prepare_chain`, which
recovers forwards and discount rates from put-call parity, keeps OTM quotes,
filters bad ones (with a counted rejection report) and attaches implied vols
and vegas. Business-day time is one keyword away:

```julia
raw = read_cboe("data/spx_quotedata.csv")
chain = prepare_chain(raw; valuation_date = Date(2026, 7, 2), daycount = :business)
chain.rejects            # why each discarded row was discarded
fit = calibrate_heston(chain.quotes, chain.spot; method = :lm, pricer = :batch)
```

See `data/README.md` for the exact download and the [results write-up](generated/rough_heston.md)
for what the SPX chain says about roughness.
