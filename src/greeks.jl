# =============================================================================
# Model Greeks by automatic differentiation
# =============================================================================
# Every pricer in this package is Dual-safe end to end (that is what makes the
# calibrations differentiable), so model Greeks come for free: differentiate
# the price itself. No bump sizes to tune, no cancellation error, and the same
# code serves any model — classical Heston through adaptive Gil-Pelaez,
# rough Heston through the fractional solver and the batch pricer.
#
# Conventions match the Black-Scholes Greeks in src/blackscholes.jl:
#   delta = ∂V/∂S, gamma = ∂²V/∂S², theta = ∂V/∂t = −∂V/∂T (per year),
#   rho = ∂V/∂r. Model-parameter sensitivities are returned alongside, plus
#   a BS-style "vega" = ∂V/∂√v0 (sensitivity to the spot vol level).
# =============================================================================

"""
    ad_greeks(price, S, r, q, T)

Delta, gamma, theta and rho of any pricer `price(S, r, q, T)` by ForwardDiff
(gamma is a nested derivative). `price` must be generic in all four
arguments. Returns `(price, delta, gamma, theta, rho)`.
"""
function ad_greeks(price, S, r, q, T)
    v = price(S, r, q, T)
    dS(s) = ForwardDiff.derivative(s2 -> price(s2, r, q, T), s)
    delta = dS(S)
    gamma = ForwardDiff.derivative(dS, S)
    theta = -ForwardDiff.derivative(t -> price(S, r, q, t), T)
    rho = ForwardDiff.derivative(x -> price(S, x, q, T), r)
    return (; price = v, delta, gamma, theta, rho)
end

"""
    heston_greeks(S, K, r, q, p::HestonParams, T; call=true, rtol=1e-10)

Classical Heston Greeks by AD through `heston_price`. Returns
`(price, delta, gamma, theta, rho, vega, sens)` where `vega = ∂V/∂√v0` and
`sens = (κ, θ, ξ, ρ, v0)` holds ∂V/∂(each model parameter).
"""
function heston_greeks(S, K, r, q, p::HestonParams, T; call::Bool = true, rtol = 1e-10)
    g = ad_greeks((s, x, y, t) -> heston_price(s, K, x, y, p, t; call, rtol), S, r, q, T)
    x0 = [p.κ, p.θ, p.ξ, p.ρ, p.v0]
    d = ForwardDiff.gradient(x -> heston_price(S, K, r, q, HestonParams(x...), T; call, rtol), x0)
    sens = (κ = d[1], θ = d[2], ξ = d[3], ρ = d[4], v0 = d[5])
    return merge(g, (; vega = 2 * sqrt(p.v0) * sens.v0, sens))
end

# Rough Heston single-option price via the batch pricer (+ parity for puts).
# `iv_hint` only steers the quadrature node budget; it is a plain Float64 so
# the node set never depends on the parameters being differentiated.
function _rough_price(S, K, r, q, p::RoughHestonParams, T, call, N, iv_hint)
    F = S * exp((r - q) * T)
    disc = exp(-r * T)
    c = batch_call_prices(make_rough_cf(T, p; N = N), F, disc, (K,), T; iv_hint = iv_hint)[1]
    return call ? c : c - disc * (F - K)
end

"""
    rough_heston_greeks(S, K, r, q, p::RoughHestonParams, T; call=true, N=128)

Rough Heston Greeks by AD through the fractional Riccati solver and the batch
pricer. Same return shape as `heston_greeks`, with `H` added to `sens`.
"""
function rough_heston_greeks(S, K, r, q, p::RoughHestonParams, T;
                             call::Bool = true, N::Int = 128)
    ivh = Float64(sqrt(p.v0))
    g = ad_greeks((s, x, y, t) -> _rough_price(s, K, x, y, p, t, call, N, ivh), S, r, q, T)
    x0 = [p.κ, p.θ, p.ξ, p.ρ, p.v0, p.H]
    d = ForwardDiff.gradient(x -> _rough_price(S, K, r, q, RoughHestonParams(x...), T,
                                               call, N, ivh), x0)
    sens = (κ = d[1], θ = d[2], ξ = d[3], ρ = d[4], v0 = d[5], H = d[6])
    return merge(g, (; vega = 2 * sqrt(p.v0) * sens.v0, sens))
end
