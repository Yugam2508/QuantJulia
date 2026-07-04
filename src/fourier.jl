# =============================================================================
# Model-agnostic Fourier pricer — Gil-Pelaez inversion  (Stage 3)
# =============================================================================
# This file prices from *a characteristic function*, not from a model. That is
# the one binding constraint the rough-Heston spec places on this stage: when
# rough Heston arrives, it plugs in as `u -> rough_heston_cf(u, T, p)` and this
# file does not change.
#
# Input convention (matches src/heston.jl): ψ(u) = E[e^{iuX}] with
# X = log(S_T/F) the de-drifted log return, ψ(−i) = 1 (martingale).
# With m = log(F/K):
#   P₂ = 1/2 + (1/π) ∫₀^∞ Re[ e^{ium} ψ(u)   / (iu) ] du   — P(S_T > K)
#   P₁ = 1/2 + (1/π) ∫₀^∞ Re[ e^{ium} ψ(u−i) / (iu) ] du   — same probability
#                                                             under the stock
#                                                             numeraire
#   call = e^{−rT} (F·P₁ − K·P₂)
# The u ↦ u−i shift is the same measure change that separates d₁ from d₂ in
# Black-Scholes, one abstraction level up.
#
# Both integrands have finite u→0 limits (the 1/(iu) pole is purely imaginary
# and Re[·] kills it); quadgk's Gauss-Kronrod nodes are interior, so u = 0 is
# never evaluated. Quadrature is adaptive over (0, ∞) — no manual truncation
# to tune, at the cost of adaptivity; if calibration profiling later shows the
# pricer dominating, swap in fixed Gauss-Legendre nodes (AD-safe either way,
# since ForwardDiff Duals ride through the integrand evaluations).
# =============================================================================

using QuadGK: quadgk

"""
    price_from_cf(ψ, S, K, r, q, T; call=true, rtol=1e-9)

European option price by Gil-Pelaez inversion of the characteristic function
`ψ(u) = E[e^{iuX}]`, X = log(S_T/F), ψ(−i) = 1. `ψ` must accept complex
arguments (it is evaluated at u − i). Model-independent by construction.

The put comes via parity, P = C − e^{−rT}(F − K), which is exact here because
both legs share the same two integrals.
"""
function price_from_cf(ψ, S, K, r, q, T; call::Bool = true, rtol = 1e-9)
    (S > 0 && K > 0 && T > 0) || throw(DomainError((S, K, T), "need S, K, T > 0"))
    F = S * exp((r - q) * T)
    m = log(F / K)
    disc = exp(-r * T)

    integrand2(u) = real(exp(im * u * m) * ψ(u) / (im * u))
    integrand1(u) = real(exp(im * u * m) * ψ(u - im) / (im * u))

    I1, _ = quadgk(integrand1, 0.0, Inf; rtol = rtol)
    I2, _ = quadgk(integrand2, 0.0, Inf; rtol = rtol)
    P1 = 1 / 2 + I1 / π
    P2 = 1 / 2 + I2 / π

    c = disc * (F * P1 - K * P2)
    return call ? c : c - disc * (F - K)
end

"""
    heston_price(S, K, r, q, p::HestonParams, T; call=true, rtol=1e-9)

Classical Heston price: `heston_cf` plugged into `price_from_cf`. The params
argument sits where σ sits in `bs_price` — same call shape, model swapped.
"""
heston_price(S, K, r, q, p::HestonParams, T; call::Bool = true, rtol = 1e-9) =
    price_from_cf(u -> heston_cf(u, T, p), S, K, r, q, T; call = call, rtol = rtol)
