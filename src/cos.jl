# =============================================================================
# COS method (Fang & Oosterlee 2008) — a second, faster Fourier pricer
# =============================================================================
# Gil-Pelaez (src/fourier.jl) integrates the CF against a slowly decaying
# kernel. COS instead expands the DENSITY of the log return in a Fourier-
# cosine series on a truncated interval [a, b]; for the smooth densities of
# diffusion models the series converges exponentially in the number of terms
# N, so 512 CF evaluations price an entire smile to ~1e-12 (defaults
# N = 512, L = 24: the wide interval is for fat left tails — Heston with
# ρ = −0.7 needs ±20σ before truncation error drops below 1e-10).
#
# Setup in the package convention, X = log(S_T/F) with CF ψ:
#   * Truncation: [a, b] = c₁ ± L·√c₂, with the cumulants c₁ = E[X] and
#     c₂ = Var[X] read off ψ by central differences of log ψ at 0. Model-
#     agnostic: nothing beyond ψ is needed.
#   * For strike K let y = log(S_T/K) = x₀ + X, x₀ = log(F/K). The put payoff
#     K(1 − eʸ)⁺ has closed-form cosine coefficients V_k (χ/φ integrals of
#     the original paper). Writing the interval for y as [x₀+a, x₀+b], the CF
#     factor in each term is Re[ψ(u_k)·e^{−iu_k a}], u_k = kπ/(b−a) — the
#     strike cancels out of it. So, as in `batch_call_prices`, ALL strikes of
#     an expiry share the same N CF evaluations.
#   * Puts are priced directly (bounded payoff: the stable choice); calls by
#     exact parity C = P + disc·(F − K).
#
# AD: the truncation interval is computed from the primal (non-Dual) CF
# values, so it never moves with the parameters being differentiated — the
# same fixed-node principle as the batch pricer.
# =============================================================================

_primal(x) = x
_primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))
_primal(z::Complex) = complex(_primal(real(z)), _primal(imag(z)))

"""
    cf_cumulants(ψ; h=1e-3)

First two cumulants (mean, variance) of X from its characteristic function,
by central differences of log ψ at 0. Returned as plain floats.
"""
function cf_cumulants(ψ; h = 1e-3)
    lp, l0, lm = _primal(log(ψ(h))), _primal(log(ψ(0.0))), _primal(log(ψ(-h)))
    c1 = imag(lp - lm) / (2h)
    c2 = -real(lp - 2l0 + lm) / h^2
    return (c1 = c1, c2 = max(c2, 0.0))
end

# Σₖ' fac[k]·V_k for the put payoff (1 − eʸ)⁺ (per unit strike) on the
# interval [α, β] for y, with the payoff support clipped to [α, min(0, β)].
# V_k = 2/(β−α)·(ψ_k − χ_k), the cosine coefficients of the original paper.
#
# Performance: every angle is k·φ, φ = π(d − α)/(β − α), so cos(kφ) and
# sin(kφ) come from one complex rotation per term (no trig in the loop), the
# exponentials are hoisted, and nothing is allocated — the per-strike cost is
# a handful of flops per term. The recurrence's rounding error grows like
# k·eps, ~1e-13 at N = 512.
function _put_sum(fac, α, β)
    d = min(0.0, β)
    z0 = zero(eltype(fac))
    d <= α && return z0                          # put is worthless on [α, β]
    w = β - α
    ed, ea = exp(d), exp(α)
    rot = cis(π * (d - α) / w)
    z = one(rot)                                 # e^{i k φ}
    acc = fac[1] * ((d - α) - (ed - ea))         # k = 0: ψ₀ = d − α, χ₀ = eᵈ − eᵅ
    @inbounds for k in 1:length(fac)-1
        z *= rot
        c, s = real(z), imag(z)
        uk = k * π / w
        ψk = s / uk
        χk = (c * ed - ea + uk * s * ed) / (1 + uk^2)
        acc += fac[k+1] * (ψk - χk)
    end
    return 2 / w * acc
end

"""
    cos_call_prices(ψ, F, disc, Ks, T; N=512, L=24, interval=nothing)

Call prices for all strikes `Ks` of one expiry by the COS method, sharing `N`
CF evaluations. Same signature and conventions as `batch_call_prices`
(ψ the de-drifted log-return CF, `F` the forward, `disc = e^{−rT}`).
`interval = (a, b)` overrides the cumulant-based truncation of X.
"""
function cos_call_prices(ψ, F, disc, Ks, T; N::Int = 512, L = 24, interval = nothing)
    a, b = if interval === nothing
        c = cf_cumulants(ψ)
        half = L * sqrt(max(c.c2, 1e-10))
        (c.c1 - half, c.c1 + half)
    else
        Float64.(interval)
    end
    w = b - a
    us = [k * π / w for k in 0:N-1]
    # Strike-independent CF factor Re[ψ(u_k) e^{−i u_k a}], first term halved.
    fac = [real(ψ(us[k]) * exp(-im * us[k] * a)) for k in 1:N]
    fac[1] /= 2
    return map(Ks) do K
        x0 = log(F / K)
        put = disc * K * _put_sum(fac, x0 + a, x0 + b)
        put + disc * (F - K)
    end
end

"""
    cos_price(ψ, S, K, r, q, T; call=true, N=512, L=24)

Single European option by the COS method; same call shape as
`price_from_cf`.
"""
function cos_price(ψ, S, K, r, q, T; call::Bool = true, N::Int = 512, L = 24)
    F = S * exp((r - q) * T)
    disc = exp(-r * T)
    c = cos_call_prices(ψ, F, disc, (K,), T; N, L)[1]
    return call ? c : c - disc * (F - K)
end
