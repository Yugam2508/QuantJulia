# =============================================================================
# Jump models — Merton, Bates, Variance Gamma — as characteristic functions
# =============================================================================
# The package's organizing claim is that a model is just its characteristic
# function: the Fourier pricers, batch pricer, Greeks and calibration never
# look inside. These three models exercise that claim on dynamics with JUMPS,
# which no diffusion (classical or rough) produces — the textbook explanation
# for the very steep short-dated skew.
#
# Same convention as src/heston.jl: ψ(u) = E[e^{iuX_T}] for the de-drifted log
# return X_T = log(S_T/F), so ψ(0) = 1 and ψ(−i) = 1 (martingale). Each CF
# below carries its own compensator so that holds exactly.
#
#   Merton (1976): Black-Scholes + compound Poisson log-normal jumps,
#       log J ~ N(μJ, δJ²), intensity λ, mean jump size k = e^{μJ+δJ²/2} − 1.
#   Bates (1996): Heston variance + the same Merton jumps in the price
#       (independent of the diffusion) — ψ_Bates = ψ_Heston · ψ_jumps.
#   Variance Gamma (Madan, Carr & Chang 1998): Brownian motion with drift θ
#       and vol σ, run on a gamma clock with variance rate ν. A pure-jump
#       process with infinite activity; θ < 0 gives the left skew.
# =============================================================================

"""
    MertonParams(σ, λ, μJ, δJ)

Merton jump-diffusion: diffusion vol `σ`, jump intensity `λ` (per year), and
log-jump size ~ N(`μJ`, `δJ`²).
"""
struct MertonParams{T<:Real}
    σ::T
    λ::T
    μJ::T
    δJ::T
end
MertonParams(σ, λ, μJ, δJ) = MertonParams(promote(σ, λ, μJ, δJ)...)

"""
    BatesParams(κ, θ, ξ, ρ, v0, λ, μJ, δJ)

Bates model: Heston parameters `(κ, θ, ξ, ρ, v0)` plus Merton jumps
`(λ, μJ, δJ)` in the price.
"""
struct BatesParams{T<:Real}
    κ::T
    θ::T
    ξ::T
    ρ::T
    v0::T
    λ::T
    μJ::T
    δJ::T
end
BatesParams(κ, θ, ξ, ρ, v0, λ, μJ, δJ) =
    BatesParams(promote(κ, θ, ξ, ρ, v0, λ, μJ, δJ)...)

"""
    VGParams(σ, ν, θ)

Variance Gamma: Brownian vol `σ`, gamma-clock variance rate `ν`, drift `θ`
(θ < 0 skews left). Requires `1 − θν − σ²ν/2 > 0` for the martingale
correction to exist.
"""
struct VGParams{T<:Real}
    σ::T
    ν::T
    θ::T
end
VGParams(σ, ν, θ) = VGParams(promote(σ, ν, θ)...)

# Compensated log-normal compound-Poisson exponent per unit time:
#   λ(e^{iuμJ − δJ²u²/2} − 1) − iu·λk,  k = e^{μJ + δJ²/2} − 1.
function _jump_exponent(u, λ, μJ, δJ)
    k = exp(μJ + δJ^2 / 2) - 1
    return λ * (exp(im * u * μJ - δJ^2 * u^2 / 2) - 1) - im * u * λ * k
end

"""
    merton_cf(u, T, p::MertonParams)

ψ(u) for Merton's jump-diffusion (de-drifted log return, ψ(−i) = 1).
"""
merton_cf(u, T, p::MertonParams) =
    exp(T * (-(im * u + u^2) * p.σ^2 / 2 + _jump_exponent(u, p.λ, p.μJ, p.δJ)))

"""
    bates_cf(u, T, p::BatesParams)

ψ(u) for Bates: the Heston little-trap CF times the compensated jump factor.
"""
bates_cf(u, T, p::BatesParams) =
    heston_cf(u, T, HestonParams(p.κ, p.θ, p.ξ, p.ρ, p.v0)) *
    exp(T * _jump_exponent(u, p.λ, p.μJ, p.δJ))

"""
    vg_cf(u, T, p::VGParams)

ψ(u) for Variance Gamma, with martingale correction
ω = log(1 − θν − σ²ν/2)/ν.
"""
function vg_cf(u, T, p::VGParams)
    base = 1 - p.θ * p.ν - p.σ^2 * p.ν / 2
    base > 0 || throw(DomainError(base, "VG needs 1 − θν − σ²ν/2 > 0"))
    ω = log(base) / p.ν
    return exp(im * u * ω * T) * (1 - im * u * p.θ * p.ν + p.σ^2 * p.ν * u^2 / 2)^(-T / p.ν)
end

"""
    merton_price(S, K, r, q, p::MertonParams, T; call=true, tol=1e-14, nmax=500)

Merton's closed form: a Poisson-weighted sum of Black-Scholes prices,
conditioning on the number of jumps n,

    C = Σₙ e^{−λ'T}(λ'T)ⁿ/n! · BS(S, K, rₙ, q, σₙ, T),
    λ' = λ(1+k),  σₙ² = σ² + nδJ²/T,  rₙ = r − λk + n·log(1+k)/T.

Independent of the Fourier machinery — the reference the jump CFs are tested
against. Summed until the Poisson weight falls below `tol`.
"""
function merton_price(S, K, r, q, p::MertonParams, T; call::Bool = true,
                      tol = 1e-14, nmax::Int = 500)
    k = exp(p.μJ + p.δJ^2 / 2) - 1
    Λ = p.λ * (1 + k) * T
    total = zero(S * K * r * q * T * p.σ)
    w = exp(-Λ)                              # Poisson weight, n = 0
    for n in 0:nmax
        σn = sqrt(p.σ^2 + n * p.δJ^2 / T)
        rn = r - p.λ * k + n * log(1 + k) / T
        total += w * bs_price(S, K, rn, q, σn, T; call)
        n > Λ && w < tol && break
        w *= Λ / (n + 1)
    end
    return total
end
