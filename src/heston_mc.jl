# =============================================================================
# Heston Monte Carlo — Andersen's QE scheme (and Euler, for comparison)
# =============================================================================
# The CIR variance process is the hard part of simulating Heston: its exact
# transition is a scaled non-central χ², and naive Euler goes negative, so it
# must be floored ("full truncation") — which biases prices, badly so when the
# Feller condition 2κθ ≥ ξ² fails, i.e. for almost every index calibration.
#
# Andersen (2008), "Simple and efficient simulation of the Heston stochastic
# volatility model": the Quadratic-Exponential (QE) scheme matches the first
# two moments of the exact transition,
#     m  = θ + (v − θ)e^{−κΔ}
#     s² = v ξ² e^{−κΔ}(1 − e^{−κΔ})/κ + θ ξ²(1 − e^{−κΔ})²/(2κ),   ψ = s²/m²,
# with a squared Gaussian when ψ ≤ ψc = 1.5 (variance well away from 0),
#     v' = a(b + Z)²,  b² = 2/ψ − 1 + √(2/ψ)√(2/ψ − 1),  a = m/(1 + b²),
# and a point mass at 0 plus an exponential tail otherwise,
#     p = (ψ − 1)/(ψ + 1),  β = (1 − p)/m,  v' = 0 if U ≤ p else log((1−p)/(1−U))/β.
# Never negative, no floor. The log price then uses Andersen's central
# discretization (γ₁ = γ₂ = ½), which integrates ∫v dt by the trapezoid rule
# and recovers the correlated part of the price shock from the variance
# increment rather than from a separate Gaussian:
#     X' = X + K₀ + K₁v + K₂v' + √(K₃v + K₄v')·Z,
#     K₀ = −ρκθΔ/ξ,  K₁ = γ₁Δ(κρ/ξ − ½) − ρ/ξ,  K₂ = γ₂Δ(κρ/ξ − ½) + ρ/ξ,
#     K₃ = γ₁Δ(1 − ρ²),  K₄ = γ₂Δ(1 − ρ²).
# X is the de-drifted log return, as in every other module here.
# =============================================================================

_rand(rng) = rng === nothing ? rand() : rand(rng)

"""
    simulate_heston(p::HestonParams, T, nsteps, npaths; scheme=:qe, rng=nothing)

Simulate `npaths` de-drifted log returns X_T = log(S_T/F) under Heston.
`scheme = :qe` is Andersen's Quadratic-Exponential scheme (accurate with few
steps, no positivity floor); `scheme = :euler` is full-truncation Euler, kept
as the baseline QE is measured against. `rng` is any RNG accepted by
`randn(rng)`/`rand(rng)` (`nothing` uses the global RNG).
"""
function simulate_heston(p::HestonParams, T, nsteps::Int, npaths::Int;
                         scheme::Symbol = :qe, rng = nothing)
    nsteps >= 1 && npaths >= 1 || throw(ArgumentError("need nsteps, npaths ≥ 1"))
    scheme in (:qe, :euler) || throw(ArgumentError("scheme must be :qe or :euler"))
    κ, θ, ξ, ρ, v0 = p.κ, p.θ, p.ξ, p.ρ, p.v0
    Δ = T / nsteps
    X = zeros(npaths)
    if scheme === :euler
        sρ = sqrt(1 - ρ^2)
        for i in 1:npaths
            x = 0.0; v = v0
            for _ in 1:nsteps
                vp = max(v, 0.0)
                z1 = _randn(rng); z2 = _randn(rng)
                x += -vp * Δ / 2 + sqrt(vp * Δ) * z1
                v += κ * (θ - vp) * Δ + ξ * sqrt(vp * Δ) * (ρ * z1 + sρ * z2)
            end
            X[i] = x
        end
        return X
    end
    E = exp(-κ * Δ)
    c1 = ξ^2 * E * (1 - E) / κ
    c2 = θ * ξ^2 * (1 - E)^2 / (2κ)
    γ1 = γ2 = 0.5
    K0 = -ρ * κ * θ * Δ / ξ
    K1 = γ1 * Δ * (κ * ρ / ξ - 0.5) - ρ / ξ
    K2 = γ2 * Δ * (κ * ρ / ξ - 0.5) + ρ / ξ
    K3 = γ1 * Δ * (1 - ρ^2)
    K4 = γ2 * Δ * (1 - ρ^2)
    for i in 1:npaths
        x = 0.0; v = v0
        for _ in 1:nsteps
            m = θ + (v - θ) * E
            s2 = v * c1 + c2
            ψ = s2 / m^2
            vn = if ψ <= 1.5
                b2 = 2 / ψ - 1 + sqrt(2 / ψ) * sqrt(2 / ψ - 1)
                a = m / (1 + b2)
                a * (sqrt(b2) + _randn(rng))^2
            else
                pz = (ψ - 1) / (ψ + 1)
                β = (1 - pz) / m
                u = _rand(rng)
                u <= pz ? 0.0 : log((1 - pz) / (1 - u)) / β
            end
            x += K0 + K1 * v + K2 * vn + sqrt(K3 * v + K4 * vn) * _randn(rng)
            v = vn
        end
        X[i] = x
    end
    return X
end

"""
    heston_mc_price(S, K, r, q, p::HestonParams, T; call=true, nsteps=32,
                    npaths=100_000, scheme=:qe, rng=nothing)

Monte Carlo price of a European option under Heston, with the forward as a
control variate (E[S_T] = F). Returns `(price, stderr)`.
"""
function heston_mc_price(S, K, r, q, p::HestonParams, T; call::Bool = true,
                         nsteps::Int = 32, npaths::Int = 100_000,
                         scheme::Symbol = :qe, rng = nothing)
    F = S * exp((r - q) * T)
    disc = exp(-r * T)
    ST = F .* exp.(simulate_heston(p, T, nsteps, npaths; scheme, rng))
    pay = call ? max.(ST .- K, 0.0) : max.(K .- ST, 0.0)
    cv = ST .- F
    β = sum((pay .- sum(pay) / npaths) .* cv) / max(sum(abs2, cv), eps())
    adj = pay .- β .* cv
    m = sum(adj) / npaths
    se = sqrt(sum(abs2, adj .- m) / (npaths - 1) / npaths)
    return (price = disc * m, stderr = disc * se)
end
