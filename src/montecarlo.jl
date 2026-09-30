# =============================================================================
# Rough Heston Monte Carlo — an independent check on the fractional Riccati CF
# =============================================================================
# The classical Heston CF has a Monte Carlo cross-check (test_heston.jl); this
# gives rough Heston the same: simulate the Volterra SDE directly, a code path
# sharing nothing with the fractional Riccati solver or the Fourier pricer.
#
# Dynamics (de-drifted log price X, α = H + 1/2):
#   V_t = V0 + (1/Γ(α)) ∫₀ᵗ (t−s)^{α−1} [ κ(θ − V_s) ds + ξ √V_s dW_s ]
#   dX_t = −V_t/2 dt + √V_t (ρ dW_t + √(1−ρ²) dZ_t)
#
# Discretization — Volterra Euler with a hybrid kernel (Bennedsen, Lunde &
# Pakkanen 2017, κ = 1) on t_i = iΔ, V frozen (and floored at 0) per cell:
#   * drift: the kernel is integrated exactly over each cell,
#       D_k = ∫_{(k−1)Δ}^{kΔ} u^{α−1} du = Δ^α (k^α − (k−1)^α)/α
#   * noise, lag 1: the singular part is simulated EXACTLY — the pair
#       (ΔW_j, I_j = ∫_{t_j}^{t_{j+1}} (t_{j+1} − s)^{α−1} dW_s)
#     is jointly Gaussian with Var ΔW = Δ, Var I = Δ^{2α−1}/(2α−1),
#     Cov = Δ^α/α;
#   * noise, lag ≥ 2: kernel replaced by its cell average, D_k/Δ · ΔW_j.
# At α = 1 every weight collapses to Δ and I_j = ΔW_j: plain full-truncation
# Euler for classical Heston, the scheme test_heston.jl already trusts.
#
# Cost O(n²) per path (the Volterra memory); the pricer uses the forward as a
# control variate (E[S_T] = F exactly), which removes most of the variance of
# near-the-money calls.
#
# KNOWN LIMITATION — the positivity floor. The discretized V can go negative,
# and flooring it (V⁺) biases prices UP. For very rough, high vol-of-vol
# parameters (H = 0.1 with ξ ≳ 0.3 at θ = 0.04) the floor binds so often that
# the bias is many standard errors and decays only slowly with n. Where the
# floor rarely binds (H = 0.1 with ξ = 0.15, or H = 0.2 with ξ = 0.3) the
# scheme matches the Fourier price to within 1σ at n = 400. The tests
# cross-check in that regime; treat MC prices outside it as biased high.
# =============================================================================

_randn(rng) = rng === nothing ? randn() : randn(rng)

"""
    simulate_rough_heston(p::RoughHestonParams, T, nsteps, npaths; rng=nothing)

Simulate `npaths` paths of the de-drifted log return X_T = log(S_T/F) under
rough Heston with the hybrid Volterra-Euler scheme (see the file header).
Returns the vector of X_T. `rng` is any RNG accepted by `randn(rng)`
(`nothing` uses the global RNG). Requires H ∈ (0, 1/2].
"""
function simulate_rough_heston(p::RoughHestonParams, T, nsteps::Int, npaths::Int; rng = nothing)
    0 < p.H <= 0.5 || throw(DomainError(p.H, "simulate_rough_heston needs H ∈ (0, 1/2]"))
    nsteps >= 1 && npaths >= 1 || throw(ArgumentError("need nsteps, npaths ≥ 1"))
    α = p.H + 0.5
    Δ = T / nsteps
    D = [Δ^α * (k^α - (k - 1)^α) / α for k in 1:nsteps]      # D[k]: lag-k kernel mass
    G = gamma(α)
    # Cholesky of Cov(ΔW, I): I = c1·Z1 + c2·Z2 with ΔW = √Δ·Z1.
    varI = Δ^(2α - 1) / (2α - 1)
    c1 = (Δ^α / α) / sqrt(Δ)
    c2 = sqrt(max(varI - c1^2, 0.0))
    sρ = sqrt(1 - p.ρ^2)
    sqΔ = sqrt(Δ)

    X = zeros(npaths)
    Vp = zeros(nsteps)          # V⁺_j, the floored variance per cell
    dW = zeros(nsteps)
    for path in 1:npaths
        x = 0.0
        v = p.v0
        for i in 1:nsteps
            vp = max(v, 0.0)
            z1 = _randn(rng); z2 = _randn(rng); z3 = _randn(rng)
            Vp[i] = vp
            dW[i] = sqΔ * z1
            I = c1 * z1 + c2 * z2
            x += -vp * Δ / 2 + sqrt(vp) * (p.ρ * dW[i] + sρ * sqΔ * z3)
            # V at t_i from the whole history (cells 1..i), lag-1 cell exact.
            acc = p.κ * (p.θ - vp) * D[1] + p.ξ * sqrt(vp) * I
            @inbounds for j in 1:i-1
                k = i - j + 1
                acc += p.κ * (p.θ - Vp[j]) * D[k] + p.ξ * sqrt(Vp[j]) * (D[k] / Δ) * dW[j]
            end
            v = p.v0 + acc / G
        end
        X[path] = x
    end
    return X
end

"""
    rough_heston_mc_price(S, K, r, q, p::RoughHestonParams, T; call=true,
                          nsteps=200, npaths=50_000, rng=nothing)

Monte Carlo price of a European option under rough Heston, with the forward
as control variate. Returns `(price, stderr)`.
"""
function rough_heston_mc_price(S, K, r, q, p::RoughHestonParams, T; call::Bool = true,
                               nsteps::Int = 200, npaths::Int = 50_000, rng = nothing)
    F = S * exp((r - q) * T)
    disc = exp(-r * T)
    X = simulate_rough_heston(p, T, nsteps, npaths; rng = rng)
    ST = F .* exp.(X)
    pay = call ? max.(ST .- K, 0.0) : max.(K .- ST, 0.0)
    # Control variate S_T with known mean F.
    cv = ST .- F
    β = sum((pay .- sum(pay) / npaths) .* cv) / max(sum(abs2, cv), eps())
    adj = pay .- β .* cv
    m = sum(adj) / npaths
    se = sqrt(sum(abs2, adj .- m) / (npaths - 1) / npaths)
    return (price = disc * m, stderr = disc * se)
end
