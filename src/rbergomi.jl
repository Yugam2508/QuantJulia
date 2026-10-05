# =============================================================================
# Rough Bergomi (Bayer, Friz & Gatheral 2016) — hybrid-scheme Monte Carlo
# =============================================================================
# The other canonical rough-volatility model. Where rough Heston keeps an
# affine structure (and so a characteristic function), rough Bergomi models
# the forward variance directly as a lognormal of a Riemann-Liouville fBM:
#   V_t = ξ₀ · exp( η W̃_t − η² t^{2H}/2 ),   W̃_t = √(2H) ∫₀ᵗ (t−s)^{H−½} dW_s,
#   dS_t/S_t = √V_t ( ρ dW_t + √(1−ρ²) dW⊥_t ).
# Var W̃_t = t^{2H}, so E[V_t] = ξ₀ (flat forward variance curve). No CF, no
# Fourier pricing: Monte Carlo is the method, and the model's signature —
# ATM skew ∝ T^{H−½} as T → 0 — is the test.
#
# Volterra process by the hybrid scheme (Bennedsen, Lunde & Pakkanen 2017,
# κ = 1), α = H − ½, on t_i = iΔ:
#   * the nearest cell is simulated EXACTLY: (ΔW_j, I_j = ∫ (t_j − s)^α dW_s)
#     jointly Gaussian, Var I = Δ^{2α+1}/(2α+1), Cov(ΔW, I) = Δ^{α+1}/(α+1);
#   * cells at lag k ≥ 2 use the kernel at the optimal point b_k·Δ,
#       b_k = ((k^{α+1} − (k−1)^{α+1})/(α+1))^{1/α},
#     which makes the kernel's cell integral exact.
# The same two ingredients as the rough Heston Monte Carlo (src/montecarlo.jl).
# Cost O(n²) per path (the Volterra memory).
#
# Log price: left-point Itô, X += −V_{i−1}Δ/2 + √V_{i−1}·(ρΔW_i + √(1−ρ²)ΔW⊥_i)
# — V_{i−1} only uses noise up to t_{i−1}, so the step is adapted.
# =============================================================================

"""
    RBergomiParams(ξ0, η, ρ, H)

Rough Bergomi: flat forward variance `ξ0`, vol-of-vol `η`, spot-vol
correlation `ρ`, Hurst exponent `H ∈ (0, 1/2]`.
"""
struct RBergomiParams{T<:Real}
    ξ0::T
    η::T
    ρ::T
    H::T
end
RBergomiParams(ξ0, η, ρ, H) = RBergomiParams(promote(ξ0, η, ρ, H)...)

"""
    simulate_rbergomi(p::RBergomiParams, T, nsteps, npaths; rng=nothing)

Simulate rough Bergomi with the hybrid scheme. Returns `(X, V)`: the
de-drifted log returns X_T = log(S_T/F) and the terminal variances V_T, one
per path.
"""
function simulate_rbergomi(p::RBergomiParams, T, nsteps::Int, npaths::Int; rng = nothing)
    0 < p.H <= 0.5 || throw(DomainError(p.H, "rough Bergomi needs H ∈ (0, 1/2]"))
    nsteps >= 1 && npaths >= 1 || throw(ArgumentError("need nsteps, npaths ≥ 1"))
    α = p.H - 0.5
    Δ = T / nsteps
    sqΔ = sqrt(Δ)
    # Exact nearest cell: I = c1·Z1 + c2·Z2 with ΔW = √Δ·Z1.
    c1 = (Δ^(α + 1) / (α + 1)) / sqΔ
    varI = Δ^(2α + 1) / (2α + 1)
    c2 = sqrt(max(varI - c1^2, 0.0))
    # Kernel weights for lags k ≥ 2 (index k): (b_k Δ)^α.
    g = zeros(nsteps)
    for k in 2:nsteps
        bk = α == 0 ? 1.0 : ((k^(α + 1) - (k - 1)^(α + 1)) / (α + 1))^(1 / α)
        g[k] = (bk * Δ)^α
    end
    s2H = sqrt(2 * p.H)
    sρ = sqrt(1 - p.ρ^2)
    X = zeros(npaths)
    VT = zeros(npaths)
    dW = zeros(nsteps)
    for path in 1:npaths
        x = 0.0
        v = p.ξ0                                     # V at t = 0
        for i in 1:nsteps
            z1 = _randn(rng); z2 = _randn(rng); z3 = _randn(rng)
            dW[i] = sqΔ * z1
            # price step on [t_{i−1}, t_i] with the left-point variance
            x += -v * Δ / 2 + sqrt(v) * (p.ρ * dW[i] + sρ * sqΔ * z3)
            # Volterra process at t_i: exact nearest cell + hybrid lags
            y = c1 * z1 + c2 * z2
            @inbounds for j in 1:i-1
                y += g[i-j+1] * dW[j]
            end
            t = i * Δ
            v = p.ξ0 * exp(p.η * s2H * y - p.η^2 * t^(2 * p.H) / 2)
        end
        X[path] = x
        VT[path] = v
    end
    return (X = X, V = VT)
end

"""
    rbergomi_mc_prices(S, Ks, r, q, p::RBergomiParams, T; nsteps=200,
                       npaths=50_000, rng=nothing)

Call prices for all strikes `Ks` from one set of rough Bergomi paths, with
the forward as control variate. Returns `(prices, stderrs)`; puts follow by
parity, `P = C − e^{−rT}(F − K)`.
"""
function rbergomi_mc_prices(S, Ks, r, q, p::RBergomiParams, T; nsteps::Int = 200,
                            npaths::Int = 50_000, rng = nothing)
    F = S * exp((r - q) * T)
    disc = exp(-r * T)
    ST = F .* exp.(simulate_rbergomi(p, T, nsteps, npaths; rng).X)
    cv = ST .- F
    vv = max(sum(abs2, cv), eps())
    prices = Float64[]; ses = Float64[]
    for K in Ks
        pay = max.(ST .- K, 0.0)
        β = sum((pay .- sum(pay) / npaths) .* cv) / vv
        adj = pay .- β .* cv
        m = sum(adj) / npaths
        push!(prices, disc * m)
        push!(ses, disc * sqrt(sum(abs2, adj .- m) / (npaths - 1) / npaths))
    end
    return (prices = prices, stderrs = ses)
end
