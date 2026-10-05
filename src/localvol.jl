# =============================================================================
# Local volatility (Dupire) from an implied total-variance surface
# =============================================================================
# Dupire (1994): the unique diffusion dS/S = σ_loc(S, t) dW (in the forward
# measure) that reprices a given surface of European options. In Gatheral's
# total-variance form, with w(k, T) = σ_imp²(k, T)·T and k = log(K/F_T):
#
#     σ_loc²(k, T) = ∂_T w / g(k, T),
#     g = (1 − k w_k/(2w))² − (w_k²/4)(1/w + 1/4) + w_kk/2.
#
# g is exactly Durrleman's butterfly function (src/svi.jl), so the local
# variance exists and is positive precisely where the surface is free of
# butterfly arbitrage (g > 0) and calendar arbitrage (∂_T w ≥ 0). SSVI fits
# from `fit_ssvi` satisfy both by construction, which makes them the natural
# input. Derivatives are taken by ForwardDiff on any `w(k, T)`.
#
# Round trip: `local_vol_mc_prices` simulates the local-vol diffusion and
# prices calls; for a correct σ_loc they reproduce the input surface — the
# defining property of Dupire's construction, and the test.
# =============================================================================

"""
    dupire_local_variance(w, k, T)

Local variance σ_loc²(k, T) from an implied total-variance surface `w(k, T)`
(log-moneyness `k = log(K/F_T)`), by Gatheral's form of Dupire's formula with
derivatives from ForwardDiff. `w` must be generic in both arguments.
"""
function dupire_local_variance(w, k, T)
    W = w(k, T)
    wk(x) = w(x, T)
    w1 = ForwardDiff.derivative(wk, k)
    w2 = ForwardDiff.derivative(x -> ForwardDiff.derivative(wk, x), k)
    wT = ForwardDiff.derivative(t -> w(k, t), T)
    g = (1 - k * w1 / (2W))^2 - (w1^2 / 4) * (1 / W + 1 / 4) + w2 / 2
    return wT / g
end

"""
    ssvi_surface(p::SSVIParams, Ts, θs)

The SSVI total-variance surface as a function `(k, T) -> w`, with the ATM
total variance θ(T) piecewise-linear through `(0, 0)` and the nodes
`(Ts, θs)` (extended beyond the last node with the last slope). `θs` must be
non-decreasing — as `fit_ssvi` returns them — for the surface to be free of
calendar arbitrage.
"""
function ssvi_surface(p::SSVIParams, Ts, θs)
    o = sortperm(collect(Ts))
    tn = [0.0; collect(Float64, Ts)[o]]
    θn = [0.0; collect(Float64, θs)[o]]
    issorted(θn) || throw(ArgumentError("ssvi_surface: θ must be non-decreasing in T"))
    function θ(T)
        j = clamp(searchsortedlast(tn, T), 1, length(tn) - 1)
        slope = (θn[j+1] - θn[j]) / (tn[j+1] - tn[j])
        return θn[j] + slope * (T - tn[j])
    end
    return (k, T) -> ssvi_total_variance(p, θ(T), k)
end

"""
    local_vol_mc_prices(S, Ks, r, q, w, T; nsteps=400, npaths=100_000,
                        kmax=1.5, nk=301, rng=nothing)

Call prices for strikes `Ks` at maturity `T` under the Dupire local-volatility
diffusion of the total-variance surface `w(k, T)`, by Euler Monte Carlo on the
de-drifted log return X_t = log(S_t/F_t) (dX = −σ²/2 dt + σ dW). The local
variance is tabulated once on a `nk`-point grid in k ∈ [−kmax, kmax] at each
step midpoint and interpolated linearly. Forward as control variate. Returns
`(prices, stderrs)`.

Euler's time-discretization bias is O(Δ): on the SSVI test surface it tilts
the smile by about ±6 bps of implied vol at 100 steps and is below 2 bps at
the default 400.
"""
function local_vol_mc_prices(S, Ks, r, q, w, T; nsteps::Int = 400, npaths::Int = 100_000,
                             kmax = 1.5, nk::Int = 301, rng = nothing)
    Δ = T / nsteps
    kg = range(-kmax, kmax, length = nk)
    dk = step(kg)
    # local variance table at step midpoints (avoids T = 0), floored at 0
    tab = [max(dupire_local_variance(w, k, (i - 0.5) * Δ), 0.0) for k in kg, i in 1:nsteps]
    lv(x, i) = begin
        u = clamp((x + kmax) / dk, 0.0, nk - 1.0)
        j = min(floor(Int, u), nk - 2)
        f = u - j
        (1 - f) * tab[j+1, i] + f * tab[j+2, i]
    end
    sqΔ = sqrt(Δ)
    F = S * exp((r - q) * T)
    disc = exp(-r * T)
    ST = zeros(npaths)
    for p in 1:npaths
        x = 0.0
        for i in 1:nsteps
            v = lv(x, i)
            x += -v * Δ / 2 + sqrt(v) * sqΔ * _randn(rng)
        end
        ST[p] = F * exp(x)
    end
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
