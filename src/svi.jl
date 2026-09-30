# =============================================================================
# SVI / SSVI implied-volatility surfaces — the model-free baseline
# =============================================================================
# The Heston fits report an IV RMSE; SVI answers "compared to what?". Raw SVI
# (Gatheral 2004) fits each expiry's smile in total implied variance
# w(k) = σ_iv²(k)·T, k = log(K/F):
#     w(k) = a + b·( ρ(k − m) + √((k − m)² + σ²) )
# Five numbers per expiry fit almost any liquid smile, so an SVI slice's RMSE
# is a floor no one-factor stochastic-vol model should be expected to beat.
#
# SSVI (Gatheral & Jacquier 2014) ties the slices together with three global
# numbers and the ATM total-variance curve θ_t:
#     w(k, θ) = (θ/2)·( 1 + ρφk + √((φk + ρ)² + 1 − ρ²) ),
#     φ(θ) = η / ( θ^γ (1 + θ)^{1−γ} )        (power-law φ)
# which is free of static arbitrage when θ_t is non-decreasing and
# η(1 + |ρ|) ≤ 2, γ ∈ (0, 1/2]. `fit_ssvi` enforces all three through its
# parametrization, so every fitted SSVI surface is arbitrage-free by
# construction.
#
# Arbitrage diagnostics:
#   butterfly — Durrleman's condition g(k) ≥ 0 (density non-negative):
#       g = (1 − k w′/(2w))² − (w′²/4)(1/w + 1/4) + w″/2
#   calendar  — total variance non-decreasing in T at every k.
# =============================================================================

"""
    SVIParams(a, b, ρ, m, σ)

Raw-SVI slice: `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))` in total variance.
"""
struct SVIParams{T<:Real}
    a::T
    b::T
    ρ::T
    m::T
    σ::T
end
SVIParams(a, b, ρ, m, σ) = SVIParams(promote(a, b, ρ, m, σ)...)

"""
    svi_total_variance(p::SVIParams, k)

Total implied variance w(k) = σ_iv²·T at log-moneyness `k`.
"""
svi_total_variance(p::SVIParams, k) = p.a + p.b * (p.ρ * (k - p.m) + sqrt((k - p.m)^2 + p.σ^2))

"""
    svi_iv(p::SVIParams, k, T)

Implied volatility √(w(k)/T).
"""
svi_iv(p::SVIParams, k, T) = sqrt(svi_total_variance(p, k) / T)

"""
    svi_butterfly_g(p::SVIParams, k)

Durrleman's function g(k); the slice is free of butterfly arbitrage iff
g ≥ 0 everywhere.
"""
function svi_butterfly_g(p::SVIParams, k)
    x = k - p.m
    s = sqrt(x^2 + p.σ^2)
    w = svi_total_variance(p, k)
    w1 = p.b * (p.ρ + x / s)
    w2 = p.b * p.σ^2 / s^3
    return (1 - k * w1 / (2w))^2 - (w1^2 / 4) * (1 / w + 1 / 4) + w2 / 2
end

"""
    svi_butterfly_free(p::SVIParams; ks=range(-2, 2, length=801), tol=0)

`true` when Durrleman's g(k) ≥ −tol on the grid `ks` and w > 0 there.
"""
svi_butterfly_free(p::SVIParams; ks = range(-2, 2, length = 801), tol = 0) =
    all(k -> svi_total_variance(p, k) > 0 && svi_butterfly_g(p, k) >= -tol, ks)

# Raw-SVI parametrization for unconstrained fitting: b, σ > 0, |ρ| < 1 and
# a ≥ −bσ√(1−ρ²) (the minimum total variance stays positive).
function _svi_unpack(x)
    b, ρ, m, σ = exp(x[2]), tanh(x[3]), x[4], exp(x[5])
    a = exp(x[1]) - b * σ * sqrt(1 - ρ^2)
    return SVIParams(a, b, ρ, m, σ)
end
_svi_pack(p::SVIParams) = [log(p.a + p.b * p.σ * sqrt(1 - p.ρ^2)), log(p.b),
                           atanh(p.ρ), p.m, log(p.σ)]

"""
    fit_svi(ks, ivs, T; weights=nothing, maxiter=500)

Fit a raw-SVI slice to implied vols `ivs` at log-moneyness `ks` by weighted
least squares on implied vol (so the RMSE is comparable to the Heston IV
RMSEs). Multi-start LBFGS over a small grid of (m, σ) seeds. Returns
`(params, rmse)`; `rmse` is the (weighted) IV RMSE.
"""
function fit_svi(ks, ivs, T; weights = nothing, maxiter::Int = 500)
    length(ks) == length(ivs) || throw(ArgumentError("fit_svi: ks and ivs differ in length"))
    length(ks) >= 5 || throw(ArgumentError("fit_svi: need at least 5 quotes for 5 parameters"))
    ws = weights === nothing ? ones(length(ks)) : collect(float.(weights))
    W = sum(ws)
    f(x) = (p = _svi_unpack(x);
            sum(ws[i] * (svi_iv(p, ks[i], T) - ivs[i])^2 for i in eachindex(ks)) / W)
    wmin = minimum(ivs)^2 * T
    kspan = max(maximum(ks) - minimum(ks), 1e-3)
    best = nothing
    for m0 in (-0.1, 0.0, 0.1) .* kspan, σ0 in (0.05, 0.25) .* kspan
        b0 = max(maximum(ivs)^2 * T - wmin, 1e-4) / kspan
        p0 = SVIParams(0.5wmin, b0, -0.5, m0, σ0)
        od = OnceDifferentiable(f, _svi_pack(p0); autodiff = :forward)
        res = optimize(od, _svi_pack(p0), LBFGS(),
                       Optim.Options(iterations = maxiter, g_tol = 1e-12))
        if best === nothing || Optim.minimum(res) < Optim.minimum(best)
            best = res
        end
    end
    return (params = _svi_unpack(Optim.minimizer(best)), rmse = sqrt(Optim.minimum(best)))
end

# -----------------------------------------------------------------------------
# SSVI
# -----------------------------------------------------------------------------

"""
    SSVIParams(ρ, η, γ)

Global SSVI parameters with power-law φ(θ) = η/(θ^γ (1+θ)^{1−γ}).
"""
struct SSVIParams{T<:Real}
    ρ::T
    η::T
    γ::T
end
SSVIParams(ρ, η, γ) = SSVIParams(promote(ρ, η, γ)...)

_ssvi_φ(p::SSVIParams, θ) = p.η / (θ^p.γ * (1 + θ)^(1 - p.γ))

"""
    ssvi_total_variance(p::SSVIParams, θ, k)

SSVI total variance at ATM total variance `θ` and log-moneyness `k`.
"""
function ssvi_total_variance(p::SSVIParams, θ, k)
    φ = _ssvi_φ(p, θ)
    return θ / 2 * (1 + p.ρ * φ * k + sqrt((φ * k + p.ρ)^2 + 1 - p.ρ^2))
end

"""
    ssvi_slice(p::SSVIParams, θ)

The raw-SVI slice equal to SSVI at ATM total variance θ, so every SVI tool
(`svi_iv`, `svi_butterfly_g`, …) applies to SSVI slices too.
"""
function ssvi_slice(p::SSVIParams, θ)
    φ = _ssvi_φ(p, θ)
    r = sqrt(1 - p.ρ^2)
    return SVIParams(θ / 2 * (1 - p.ρ^2), θ * φ / 2, p.ρ, -p.ρ / φ, r / φ)
end

"""
    ssvi_arbitrage_free(p::SSVIParams, θs)

Gatheral–Jacquier sufficient conditions: θ non-decreasing, γ ∈ (0, 1/2],
η(1 + |ρ|) ≤ 2.
"""
ssvi_arbitrage_free(p::SSVIParams, θs) =
    issorted(θs) && 0 < p.γ <= 0.5 && p.η * (1 + abs(p.ρ)) <= 2 + 1e-12

# Unconstrained ↔ arbitrage-free SSVI: ρ = tanh, γ = logistic/2,
# η = 2·logistic/(1+|ρ|).
function _ssvi_unpack(x)
    ρ = tanh(x[1])
    γ = _logistic(x[3]) / 2
    η = 2 * _logistic(x[2]) / (1 + abs(ρ))
    return SSVIParams(ρ, η, γ)
end
_ssvi_pack(p::SSVIParams) = [atanh(p.ρ), _logit(p.η * (1 + abs(p.ρ)) / 2), _logit(2p.γ)]

"""
    fit_ssvi(slices; p0=SSVIParams(-0.6, 1.0, 0.4), maxiter=500)

Fit SSVI to `slices`, a vector of `(T, ks, ivs, θ)` with `θ` the slice's ATM
total variance (see `atm_total_variance`). The θ curve is made non-decreasing
(running maximum) before fitting, and the parametrization keeps η(1+|ρ|) ≤ 2
and γ ∈ (0, 1/2], so the result is free of static arbitrage by construction.
Returns `(params, θs, rmse)` with `rmse` the IV RMSE over all quotes.
"""
function fit_ssvi(slices; p0::SSVIParams = SSVIParams(-0.6, 1.0, 0.4), maxiter::Int = 500)
    isempty(slices) && throw(ArgumentError("fit_ssvi: no slices"))
    order = sortperm([s.T for s in slices])
    ss = slices[order]
    θs = accumulate(max, [s.θ for s in ss])
    n = sum(s -> length(s.ks), ss)
    function f(x)
        p = _ssvi_unpack(x)
        acc = zero(eltype(x))
        for (s, θ) in zip(ss, θs), (k, iv) in zip(s.ks, s.ivs)
            acc += (sqrt(ssvi_total_variance(p, θ, k) / s.T) - iv)^2
        end
        return acc / n
    end
    x0 = _ssvi_pack(p0)
    od = OnceDifferentiable(f, x0; autodiff = :forward)
    res = optimize(od, x0, LBFGS(), Optim.Options(iterations = maxiter, g_tol = 1e-12))
    return (params = _ssvi_unpack(Optim.minimizer(res)), θs = θs[invperm(order)],
            rmse = sqrt(Optim.minimum(res)))
end

"""
    atm_total_variance(ks, ivs, T)

ATM total variance σ_atm²·T, with σ_atm linearly interpolated at k = 0 from
the two quotes bracketing it (nearest quote if k = 0 is not bracketed).
"""
function atm_total_variance(ks, ivs, T)
    o = sortperm(ks)
    k, v = ks[o], ivs[o]
    j = searchsortedfirst(k, 0.0)
    σ = if j == 1 || j > length(k)
        v[argmin(abs.(k))]
    else
        v[j-1] + (v[j] - v[j-1]) * (0 - k[j-1]) / (k[j] - k[j-1])
    end
    return σ^2 * T
end

"""
    calendar_violations(ws, Ts; ks=range(-1, 1, length=201), tol=1e-10)

Calendar-arbitrage check across slices: `ws[i]` is a function k ↦ total
variance for maturity `Ts[i]`. Returns the `(T_short, T_long, k)` triples
where total variance DEcreases with maturity; empty means calendar-free.
"""
function calendar_violations(ws, Ts; ks = range(-1, 1, length = 201), tol = 1e-10)
    o = sortperm(collect(Ts))
    out = Tuple{Float64,Float64,Float64}[]
    for j in 1:length(o)-1
        i1, i2 = o[j], o[j+1]
        for k in ks
            ws[i1](k) > ws[i2](k) + tol && push!(out, (Ts[i1], Ts[i2], k))
        end
    end
    return out
end

"""
    fit_svi_surface(quotes; minquotes=5)

Fit raw SVI per expiry and SSVI globally to `prepare_chain`-style quotes
(fields `T, K, F, iv`). Returns `(slices, ssvi, svi_rmse, ssvi_rmse,
butterfly_free, calendar_violations)`: `slices` holds
`(T, params, rmse, n)` per expiry, `svi_rmse`/`ssvi_rmse` are pooled IV RMSEs,
`butterfly_free[i]` flags each SVI slice, and `calendar_violations` lists
calendar-arbitrage points between consecutive SVI slices.
"""
function fit_svi_surface(quotes; minquotes::Int = 5)
    byT = Dict{Float64,Vector{Int}}()
    for (i, qt) in enumerate(quotes)
        push!(get!(byT, qt.T, Int[]), i)
    end
    Ts = sort!([T for (T, idx) in byT if length(idx) >= minquotes])
    isempty(Ts) && throw(ArgumentError("fit_svi_surface: no expiry has $minquotes quotes"))
    data = map(Ts) do T
        qs = quotes[byT[T]]
        ks = [log(q.K / q.F) for q in qs]
        ivs = [q.iv for q in qs]
        (T = T, ks = ks, ivs = ivs, θ = atm_total_variance(ks, ivs, T))
    end
    slices = map(data) do d
        fit = fit_svi(d.ks, d.ivs, d.T)
        (T = d.T, params = fit.params, rmse = fit.rmse, n = length(d.ks))
    end
    ssvi = fit_ssvi(data)
    ntot = sum(s -> s.n, slices)
    svi_rmse = sqrt(sum(s -> s.rmse^2 * s.n, slices) / ntot)
    return (slices = slices, ssvi = ssvi, svi_rmse = svi_rmse, ssvi_rmse = ssvi.rmse,
            butterfly_free = [svi_butterfly_free(s.params) for s in slices],
            calendar_violations = calendar_violations(
                [k -> svi_total_variance(s.params, k) for s in slices], Ts))
end
