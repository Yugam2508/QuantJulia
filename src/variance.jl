# =============================================================================
# Variance swaps — model fair strikes and model-free replication
# =============================================================================
# A variance swap pays realized variance minus a strike fixed today; the fair
# strike is the risk-neutral expected realized variance. Two independent
# routes to it, which is what makes it a good calibration diagnostic:
#
#   1. From the MODEL — for continuous-path models (Black-Scholes, Heston,
#      rough Heston), Itô on log S gives
#          E[∫₀ᵀ v_t dt] = −2·E[log(S_T/F)] = −2·c₁,
#      and c₁ = E[X] is read off the characteristic function. So every model
#      in the package gets a variance term structure for free.
#   2. From the MARKET — the same log contract is replicated statically by a
#      strip of out-of-the-money options (Carr & Madan 1998; the VIX recipe):
#          K_var·T = (2/disc)·[ ∫₀^F P(K)/K² dK + ∫_F^∞ C(K)/K² dK ].
#      No model at all — only the chain.
#
# Comparing the two term structures shows where a calibrated model's variance
# LEVEL drifts from the market's, independently of how well it fits the smile.
#
# Jumps: for models with jumps the log contract and realized variance differ
# (a jump of size J contributes J² to realized variance but 2(eᴶ − 1 − J) to
# the log contract). `variance_swap_strike` returns the log-contract value —
# what replication prices and what the VIX measures — and says so.
# =============================================================================

"""
    variance_swap_strike(ψ, T; h=1e-4)

Fair variance strike (annualized, in variance units) of a log-contract
variance swap to maturity `T` under the model with characteristic function
`ψ`: −2·E[log(S_T/F)]/T, with the mean read off ψ by a central difference
of log ψ at 0. Equals expected realized variance for continuous-path models;
for jump models it is the log-contract (VIX-style) value. Type-generic, so
ForwardDiff parameters flow through `ψ`.
"""
function variance_swap_strike(ψ, T; h = 1e-4)
    c1 = imag(log(ψ(h)) - log(ψ(-h))) / (2h)
    return -2 * c1 / T
end

"""
    replicate_variance(Ks, otm, F, disc, T)

Model-free variance strike from a strip of out-of-the-money option prices
`otm` (puts below the forward, calls at or above) at strikes `Ks`, by the
CBOE VIX discretization:

    K_var = (2/T) Σᵢ (ΔKᵢ/Kᵢ²)·otmᵢ/disc − (1/T)(F/K₀ − 1)²,

where ΔKᵢ is the central strike spacing and K₀ the first strike at or below
`F` (the correction term fixes the put/call switch not being exactly at F).
Strikes must be sorted. Truncation: the strip only sees variance between its
lowest and highest strikes, so a narrow strip UNDER-states the strike.
"""
function replicate_variance(Ks, otm, F, disc, T)
    n = length(Ks)
    n >= 3 || throw(ArgumentError("replicate_variance: need at least 3 strikes"))
    issorted(Ks) || throw(ArgumentError("replicate_variance: strikes must be sorted"))
    j0 = findlast(K -> K <= F, Ks)
    j0 === nothing && throw(ArgumentError("replicate_variance: no strike at or below F"))
    s = zero(float(first(otm)))
    for i in 1:n
        ΔK = i == 1 ? Ks[2] - Ks[1] : i == n ? Ks[n] - Ks[n-1] : (Ks[i+1] - Ks[i-1]) / 2
        s += ΔK / Ks[i]^2 * otm[i]
    end
    return 2 / T * s / disc - (F / Ks[j0] - 1)^2 / T
end

"""
    market_variance_term_structure(quotes; minquotes=5)

Model-free variance strikes per expiry from `prepare_chain` quotes (fields
`T, K, F, r, side, mid`): OTM mids replicated by `replicate_variance`.
Returns `(T, kvar, vol)` per expiry, `vol = √kvar` (the VIX-style level).
"""
function market_variance_term_structure(quotes; minquotes::Int = 5)
    byT = Dict{Float64,Vector{Int}}()
    for (i, qt) in enumerate(quotes)
        push!(get!(byT, qt.T, Int[]), i)
    end
    out = @NamedTuple{T::Float64, kvar::Float64, vol::Float64}[]
    for T in sort!(collect(keys(byT)))
        idx = byT[T]
        length(idx) < minquotes && continue
        qs = sort(quotes[idx]; by = q -> q.K)
        q1 = first(qs)
        F = q1.F
        any(q -> q.K <= F, qs) || continue
        kv = replicate_variance([q.K for q in qs], [q.mid for q in qs], F, exp(-q1.r * T), T)
        push!(out, (T = T, kvar = kv, vol = sqrt(max(kv, 0.0))))
    end
    return out
end

"""
    model_vix(make_cf; days=30)

VIX-style index level implied by a model: 100·√(variance strike) at a
`days`-day horizon, where `make_cf(T)` returns the model's CF at maturity T.
"""
model_vix(make_cf; days = 30) = (T = days / 365; 100 * sqrt(variance_swap_strike(make_cf(T), T)))
