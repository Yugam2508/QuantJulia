# =============================================================================
# Identifying H: skew-slope loss term and joint multi-date rough fits
# =============================================================================
# docs/rough_heston.md found H unidentified by the vega-weighted price loss on
# one weekend snapshot: front-end calendar distortions dominate the loss and
# the one thing roughness explains — how ATM skew SCALES with maturity,
# |skew| ~ T^{H−1/2} — is a small part of it. Three remedies are listed there;
# this file implements the two that are model-side (the third, business-time
# day counts, lives in src/cboe.jl as `prepare_chain(...; daycount=:business)`):
#
#   1. A skew-slope term. For each expiry, the model's ATM skew ∂σ/∂k is
#      compared to the market's, both measured as the least-squares slope of
#      implied vol on k = log(K/F) over the same ATM window. The mismatch is
#      scaled by the ATM width σ_atm·√T so it is in implied-vol units, like
#      the price residuals, and `skew_weight` sets the balance.
#   2. Joint fits across snapshot dates. Structural parameters (κ, θ, ξ, ρ, H)
#      are shared; v0 — the state variable — is free per date. Several dates
#      constrain the term-structure shape far more than one.
#
# The skew mismatch needs no extra pricing. The vega-normalized residual is
# already the IV error to first order, r_i ≈ σ_model(K_i) − σ_mkt(K_i), and a
# least-squares slope is linear in the data, so
#   skew_model − skew_mkt ≈ slope of r_i on k_i over the ATM window —
# exactly zero at a perfect fit, no IV inversion inside AD.
#
# `model_atm_skew` (a local ±1% stencil, linearized around σ_atm) is kept for
# reporting a model's skew term structure on its own.
# =============================================================================

const _SKEW_DK = 0.01          # log-moneyness half-width of the model skew stencil

# Spot implied by a group's forward and carry.
_group_spot(g) = g.F * exp(-(g.r - g.q) * g.T)

"""
    market_atm_skew(g; minpts=4)

Model-free ATM skew ∂σ/∂k (k = log K/F) of one `group_quotes` expiry: slope of
the least-squares line of market implied vol on k over
|k| ≤ max(0.02, 1.5·σ_atm·√T) — the same window `scripts/03` uses. Returns
`nothing` when fewer than `minpts` quotes fall in the window.
"""
_skew_window(g) = (win = max(0.02, 1.5 * g.iv_atm * sqrt(g.T));
                   [i for i in eachindex(g.Ks) if abs(log(g.Ks[i] / g.F)) <= win])

function market_atm_skew(g; minpts::Int = 4)
    sel = _skew_window(g)
    length(sel) < minpts && return nothing
    S = _group_spot(g)
    ks = [log(g.Ks[i] / g.F) for i in sel]
    ivs = [implied_vol(g.cmids[i], S, g.Ks[i], g.r, g.q, g.T) for i in sel]
    _, b = _lsq_line(ks, ivs)
    return b
end

"""
    skew_powerlaw_H(Ts, skews)

Fit |skew| = c·T^{H−1/2} by least squares in log-log space. Returns `(H, c)`.
The model-free roughness estimate of `scripts/04`.
"""
function skew_powerlaw_H(Ts, skews)
    a, b = _lsq_line(log.(Ts), log.(abs.(skews)))
    return (H = b + 0.5, c = exp(a))
end

# Linearized model ATM skew from the two stencil call prices c₋, c₊.
function _linear_skew(g, cm, cp)
    S = _group_spot(g)
    σ0 = g.iv_atm
    iv(K, c) = σ0 + (c - bs_price(S, K, g.r, g.q, σ0, g.T)) / bs_vega(S, K, g.r, g.q, σ0, g.T)
    Km, Kp = g.F * exp(-_SKEW_DK), g.F * exp(_SKEW_DK)
    return (iv(Kp, cp) - iv(Km, cm)) / (2 * _SKEW_DK)
end

"""
    model_atm_skew(ψ, g)

Linearized model ATM skew ∂σ/∂k for characteristic function `ψ` on expiry
group `g` (see the file header). AD-safe.
"""
function model_atm_skew(ψ, g)
    Ks = (g.F * exp(-_SKEW_DK), g.F * exp(_SKEW_DK))
    cs = batch_call_prices(ψ, g.F, g.disc, Ks, g.T; iv_hint = g.iv_atm)
    return _linear_skew(g, cs[1], cs[2])
end

"""
    prepare_identification_set(quotes; minpts=4)

Group one snapshot's quotes by expiry and precompute each expiry's skew
stencil. Returns `(groups, skews, stencils)`: `skews[i]` is the market ATM
skew (`nothing` where fewer than `minpts` quotes sit in the ATM window) and
`stencils[i]` the `(idx, w)` pair turning residuals into a skew mismatch,
`Σ w·r[idx]` (the least-squares slope on k), scaled by σ_atm·√T.
"""
function prepare_identification_set(quotes; minpts::Int = 4)
    groups = group_quotes(quotes)
    skews = [market_atm_skew(g; minpts) for g in groups]
    stencils = map(groups) do g
        idx = _skew_window(g)
        length(idx) < minpts && return nothing
        ks = [log(g.Ks[i] / g.F) for i in idx]
        kc = ks .- sum(ks) / length(ks)
        (idx = idx, w = kc ./ sum(abs2, kc) .* (g.iv_atm * sqrt(g.T)))
    end
    return (groups = groups, skews = skews, stencils = stencils)
end

_pack_joint(p::RoughHestonParams, v0s) =
    [log(p.κ), log(p.θ), log(p.ξ), atanh(p.ρ), _logit((p.H - 0.01) / 0.48), log.(v0s)...]

function _unpack_joint(x, D::Int)
    κ, θ, ξ, ρ = exp(x[1]), exp(x[2]), exp(x[3]), tanh(x[4])
    H = 0.01 + 0.48 * _logistic(x[5])
    return [RoughHestonParams(κ, θ, ξ, ρ, exp(x[5+d]), H) for d in 1:D]
end

"""
    rough_joint_loss(x, datasets; skew_weight=0.0, N=96, parts=false)

Loss for a joint rough-Heston fit over snapshot `datasets` (each from
`prepare_identification_set`) at packed parameters
`x = [log κ, log θ, log ξ, atanh ρ, logit H, log v0₁, …, log v0_D]`:

    mean_i r_i²  +  skew_weight · mean_e [ (skew_model − skew_mkt)·σ_atm·√T ]²

with r_i the vega-normalized price residuals and the skew mismatch taken as
the slope of r on k over each expiry's ATM window (see the file header). `parts = true` returns
`(price_mse, skew_mse)` instead of the combined value.
"""
function rough_joint_loss(x, datasets; skew_weight = 0.0, N::Int = 96, parts::Bool = false)
    ps = _unpack_joint(x, length(datasets))
    sp = zero(eltype(x)); np = 0
    ss = zero(eltype(x)); ns = 0
    for (p, ds) in zip(ps, datasets)
        for (g, st) in zip(ds.groups, ds.stencils)
            ψ = make_rough_cf(g.T, p; N = N)
            cs = batch_call_prices(ψ, g.F, g.disc, g.Ks, g.T; iv_hint = g.iv_atm)
            r = (cs .- g.cmids) ./ g.vegas
            sp += sum(abs2, r)
            np += length(r)
            if st !== nothing
                ss += sum(st.w .* r[st.idx])^2
                ns += 1
            end
        end
    end
    mp = sp / np
    ms = ns == 0 ? zero(ss) : ss / ns
    return parts ? (mp, ms) : mp + skew_weight * ms
end

"""
    calibrate_rough_heston_joint(quote_sets; p0=RoughHestonParams(1.5,0.05,0.35,-0.65,0.011,0.12),
                                 skew_weight=0.0, maxiter=60, N=96)

Fit rough Heston jointly to several snapshots: (κ, θ, ξ, ρ, H) shared, v0 per
snapshot (initialized from each snapshot's shortest-expiry ATM variance).
`quote_sets` is a vector of `prepare_chain`-style quote vectors (fields
`T, K, F, r, q, side, mid, iv, vega`); a single snapshot is the D = 1 case,
where `skew_weight > 0` alone adds the skew-slope term.

Returns `(params, rmse, skew_rmse, converged, iterations)`: `params` holds one
`RoughHestonParams` per snapshot; `rmse` is the price-residual IV RMSE and
`skew_rmse` the RMS scaled skew mismatch, both in vol units.
"""
function calibrate_rough_heston_joint(quote_sets;
                                      p0::RoughHestonParams = RoughHestonParams(1.5, 0.05, 0.35, -0.65, 0.011, 0.12),
                                      skew_weight::Real = 0.0, maxiter::Int = 60, N::Int = 96)
    isempty(quote_sets) && throw(ArgumentError("calibrate_rough_heston_joint: no snapshots"))
    any(isempty, quote_sets) && throw(ArgumentError("calibrate_rough_heston_joint: empty snapshot"))
    datasets = [prepare_identification_set(qs) for qs in quote_sets]
    v0s = [first(ds.groups).iv_atm^2 for ds in datasets]
    f(x) = rough_joint_loss(x, datasets; skew_weight = skew_weight, N = N)
    x0 = _pack_joint(p0, v0s)
    od = OnceDifferentiable(f, x0; autodiff = :forward)
    res = optimize(od, x0, LBFGS(), Optim.Options(iterations = maxiter, g_tol = 1e-8))
    x = Optim.minimizer(res)
    mp, ms = rough_joint_loss(x, datasets; N = N, parts = true)
    return (params = _unpack_joint(x, length(datasets)), rmse = sqrt(mp),
            skew_rmse = sqrt(ms), converged = Optim.converged(res),
            iterations = Optim.iterations(res))
end
