# =============================================================================
# The H-identifiability study: one function per snapshot
# =============================================================================
# docs/rough_heston.md found H unidentified on one weekend SPX snapshot and
# named three remedies (business-day time, a skew-slope loss term, several
# snapshots). This file turns the full analysis of one CBOE snapshot into a
# single call, so the study runs the moment real data is available
# (scripts/05_identify_H.jl loops it over a folder of snapshots):
#
#   1. prepare the chain under a chosen day count (calendar or business);
#   2. model-free: fit |ATM skew| ∝ T^{H−½} on the short end;
#   3. pick a calibration set (strike window ∝ σ_atm√T, thinned per expiry);
#   4. calibrate rough Heston, optionally with the skew-slope term;
#   5. report the fitted H with its standard error from the calibration
#      Jacobian (src/diagnostics.jl), at the fit's own residual noise level.
# =============================================================================

using Dates: Date, Day, dayofweek, Friday, year

"""
    last_trading_day(d; holidays=nothing)

The latest NYSE trading day on or before `d` (weekends and `nyse_holidays`
skipped). Weekend/holiday CBOE snapshots carry the previous session's close.
"""
function last_trading_day(d::Date; holidays = nothing)
    hs = holidays === nothing ? Set(vcat(nyse_holidays(year(d) - 1), nyse_holidays(year(d)))) :
         Set(holidays)
    while dayofweek(d) > Friday || d in hs
        d -= Day(1)
    end
    return d
end

"""
    select_calibration_set(quotes; per_expiry=15, width=5.0, max_T=Inf)

Calibration subset of `prepare_chain` quotes: per expiry, strikes with
|log(K/F)| ≤ `width`·σ_atm·√T, thinned evenly to at most `per_expiry`;
expiries beyond `max_T` or with fewer than 5 quotes are dropped.
"""
function select_calibration_set(quotes; per_expiry::Int = 15, width = 5.0, max_T = Inf)
    out = eltype(quotes)[]
    for T in sort!(unique(q.T for q in quotes))
        T <= max_T || continue
        qs = [q for q in quotes if q.T == T]
        atm = qs[argmin([abs(log(q.K / q.F)) for q in qs])]
        win = width * atm.iv * sqrt(T)
        qs = sort!([q for q in qs if abs(log(q.K / q.F)) <= win]; by = q -> q.K)
        length(qs) < 5 && continue
        step = max(1, cld(length(qs), per_expiry))
        append!(out, qs[1:step:end])
    end
    return out
end

"""
    identification_study(raw; valuation_date=raw.quote_date, daycount=:calendar,
                         skew_weight=0.0, short_T=0.25, per_expiry=15,
                         p0=RoughHestonParams(1.5, 0.05, 0.35, -0.65, 0.011, 0.12),
                         N=96, maxiter=60)

Full H-identifiability analysis of one snapshot (`read_cboe` output). Returns
`(daycount, skew_weight, H_modelfree, n_skew, H_fit, se_H, rmse, skew_rmse,
params, n_quotes, n_expiries, converged)`: the model-free power-law H from
expiries with T ≤ `short_T`, the calibrated H and its standard error at the
fit's residual noise level, and fit quality in implied-vol units.
"""
function identification_study(raw; valuation_date = raw.quote_date, daycount::Symbol = :calendar,
                              skew_weight = 0.0, short_T = 0.25, per_expiry::Int = 15,
                              p0::RoughHestonParams = RoughHestonParams(1.5, 0.05, 0.35, -0.65, 0.011, 0.12),
                              N::Int = 96, maxiter::Int = 60)
    chain = prepare_chain(raw; valuation_date, daycount)
    isempty(chain.quotes) && throw(ArgumentError("identification_study: no usable quotes"))
    groups = group_quotes(chain.quotes)
    sk = [(g.T, market_atm_skew(g)) for g in groups if g.T <= short_T]
    sk = [(T, s) for (T, s) in sk if s !== nothing && s < 0]
    Hmf = length(sk) >= 3 ? skew_powerlaw_H(first.(sk), last.(sk)).H : NaN
    cal = select_calibration_set(chain.quotes; per_expiry)
    fit = calibrate_rough_heston_joint([cal]; p0, skew_weight, N, maxiter)
    p = fit.params[1]
    d = rough_heston_diagnostics(cal, p; noise = max(fit.rmse, 1e-6), N)
    return (daycount = daycount, skew_weight = skew_weight, H_modelfree = Hmf, n_skew = length(sk),
            H_fit = p.H, se_H = d.stderr[6], rmse = fit.rmse, skew_rmse = fit.skew_rmse,
            params = p, n_quotes = length(cal), n_expiries = length(unique(q.T for q in cal)),
            converged = fit.converged)
end

# =============================================================================
# How biased is the power-law H? The H_eff(H) map
# =============================================================================
# `skew_powerlaw_H` reads H off the slope of log|skew| against log T. In rough
# Heston that slope is H − ½ only as T → 0; at quoted maturities mean
# reversion (κ) bends the skew term structure down and a large vol-of-vol
# relative to spot vol (ξ/√v0) flattens its short end. `powerlaw_bias_map`
# measures this: it prices the model's ATM skew on a given expiry grid for a
# range of true H (all other parameters fixed), applies the market estimator
# window by window, and returns the effective H the estimator would report.
# `invert_powerlaw_H` reads a market H_eff back through that map.
# =============================================================================

"""
    model_window_skew(ψ, T; npts=11, width=1.5, minwin=0.02)

ATM skew ∂σ/∂k of characteristic function `ψ` at maturity `T`, measured like
`market_atm_skew`: the least-squares slope of model implied vol on
k = log(K/F) over |k| ≤ max(`minwin`, `width`·σ_atm·√T), with `npts` equally
spaced strikes. The defaults are the market estimator's window. Scale-free
(F = 1, no discounting).
"""
function model_window_skew(ψ, T; npts::Int = 11, width = 1.5, minwin = 0.02)
    c = batch_call_prices(ψ, 1.0, 1.0, [1.0], T; iv_hint = 0.2)[1]
    σ = implied_vol(c, 1.0, 1.0, 0.0, 0.0, T)
    win = max(minwin, width * σ * sqrt(T))
    ks = collect(range(-win, win; length = npts))
    cs = batch_call_prices(ψ, 1.0, 1.0, exp.(ks), T; iv_hint = σ)
    ivs = [implied_vol(cs[i], 1.0, exp(ks[i]), 0.0, 0.0, T) for i in eachindex(ks)]
    _, b = _lsq_line(ks, ivs)
    return b
end

"""
    powerlaw_bias_map(κ, θ, ξ, ρ, v0, Ts; Hs=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
                      windows=["all" => T -> true], N=192, width=1.5, minwin=0.02)

Effective power-law H (`skew_powerlaw_H` applied to the model's ATM skews on
the expiry grid `Ts`, restricted to each window) for rough Heston with true
roughness `H ∈ Hs` and the other parameters fixed. H = 0.5 uses the classical
Heston CF. `width` and `minwin` set the skew window (`model_window_skew`).
Returns `(Hs, windows, Heff, skews)` with `Heff[i, j]` for `Hs[i]`
and window `j`, and `skews[i]` the model skews on `Ts`.
"""
function powerlaw_bias_map(κ, θ, ξ, ρ, v0, Ts; Hs = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
                           windows = ["all" => T -> true], N::Int = 192, width = 1.5, minwin = 0.02)
    Ts = collect(float.(Ts))
    skews = map(collect(Hs)) do H
        map(Ts) do T
            ψ = H >= 0.5 ? (u -> heston_cf(u, T, HestonParams(κ, θ, ξ, ρ, v0))) :
                make_rough_cf(T, RoughHestonParams(κ, θ, ξ, ρ, v0, H); N)
            model_window_skew(ψ, T; width, minwin)
        end
    end
    Heff = [(m = last(w).(Ts); count(m) >= 3 ? skew_powerlaw_H(Ts[m], s[m]).H : NaN)
            for s in skews, w in windows]
    return (Hs = collect(Hs), windows = first.(windows), Heff = Heff, skews = skews)
end

"""
    invert_powerlaw_H(Heff_market, Hs, Heff_model)

The true H at which the model's effective power-law H equals `Heff_market`,
by linear interpolation along one column of a `powerlaw_bias_map`. Returns
`nothing` when the map is not monotone or the market value lies outside its
range: no H in the model reproduces the observed slope.
"""
function invert_powerlaw_H(Heff_market, Hs, Heff_model)
    d = diff(Heff_model)
    (all(>(0), d) || all(<(0), d)) || return nothing
    lo, hi = extrema(Heff_model)
    lo <= Heff_market <= hi || return nothing
    for i in 1:length(Hs)-1
        a, b = Heff_model[i], Heff_model[i+1]
        if min(a, b) <= Heff_market <= max(a, b)
            return Hs[i] + (Heff_market - a) / (b - a) * (Hs[i+1] - Hs[i])
        end
    end
    return nothing
end
